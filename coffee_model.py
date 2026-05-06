#!/usr/bin/env python3
"""
COFFEE PROBABILITY PREDICTION MODEL - FINAL COMPLETE VERSION
================================================================
SUPPORTS ANY ZOOM LEVEL (up to 25+):
  - Auto-detects AOI size and uses tiled processing for large areas
  - Memory limit: 4GB per tile, processes in 2000×2000 chunks if larger

TRAINING DATA LOGIC:
  - class=1: Coffee (digitized, confirmed) → POSITIVE samples
  - class=2: Non-coffee but UNSURE (digitized, uncertain) → SKIPPED
  - UNDIGITIZED areas: 100% sure NOT coffee → NEGATIVE samples (balanced 1:1)

PREDICTION (MEMORY-SAFE FOR ANY ZOOM):
  - MBTiles: read_aoi() for small areas, tiled read for large areas
  - GeoTIFF: windowed read
  - On-the-fly batch processing - NO storing all patches
  - NO Gaussian smoothing (memory-safe)
  - Polygon simplification + minimum area filter 1e-8
  - Incremental saving: every 10%
================================================================
"""

import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.features import shapes
from rasterio.windows import Window
from rasterio.coords import BoundingBox
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from shapely.geometry import Polygon, box, Point, shape
from shapely.ops import unary_union
from scipy.ndimage import binary_closing
from scipy.ndimage import zoom as scipy_zoom
import sqlite3, sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import io, gc, os, random, pickle, json
from datetime import datetime
from PIL import Image
import mercantile


# ============================================================================
# 1. MBTILES HANDLER
# ============================================================================

class MBTilesVirtualRaster:
    """MBTiles reader with TMS→XYZ fix and memory-safe AOI reading."""
    
    def __init__(self, mbtile_path):
        self.mbtile_path = mbtile_path
        self.conn = sqlite3.connect(mbtile_path)
        c = self.conn.cursor()
        try:
            c.execute("SELECT name, value FROM metadata")
            self.metadata = dict(c.fetchall())
        except: self.metadata = {}
        c.execute("SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level DESC")
        zs = [r[0] for r in c.fetchall()]
        if not zs: raise ValueError("No tiles")
        self.zoom_level = zs[0]
        self.max_tiles = 2**self.zoom_level
        c.execute("SELECT tile_column, tile_row, tile_data FROM tiles WHERE zoom_level=?", (self.zoom_level,))
        tiles = c.fetchall()
        self.min_col = min(t[0] for t in tiles)
        self.max_col = max(t[0] for t in tiles)
        tmn = min(t[1] for t in tiles)
        tmx = max(t[1] for t in tiles)
        xyz_min = self.max_tiles-1-tmx
        xyz_max = self.max_tiles-1-tmn
        sw = mercantile.bounds(self.min_col, xyz_max, self.zoom_level)
        ne = mercantile.bounds(self.max_col, xyz_min, self.zoom_level)
        self._bounds = BoundingBox(min(sw.west,ne.west), min(sw.south,ne.south), max(sw.east,ne.east), max(sw.north,ne.north))
        if (self._bounds.bottom+self._bounds.top)/2 > 0:
            self._bounds = BoundingBox(self._bounds.left, -self._bounds.top, self._bounds.right, -self._bounds.bottom)
        self.tile_size = 256
        self.width = (self.max_col-self.min_col+1)*self.tile_size
        self.height = (tmx-tmn+1)*self.tile_size
        self.transform = rasterio.transform.from_bounds(self._bounds.left, self._bounds.bottom, self._bounds.right, self._bounds.top, self.width, self.height)
        self.tile_index = {(t[0],t[1]):t[2] for t in tiles}
        self.tms_rows_sorted = sorted(set(t[1] for t in tiles), reverse=True)
        self.crs = rasterio.crs.CRS.from_epsg(4326)
        self.count = 3
        print(f"    ✅ MBTiles: zoom={self.zoom_level}, size={self.width:,}×{self.height:,}")
        print(f"    Bounds: [{self._bounds.left:.6f}, {self._bounds.bottom:.6f}, {self._bounds.right:.6f}, {self._bounds.top:.6f}]")
    
    @property
    def bounds(self): return self._bounds
    
    def _load_tile(self, col, row):
        if (col,row) not in self.tile_index: return None
        try:
            img = Image.open(io.BytesIO(self.tile_index[(col,row)]))
            a = np.array(img)
            if len(a.shape)==2: a = np.stack([a]*3, axis=2)
            elif a.shape[2]>=4: a = a[:,:,:3]
            return np.transpose(a, (2,0,1))
        except: return None
    
    def read_aoi(self, min_lon, min_lat, max_lon, max_lat):
        """Read ONLY tiles covering an AOI - memory-safe for small AOIs."""
        rt, ct = self.index(min_lon, max_lat)
        rb, cb = self.index(max_lon, min_lat)
        cl = max(0, int(ct)); rt = max(0, int(rt))
        cr = min(self.width, int(cb)+1); rb = min(self.height, int(rb)+1)
        aoi_w, aoi_h = cr-cl, rb-rt
        
        # Memory check
        aoi_memory_gb = (aoi_w * aoi_h * 3) / (1024**3)
        if aoi_memory_gb > 4:
            raise MemoryError(f"AOI too large for single read: {aoi_memory_gb:.1f} GB. Use tiled processing.")
        
        data = np.zeros((3, aoi_h, aoi_w), dtype=np.uint8)
        tiles_read = 0
        for ri, tms_row in enumerate(self.tms_rows_sorted):
            tile_y = ri*self.tile_size
            if tile_y+self.tile_size<=rt or tile_y>=rb: continue
            for col in range(self.min_col, self.max_col+1):
                if (col,tms_row) not in self.tile_index: continue
                tile_x = (col-self.min_col)*self.tile_size
                if tile_x+self.tile_size<=cl or tile_x>=cr: continue
                tile = self._load_tile(col, tms_row)
                if tile is None: continue
                x1,y1 = max(cl,tile_x), max(rt,tile_y)
                x2,y2 = min(cr,tile_x+self.tile_size), min(rb,tile_y+self.tile_size)
                if x2>x1 and y2>y1:
                    data[:, y1-rt:y2-rt, x1-cl:x2-cl] = tile[:, y1-tile_y:y2-tile_y, x1-tile_x:x2-tile_x]
                    tiles_read += 1
        print(f"    Tiles read: {tiles_read} (of {len(self.tile_index)} total)")
        return data
    
    def read(self, window=None):
        if window is not None:
            wc,wr,ww,wh = int(window.col_off),int(window.row_off),int(window.width),int(window.height)
            d = np.zeros((3,wh,ww), dtype=np.uint8)
            for ri, row in enumerate(self.tms_rows_sorted):
                ty = ri*self.tile_size
                if ty+self.tile_size<=wr or ty>=wr+wh: continue
                for col in range(self.min_col, self.max_col+1):
                    if (col,row) not in self.tile_index: continue
                    tx = (col-self.min_col)*self.tile_size
                    if tx+self.tile_size<=wc or tx>=wc+ww: continue
                    t = self._load_tile(col,row)
                    if t is None: continue
                    x1,y1 = max(wc,tx), max(wr,ty)
                    x2,y2 = min(wc+ww,tx+self.tile_size), min(wr+wh,ty+self.tile_size)
                    if x2>x1 and y2>y1:
                        d[:, y1-wr:y2-wr, x1-wc:x2-wc] = t[:, y1-ty:y2-ty, x1-tx:x2-tx]
            return d
        d = np.zeros((3,self.height,self.width), dtype=np.uint8)
        for ri, row in enumerate(self.tms_rows_sorted):
            yo = ri*self.tile_size
            for col in range(self.min_col, self.max_col+1):
                if (col,row) not in self.tile_index: continue
                xo = (col-self.min_col)*self.tile_size
                t = self._load_tile(col,row)
                if t is not None:
                    h = min(self.tile_size, self.height-yo)
                    w = min(self.tile_size, self.width-xo)
                    d[:, yo:yo+h, xo:xo+w] = t[:, :h, :w]
        return d
    
    def index(self, x, y): return rowcol(self.transform, x, y)
    def close(self):
        if hasattr(self,'conn'): self.conn.close()
    def __enter__(self): return self
    def __exit__(self,*a): self.close()
    def __del__(self): self.close()


# ============================================================================
# 2. CNN MODEL
# ============================================================================

class DualCoffeeCNN(nn.Module):
    def __init__(self, nc=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2); self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128*4*4,256)
        self.binary_fc = nn.Linear(256+22,2)
        self.cultivation_fc = nn.Linear(256+22,nc)
    def forward(self, x, rf):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool(x)
        x = x.view(x.size(0),-1); x = self.dropout(x); x = F.relu(self.fc1(x))
        c = torch.cat([x,rf], dim=1)
        return self.binary_fc(c), self.cultivation_fc(c)


# ============================================================================
# 3. MAIN MODEL
# ============================================================================

class CoffeeProbabilityModel:
    
    CT = {0:"Monoculture Coffee", 1:"Coffee Mixed Cropping", 2:"Coffee Agroforestry", 3:"Coffee + Agroforestry + Mixed"}
    
    def __init__(self, use_cuda=True):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.binary_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1, class_weight='balanced')
        self.cultivation_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1, class_weight='balanced')
        self.scaler = StandardScaler()
        self.cnn_model = DualCoffeeCNN().to(self.device)
        self.is_trained = False
        self.training_history = {}
        self.ensemble_models = []  # ADDED FOR ENSEMBLE SUPPORT
        print(f"✅ Model on {self.device}")
        print(f"   Training: class=1(coffee), class=2(uncertain→skipped), undigitized(non-coffee)")
        print(f"   Prediction: Auto-tiled for any zoom level (memory-safe)")
    
    def _open_image(self, p):
        p = Path(p)
        if p.suffix.lower() in ['.mbtiles','.mbtile']: return MBTilesVirtualRaster(p)
        return rasterio.open(p)
    
    def _extract_rf_features(self, patch):
        try:
            img = patch.astype(np.float32)
            if img.max()>1.0: img/=255.0
            r,g,b = img[0],img[1],img[2]
            feats = [float(np.mean(r)),float(np.mean(g)),float(np.mean(b)),float(np.std(r)),float(np.std(g)),float(np.std(b)),float(np.median(r)),float(np.median(g)),float(np.median(b)),float(np.percentile(r,25)),float(np.percentile(g,25)),float(np.percentile(b,25))]
            t = r+g+b+1e-10
            feats.extend([float(np.mean((g-r)/(g+r+1e-10))),float(np.mean(2*g/t-r/t-b/t)),float(np.mean((2*g-r-b)/(2*g+r+b+1e-10))),float(np.mean((g-r)/(g+r-b+1e-10)))])
            for bd in [r,g,b]:
                feats.append(float(np.mean(np.abs(np.diff(bd,axis=1)))))
                feats.append(float(np.mean(np.abs(np.diff(bd,axis=0)))))
            return np.array(feats, dtype=np.float32)
        except: return np.zeros(22, dtype=np.float32)
    
    def _extract_patch(self, src, col, row, ps=32):
        h = ps//2; cs = max(0, min(col-h, src.width-ps)); rs = max(0, min(row-h, src.height-ps))
        try:
            p = src.read(window=Window(cs, rs, ps, ps))
            if p.shape[0]==1: p = np.repeat(p,3,axis=0)
            elif p.shape[0]>3: p = p[:3]
            return p
        except: return None
    
    # ==================== TRAINING ====================
    
    def extract_training_data(self, image_path, geopackage_paths, coffee_column, cultivation_column=None):
        print("\n"+"="*60+"\n📊 EXTRACTING TRAINING DATA\n"+"="*60)
        ckpt = Path(image_path).parent/"training_checkpoint.pkl"
        if ckpt.exists():
            if input("Resume? (y/n): ").lower()=='y':
                with open(ckpt,'rb') as f: ch = pickle.load(f)
                print(f"✅ Loaded {ch['samples_extracted']} samples")
                return ch['data']
        src = self._open_image(image_path)
        print(f"   Size: {src.width:,} × {src.height:,} pixels")
        allp = []
        for gp in geopackage_paths:
            gp = Path(gp); gdf = gpd.read_file(gp)
            if gdf.crs and gdf.crs.to_epsg()!=4326: gdf = gdf.to_crs(epsg=4326)
            allp.append(gdf)
        pgdf = pd.concat(allp, ignore_index=True)
        ib = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        oi = [i for i,r in pgdf.iterrows() if r.geometry and r.geometry.is_valid and r.geometry.intersects(ib)]
        if not oi: raise ValueError("NO OVERLAP")
        ov = pgdf.iloc[oi].copy()
        print(f"   ✅ {len(ov)} polygons overlap")
        
        print(f"\n   '{coffee_column}' values:")
        for val, count in ov[coffee_column].value_counts().items():
            label = "Coffee (positive)" if int(float(val))==1 else "Uncertain (skipped)" if int(float(val))==2 else "Other"
            print(f"      {val} → {label}: {count} polygons")
        
        if cultivation_column and cultivation_column in ov.columns:
            coffee_mask = ov[coffee_column]==1
            cult_counts = ov.loc[coffee_mask, cultivation_column].value_counts()
            print(f"\n   '{cultivation_column}' values (coffee only):")
            for val, count in cult_counts.items():
                cult_num = int(float(val))
                if cult_num==5: type_name = "NA (skipped)"
                else: type_name = self.CT.get(cult_num-1, f"Type {val}")
                print(f"      {val} → {type_name}: {count}")
        
        print(f"\n   Extracting coffee samples from class=1 polygons...")
        Xrb,Xcb,Yb = [],[],[]
        Xrc,Xcc,Yc = [],[],[]
        total, se = len(ov), 0
        
        for idx,(_,row) in enumerate(ov.iterrows()):
            g = row.geometry
            if g is None or not g.is_valid: continue
            try: cl = int(float(row[coffee_column]))
            except: cl = 2
            if cl != 1: continue
            coffee_label = 1
            ct = None
            if cultivation_column and cultivation_column in ov.columns:
                try:
                    cv = row[cultivation_column]
                    if pd.notna(cv):
                        cn = int(float(cv))
                        if cn in [1,2,3,4]: ct = cn-1
                except: pass
            pc, at = 0, 0
            mx,my,Mx,My = g.bounds
            while pc<5 and at<100:
                at+=1
                rx,ry = random.uniform(mx,Mx), random.uniform(my,My)
                if not g.contains(Point(rx,ry)): continue
                pr,pc2 = src.index(rx,ry)
                if 0<=pr<src.height and 0<=pc2<src.width:
                    patch = self._extract_patch(src, pc2, pr)
                    if patch is not None and patch.shape==(3,32,32) and patch.max()>0:
                        rf = self._extract_rf_features(patch)
                        Xrb.append(rf); Xcb.append(patch); Yb.append(coffee_label); pc+=1; se+=1
                        if ct is not None: Xrc.append(rf); Xcc.append(patch); Yc.append(ct)
            if (idx+1)%100==0: print(f"   Processed {idx+1}/{total} polygons, {se} samples")
            if (idx+1)%500==0:
                dd = {'binary':(np.array(Xrb,dtype=np.float32),Xcb,np.array(Yb)),'cultivation':None}
                if Xrc: dd['cultivation']=(np.array(Xrc,dtype=np.float32),Xcc,np.array(Yc))
                with open(ckpt,'wb') as f: pickle.dump({'data':dd,'polygons_processed':idx+1,'total_polygons':total,'samples_extracted':se},f)
                print(f"   💾 Checkpoint: {idx+1}/{total}")
        
        coffee_count = len(Yb)
        print(f"   Coffee samples extracted: {coffee_count}")
        
        print(f"\n   Adding negative samples from undigitized areas...")
        print(f"   (Areas outside ALL digitized polygons = definitely NOT coffee)")
        all_digitized = unary_union(ov.geometry)
        buffered = all_digitized.buffer(0.0005)
        neg_target = coffee_count * 2   # CHANGED: 2:1 ratio (was coffee_count)
        neg_added, neg_attempts = 0, 0
        while neg_added<neg_target and neg_attempts<neg_target*10:
            neg_attempts+=1
            rand_lon = random.uniform(src.bounds.left, src.bounds.right)
            rand_lat = random.uniform(src.bounds.bottom, src.bounds.top)
            if buffered.contains(Point(rand_lon, rand_lat)): continue
            pix_row, pix_col = src.index(rand_lon, rand_lat)
            if 0<=pix_row<src.height and 0<=pix_col<src.width:
                patch = self._extract_patch(src, pix_col, pix_row, 32)
                if patch is not None and patch.shape==(3,32,32) and patch.max()>0:
                    rf = self._extract_rf_features(patch)
                    Xrb.append(rf); Xcb.append(patch); Yb.append(0); neg_added+=1; se+=1
            if neg_added%5000==0 and neg_added>0: print(f"      {neg_added}/{neg_target} negative samples...")
        
        print(f"   Negative samples added: {neg_added}")
        src.close()
        if ckpt.exists(): ckpt.unlink()
        Xrb = np.array(Xrb,dtype=np.float32); Yb = np.array(Yb)
        print(f"\n✅ Binary: {len(Xrb)} samples total")
        print(f"      Coffee: {coffee_count}, Non-coffee: {neg_added}")
        res = {'binary':(Xrb,Xcb,Yb),'cultivation':None}
        if Xrc:
            Xrc = np.array(Xrc,dtype=np.float32); Yc = np.array(Yc)
            res['cultivation'] = (Xrc,Xcc,Yc)
            print(f"   Cultivation: {len(Xrc)} samples")
            for i in range(4):
                c = np.sum(Yc==i)
                if c>0: print(f"      {self.CT[i]}: {c}")
        return res
    
    def _train_cnn(self, Xcnn, Xrf, y, epochs, bs, is_binary=True, prefix="m"):
        if len(Xcnn)<10: return 0.0
        ckpt = f"{prefix}_cnn_checkpoint.pth"
        se, ba = 0, 0.0
        if os.path.exists(ckpt):
            ch = torch.load(ckpt); self.cnn_model.load_state_dict(ch['model_state'])
            se = ch['epoch']+1; ba = ch.get('best_acc',0.0)
        ct = []
        for p in Xcnn:
            if p.shape!=(3,32,32): p = scipy_zoom(p,(1,32/p.shape[1],32/p.shape[2]),order=1)
            ct.append(torch.FloatTensor(p))
        rt = [torch.FloatTensor(r) for r in Xrf]
        yt = torch.LongTensor(y)
        n = len(ct); idxs = list(range(n)); random.shuffle(idxs); sp = int(0.8*n)
        tr = TensorDataset(torch.stack([ct[i] for i in idxs[:sp]]),torch.stack([rt[i] for i in idxs[:sp]]),torch.stack([yt[i] for i in idxs[:sp]]))
        vl = TensorDataset(torch.stack([ct[i] for i in idxs[sp:]]),torch.stack([rt[i] for i in idxs[sp:]]),torch.stack([yt[i] for i in idxs[sp:]]))
        tl = DataLoader(tr, batch_size=min(bs,len(tr)), shuffle=True)
        vl2 = DataLoader(vl, batch_size=min(bs,len(vl)), shuffle=False)
        opt = torch.optim.Adam(self.cnn_model.parameters(), lr=0.001)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        cr = nn.CrossEntropyLoss()
        for ep in range(se, epochs):
            self.cnn_model.train(); tlc,tcc,ttc=0,0,0
            for pt,rf,lb in tl:
                pt,rf,lb=pt.to(self.device),rf.to(self.device),lb.to(self.device)
                opt.zero_grad(); bo,co=self.cnn_model(pt,rf); o=bo if is_binary else co
                l=cr(o,lb); l.backward(); opt.step()
                tlc+=l.item(); _,pr=o.max(1); ttc+=lb.size(0); tcc+=pr.eq(lb).sum().item()
            self.cnn_model.eval(); vlc,vcc,vtc=0,0,0
            with torch.no_grad():
                for pt,rf,lb in vl2:
                    pt,rf,lb=pt.to(self.device),rf.to(self.device),lb.to(self.device)
                    bo,co=self.cnn_model(pt,rf); o=bo if is_binary else co
                    vlc+=cr(o,lb).item(); _,pr=o.max(1); vtc+=lb.size(0); vcc+=pr.eq(lb).sum().item()
            ta=100.*tcc/ttc if ttc>0 else 0; va=100.*vcc/vtc if vtc>0 else 0
            sch.step(vlc)
            print(f"      Epoch {ep+1}/{epochs}: Train={ta:.1f}%, Val={va:.1f}%" + (" ✓" if va>ba else ""))
            torch.save({'epoch':ep,'model_state':self.cnn_model.state_dict(),'best_acc':max(ba,va)},ckpt)
            if va>ba: ba=va; torch.save({'epoch':ep,'model_state':self.cnn_model.state_dict(),'best_acc':ba},f"{prefix}_best.pth")
            gc.collect()
            if self.device.type=='cuda': torch.cuda.empty_cache()
        if os.path.exists(ckpt): os.remove(ckpt)
        return ba/100.0
    
    def train(self, image_path, geopackage_paths, coffee_column, cultivation_column=None, epochs=20, batch_size=32):
        print("\n"+"="*60+"\n🌿 TRAINING\n"+"="*60)
        print(f"   class=1: Coffee → positive samples")
        print(f"   class=2: Uncertain → SKIPPED")
        print(f"   Undigitized areas → negative samples (balanced 2:1)")
        data = self.extract_training_data(image_path, geopackage_paths, coffee_column, cultivation_column)
        print("\n🔵 Binary Classifier...")
        Xr,Xc,y = data['binary']; Xrs = self.scaler.fit_transform(Xr)
        self.binary_model.fit(Xrs, y); ra = accuracy_score(y, self.binary_model.predict(Xrs))
        print(f"   RF: {ra:.3f}")
        ca = self._train_cnn(Xc, Xrs, y, epochs, batch_size, True, "binary")
        self.training_history['binary'] = {'rf_accuracy':float(ra),'cnn_accuracy':float(ca)}
        if data['cultivation'] and len(data['cultivation'][0])>10:
            print("\n🟢 Cultivation Classifier...")
            Xrc,Xcc,yc = data['cultivation']; Xrcs = self.scaler.transform(Xrc)
            self.cultivation_model.fit(Xrcs, yc); rca = accuracy_score(yc, self.cultivation_model.predict(Xrcs))
            print(f"   RF: {rca:.3f}")
            for i in range(4):
                cnt = np.sum(yc==i)
                if cnt>0: print(f"      {self.CT[i]}: {cnt} samples")
            cca = self._train_cnn(Xcc, Xrcs, yc, epochs, batch_size, False, "cultivation")
            self.training_history['cultivation'] = {'rf_accuracy':float(rca),'cnn_accuracy':float(cca)}
        self.is_trained = True
        print(f"\n✅ Done | Binary: RF={ra:.3f} CNN={ca:.3f}")
        return self.training_history
    
    # ==================== PREDICTION (AUTO-TILED FOR ANY ZOOM) ====================
    
    def predict(self, image_path, aoi_path, output_path, threshold=0.5, tile_size=256, save_probability_map=True):
        print("\n"+"="*60+"\n🔍 PREDICTING\n"+"="*60)
        if not self.is_trained: raise ValueError("Not trained!")
        
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_file = output_path.parent/f"{output_path.stem}_pred_checkpoint.json"
        
        allp = []
        if ckpt_file.exists():
            with open(ckpt_file) as f: ck = json.load(f)
            print(f"📁 Resuming from {ck.get('percent',0)}%")
            if output_path.exists():
                try: allp = gpd.read_file(output_path).to_dict('records')
                except: pass
        
        print("\n[1/4] Loading AOI...")
        aoi_gdf = gpd.read_file(aoi_path)
        if aoi_gdf.crs and aoi_gdf.crs.to_epsg()!=4326: aoi_gdf = aoi_gdf.to_crs(epsg=4326)
        aoi_geom = unary_union(aoi_gdf.geometry)
        b = aoi_gdf.total_bounds
        mnx,mny,mxx,mxy = b
        print(f"   AOI: [{mnx:.4f},{mny:.4f}] to [{mxx:.4f},{mxy:.4f}]")
        
        print("\n[2/4] Opening image...")
        src = self._open_image(image_path)
        
        print("\n[3/4] Checking AOI size...")
        rt, ct = src.index(mnx, mxy)
        rb, cb = src.index(mxx, mny)
        cl = max(0, int(ct)); rt = max(0, int(rt))
        cr = min(src.width, int(cb)+1); rb = min(src.height, int(rb)+1)
        aoi_w = cr - cl; aoi_h = rb - rt
        aoi_memory_gb = (aoi_w * aoi_h * 3) / (1024**3)
        
        print(f"   AOI pixels: {aoi_w:,} × {aoi_h:,} ({aoi_w*aoi_h/1e6:.1f} MP)")
        print(f"   Memory needed: ~{aoi_memory_gb:.1f} GB")
        
        # DECIDE: Single read or tiled processing
        MAX_MEMORY_GB = 4
        
        if aoi_memory_gb > MAX_MEMORY_GB:
            print(f"   ⚠️ AOI too large for single read (> {MAX_MEMORY_GB} GB)")
            print(f"   Using tiled processing...")
            allp = self._predict_tiled(src, cl, rt, cr, rb, mnx, mny, mxx, mxy, 
                                       aoi_geom, threshold, output_path, ckpt_file)
        else:
            print(f"   ✅ AOI fits in memory")
            print(f"   Reading AOI...")
            is_mbtiles = hasattr(src, 'read_aoi')
            if is_mbtiles:
                img = src.read_aoi(mnx, mny, mxx, mxy)
            else:
                img = src.read(window=Window(cl, rt, aoi_w, aoi_h))
            
            h, w = img.shape[1], img.shape[2]
            atrans = rasterio.transform.from_bounds(mnx, mny, mxx, mxy, w, h)
            print(f"   Cropped: {w}×{h} pixels")
            
            print(f"\n[4/4] Processing (memory-safe, on-the-fly)...")
            allp = self._process_image(img, h, w, atrans, aoi_geom, threshold, output_path, ckpt_file)
        
        src.close()
        
        # Final save
        prob_path = output_path.parent/f"{output_path.stem}_probability.tif"
        res = gpd.GeoDataFrame(allp, crs='EPSG:4326') if allp else gpd.GeoDataFrame()
        if len(res)>0:
            res.to_file(output_path, driver='GPKG')
            print(f"\n✅ Done! {len(res)} polygons → {output_path}")
            if 'cultivation' in res.columns:
                print("   Cultivation types:")
                for n,c in res['cultivation'].value_counts().items(): print(f"      {n}: {c}")
        else:
            print(f"\n⚠️ No coffee detected above threshold {threshold}")
        
        if ckpt_file.exists(): ckpt_file.unlink()
        return res, {'probability_map': str(prob_path)}
    
    def _process_image(self, img, h, w, atrans, aoi_geom, threshold, output_path, ckpt_file):
        """Process a single image array."""
        patch_size, step, batch_size = 32, 16, 64
        
        prob_map = np.zeros((h, w), dtype=np.uint8)
        cult_map = np.zeros((h, w), dtype=np.uint8)
        
        total = ((h - patch_size) // step + 1) * ((w - patch_size) // step + 1)
        print(f"   Patches: {total:,} | Batch: {batch_size}")
        
        batch_patches, batch_positions = [], []
        last_save_pct = 0
        patch_idx = 0
        
        for i in range(0, h - patch_size + 1, step):
            for j in range(0, w - patch_size + 1, step):
                patch = img[:, i:i+patch_size, j:j+patch_size]
                if patch.max() > 0:
                    batch_patches.append(patch)
                    batch_positions.append((i, j))
                
                if len(batch_patches) >= batch_size:
                    self._process_batch(batch_patches, batch_positions, prob_map, cult_map, h, w, step)
                    patch_idx += len(batch_patches)
                    batch_patches, batch_positions = [], []
                    
                    pct = 100 * patch_idx // total
                    if pct >= last_save_pct + 10:
                        last_save_pct = pct - (pct % 10)
                        self._save_progress(prob_map, cult_map, h, w, atrans, aoi_geom, threshold, output_path, ckpt_file, pct, patch_idx, total)
                        print(f"   💾 {pct}% ({patch_idx:,}/{total:,})")
        
        if batch_patches:
            self._process_batch(batch_patches, batch_positions, prob_map, cult_map, h, w, step)
        
        print(f"   Vectorizing...")
        return self._vectorize(prob_map, cult_map, h, w, atrans, aoi_geom, threshold)
    
    def _predict_tiled(self, src, cl, rt, cr, rb, mnx, mny, mxx, mxy, aoi_geom, threshold, output_path, ckpt_file):
        """Process large AOI in 2000×2000 pixel tiles."""
        aoi_w = cr - cl
        aoi_h = rb - rt
        
        TILE_SIZE = 2000
        ntiles_x = max(1, int(np.ceil(aoi_w / TILE_SIZE)))
        ntiles_y = max(1, int(np.ceil(aoi_h / TILE_SIZE)))
        
        print(f"   Tiled: {ntiles_x}×{ntiles_y} = {ntiles_x*ntiles_y} tiles of ~{TILE_SIZE}×{TILE_SIZE}")
        print(f"\n[4/4] Processing tiles...")
        
        allp = []
        tile_count = 0
        total_tiles = ntiles_x * ntiles_y
        
        for ty in range(ntiles_y):
            for tx in range(ntiles_x):
                tile_count += 1
                
                t_cl = cl + tx * TILE_SIZE
                t_rt = rt + ty * TILE_SIZE
                t_cr = min(cr, t_cl + TILE_SIZE)
                t_rb = min(rb, t_rt + TILE_SIZE)
                
                t_mnx = mnx + (mxx - mnx) * (tx / ntiles_x)
                t_mxx = mnx + (mxx - mnx) * ((tx + 1) / ntiles_x)
                t_mxy = mny + (mxy - mny) * (ty / ntiles_y)
                t_mny = mny + (mxy - mny) * ((ty + 1) / ntiles_y)
                
                print(f"   Tile {tile_count}/{total_tiles}: {t_cr-t_cl}×{t_rb-t_rt} pixels")
                
                if hasattr(src, 'read_aoi'):
                    img = src.read_aoi(t_mnx, t_mxy, t_mxx, t_mny)
                else:
                    img = src.read(window=Window(t_cl, t_rt, t_cr-t_cl, t_rb-t_rt))
                
                h, w = img.shape[1], img.shape[2]
                atrans = rasterio.transform.from_bounds(t_mnx, t_mny, t_mxx, t_mxy, w, h)
                
                patch_size, step, batch_size = 32, 16, 64
                prob_map = np.zeros((h, w), dtype=np.uint8)
                cult_map = np.zeros((h, w), dtype=np.uint8)
                
                batch_patches, batch_positions = [], []
                
                for i in range(0, h - patch_size + 1, step):
                    for j in range(0, w - patch_size + 1, step):
                        patch = img[:, i:i+patch_size, j:j+patch_size]
                        if patch.max() > 0:
                            batch_patches.append(patch)
                            batch_positions.append((i, j))
                        
                        if len(batch_patches) >= batch_size:
                            self._process_batch(batch_patches, batch_positions, prob_map, cult_map, h, w, step)
                            batch_patches, batch_positions = [], []
                
                if batch_patches:
                    self._process_batch(batch_patches, batch_positions, prob_map, cult_map, h, w, step)
                
                tile_polys = self._vectorize(prob_map, cult_map, h, w, atrans, aoi_geom, threshold)
                allp.extend(tile_polys)
                
                # Save incrementally
                if tile_count % max(1, total_tiles // 10) == 0:
                    temp_gdf = gpd.GeoDataFrame(allp, crs='EPSG:4326') if allp else gpd.GeoDataFrame()
                    if len(temp_gdf) > 0:
                        temp_gdf.to_file(output_path, driver='GPKG')
                    print(f"      Saved: {len(allp)} polygons ({100*tile_count//total_tiles}%)")
                
                del img, prob_map, cult_map
                gc.collect()
        
        return allp
    
    def _process_batch(self, batch_patches, batch_positions, prob_map, cult_map, h, w, step):
        """Process a batch through CNN+RF."""
        brf = np.array([self._extract_rf_features(p) for p in batch_patches])
        brfs = self.scaler.transform(brf)
        bt = torch.FloatTensor(np.array(batch_patches)).to(self.device)
        brt = torch.FloatTensor(brfs).to(self.device)
        with torch.no_grad():
            bo, co = self.cnn_model(bt, brt)
            cprobs = torch.softmax(bo, dim=1)[:, 1].cpu().numpy()
            cults = torch.argmax(co, dim=1).cpu().numpy()
        rprobs = self.binary_model.predict_proba(brfs)[:, 1]
        fps = (cprobs + rprobs) / 2
        
        for k, (i, j) in enumerate(batch_positions):
            pv = int(fps[k] * 255)
            ie, je = min(i + step, h), min(j + step, w)
            prob_map[i:ie, j:je] = pv
            cult_map[i:ie, j:je] = cults[k]
        
        del brf, brfs, bt, brt, bo, co, cprobs, cults, rprobs, fps
        gc.collect()
    
    def _save_progress(self, prob_map, cult_map, h, w, atrans, aoi_geom, threshold, output_path, ckpt_file, pct, ei, total):
        """Save incremental progress."""
        prob_path = output_path.parent/f"{output_path.stem}_probability.tif"
        with rasterio.open(prob_path,'w',driver='GTiff',height=h,width=w,count=1,dtype=np.uint8,crs='EPSG:4326',transform=atrans,compress='lzw') as dst:
            dst.write(prob_map,1)
        
        coffee_mask = prob_map >= int(threshold*255)
        temp_polys = []
        if np.any(coffee_mask):
            cleaned = binary_closing(coffee_mask, iterations=1)
            for g,v in shapes(cleaned.astype(np.uint8),mask=cleaned,transform=atrans):
                if v==1 and g['type']=='Polygon':
                    poly = shape(g)
                    if poly.area<1e-8: continue
                    cx,cy = poly.centroid.x, poly.centroid.y
                    px,py = ~atrans*(cx,cy); px,py = int(px),int(py)
                    cn = self.CT.get(int(cult_map[py,px]),"Unknown") if 0<=px<w and 0<=py<h else "Unknown"
                    pvl = float(prob_map[py,px])/255.0 if 0<=py<h and 0<=px<w else threshold
                    if not aoi_geom.contains(poly):
                        try:
                            poly = poly.intersection(aoi_geom)
                            if poly.is_empty or poly.area<1e-8: continue
                        except: continue
                    temp_polys.append({'geometry':poly,'probability':round(pvl,3),'cultivation':cn,'area_sq_deg':poly.area})
        
        temp_gdf = gpd.GeoDataFrame(temp_polys,crs='EPSG:4326') if temp_polys else gpd.GeoDataFrame()
        if len(temp_gdf)>0: temp_gdf.to_file(output_path,driver='GPKG')
        with open(ckpt_file,'w') as f:
            json.dump({'percent':pct,'polygons':len(temp_polys),'patches':f"{ei}/{total}",'time':datetime.now().isoformat()},f)
    
    def _vectorize(self, prob_map, cult_map, h, w, atrans, aoi_geom, threshold):
        """Vectorize without smoothing - memory safe."""
        coffee_mask = prob_map >= int(threshold * 255)
        allp = []
        
        if np.any(coffee_mask):
            cleaned = binary_closing(coffee_mask, iterations=1)
            for g, v in shapes(cleaned.astype(np.uint8), mask=cleaned, transform=atrans):
                if v == 1 and g['type'] == 'Polygon':
                    poly = shape(g)
                    poly = poly.simplify(0.00001, preserve_topology=True)
                    if poly.area < 1e-8: continue
                    
                    cx, cy = poly.centroid.x, poly.centroid.y
                    px, py = ~atrans * (cx, cy)
                    px, py = int(px), int(py)
                    
                    if 0 <= px < w and 0 <= py < h:
                        cn = self.CT.get(int(cult_map[py, px]), "Unknown")
                        pv = float(prob_map[py, px]) / 255.0
                    else:
                        cn = "Unknown"
                        pv = float(threshold)
                    
                    if not aoi_geom.contains(poly):
                        try:
                            poly = poly.intersection(aoi_geom)
                            if poly.is_empty or poly.area < 1e-8: continue
                        except: continue
                    
                    allp.append({
                        'geometry': poly,
                        'probability': round(pv, 3),
                        'cultivation': cn,
                        'area_sq_deg': poly.area
                    })
        return allp
    
    # ==================== ENSEMBLE SUPPORT ====================
    
    def load_ensemble(self, model_paths):
        """Load multiple models for ensemble prediction."""
        self.ensemble_models = []
        for path in model_paths:
            print(f"📂 Loading: {Path(path).name}")
            temp = CoffeeProbabilityModel(use_cuda=(self.device.type == 'cuda'))
            temp.load_model(path)
            self.ensemble_models.append((temp.binary_model, temp.cultivation_model,
                                        temp.scaler, temp.cnn_model))
        print(f"✅ {len(self.ensemble_models)} models loaded for ensemble")
    
    def add_ensemble_model(self, path):
        """Add a single model to ensemble."""
        temp = CoffeeProbabilityModel(use_cuda=(self.device.type == 'cuda'))
        temp.load_model(path)
        self.ensemble_models.append((temp.binary_model, temp.cultivation_model,
                                    temp.scaler, temp.cnn_model))
        print(f"✅ Added (total: {len(self.ensemble_models)})")
    
    def clear_ensemble(self):
        """Clear all ensemble models."""
        self.ensemble_models = []
        print("✅ Ensemble cleared")
    
    def predict_ensemble(self, image_path, aoi_path, output_path, threshold=0.5):
        """Ensemble prediction with multiple models."""
        if not self.ensemble_models:
            print("⚠️ No ensemble models loaded. Using single model.")
            return self.predict(image_path, aoi_path, output_path, threshold)
        
        print("\n" + "=" * 60)
        print(f"🔍 ENSEMBLE PREDICTION ({len(self.ensemble_models)} models)")
        print("=" * 60)
        
        # Store original models
        original_binary = self.binary_model
        original_cultivation = self.cultivation_model
        original_scaler = self.scaler
        original_cnn = self.cnn_model
        
        all_predictions = []
        
        for idx, (bin_m, cult_m, scaler_m, cnn_m) in enumerate(self.ensemble_models):
            print(f"\n   Model {idx+1}/{len(self.ensemble_models)}")
            
            # Swap in ensemble model
            self.binary_model = bin_m
            self.cultivation_model = cult_m
            self.scaler = scaler_m
            self.cnn_model = cnn_m
            
            # Predict with this model
            temp_output = output_path.parent / f"{output_path.stem}_model_{idx}.gpkg"
            try:
                res, _ = self.predict(image_path, aoi_path, temp_output, threshold)
                if len(res) > 0:
                    all_predictions.append(res)
            except Exception as e:
                print(f"   ⚠️ Model {idx+1} failed: {e}")
            
            # Clean up temp file
            if temp_output.exists():
                temp_output.unlink()
            
            gc.collect()
        
        # Restore original models
        self.binary_model = original_binary
        self.cultivation_model = original_cultivation
        self.scaler = original_scaler
        self.cnn_model = original_cnn
        
        # Combine predictions
        if all_predictions:
            combined = pd.concat(all_predictions, ignore_index=True)
            combined = combined.drop_duplicates(subset='geometry')
            combined.to_file(output_path, driver='GPKG')
            print(f"\n✅ Ensemble complete! {len(combined)} unique polygons → {output_path}")
            return combined, {}
        else:
            print(f"\n⚠️ No coffee detected by any model")
            return gpd.GeoDataFrame(), {}
    
    # ==================== SAVE / LOAD ====================
    
    def save_model(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'cnn_state_dict':self.cnn_model.state_dict(),'binary_model':self.binary_model,'cultivation_model':self.cultivation_model,'scaler':self.scaler,'is_trained':self.is_trained,'training_history':self.training_history},path)
        print(f"✅ Saved to {path}")
    
    def load_model(self, path):
        path = Path(path)
        if not path.exists(): raise FileNotFoundError(f"Not found: {path}")
        for strategy in ['cpu_full','cpu_simple','safe_weights']:
            try:
                if strategy=='cpu_full': data = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
                elif strategy=='cpu_simple': data = torch.load(path, map_location=torch.device('cpu'))
                else:
                    data_w = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
                    data_f = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
                    self.cnn_model.load_state_dict(data_w['cnn_state_dict'])
                    self.binary_model = data_f['binary_model']
                    self.cultivation_model = data_f['cultivation_model']
                    self.scaler = data_f['scaler']
                    self.is_trained = data_f.get('is_trained',True)
                    self.training_history = data_f.get('training_history',{})
                    self.cnn_model = self.cnn_model.to(self.device)
                    print(f"✅ Loaded!"); return
                self.cnn_model.load_state_dict(data['cnn_state_dict'])
                self.cnn_model = self.cnn_model.to(self.device)
                self.binary_model = data['binary_model']
                self.cultivation_model = data['cultivation_model']
                self.scaler = data['scaler']
                self.is_trained = data.get('is_trained',True)
                self.training_history = data.get('training_history',{})
                print(f"✅ Loaded!"); return
            except: continue
        raise RuntimeError("Failed to load model.")


if __name__=="__main__":
    print("="*60+"\n☕ COFFEE PROBABILITY MODEL\n"+"="*60)
    print("Supports: Any zoom level (auto-tiled for large AOIs)")
    print("Training: class=1(coffee), class=2(skipped), undigitized(non-coffee)")
    print("Prediction: Memory-safe on-the-fly + tiled for large areas")
    print("Ensemble: Load multiple models for combined predictions")
    print("="*60)
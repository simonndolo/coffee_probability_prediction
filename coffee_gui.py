#!/usr/bin/env python3
"""
COFFEE PROBABILITY PREDICTION - COMPLETE GUI WITH ENSEMBLE SUPPORT
================================================================
FEATURES:
  - CRS and coordinate overlap check FIRST
  - Single model training and prediction
  - Multi-model ensemble prediction (2+ models)
  - Handles class=1 (coffee), class=2 (uncertain→skipped)
  - Timing for every major process
  - Checkpoint resume support
  - Robust model loading
================================================================
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from pathlib import Path
import sys
import os
import time
import geopandas as gpd
import rasterio
import sqlite3
import json
from datetime import datetime
from shapely.geometry import box

class CoffeeProbabilityApp:
    def __init__(self):
        self.model = None
        self.ensemble_paths = []
        self.root = tk.Tk()
        self.root.title("🌿 Coffee Probability Prediction System")
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width-100}x{screen_height-100}")
        self.root.minsize(1000, 700)
        
        self.training_image = None
        self.training_polygons = []
        self.available_columns = []
        self.prediction_image = None
        self.prediction_aoi = None
        self.timings = {}
        
        self.bg_color = "#f5f5f5"
        self.root.configure(bg=self.bg_color)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.setup_ui()
    
    def format_time(self, seconds):
        if seconds < 60: return f"{seconds:.0f}s"
        elif seconds < 3600: return f"{seconds/60:.1f}min"
        else: return f"{seconds/3600:.1f}hr"
    
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.grid(row=0, column=0, sticky="nsew")
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Title
        title_frame = tk.Frame(main_container, bg="#2c3e50", height=70)
        title_frame.grid(row=0, column=0, sticky="ew")
        title_frame.grid_propagate(False)
        tk.Label(title_frame, text="🌿 COFFEE PROBABILITY PREDICTION MODEL", 
                font=("Arial", 20, "bold"), bg="#2c3e50", fg="white").pack(pady=(12,3))
        tk.Label(title_frame, text="Single Model + Multi-Model Ensemble | High Accuracy", 
                font=("Arial", 11), bg="#2c3e50", fg="#ecf0f1").pack()
        
        # Scrollable content
        content_frame = tk.Frame(main_container, bg=self.bg_color)
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(content_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.bg_color)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Status bar
        status_frame = tk.Frame(main_container, bg="#34495e", height=45)
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.grid_propagate(False)
        self.status_label = tk.Label(status_frame, text="✅ Ready", font=("Arial", 10), bg="#34495e", fg="white")
        self.status_label.pack(side='left', padx=15, pady=10)
        self.timer_label = tk.Label(status_frame, text="", font=("Arial", 9), bg="#34495e", fg="#f39c12")
        self.timer_label.pack(side='left', padx=15)
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(side='right', padx=15, pady=10)
        
        # Sections
        self.create_training_section()
        self.create_prediction_section()
        self.create_model_section()
        self.create_timing_section()
        self.create_info_section()
        
        tk.Button(self.scrollable_frame, text="Exit Application", command=self.exit_app,
                 font=("Arial", 12, "bold"), bg="#e74c3c", fg="white",
                 padx=50, pady=12, cursor="hand2").pack(pady=20)
        
        self.update_train_button()
        self.update_predict_button()
    
    def create_training_section(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="📚 TRAINING SECTION", 
                             font=("Arial", 14, "bold"), bg=self.bg_color, padx=25, pady=20, fg="#2c3e50")
        frame.pack(fill='x', padx=30, pady=15)
        
        # Image
        f1 = tk.Frame(frame, bg=self.bg_color)
        f1.pack(fill='x', pady=8)
        tk.Label(f1, text="Satellite Image:", font=("Arial", 11, "bold"), bg=self.bg_color, width=18, anchor='w').pack(side='left')
        self.train_img_label = tk.Label(f1, text="No file attached", bg="#ecf0f1", fg="gray", relief='sunken',
                                        width=50, anchor='w', padx=10, font=("Arial", 10))
        self.train_img_label.pack(side='left', padx=10, fill='x', expand=True)
        bf = tk.Frame(f1, bg=self.bg_color); bf.pack(side='left')
        tk.Button(bf, text="📎 Attach", command=self.attach_training_image, bg="#3498db", fg="white",
                 font=("Arial", 10, "bold"), width=10, cursor="hand2").pack(side='left', padx=5)
        tk.Button(bf, text="❌ Clear", command=self.clear_training_image, bg="#e74c3c", fg="white",
                 font=("Arial", 10, "bold"), width=8, cursor="hand2").pack(side='left')
        
        # Polygons
        f2 = tk.Frame(frame, bg=self.bg_color)
        f2.pack(fill='x', pady=8)
        tk.Label(f2, text="Training Polygons:", font=("Arial", 11, "bold"), bg=self.bg_color, width=18, anchor='w').pack(side='left')
        lf = tk.Frame(f2, bg=self.bg_color); lf.pack(side='left', padx=10, fill='x', expand=True)
        self.poly_listbox = tk.Listbox(lf, height=3, width=50, bg="#ecf0f1", font=("Arial", 10))
        self.poly_listbox.pack(side='left', fill='x', expand=True)
        pf = tk.Frame(lf, bg=self.bg_color); pf.pack(side='right', padx=10)
        tk.Button(pf, text="📎 Add", command=self.attach_polygons, bg="#27ae60", fg="white",
                 font=("Arial", 10, "bold"), width=8, cursor="hand2").pack(pady=3)
        tk.Button(pf, text="🗑️ Clear", command=self.clear_polygons, bg="#e74c3c", fg="white",
                 font=("Arial", 10, "bold"), width=8, cursor="hand2").pack(pady=3)
        
        # Columns
        f3 = tk.Frame(frame, bg=self.bg_color)
        f3.pack(fill='x', pady=10)
        tk.Button(f3, text="🔍 Detect Columns", command=self.detect_columns, bg="#3498db", fg="white",
                 font=("Arial", 11, "bold"), width=18, cursor="hand2").pack(side='left', padx=5)
        self.column_info = tk.Label(f3, text="", font=("Arial", 10), bg=self.bg_color, fg="blue")
        self.column_info.pack(side='left', padx=20)
        
        f4 = tk.Frame(frame, bg=self.bg_color)
        f4.pack(fill='x', pady=6)
        tk.Label(f4, text="Coffee Column:", font=("Arial", 11, "bold"), bg=self.bg_color, width=18, anchor='w').pack(side='left')
        self.coffee_column_combo = ttk.Combobox(f4, width=40, state='disabled', font=("Arial", 10))
        self.coffee_column_combo.pack(side='left', padx=10)
        tk.Label(f4, text="(1=coffee, 2=uncertain→skipped)", font=("Arial", 9), bg=self.bg_color, fg="#7f8c8d").pack(side='left')
        
        f5 = tk.Frame(frame, bg=self.bg_color)
        f5.pack(fill='x', pady=6)
        tk.Label(f5, text="Cultivation Column:", font=("Arial", 11, "bold"), bg=self.bg_color, width=18, anchor='w').pack(side='left')
        self.cultivation_column_combo = ttk.Combobox(f5, width=40, state='disabled', font=("Arial", 10))
        self.cultivation_column_combo.pack(side='left', padx=10)
        tk.Label(f5, text="(1=mono, 2=mixed, 3=agro, 4=mixed+agro, 5=NA)", font=("Arial", 9), bg=self.bg_color, fg="#7f8c8d").pack(side='left')
        
        self.train_button = tk.Button(frame, text="🚀 START TRAINING", command=self.train_model,
                                     font=("Arial", 14, "bold"), pady=15, bg="#2c3e50", fg="white",
                                     cursor="hand2", state='disabled', width=40)
        self.train_button.pack(pady=15)
    
    def create_prediction_section(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="🔍 PREDICTION SECTION", 
                             font=("Arial", 14, "bold"), bg=self.bg_color, padx=25, pady=20, fg="#2c3e50")
        frame.pack(fill='x', padx=30, pady=15)
        
        f1 = tk.Frame(frame, bg=self.bg_color)
        f1.pack(fill='x', pady=8)
        tk.Label(f1, text="Satellite Image:", font=("Arial", 11, "bold"), bg=self.bg_color, width=18, anchor='w').pack(side='left')
        self.pred_img_label = tk.Label(f1, text="No file attached", bg="#ecf0f1", fg="gray", relief='sunken',
                                       width=50, anchor='w', padx=10, font=("Arial", 10))
        self.pred_img_label.pack(side='left', padx=10, fill='x', expand=True)
        bf = tk.Frame(f1, bg=self.bg_color); bf.pack(side='left')
        tk.Button(bf, text="📎 Attach", command=self.attach_prediction_image, bg="#3498db", fg="white",
                 font=("Arial", 10, "bold"), width=10, cursor="hand2").pack(side='left', padx=5)
        tk.Button(bf, text="❌ Clear", command=self.clear_prediction_image, bg="#e74c3c", fg="white",
                 font=("Arial", 10, "bold"), width=8, cursor="hand2").pack(side='left')
        
        f2 = tk.Frame(frame, bg=self.bg_color)
        f2.pack(fill='x', pady=8)
        tk.Label(f2, text="AOI Polygon:", font=("Arial", 11, "bold"), bg=self.bg_color, width=18, anchor='w').pack(side='left')
        self.aoi_label = tk.Label(f2, text="No file attached", bg="#ecf0f1", fg="gray", relief='sunken',
                                  width=50, anchor='w', padx=10, font=("Arial", 10))
        self.aoi_label.pack(side='left', padx=10, fill='x', expand=True)
        af = tk.Frame(f2, bg=self.bg_color); af.pack(side='left')
        tk.Button(af, text="📎 Attach", command=self.attach_aoi, bg="#3498db", fg="white",
                 font=("Arial", 10, "bold"), width=10, cursor="hand2").pack(side='left', padx=5)
        tk.Button(af, text="❌ Clear", command=self.clear_aoi, bg="#e74c3c", fg="white",
                 font=("Arial", 10, "bold"), width=8, cursor="hand2").pack(side='left')
        
        f3 = tk.Frame(frame, bg=self.bg_color)
        f3.pack(fill='x', pady=10)
        tk.Label(f3, text="Threshold:", font=("Arial", 11, "bold"), bg=self.bg_color, width=18, anchor='w').pack(side='left')
        self.threshold_var = tk.DoubleVar(value=0.6)
        sf = tk.Frame(f3, bg=self.bg_color); sf.pack(side='left', padx=10)
        tk.Scale(sf, from_=0.0, to=1.0, resolution=0.05, orient='horizontal',
                variable=self.threshold_var, length=300, bg=self.bg_color).pack(side='left')
        self.threshold_label = tk.Label(sf, text="0.60", font=("Arial", 13, "bold"), bg=self.bg_color, width=5, fg="#2c3e50")
        self.threshold_label.pack(side='left', padx=10)
        self.threshold_var.trace('w', lambda *a: self.threshold_label.config(text=f"{self.threshold_var.get():.2f}"))
        
        # Prediction buttons
        f4 = tk.Frame(frame, bg=self.bg_color)
        f4.pack(pady=10)
        self.predict_button = tk.Button(f4, text="🔍 SINGLE MODEL PREDICT", command=self.predict,
                                       font=("Arial", 13, "bold"), pady=12, bg="#2c3e50", fg="white",
                                       cursor="hand2", state='disabled', width=25)
        self.predict_button.pack(side='left', padx=10)
        
        self.ensemble_button = tk.Button(f4, text="🔗 ENSEMBLE PREDICT", command=self.predict_ensemble,
                                        font=("Arial", 13, "bold"), pady=12, bg="#e67e22", fg="white",
                                        cursor="hand2", state='disabled', width=25)
        self.ensemble_button.pack(side='left', padx=10)
    
    def create_model_section(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="💾 MODEL MANAGEMENT", 
                             font=("Arial", 14, "bold"), bg=self.bg_color, padx=25, pady=20, fg="#2c3e50")
        frame.pack(fill='x', padx=30, pady=15)
        
        # Row 1: Save/Load/Clear
        bf1 = tk.Frame(frame, bg=self.bg_color)
        bf1.pack(pady=8)
        tk.Button(bf1, text="💾 Save Model", command=self.save_model, bg="#f39c12", fg="white",
                 font=("Arial", 11, "bold"), pady=10, padx=20, cursor="hand2", width=18).pack(side='left', padx=10)
        tk.Button(bf1, text="📂 Load Model", command=self.load_model, bg="#9b59b6", fg="white",
                 font=("Arial", 11, "bold"), pady=10, padx=20, cursor="hand2", width=18).pack(side='left', padx=10)
        tk.Button(bf1, text="🗑️ Clear Ensemble", command=self.clear_ensemble, bg="#c0392b", fg="white",
                 font=("Arial", 11, "bold"), pady=10, padx=20, cursor="hand2", width=18).pack(side='left', padx=10)
        
        # Row 2: Ensemble load
        bf2 = tk.Frame(frame, bg=self.bg_color)
        bf2.pack(pady=8)
        tk.Button(bf2, text="📂 Load Ensemble Models (2+)", command=self.load_ensemble_models, bg="#2ecc71", fg="white",
                 font=("Arial", 11, "bold"), pady=10, padx=20, cursor="hand2", width=25).pack(side='left', padx=10)
        tk.Button(bf2, text="➕ Add to Ensemble", command=self.add_ensemble_model, bg="#27ae60", fg="white",
                 font=("Arial", 11, "bold"), pady=10, padx=20, cursor="hand2", width=20).pack(side='left', padx=10)
        
        # Status
        self.model_status = tk.Label(frame, text="⚫ No model loaded", font=("Arial", 11), bg=self.bg_color, fg="red")
        self.model_status.pack(pady=10)
        self.ensemble_status = tk.Label(frame, text="", font=("Arial", 10), bg=self.bg_color, fg="#e67e22")
        self.ensemble_status.pack()
    
    def create_timing_section(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="⏱️ PROCESS TIMINGS", 
                             font=("Arial", 13, "bold"), bg="#ecf0f1", padx=25, pady=15, fg="#2c3e50")
        frame.pack(fill='x', padx=30, pady=15)
        self.timing_text = tk.Text(frame, height=6, width=80, font=("Consolas", 10), bg="white", fg="#2c3e50",
                                   relief='sunken', padx=10, pady=10)
        self.timing_text.pack(fill='x')
        self.timing_text.insert('1.0', "No processes run yet.\nTimings will appear here.\n")
        self.timing_text.config(state='disabled')
    
    def add_timing(self, name, seconds):
        self.timings[name] = seconds
        self.timing_text.config(state='normal')
        self.timing_text.insert('end', f"✅ {name}: {self.format_time(seconds)}\n")
        self.timing_text.see('end')
        self.timing_text.config(state='disabled')
        self.timer_label.config(text=f"Last: {name} ({self.format_time(seconds)})")
    
    def create_info_section(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="📖 INFO", font=("Arial", 13, "bold"),
                             bg="#ecf0f1", padx=25, pady=15, fg="#2c3e50")
        frame.pack(fill='x', padx=30, pady=15)
        info = """📋 Schema: class=1(Coffee), class=2(Uncertain→skipped) | cultivation: 1=mono, 2=mixed, 3=agro, 4=agro+mixed, 5=NA
🖼️ Images: MBTiles (.mbtiles) or GeoTIFF (.tif) | Polygons: GeoPackage (.gpkg) or Shapefile (.shp)
🔗 Ensemble: Load 2+ models → average probabilities → fewer false positives
🎯 Default threshold: 0.6 (higher = more accurate, fewer polygons)"""
        tk.Label(frame, text=info, font=("Consolas", 9), justify=tk.LEFT, bg="#ecf0f1", fg="#2c3e50").pack(anchor='w')
    
    # ==================== CRS CHECK ====================
    
    def check_crs_and_overlap(self):
        print("\n" + "="*60)
        print("📍 CRS & COORDINATE OVERLAP CHECK")
        print("="*60)
        try:
            if str(self.training_image).lower().endswith(('.mbtile', '.mbtiles')):
                img_crs = "EPSG:4326"
                conn = sqlite3.connect(self.training_image)
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM metadata WHERE name = 'bounds'")
                result = cursor.fetchone()
                if result:
                    bounds_str = result[0]
                    min_lon, min_lat, max_lon, max_lat = map(float, bounds_str.split(','))
                else:
                    import mercantile
                    cursor.execute("SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level DESC")
                    zoom = cursor.fetchone()[0]
                    cursor.execute("SELECT MIN(tile_column), MAX(tile_column), MIN(tile_row), MAX(tile_row) FROM tiles WHERE zoom_level = ?", (zoom,))
                    min_col, max_col, min_row, max_row = cursor.fetchone()
                    max_tiles = 2**zoom
                    xyz_min_row = max_tiles - 1 - max_row
                    xyz_max_row = max_tiles - 1 - min_row
                    sw = mercantile.bounds(min_col, xyz_max_row, zoom)
                    ne = mercantile.bounds(max_col, xyz_min_row, zoom)
                    min_lon, min_lat = sw.west, sw.south
                    max_lon, max_lat = ne.east, ne.north
                    if (min_lat+max_lat)/2 > 0: min_lat, max_lat = -max_lat, -min_lat
                conn.close()
            else:
                with rasterio.open(self.training_image) as src:
                    img_crs = src.crs.to_string()
                    min_lon, min_lat, max_lon, max_lat = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            print(f"    Image CRS: {img_crs}")
            print(f"    Bounds: L:{min_lon:.4f} R:{max_lon:.4f} B:{min_lat:.4f} T:{max_lat:.4f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read image: {e}")
            return False
        
        try:
            gdf = gpd.read_file(self.training_polygons[0])
            data_crs = gdf.crs.to_string() if gdf.crs else "Unknown"
            if gdf.crs and gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(epsg=4326)
            d = gdf.total_bounds
            print(f"    Data CRS: {data_crs}")
            print(f"    Bounds: L:{d[0]:.4f} R:{d[2]:.4f} B:{d[1]:.4f} T:{d[3]:.4f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read data: {e}")
            return False
        
        print("\n[3] Checking CRS...")
        if "4326" in img_crs and "4326" in data_crs: print("    ✅ Compatible (EPSG:4326)")
        
        print("\n[4] Checking overlap...")
        img_box = box(min_lon, min_lat, max_lon, max_lat)
        data_box = box(d[0], d[1], d[2], d[3])
        
        if not img_box.intersects(data_box):
            messagebox.showerror("❌ NO OVERLAP", "Image and training data do not overlap!")
            return False
        
        overlap_km2 = img_box.intersection(data_box).area * 111 * 111
        print(f"    ✅ OVERLAP: ~{overlap_km2:.0f} km²")
        return True
    
    # ==================== ATTACHMENTS ====================
    
    def attach_training_image(self):
        path = filedialog.askopenfilename(title="Select Training Image",
            filetypes=[("Image files", "*.tif *.tiff *.mbtile *.mbtiles"), ("All files", "*.*")])
        if path:
            self.training_image = path
            self.train_img_label.config(text=Path(path).name[:55], fg="green")
            self.update_train_button()
    
    def clear_training_image(self):
        self.training_image = None
        self.train_img_label.config(text="No file attached", fg="gray")
        self.update_train_button()
    
    def attach_polygons(self):
        paths = filedialog.askopenfilenames(title="Select Training Polygons",
            filetypes=[("GeoPackage", "*.gpkg"), ("Shapefile", "*.shp"), ("All files", "*.*")])
        for p in paths:
            if p not in self.training_polygons:
                self.training_polygons.append(p)
                self.poly_listbox.insert(tk.END, Path(p).name[:55])
        self.update_train_button()
    
    def clear_polygons(self):
        self.poly_listbox.delete(0, tk.END)
        self.training_polygons = []
        self.coffee_column_combo.set('')
        self.cultivation_column_combo.set('')
        self.column_info.config(text="")
        self.update_train_button()
    
    def detect_columns(self):
        if not self.training_polygons:
            messagebox.showwarning("No Files", "Attach polygons first")
            return
        try:
            gdf = gpd.read_file(self.training_polygons[0])
            cols = list(gdf.columns)
            self.coffee_column_combo['values'] = cols
            self.coffee_column_combo['state'] = 'readonly'
            self.cultivation_column_combo['values'] = cols
            self.cultivation_column_combo['state'] = 'readonly'
            self.column_info.config(text=f"✓ {len(cols)} columns", fg="green")
            for c in ['class', 'coffee']:
                if c in cols: self.coffee_column_combo.set(c); break
            for c in ['cultivation_system', 'cultivation']:
                if c in cols: self.cultivation_column_combo.set(c); break
            self.update_train_button()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def update_train_button(self):
        ready = self.training_image and self.training_polygons and self.coffee_column_combo.get()
        self.train_button.config(state='normal' if ready else 'disabled',
                                bg="#27ae60" if ready else "#2c3e50")
    
    def update_predict_button(self):
        ready = self.model and getattr(self.model, 'is_trained', False) and self.prediction_image and self.prediction_aoi
        self.predict_button.config(state='normal' if ready else 'disabled',
                                  bg="#27ae60" if ready else "#2c3e50")
        # Ensemble button
        ens_ready = self.model and self.model.ensemble_models and self.prediction_image and self.prediction_aoi
        self.ensemble_button.config(state='normal' if ens_ready else 'disabled',
                                   bg="#e67e22" if ens_ready else "#2c3e50")
    
    def update_status(self, msg, error=False):
        ts = datetime.now().strftime("%H:%M:%S")
        self.status_label.config(text=f"{'❌' if error else '✅'} [{ts}] {msg}", fg="#e74c3c" if error else "white")
        self.root.update()
    
    def start_progress(self): self.progress.start(10)
    def stop_progress(self): self.progress.stop()
    
    # ==================== TRAINING ====================
    
    def train_model(self):
        if not self.training_image or not self.training_polygons: return
        coffee_col = self.coffee_column_combo.get()
        if not coffee_col: return
        cult_col = self.cultivation_column_combo.get()
        
        self.update_status("Checking CRS...")
        if not self.check_crs_and_overlap(): return
        
        ckpt = Path(self.training_image).parent / "training_checkpoint.pkl"
        resume_msg = "\n\n⚠️ Checkpoint found! Will resume." if ckpt.exists() else ""
        
        if not messagebox.askyesno("Confirm", 
            f"✅ CRS check passed!\n\nTrain with:\n📷 {Path(self.training_image).name}\n"
            f"📊 {len(self.training_polygons)} file(s)\n🏷️ {coffee_col}\n🌿 {cult_col or 'None'}"
            f"{resume_msg}\n\nHigh Accuracy: 20 epochs, 3x negatives\n\nContinue?"): return
        
        self.start_progress()
        self.train_button.config(state='disabled', text="⏳ TRAINING...")
        
        def run():
            t0 = time.time()
            try:
                from coffee_model import CoffeeProbabilityModel
                self.model = CoffeeProbabilityModel(use_cuda=True)
                history = self.model.train(self.training_image, self.training_polygons, coffee_col, cult_col or None)
                tt = time.time() - t0
                
                brf = history.get('binary',{}).get('rf_accuracy',0)
                bcnn = history.get('binary',{}).get('cnn_accuracy',0)
                crf = history.get('cultivation',{}).get('rf_accuracy')
                
                self.model_status.config(text="✅ Trained & ready", fg="green")
                self.update_status(f"Done! {self.format_time(tt)}")
                self.stop_progress()
                self.update_predict_button()
                self.add_timing("Training", tt)
                
                msg = f"✅ Training Complete!\n\n⏱️ {self.format_time(tt)}\n\n📊 Binary:\n  RF: {brf:.1%}\n  CNN: {bcnn:.1%}\n  Best: {max(brf,bcnn):.1%}"
                if crf: msg += f"\n\n🌿 Cultivation:\n  RF: {crf:.1%}"
                msg += "\n\nReady for prediction!"
                messagebox.showinfo("Success", msg)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.update_status(f"Failed: {e}", True)
                self.stop_progress()
                messagebox.showerror("Error", str(e))
            finally:
                self.train_button.config(state='normal', text="🚀 START TRAINING")
                self.update_train_button()
        
        threading.Thread(target=run, daemon=True).start()
    
    # ==================== PREDICTION ====================
    
    def attach_prediction_image(self):
        path = filedialog.askopenfilename(title="Select Prediction Image",
            filetypes=[("Image files", "*.tif *.tiff *.mbtile *.mbtiles"), ("All files", "*.*")])
        if path:
            self.prediction_image = path
            self.pred_img_label.config(text=Path(path).name[:55], fg="green")
            self.update_predict_button()
    
    def clear_prediction_image(self):
        self.prediction_image = None
        self.pred_img_label.config(text="No file attached", fg="gray")
        self.update_predict_button()
    
    def attach_aoi(self):
        path = filedialog.askopenfilename(title="Select AOI",
            filetypes=[("GeoPackage", "*.gpkg"), ("Shapefile", "*.shp"), ("All files", "*.*")])
        if path:
            self.prediction_aoi = path
            self.aoi_label.config(text=Path(path).name[:55], fg="green")
            self.update_predict_button()
    
    def clear_aoi(self):
        self.prediction_aoi = None
        self.aoi_label.config(text="No file attached", fg="gray")
        self.update_predict_button()
    
    def predict(self):
        """Single model prediction."""
        if not self.model or not getattr(self.model, 'is_trained', False):
            messagebox.showerror("Error", "No trained model!")
            return
        if not self.prediction_image or not self.prediction_aoi:
            messagebox.showerror("Error", "Attach image and AOI!")
            return
        
        output = filedialog.asksaveasfilename(title="Save Results", defaultextension=".gpkg",
                                               filetypes=[("GeoPackage", "*.gpkg")])
        if not output: return
        
        self.start_progress()
        self.predict_button.config(state='disabled', text="⏳ PREDICTING...")
        self.ensemble_button.config(state='disabled')
        
        def run():
            t0 = time.time()
            try:
                result, stats = self.model.predict(self.prediction_image, self.prediction_aoi, output, self.threshold_var.get())
                tt = time.time() - t0
                self.update_status(f"Done! {len(result)} polygons in {self.format_time(tt)}")
                self.stop_progress()
                self.add_timing("Single Prediction", tt)
                
                msg = f"✅ Complete!\n\n⏱️ {self.format_time(tt)}\n📍 {len(result)} coffee areas"
                if len(result) > 0 and 'cultivation' in result.columns:
                    msg += "\n\n🌿 Cultivation:\n"
                    for n, c in result['cultivation'].value_counts().items(): msg += f"  {n}: {c}\n"
                msg += f"\n📁 {output}"
                if stats.get('probability_map'): msg += f"\n🗺️ {stats['probability_map']}"
                messagebox.showinfo("Success", msg)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.update_status(f"Failed: {e}", True); self.stop_progress()
                messagebox.showerror("Error", str(e))
            finally:
                self.predict_button.config(state='normal', text="🔍 SINGLE MODEL PREDICT")
                self.update_predict_button()
        
        threading.Thread(target=run, daemon=True).start()
    
    # ==================== ENSEMBLE ====================
    
    def load_ensemble_models(self):
        """Load multiple models for ensemble prediction."""
        paths = filedialog.askopenfilenames(
            title="Select Model Files for Ensemble (2 or more)",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        if paths and len(paths) >= 2:
            self.ensemble_paths = list(paths)
            if not self.model:
                from coffee_model import CoffeeProbabilityModel
                self.model = CoffeeProbabilityModel()
            self.model.load_ensemble(self.ensemble_paths)
            self.ensemble_status.config(text=f"🔗 Ensemble: {len(paths)} models loaded")
            self.model_status.config(text="✅ Ensemble ready", fg="green")
            self.update_status(f"Loaded {len(paths)} models for ensemble")
            self.update_predict_button()
        elif paths:
            messagebox.showwarning("Need More", "Please select at least 2 models for ensemble.")
    
    def add_ensemble_model(self):
        """Add one more model to existing ensemble."""
        path = filedialog.askopenfilename(
            title="Select Model to Add",
            filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")]
        )
        if path:
            if not self.model:
                from coffee_model import CoffeeProbabilityModel
                self.model = CoffeeProbabilityModel()
            self.model.add_ensemble_model(path)
            self.ensemble_paths.append(path)
            self.ensemble_status.config(text=f"🔗 Ensemble: {len(self.ensemble_paths)} models")
            self.model_status.config(text="✅ Ensemble ready", fg="green")
            self.update_predict_button()
    
    def clear_ensemble(self):
        """Clear all ensemble models."""
        if self.model:
            self.model.clear_ensemble()
        self.ensemble_paths = []
        self.ensemble_status.config(text="")
        self.update_predict_button()
        self.update_status("Ensemble cleared")
    
    def predict_ensemble(self):
        """Ensemble prediction using multiple models."""
        if not self.model or not self.model.ensemble_models:
            messagebox.showerror("Error", "Load ensemble models first!\n\nUse 'Load Ensemble Models (2+)' button.")
            return
        if not self.prediction_image or not self.prediction_aoi:
            messagebox.showerror("Error", "Attach image and AOI!")
            return
        
        output = filedialog.asksaveasfilename(title="Save Ensemble Results", defaultextension=".gpkg",
                                               filetypes=[("GeoPackage", "*.gpkg")])
        if not output: return
        
        self.start_progress()
        self.predict_button.config(state='disabled')
        self.ensemble_button.config(state='disabled', text="⏳ ENSEMBLE...")
        
        def run():
            t0 = time.time()
            try:
                result, stats = self.model.predict_ensemble(
                    self.prediction_image, self.prediction_aoi, output, self.threshold_var.get()
                )
                tt = time.time() - t0
                self.update_status(f"Ensemble done! {len(result)} polygons in {self.format_time(tt)}")
                self.stop_progress()
                self.add_timing("Ensemble Prediction", tt)
                
                msg = f"✅ Ensemble Complete!\n\n⏱️ {self.format_time(tt)}\n📍 {len(result)} coffee areas"
                if len(result) > 0 and 'cultivation' in result.columns:
                    msg += "\n\n🌿 Cultivation:\n"
                    for n, c in result['cultivation'].value_counts().items(): msg += f"  {n}: {c}\n"
                msg += f"\n📁 {output}"
                if stats.get('probability_map'): msg += f"\n🗺️ {stats['probability_map']}"
                messagebox.showinfo("Success", msg)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.update_status(f"Failed: {e}", True); self.stop_progress()
                messagebox.showerror("Error", str(e))
            finally:
                self.ensemble_button.config(state='normal', text="🔗 ENSEMBLE PREDICT")
                self.update_predict_button()
        
        threading.Thread(target=run, daemon=True).start()
    
    # ==================== MODEL MANAGEMENT ====================
    
    def save_model(self):
        if not self.model:
            messagebox.showerror("Error", "No model!")
            return
        path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".pth",
                                             filetypes=[("PyTorch model", "*.pth")])
        if path:
            try:
                t0 = time.time()
                self.model.save_model(path)
                tt = time.time() - t0
                self.add_timing("Save", tt)
                messagebox.showinfo("Success", f"✅ Saved!\n\n{path}\n\n{self.format_time(tt)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def load_model(self):
        path = filedialog.askopenfilename(title="Load Model", filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if path:
            self.start_progress()
            def run():
                t0 = time.time()
                try:
                    from coffee_model import CoffeeProbabilityModel
                    self.model = CoffeeProbabilityModel()
                    self.model.load_model(path)
                    tt = time.time() - t0
                    self.model_status.config(text=f"✅ Loaded: {Path(path).name}", fg="green")
                    self.update_status(f"Loaded ({self.format_time(tt)})")
                    self.stop_progress()
                    self.update_predict_button()
                    self.add_timing("Load", tt)
                    messagebox.showinfo("Success", f"✅ Loaded!\n\n{Path(path).name}\n\n{self.format_time(tt)}\n\nReady!")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    self.update_status("Load failed", True); self.stop_progress()
                    messagebox.showerror("Error", f"Failed to load:\n\n{e}\n\nTry retraining.")
            threading.Thread(target=run, daemon=True).start()
    
    def exit_app(self):
        if messagebox.askyesno("Exit", "Exit?"):
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    CoffeeProbabilityApp().run()
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from datetime import datetime
from PIL import Image, ImageTk

from database import DatabaseManager
from model_handler import ModelHandler
from image_processing import enhance_image, generate_thermal_image, highlight_fracture_area
from report_generator import generate_patient_report

class BoneDetectionTab(ttk.Frame):
    def __init__(self, parent, db_manager: DatabaseManager, model_handler: ModelHandler, patient_info=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.model_handler = model_handler
        self.patient_info = patient_info
        
        self.original_image_path = None
        self.current_image_path = None
        self.highlighted_image_path = None
        self.thermal_image_path = None

        self.test_images_dir = os.path.join(os.path.dirname(__file__), 'test_images')
        os.makedirs(self.test_images_dir, exist_ok=True)
        self.test_image_files = [
            ("X-ray-1.png", "X-Ray Image 1"),
            ("MRI-1.png", "MRI Image 1")
        ]

        # Diagnosis result storage for report generation
        self.current_fracture_type = None
        self.current_confidence = None
        self.current_severity = None
        self.current_description = None
        self.current_treatment = None

        self._setup_patient_info_sync()
        self.setup_ui()

    def setup_ui(self):
        # Create canvas with scrollbars for responsive margin scaling
        self.canvas = tk.Canvas(self, bg="#f8f9fa", highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.scrollable_frame = ttk.Frame(self.canvas, padding=15)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure main grid inside scrollable frame (Left Panel & Right Panel)
        self.scrollable_frame.columnconfigure(0, weight=6)
        self.scrollable_frame.columnconfigure(1, weight=4)
        self.scrollable_frame.rowconfigure(0, weight=1)

        # Left Container
        self.left_frame = ttk.Frame(self.scrollable_frame)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15), pady=5)

        # Right Container
        self.right_frame = ttk.Frame(self.scrollable_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

        # --- LEFT PANEL CONTENTS ---
        self.setup_patient_info_section(self.left_frame)
        self.setup_test_images_section(self.left_frame)
        self.setup_control_buttons_section(self.left_frame)
        self.setup_image_previews_section(self.left_frame)
        self.setup_history_section(self.left_frame)

        # --- RIGHT PANEL CONTENTS ---
        self.setup_results_section(self.right_frame)

    def setup_test_images_section(self, parent):
        test_frame = ttk.LabelFrame(parent, text=" 🧪 Sample Test Images ", padding=10)
        test_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(test_frame, text="Select sample image:").pack(side=tk.LEFT, padx=5)
        
        self.test_combo = ttk.Combobox(test_frame, state="readonly", width=25)
        self.test_combo['values'] = [name for name, _ in self.test_image_files]
        if self.test_image_files:
            self.test_combo.current(0)
        self.test_combo.pack(side=tk.LEFT, padx=5)

        self.btn_load_sample = ttk.Button(test_frame, text="📥 Load Sample", command=self.load_test_image)
        self.btn_load_sample.pack(side=tk.LEFT, padx=5)

    def _setup_patient_info_sync(self):
        """Setup synchronization with shared patient info model."""
        if self.patient_info:
            self.patient_info.add_callback(self._on_patient_info_changed)
            # Initialize fields with current values
            self.after(100, self._sync_fields_from_model)

    def _on_patient_info_changed(self, name, age, gender):
        """Callback when patient info changes in another tab."""
        self.ent_patient_name.delete(0, tk.END)
        self.ent_patient_name.insert(0, name)
        self.ent_patient_age.delete(0, tk.END)
        self.ent_patient_age.insert(0, age)
        self.combo_gender.set(gender)

    def _sync_fields_from_model(self):
        """Sync entry fields from patient info model."""
        if self.patient_info:
            self.ent_patient_name.delete(0, tk.END)
            self.ent_patient_name.insert(0, self.patient_info.name)
            self.ent_patient_age.delete(0, tk.END)
            self.ent_patient_age.insert(0, self.patient_info.age)
            self.combo_gender.set(self.patient_info.gender)

    def _on_name_changed(self, event=None):
        if self.patient_info:
            self.patient_info.name = self.ent_patient_name.get()

    def _on_age_changed(self, event=None):
        if self.patient_info:
            self.patient_info.age = self.ent_patient_age.get()

    def _on_gender_changed(self, event=None):
        if self.patient_info:
            self.patient_info.gender = self.combo_gender.get()

    def setup_patient_info_section(self, parent):
        patient_frame = ttk.LabelFrame(parent, text=" 👤 Patient Information ", padding=10)
        patient_frame.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Name
        name_row = ttk.Frame(patient_frame)
        name_row.pack(fill=tk.X, pady=2)
        ttk.Label(name_row, text="Patient Name:", width=14).pack(side=tk.LEFT)
        self.ent_patient_name = ttk.Entry(name_row, width=30)
        self.ent_patient_name.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.ent_patient_name.bind('<KeyRelease>', self._on_name_changed)
        self.ent_patient_name.bind('<FocusOut>', self._on_name_changed)

        # Row 2: Age and Gender
        age_gender_row = ttk.Frame(patient_frame)
        age_gender_row.pack(fill=tk.X, pady=2)
        ttk.Label(age_gender_row, text="Age:", width=14).pack(side=tk.LEFT)
        self.ent_patient_age = ttk.Entry(age_gender_row, width=10)
        self.ent_patient_age.pack(side=tk.LEFT, padx=5)
        self.ent_patient_age.bind('<KeyRelease>', self._on_age_changed)
        self.ent_patient_age.bind('<FocusOut>', self._on_age_changed)
        ttk.Label(age_gender_row, text="Gender:", width=8).pack(side=tk.LEFT, padx=(10, 0))
        self.combo_gender = ttk.Combobox(age_gender_row, state="readonly", width=12, values=["Male", "Female", "Other"])
        self.combo_gender.current(0)
        self.combo_gender.pack(side=tk.LEFT, padx=5)
        self.combo_gender.bind('<<ComboboxSelected>>', self._on_gender_changed)

    def setup_control_buttons_section(self, parent):
        controls_frame = ttk.LabelFrame(parent, text=" ⚙️ Controls & Operations ", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        btn_grid = ttk.Frame(controls_frame)
        btn_grid.pack(fill=tk.X)

        self.btn_upload = ttk.Button(btn_grid, text="📂 Upload X-Ray/MRI", command=self.upload_image)
        self.btn_upload.pack(side=tk.LEFT, padx=4, pady=4)

        self.btn_enhance = ttk.Button(btn_grid, text="✨ Enhance Contrast", command=self.on_enhance_click)
        self.btn_enhance.pack(side=tk.LEFT, padx=4, pady=4)

        self.btn_detect = ttk.Button(btn_grid, text="🔬 Detect Fracture", command=self.on_detect_click)
        self.btn_detect.pack(side=tk.LEFT, padx=4, pady=4)

        self.btn_thermal = ttk.Button(btn_grid, text="🔥 Thermal View", command=self.on_thermal_click)
        self.btn_thermal.pack(side=tk.LEFT, padx=4, pady=4)

        self.btn_report = ttk.Button(btn_grid, text="📄 Generate PDF Report", command=self.on_generate_report)
        self.btn_report.pack(side=tk.LEFT, padx=4, pady=4)

        self.btn_clear = ttk.Button(btn_grid, text="🧹 Clear All", command=self.clear_all)
        self.btn_clear.pack(side=tk.LEFT, padx=4, pady=4)

        # Processing Status Indicator Banner
        self.status_banner = tk.Label(
            controls_frame, 
            text=" READY FOR SCAN ANALYSIS ", 
            font=('Segoe UI', 10, 'bold'),
            bg="#e2e8f0", 
            fg="#334155",
            pady=4
        )
        self.status_banner.pack(fill=tk.X, pady=(6, 0))

    def setup_image_previews_section(self, parent):
        preview_frame = ttk.LabelFrame(parent, text=" 🖼️ Multi-Pane Image Workstation (Click Image to Enlarge) ", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.columnconfigure(2, weight=1)

        # 1. Original Pane
        pane1 = ttk.LabelFrame(preview_frame, text="Original Image", padding=5)
        pane1.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        self.lbl_original = tk.Label(pane1, text="No Image Loaded\n(Click Upload)", bg="#1e293b", fg="#94a3b8", width=32, height=14, cursor="hand2")
        self.lbl_original.pack(fill=tk.BOTH, expand=True)
        self.lbl_original.bind("<Button-1>", lambda e: self.on_image_click("original"))

        # 2. Highlighted Fracture Pane
        pane2 = ttk.LabelFrame(preview_frame, text="Fracture Highlighted", padding=5)
        pane2.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        self.lbl_highlighted = tk.Label(pane2, text="Analysis Pending\n(Click 'Detect Fracture')", bg="#1e293b", fg="#94a3b8", width=32, height=14, cursor="hand2")
        self.lbl_highlighted.pack(fill=tk.BOTH, expand=True)
        self.lbl_highlighted.bind("<Button-1>", lambda e: self.on_image_click("highlighted"))

        # 3. Thermal View Pane
        pane3 = ttk.LabelFrame(preview_frame, text="Thermal Intensity View", padding=5)
        pane3.grid(row=0, column=2, sticky="nsew", padx=3, pady=3)
        self.lbl_thermal = tk.Label(pane3, text="Thermal View Pending\n(Click 'Thermal View')", bg="#1e293b", fg="#94a3b8", width=32, height=14, cursor="hand2")
        self.lbl_thermal.pack(fill=tk.BOTH, expand=True)
        self.lbl_thermal.bind("<Button-1>", lambda e: self.on_image_click("thermal"))

    def setup_history_section(self, parent):
        history_frame = ttk.LabelFrame(parent, text=" 📊 Recent Scans History ", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True)

        tree_frame = ttk.Frame(history_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        y_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        x_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

        self.history_tree = ttk.Treeview(
            tree_frame,
            columns=("Date", "Patient", "Age", "Gender", "Type", "Severity"),
            show="headings",
            height=5,
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
        )

        y_scroll.config(command=self.history_tree.yview)
        x_scroll.config(command=self.history_tree.xview)

        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.history_tree.heading("Date", text="Timestamp")
        self.history_tree.heading("Patient", text="Patient Name")
        self.history_tree.heading("Age", text="Age")
        self.history_tree.heading("Gender", text="Gender")
        self.history_tree.heading("Type", text="Fracture Type")
        self.history_tree.heading("Severity", text="Severity Level")

        self.history_tree.column("Date", width=150, anchor="center")
        self.history_tree.column("Patient", width=120, anchor="center")
        self.history_tree.column("Age", width=50, anchor="center")
        self.history_tree.column("Gender", width=70, anchor="center")
        self.history_tree.column("Type", width=120, anchor="center")
        self.history_tree.column("Severity", width=130, anchor="center")

        self.update_history()

    def setup_results_section(self, parent):
        results_frame = ttk.LabelFrame(parent, text=" 📝 AI Diagnostic Output ", padding=12)
        results_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(results_frame, text="Primary Classification Result:", font=('Segoe UI', 10, 'bold')).pack(anchor="w", pady=(0, 2))
        
        self.results_text = tk.Text(results_frame, height=6, font=('Consolas', 10), bg="#f1f3f5", relief="solid", bd=1)
        self.results_text.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(results_frame, text="Clinical Details & Recommended Care:", font=('Segoe UI', 10, 'bold')).pack(anchor="w", pady=(0, 2))
        
        self.detail_text = tk.Text(results_frame, height=20, font=('Segoe UI', 10), bg="#ffffff", relief="solid", bd=1)
        self.detail_text.pack(fill=tk.BOTH, expand=True)

    def prepare_image_for_display(self, path, target_size=(350, 300)):
        """Trim empty outer black borders and scale image to fill target_size cleanly."""
        if not path or not os.path.exists(path):
            return None

        img = Image.open(path).convert("RGB")

        # Trim empty black borders if present
        gray = img.convert("L")
        bbox = gray.getbbox()
        if bbox:
            w, h = img.size
            left = max(0, bbox[0] - 5)
            top = max(0, bbox[1] - 5)
            right = min(w, bbox[2] + 5)
            bottom = min(h, bbox[3] + 5)
            if (right - left) > 20 and (bottom - top) > 20:
                img = img.crop((left, top, right, bottom))

        # Scale image to fit target_size
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        return img

    def display_image_on_label(self, path, label, target_box=(350, 300)):
        """Renders image centered inside a crisp background canvas."""
        if not path or not os.path.exists(path):
            label.configure(image=None, bg="#1e293b")
            label.image = None
            label.photo = None
            return

        try:
            img = self.prepare_image_for_display(path, target_box)
            if img is None:
                return

            canvas_box = Image.new("RGB", target_box, (30, 41, 59))
            offset_x = (target_box[0] - img.width) // 2
            offset_y = (target_box[1] - img.height) // 2
            canvas_box.paste(img, (offset_x, offset_y))
            
            photo = ImageTk.PhotoImage(canvas_box)
            label.configure(image=photo, text="", bg="#0f172a")
            label.image = photo
            label.photo = photo
        except Exception as e:
            label.configure(image=None, text="Failed to render image", bg="#7f1d1d", fg="#fca5a5")
            label.image = None
            label.photo = None

    def on_image_click(self, view_type):
        """Triggers full-screen viewer or auto-generates view if missing."""
        if view_type == "original":
            if self.original_image_path and os.path.exists(self.original_image_path):
                self.open_fullscreen_viewer(self.original_image_path, "Original Image Scan")
            else:
                messagebox.showinfo("Notice", "Please upload an X-Ray or MRI image first.")

        elif view_type == "highlighted":
            if self.highlighted_image_path and os.path.exists(self.highlighted_image_path):
                self.open_fullscreen_viewer(self.highlighted_image_path, "Fracture Highlighted Analysis")
            elif self.current_image_path:
                self.on_detect_click()
            else:
                messagebox.showinfo("Notice", "Please upload an image first, then click 'Detect Fracture'.")

        elif view_type == "thermal":
            if self.thermal_image_path and os.path.exists(self.thermal_image_path):
                self.open_fullscreen_viewer(self.thermal_image_path, "Thermal Intensity Heatmap")
            elif self.original_image_path:
                self.on_thermal_click()
            else:
                messagebox.showinfo("Notice", "Please upload an image first, then click 'Thermal View'.")

    def open_fullscreen_viewer(self, image_path, title_name):
        """Opens high-resolution interactive viewer modal with zoom controls."""
        if not image_path or not os.path.exists(image_path):
            messagebox.showinfo("Notice", f"No {title_name} available to view yet.")
            return

        viewer = tk.Toplevel(self)
        viewer.title(f"🔍 High-Resolution Medical Viewer - {title_name}")
        viewer.geometry("960x820")
        viewer.configure(bg="#0f172a")
        viewer.lift()
        viewer.focus_force()

        # Top Bar (Fixed: Use ttk.Frame for proper padding support)
        top_bar = ttk.Frame(viewer, padding=10)
        top_bar.pack(fill=tk.X)

        ttk.Label(top_bar, text=f"🔍 CLINICAL VIEWER: {title_name.upper()}", font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=5)

        # Zoom Controls
        zoom_frame = ttk.Frame(top_bar)
        zoom_frame.pack(side=tk.LEFT, padx=20)

        # Main Image Label inside viewer
        img_container = ttk.Frame(viewer, padding=15)
        img_container.pack(fill=tk.BOTH, expand=True)

        lbl_full_image = tk.Label(img_container, bg="#0f172a", bd=2, relief="solid")
        lbl_full_image.pack(fill=tk.BOTH, expand=True)

        # Zoom State Management
        state = {"zoom_factor": 1.0}

        def render_viewer_image():
            try:
                base_img = self.prepare_image_for_display(image_path, target_size=(900, 700))
                if base_img is None:
                    return

                zf = state["zoom_factor"]
                new_w = max(50, int(base_img.width * zf))
                new_h = max(50, int(base_img.height * zf))
                zoomed_img = base_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Canvas container
                box_w = max(920, new_w + 40)
                box_h = max(700, new_h + 40)
                canvas_box = Image.new("RGB", (box_w, box_h), (15, 23, 42))
                ox = (box_w - new_w) // 2
                oy = (box_h - new_h) // 2
                canvas_box.paste(zoomed_img, (ox, oy))

                photo = ImageTk.PhotoImage(canvas_box)
                lbl_full_image.configure(image=photo, text="", bg="#0f172a")
                lbl_full_image.image = photo
                lbl_full_image.photo = photo
                viewer.photo = photo
            except Exception as e:
                lbl_full_image.configure(text=f"Error rendering image: {str(e)}", fg="#ef4444")

        def zoom_in():
            if state["zoom_factor"] < 3.0:
                state["zoom_factor"] += 0.25
                render_viewer_image()

        def zoom_out():
            if state["zoom_factor"] > 0.5:
                state["zoom_factor"] -= 0.25
                render_viewer_image()

        def zoom_reset():
            state["zoom_factor"] = 1.0
            render_viewer_image()

        ttk.Button(zoom_frame, text="🔍+ Zoom In", command=zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔍- Zoom Out", command=zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔄 Reset Fit", command=zoom_reset).pack(side=tk.LEFT, padx=2)

        ttk.Button(top_bar, text="❌ Close Viewer", command=viewer.destroy).pack(side=tk.RIGHT)

        # Initial render
        render_viewer_image()

    def set_processing_state(self, is_processing, message="PROCESSING SCAN... PLEASE WAIT"):
        """Updates UI visual state during background model inference."""
        all_buttons = [self.btn_upload, self.btn_enhance, self.btn_detect, self.btn_thermal, self.btn_clear, self.btn_load_sample]
        
        if is_processing:
            for btn in all_buttons:
                btn.configure(state="disabled")
            self.status_banner.configure(
                text=f" ⏳ {message.upper()} ", 
                bg="#f59e0b", 
                fg="#ffffff"
            )
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"⏳ PROCESSING IN PROGRESS:\n{message}\n\n[1/3] Loading matrix...\n[2/3] Extracting DenseNet169 features...\n[3/3] Calculating diagnosis...")
            self.config(cursor="watch")
            self.update_idletasks()
        else:
            for btn in all_buttons:
                btn.configure(state="normal")
            self.status_banner.configure(
                text=" ✅ PROCESSING COMPLETE - DIAGNOSIS READY ", 
                bg="#10b981", 
                fg="#ffffff"
            )
            self.config(cursor="")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.dcm")]
        )
        if file_path:
            self.load_image_file(file_path)

    def load_test_image(self):
        idx = self.test_combo.current()
        if idx >= 0:
            file_name, _ = self.test_image_files[idx]
            file_path = os.path.join(self.test_images_dir, file_name)
            if os.path.exists(file_path):
                self.load_image_file(file_path)
            else:
                messagebox.showerror("File Error", f"Sample image file not found: {file_name}")

    def load_image_file(self, file_path):
        self.original_image_path = file_path
        self.current_image_path = file_path
        self.highlighted_image_path = None
        self.thermal_image_path = None

        self.display_image_on_label(file_path, self.lbl_original)
        self.lbl_highlighted.configure(image=None, text="Analysis Pending\n(Click 'Detect Fracture')", bg="#1e293b")
        self.lbl_highlighted.image = None
        self.lbl_thermal.configure(image=None, text="Thermal View Pending\n(Click 'Thermal View')", bg="#1e293b")
        self.lbl_thermal.image = None

        self.status_banner.configure(text=" 📂 IMAGE LOADED - CLICK 'DETECT FRACTURE' OR 'THERMAL VIEW' ", bg="#3b82f6", fg="#ffffff")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Image loaded: {os.path.basename(file_path)}\nStatus: Ready for fracture classification or thermal heat mapping.")
        self.detail_text.delete(1.0, tk.END)

    def on_enhance_click(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload or select an image first.")
            return

        self.set_processing_state(True, "Enhancing Contrast with Adaptive Histogram Equalization")
        
        def thread_target():
            try:
                enhanced_path = enhance_image(self.current_image_path)
                self.current_image_path = enhanced_path
                self.after(0, lambda: self.finish_enhance(enhanced_path))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Enhancement failed: {str(e)}"))
                self.after(0, lambda: self.set_processing_state(False))

        threading.Thread(target=thread_target, daemon=True).start()

    def finish_enhance(self, enhanced_path):
        self.display_image_on_label(enhanced_path, self.lbl_original)
        self.set_processing_state(False)
        messagebox.showinfo("Success", "Contrast enhanced successfully!")

    def on_detect_click(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload or select an image first.")
            return

        patient_name = self.ent_patient_name.get().strip()
        patient_age = self.ent_patient_age.get().strip()
        patient_gender = self.combo_gender.get()

        if not patient_name:
            messagebox.showwarning("Warning", "Please enter patient name before detection.")
            return

        if not patient_age.isdigit():
            messagebox.showwarning("Warning", "Please enter a valid age before detection.")
            return

        self.set_processing_state(True, "Analyzing Scan with DenseNet169 Deep Learning Model")

        def thread_target():
            try:
                # Highlight region
                hl_path = highlight_fracture_area(self.current_image_path)
                self.highlighted_image_path = hl_path

                # Model prediction
                fracture_type, confidence, info = self.model_handler.predict(self.current_image_path)

                # Save DB with patient info
                self.db_manager.save_scan_result(
                    patient_name, int(patient_age), patient_gender,
                    fracture_type, confidence, info['severity'],
                    self.current_image_path, hl_path, self.thermal_image_path
                )

                # UI Update on main thread
                self.after(0, lambda: self.update_detection_ui(fracture_type, confidence, info, hl_path))
            except Exception as e:
                self.after(0, lambda: self.set_processing_state(False))
                self.after(0, lambda: messagebox.showerror("Analysis Error", f"Detection failed: {str(e)}"))

        threading.Thread(target=thread_target, daemon=True).start()

    def update_detection_ui(self, fracture_type, confidence, info, hl_path):
        self.display_image_on_label(hl_path, self.lbl_highlighted)

        # Store diagnosis results for report generation
        self.current_fracture_type = fracture_type
        self.current_confidence = confidence
        self.current_severity = info['severity']
        self.current_description = info['description']
        self.current_treatment = info['treatment']

        self.results_text.delete(1.0, tk.END)
        res_str = (
            f"STATUS: Diagnosis Complete\n"
            f"CLASSIFICATION: {fracture_type.upper()}\n"
            f"CONFIDENCE: {confidence:.2f}%\n"
            f"SEVERITY: {info['severity']}"
        )
        self.results_text.insert(tk.END, res_str)

        self.detail_text.delete(1.0, tk.END)
        det_str = (
            f"📌 DESCRIPTION:\n{info['description']}\n\n"
            f"🚨 SEVERITY LEVEL:\n{info['severity']}\n\n"
            f"💊 RECOMMENDED TREATMENT PLAN:\n{info['treatment']}\n\n"
            "📋 CLINICAL ADVISORY NOTES:\n"
            "• Immediate orthopaedic surgeon consultation advised for severe/compound cases.\n"
            "• Radiographic follow-up recommended in 2-4 weeks to monitor healing progress.\n"
            "• Immobilize joint above and below fracture zone."
        )
        self.detail_text.insert(tk.END, det_str)

        self.update_history()
        self.set_processing_state(False)

    def on_thermal_click(self):
        if not self.original_image_path:
            messagebox.showwarning("Warning", "Please upload or select an image first.")
            return

        self.set_processing_state(True, "Generating Thermal Heatmap & Intensity Gradient Map")

        def thread_target():
            try:
                th_path = generate_thermal_image(self.original_image_path)
                self.thermal_image_path = th_path
                self.after(0, lambda: self.finish_thermal(th_path))
            except Exception as e:
                self.after(0, lambda: self.set_processing_state(False))
                self.after(0, lambda: messagebox.showerror("Error", f"Thermal generation failed: {str(e)}"))

        threading.Thread(target=thread_target, daemon=True).start()

    def finish_thermal(self, th_path):
        self.display_image_on_label(th_path, self.lbl_thermal)
        self.set_processing_state(False)
        self.open_fullscreen_viewer(th_path, "Thermal Intensity Heatmap")

    def on_generate_report(self):
        if not self.original_image_path:
            messagebox.showwarning("Warning", "Please upload or select an image first.")
            return

        if not self.current_fracture_type:
            messagebox.showwarning("Warning", "Please run fracture detection first to generate a diagnosis.")
            return

        patient_name = self.ent_patient_name.get().strip()
        patient_age = self.ent_patient_age.get().strip()
        patient_gender = self.combo_gender.get()

        if not patient_name:
            messagebox.showwarning("Warning", "Please enter patient name.")
            return

        if not patient_age.isdigit():
            messagebox.showwarning("Warning", "Please enter a valid age.")
            return

        patient_age = int(patient_age)

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Report", "*.pdf")],
            initialfile=f"MediVision_Report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        if not save_path:
            return

        self.set_processing_state(True, "Generating Professional Hospital-Style PDF Report")

        def thread_target():
            try:
                generate_patient_report(
                    patient_name=patient_name,
                    patient_age=patient_age,
                    patient_gender=patient_gender,
                    fracture_type=self.current_fracture_type,
                    confidence=self.current_confidence,
                    severity=self.current_severity,
                    description=self.current_description,
                    treatment=self.current_treatment,
                    original_image_path=self.original_image_path,
                    highlighted_image_path=self.highlighted_image_path,
                    thermal_image_path=self.thermal_image_path,
                    output_path=save_path
                )
                self.after(0, lambda: self.finish_report_generation(save_path))
            except Exception as e:
                self.after(0, lambda: self.set_processing_state(False))
                self.after(0, lambda: messagebox.showerror("Report Error", f"Failed to generate report: {str(e)}"))

        threading.Thread(target=thread_target, daemon=True).start()

    def finish_report_generation(self, save_path):
        self.set_processing_state(False)
        messagebox.showinfo("Success", f"Professional hospital-style PDF report generated successfully!\n\nSaved to: {save_path}")

    def update_history(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        rows = self.db_manager.fetch_recent_scans(limit=10)
        for row in rows:
            self.history_tree.insert('', tk.END, values=row)

    def clear_all(self):
        self.original_image_path = None
        self.current_image_path = None
        self.highlighted_image_path = None
        self.thermal_image_path = None

        self.lbl_original.configure(image=None, text="No Image Loaded\n(Click Upload)", bg="#1e293b")
        self.lbl_original.image = None
        self.lbl_original.photo = None
        self.lbl_highlighted.configure(image=None, text="Analysis Pending\n(Click 'Detect Fracture')", bg="#1e293b")
        self.lbl_highlighted.image = None
        self.lbl_highlighted.photo = None
        self.lbl_thermal.configure(image=None, text="Thermal View Pending\n(Click 'Thermal View')", bg="#1e293b")
        self.lbl_thermal.image = None
        self.lbl_thermal.photo = None

        self.status_banner.configure(text=" READY FOR SCAN ANALYSIS ", bg="#e2e8f0", fg="#334155")
        self.results_text.delete(1.0, tk.END)
        self.detail_text.delete(1.0, tk.END)
        messagebox.showinfo("Cleared", "All workspace views cleared.")

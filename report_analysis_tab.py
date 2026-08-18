import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from datetime import datetime
from report_analyzer import MedicalReportAnalyzer
from report_generator import generate_report_analysis_report, generate_master_report

class ReportAnalysisTab(ttk.Frame):
    def __init__(self, parent, patient_info=None):
        super().__init__(parent)
        self.analyzer = MedicalReportAnalyzer()
        self.current_report_path = None
        self.current_summary = None
        self.patient_info = patient_info
        self._setup_patient_info_sync()
        self.setup_ui()

    def setup_ui(self):
        # Create scrollable canvas for perfect margin & dynamic layout scaling
        self.canvas = tk.Canvas(self, bg="#f8f9fa", highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas, padding=20)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollable_frame.columnconfigure(0, weight=1)

        # Header Title Banner
        header_frame = ttk.Frame(self.scrollable_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text="📄 Smart Medical Report Summarizer & AI Clinical Insights", font=('Segoe UI', 14, 'bold'))
        title_label.pack(anchor="w")
        subtitle_label = ttk.Label(header_frame, text="Upload patient lab results, MRI/CT text reports, or clinical notes for instant AI summaries, key diagnosis, and risk alerts.", font=('Segoe UI', 10), foreground="#6c757d")
        subtitle_label.pack(anchor="w", pady=(2, 0))

        # --- PATIENT INFO SECTION ---
        patient_frame = ttk.LabelFrame(self.scrollable_frame, text=" 👤 Patient Information ", padding=15)
        patient_frame.pack(fill=tk.X, pady=(0, 15))

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

    def _setup_patient_info_sync(self):
        """Setup synchronization with shared patient info model."""
        if self.patient_info:
            self.patient_info.add_callback(self._on_patient_info_changed)
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

        # --- SECTION 1: REPORT UPLOAD & API CONFIG ---
        config_frame = ttk.LabelFrame(self.scrollable_frame, text=" 📤 Upload Document & AI Settings ", padding=15)
        config_frame.pack(fill=tk.X, pady=(0, 15))

        # File Select Line
        file_row = ttk.Frame(config_frame)
        file_row.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_row, text="📁 Choose Medical File (PDF/TXT/Img)", command=self.select_report_file).pack(side=tk.LEFT, padx=(0, 10))
        self.lbl_file_status = ttk.Label(file_row, text="No document selected", font=('Segoe UI', 10, 'italic'), foreground="#495057")
        self.lbl_file_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # API Key Row
        api_row = ttk.Frame(config_frame)
        api_row.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(api_row, text="Gemini API Key (Optional):", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 8))
        self.ent_api_key = ttk.Entry(api_row, show="*", width=45)
        self.ent_api_key.pack(side=tk.LEFT, padx=(0, 10))
        
        env_key = os.getenv("GEMINI_API_KEY", "")
        if env_key:
            self.ent_api_key.insert(0, env_key)
            
        ttk.Label(api_row, text="(Leave blank for offline diagnostic summary)", font=('Segoe UI', 9), foreground="#6c757d").pack(side=tk.LEFT)

        # Notes Row
        notes_row = ttk.Frame(config_frame)
        notes_row.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(notes_row, text="Patient Notes / Symptoms:").pack(anchor="w", pady=(0, 4))
        self.txt_patient_notes = tk.Text(notes_row, height=3, font=('Segoe UI', 10), bg="#ffffff", relief="solid", bd=1)
        self.txt_patient_notes.pack(fill=tk.X)

        # Action Buttons
        btn_row = ttk.Frame(config_frame)
        btn_row.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(btn_row, text="⚡ Generate AI Summary & Key Info", command=self.start_analysis).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="📄 Generate PDF Report", command=self.on_generate_report).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="🧹 Clear", command=self.clear_fields).pack(side=tk.LEFT)

        # --- SECTION 2: AI SUMMARY & IMPORTANT INFORMATION OUTPUT ---
        output_frame = ttk.LabelFrame(self.scrollable_frame, text=" 📊 Clinical Summary & Important Medical Information ", padding=15)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_summary_output = tk.Text(output_frame, height=22, font=('Segoe UI', 10), bg="#ffffff", fg="#212529", relief="solid", bd=1, wrap="word", padx=10, pady=10)
        self.txt_summary_output.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Save Report Button
        export_row = ttk.Frame(output_frame)
        export_row.pack(fill=tk.X)
        ttk.Button(export_row, text="💾 Save Summary to Text File", command=self.save_summary).pack(side=tk.RIGHT)

    def select_report_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("All Supported Reports", "*.pdf *.txt *.jpg *.jpeg *.png *.bmp"),
                ("PDF Documents", "*.pdf"),
                ("Text Files", "*.txt"),
                ("Image Reports", "*.jpg *.jpeg *.png *.bmp")
            ]
        )
        if path:
            self.current_report_path = path
            self.lbl_file_status.configure(text=f"Selected: {os.path.basename(path)}", foreground="#198754")

    def start_analysis(self):
        if not self.current_report_path:
            messagebox.showwarning("Warning", "Please select a medical report file first.")
            return

        api_key = self.ent_api_key.get().strip()
        self.analyzer.set_api_key(api_key)

        custom_notes = self.txt_patient_notes.get(1.0, tk.END).strip()

        self.txt_summary_output.delete(1.0, tk.END)
        self.txt_summary_output.insert(tk.END, "⏳ Extracting document contents & querying Gemini AI Medical Engine...\nPlease wait a moment...\n")

        def thread_target():
            try:
                res = self.analyzer.analyze_report(self.current_report_path, custom_notes)
                self.after(0, lambda: self.display_summary_result(res))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Analysis Error", f"Failed to analyze report: {str(e)}"))

        threading.Thread(target=thread_target, daemon=True).start()

    def display_summary_result(self, summary_text):
        self.txt_summary_output.delete(1.0, tk.END)
        self.txt_summary_output.insert(tk.END, summary_text)
        self.current_summary = summary_text

    def save_summary(self):
        content = self.txt_summary_output.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No summary content to save.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("Markdown File", "*.md")]
        )
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Report summary saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")

    def on_generate_report(self):
        if not self.current_summary:
            messagebox.showwarning("Warning", "Please generate an AI summary first.")
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
            initialfile=f"MediVision_ReportAnalysis_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        if not save_path:
            return

        try:
            generate_report_analysis_report(
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                summary_text=self.current_summary,
                output_path=save_path
            )
            messagebox.showinfo("Success", f"Professional Medical Report Analysis PDF generated successfully!\n\nSaved to: {save_path}")
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {str(e)}")

    def clear_fields(self):
        self.current_report_path = None
        self.lbl_file_status.configure(text="No document selected", foreground="#495057")
        self.txt_patient_notes.delete(1.0, tk.END)
        self.txt_summary_output.delete(1.0, tk.END)

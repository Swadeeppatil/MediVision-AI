import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from datetime import datetime

from symptom_advisor import get_symptom_advice, SYMPTOM_DB
from report_generator import generate_symptom_advisor_report, generate_master_report


class SymptomAdvisorTab(ttk.Frame):
    """Tab where users type symptoms like 'I have fever' and get medicine suggestions."""

    def __init__(self, parent, patient_info=None):
        super().__init__(parent)
        self.current_advice = None
        self.current_symptoms = None
        self.patient_info = patient_info
        self._setup_patient_info_sync()
        self.setup_ui()

    def setup_ui(self):
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

        # ── Header ──
        header = ttk.Frame(self.scrollable_frame)
        header.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(header, text="💊 Symptom & Medicine Advisor", font=('Segoe UI', 14, 'bold')).pack(anchor="w")
        ttk.Label(header, text="Describe your symptoms below to get instant medicine suggestions, dosage info, specialist recommendations, and FDA drug database results.",
                  font=('Segoe UI', 10), foreground="#6c757d", wraplength=800).pack(anchor="w", pady=(2, 0))

        # ── Patient Information ──
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

        # ── Input Section ──
        input_frame = ttk.LabelFrame(self.scrollable_frame, text=" 🩺 Enter Your Symptoms ", padding=15)
        input_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(input_frame, text="What are you experiencing? (e.g., 'I have fever and headache'):", font=('Segoe UI', 10, 'bold')).pack(anchor="w", pady=(0, 6))
        self.txt_symptom_input = tk.Text(input_frame, height=3, font=('Segoe UI', 11), bg="#ffffff", fg="#1e293b", relief="solid", bd=1, padx=10, pady=10)
        self.txt_symptom_input.pack(fill=tk.X, pady=(0, 10))

        # Quick Symptom Buttons
        quick_frame = ttk.Frame(input_frame)
        quick_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(quick_frame, text="Quick Select:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 6))

        quick_symptoms = ["Fever", "Headache", "Cold", "Cough", "Stomach Pain", "Back Pain",
                          "Allergy", "Diarrhea", "Chest Pain", "Skin Rash", "Anxiety", "Vomiting"]
        for symptom in quick_symptoms:
            ttk.Button(quick_frame, text=symptom, command=lambda s=symptom: self._quick_select(s)).pack(side=tk.LEFT, padx=2, pady=2)

        # Action Buttons
        btn_row = ttk.Frame(input_frame)
        btn_row.pack(fill=tk.X)

        self.btn_search = ttk.Button(btn_row, text="⚡ Get Medicine Suggestions & FDA Info", command=self.on_search)
        self.btn_search.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_report = ttk.Button(btn_row, text="📄 Generate PDF Report", command=self.on_generate_report)
        self.btn_report.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(btn_row, text="🧹 Clear", command=self.clear_all).pack(side=tk.LEFT)

        # Status
        self.status_label = tk.Label(input_frame, text=" READY ", font=('Segoe UI', 9, 'bold'), bg="#e2e8f0", fg="#334155", pady=3)
        self.status_label.pack(fill=tk.X, pady=(8, 0))

        # ── Results Section ──
        output_frame = ttk.LabelFrame(self.scrollable_frame, text=" 📊 Medicine Suggestions & Clinical Advisory ", padding=15)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_output = tk.Text(output_frame, height=24, font=('Consolas', 10), bg="#0f172a", fg="#e2e8f0",
                                  relief="solid", bd=1, padx=12, pady=12, wrap="word", insertbackground="#38bdf8")
        self.txt_output.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Configure text tags for coloring
        self.txt_output.tag_configure("header", foreground="#38bdf8", font=('Consolas', 10, 'bold'))
        self.txt_output.tag_configure("warning", foreground="#f59e0b")
        self.txt_output.tag_configure("medicine", foreground="#10b981")
        self.txt_output.tag_configure("normal", foreground="#e2e8f0")

        # Save button
        save_row = ttk.Frame(output_frame)
        save_row.pack(fill=tk.X)
        ttk.Button(save_row, text="💾 Save Advice to File", command=self.save_output).pack(side=tk.RIGHT)

    def _quick_select(self, symptom):
        self.txt_symptom_input.delete(1.0, tk.END)
        self.txt_symptom_input.insert(tk.END, f"I have {symptom.lower()}")
        self.on_search()

    def on_search(self):
        query = self.txt_symptom_input.get(1.0, tk.END).strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter your symptoms first.")
            return

        self.btn_search.configure(state="disabled")
        self.status_label.configure(text=" ⏳ SEARCHING LOCAL DB & FDA DRUG API... ", bg="#f59e0b", fg="#ffffff")
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, "⏳ Analyzing symptoms & querying FDA drug database...\nPlease wait...\n")
        self.update_idletasks()

        def run():
            try:
                result = get_symptom_advice(query)
                self.after(0, lambda: self._display_result(result))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Search failed: {str(e)}"))
                self.after(0, lambda: self.btn_search.configure(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def _display_result(self, text):
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, text)
        self.current_advice = text
        self.current_symptoms = self.txt_symptom_input.get(1.0, tk.END).strip()
        self.btn_search.configure(state="normal")
        self.status_label.configure(text=" ✅ RESULTS READY ", bg="#10b981", fg="#ffffff")

    def save_output(self):
        content = self.txt_output.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No results to save.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt"), ("Markdown", "*.md")])
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Saved", f"Advice saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")

    def on_generate_report(self):
        if not self.current_advice:
            messagebox.showwarning("Warning", "Please get medicine suggestions first.")
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
            initialfile=f"MediVision_SymptomAdvisor_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        if not save_path:
            return

        try:
            generate_symptom_advisor_report(
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                symptoms=self.current_symptoms or "N/A",
                advice_text=self.current_advice,
                output_path=save_path
            )
            messagebox.showinfo("Success", f"Professional Symptom Advisory PDF report generated successfully!\n\nSaved to: {save_path}")
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {str(e)}")

    def clear_all(self):
        self.txt_symptom_input.delete(1.0, tk.END)
        self.txt_output.delete(1.0, tk.END)
        self.status_label.configure(text=" READY ", bg="#e2e8f0", fg="#334155")

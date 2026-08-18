import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime

from database import DatabaseManager
from model_handler import ModelHandler
from bone_detection_tab import BoneDetectionTab
from report_analysis_tab import ReportAnalysisTab
from mri_ct_scan_tab import MriCtScanTab
from symptom_advisor_tab import SymptomAdvisorTab
from report_generator import generate_master_report


class PatientInfo:
    """Shared patient information model that syncs across all tabs."""
    def __init__(self):
        self._name = ""
        self._age = ""
        self._gender = "Male"
        self._callbacks = []

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self._notify()

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        self._age = value
        self._notify()

    @property
    def gender(self):
        return self._gender

    @gender.setter
    def gender(self, value):
        self._gender = value
        self._notify()

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def _notify(self):
        for callback in self._callbacks:
            callback(self._name, self._age, self._gender)

    def get_dict(self):
        return {"name": self._name, "age": self._age, "gender": self._gender}


class MedicalDiagnosticApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MediVision AI - Advanced Bone Diagnostic & Medical Report Workstation")
        self.root.geometry("1480x860")
        self.root.minsize(1150, 720)

        # Style configuration
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        # Customizing ttk styles for clean margins & modern look
        self.style.configure('TNotebook.Tab', padding=[14, 8], font=('Segoe UI', 10, 'bold'))
        self.style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'), foreground="#0d6efd")
        self.style.configure('TButton', font=('Segoe UI', 9, 'bold'), padding=6)

        # Initialize shared components
        self.db_manager = DatabaseManager()
        self.model_handler = ModelHandler()
        
        # Shared patient info across all tabs
        self.patient_info = PatientInfo()

        self.setup_ui()

    def setup_ui(self):
        # Header banner
        header_bar = tk.Frame(self.root, bg="#1e293b", height=50)
        header_bar.pack(fill=tk.X, side=tk.TOP)
        header_bar.pack_propagate(False)

        title_lbl = tk.Label(
            header_bar,
            text=" 🏥 MediVision AI Radiology Workstation & Clinical Assistant",
            font=('Segoe UI', 14, 'bold'),
            fg="#ffffff",
            bg="#1e293b"
        )
        title_lbl.pack(side=tk.LEFT, padx=15, pady=8)

        # Master Report Button in Header
        master_report_btn = ttk.Button(
            header_bar,
            text="📋 Generate Master Report",
            command=self.on_generate_master_report
        )
        master_report_btn.pack(side=tk.RIGHT, padx=15, pady=8)

        # Notebook for dynamic tab management
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Bone Detection & Thermal Workstation
        self.tab_bone = BoneDetectionTab(self.notebook, self.db_manager, self.model_handler, self.patient_info)
        self.notebook.add(self.tab_bone, text=" 🦴 Bone Detection & Thermal Imaging ")

        # Tab 2: MRI / CT Scan Analysis
        self.tab_mri_ct = MriCtScanTab(self.notebook, self.db_manager, self.model_handler, self.patient_info)
        self.notebook.add(self.tab_mri_ct, text=" 🧠 MRI / CT Scan Analysis ")

        # Tab 3: Medical Report AI Summarizer
        self.tab_report = ReportAnalysisTab(self.notebook, self.patient_info)
        self.notebook.add(self.tab_report, text=" 📋 Medical Report Summarizer & Info ")

        # Tab 4: Symptom & Medicine Advisor
        self.tab_symptom = SymptomAdvisorTab(self.notebook, self.patient_info)
        self.notebook.add(self.tab_symptom, text=" 💊 Symptom & Medicine Advisor ")

        # Status Bar
        self.status_bar = tk.Label(
            self.root,
            text=" System Status: Ready | Database Connected | DenseNet169 Loaded",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=('Segoe UI', 9),
            bg="#f1f5f9",
            fg="#475569"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_generate_master_report(self):
        # Get patient info from Bone Detection tab (primary source)
        bone_tab = self.tab_bone
        patient_name = bone_tab.ent_patient_name.get().strip()
        patient_age = bone_tab.ent_patient_age.get().strip()
        patient_gender = bone_tab.combo_gender.get()

        if not patient_name:
            messagebox.showwarning("Warning", "Please enter patient name in the Bone Detection tab first.")
            return

        if not patient_age.isdigit():
            messagebox.showwarning("Warning", "Please enter a valid age in the Bone Detection tab.")
            return

        patient_age = int(patient_age)

        # Collect data from all tabs
        bone_data = None
        if bone_tab.current_fracture_type:
            bone_data = {
                'fracture_type': bone_tab.current_fracture_type,
                'confidence': bone_tab.current_confidence,
                'severity': bone_tab.current_severity,
                'description': bone_tab.current_description,
                'treatment': bone_tab.current_treatment,
                'original_image_path': bone_tab.original_image_path,
                'highlighted_image_path': bone_tab.highlighted_image_path,
                'thermal_image_path': bone_tab.thermal_image_path
            }

        mri_ct_tab = self.tab_mri_ct
        mri_ct_data = None
        if mri_ct_tab.current_anomaly_type:
            mri_ct_data = {
                'scan_type': mri_ct_tab.scan_type_var.get(),
                'anomaly_type': mri_ct_tab.current_anomaly_type,
                'confidence': mri_ct_tab.current_confidence,
                'severity': mri_ct_tab.current_severity,
                'description': mri_ct_tab.current_description,
                'treatment': mri_ct_tab.current_treatment,
                'original_image_path': mri_ct_tab.original_image_path,
                'highlighted_image_path': mri_ct_tab.highlighted_image_path,
                'thermal_image_path': mri_ct_tab.thermal_image_path
            }

        report_tab = self.tab_report
        report_analysis_data = None
        if report_tab.current_summary:
            report_analysis_data = {
                'summary_text': report_tab.current_summary
            }

        symptom_tab = self.tab_symptom
        symptom_advisor_data = None
        if symptom_tab.current_advice:
            symptom_advisor_data = {
                'symptoms': symptom_tab.current_symptoms,
                'advice_text': symptom_tab.current_advice
            }

        # Check if any data exists
        if not any([bone_data, mri_ct_data, report_analysis_data, symptom_advisor_data]):
            messagebox.showwarning("Warning", "No analysis data available from any tab. Please run at least one analysis first.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Report", "*.pdf")],
            initialfile=f"MediVision_MasterReport_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        if not save_path:
            return

        self.status_bar.configure(text=" Generating Master Report... ")
        self.root.update_idletasks()

        try:
            generate_master_report(
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                bone_data=bone_data,
                mri_ct_data=mri_ct_data,
                report_analysis_data=report_analysis_data,
                symptom_advisor_data=symptom_advisor_data,
                output_path=save_path
            )
            self.status_bar.configure(text=" System Status: Ready | Database Connected | DenseNet169 Loaded")
            messagebox.showinfo("Success", f"Comprehensive Master Report generated successfully!\n\nSaved to: {save_path}")
        except Exception as e:
            self.status_bar.configure(text=" System Status: Ready | Database Connected | DenseNet169 Loaded")
            messagebox.showerror("Report Error", f"Failed to generate master report: {str(e)}")

    def run(self):
        self.root.mainloop()

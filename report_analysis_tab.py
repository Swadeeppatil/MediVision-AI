import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
import json
import urllib.request
from datetime import datetime
from report_analyzer import MedicalReportAnalyzer
from report_generator import generate_report_analysis_report, generate_master_report

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class ReportAnalysisTab(ttk.Frame):
    def __init__(self, parent, patient_info=None):
        super().__init__(parent)
        self.analyzer = MedicalReportAnalyzer()
        self.current_report_path = None
        self.current_summary = None
        self.extracted_text = ""
        self.pdf_doc = None
        self.current_pdf_page = 0
        self.patient_info = patient_info
        self._setup_patient_info_sync()
        self.setup_ui()

    def setup_ui(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)

        # LEFT PANEL - Controls & Summary
        left_panel = ttk.Frame(main_container)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_panel.rowconfigure(2, weight=1)

        # RIGHT PANEL - PDF Viewer
        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_panel.rowconfigure(1, weight=1)

        self._build_left_panel(left_panel)
        self._build_right_panel(right_panel)

    def _build_left_panel(self, parent):
        # Header
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(header_frame, text="📄 Smart Medical Report Summarizer & AI Clinical Insights", font=('Segoe UI', 13, 'bold')).pack(anchor="w")
        ttk.Label(header_frame, text="Upload PDF/TXT/Image reports for AI summaries, key findings, and interactive Q&A.", font=('Segoe UI', 9), foreground="#6c757d").pack(anchor="w", pady=(2, 0))

        # Patient Info
        patient_frame = ttk.LabelFrame(parent, text=" 👤 Patient Information ", padding=12)
        patient_frame.pack(fill=tk.X, pady=(0, 12))

        name_row = ttk.Frame(patient_frame)
        name_row.pack(fill=tk.X, pady=2)
        ttk.Label(name_row, text="Patient Name:", width=14).pack(side=tk.LEFT)
        self.ent_patient_name = ttk.Entry(name_row, width=28)
        self.ent_patient_name.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.ent_patient_name.bind('<KeyRelease>', self._on_name_changed)
        self.ent_patient_name.bind('<FocusOut>', self._on_name_changed)

        age_gender_row = ttk.Frame(patient_frame)
        age_gender_row.pack(fill=tk.X, pady=2)
        ttk.Label(age_gender_row, text="Age:", width=14).pack(side=tk.LEFT)
        self.ent_patient_age = ttk.Entry(age_gender_row, width=8)
        self.ent_patient_age.pack(side=tk.LEFT, padx=5)
        self.ent_patient_age.bind('<KeyRelease>', self._on_age_changed)
        self.ent_patient_age.bind('<FocusOut>', self._on_age_changed)
        ttk.Label(age_gender_row, text="Gender:", width=8).pack(side=tk.LEFT, padx=(10, 0))
        self.combo_gender = ttk.Combobox(age_gender_row, state="readonly", width=10, values=["Male", "Female", "Other"])
        self.combo_gender.current(0)
        self.combo_gender.pack(side=tk.LEFT, padx=5)
        self.combo_gender.bind('<<ComboboxSelected>>', self._on_gender_changed)

        # Upload & Config
        config_frame = ttk.LabelFrame(parent, text=" 📤 Upload Document & AI Settings ", padding=12)
        config_frame.pack(fill=tk.X, pady=(0, 12))

        file_row = ttk.Frame(config_frame)
        file_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(file_row, text="📁 Choose Medical File (PDF/TXT/Img)", command=self.select_report_file).pack(side=tk.LEFT, padx=(0, 10))
        self.lbl_file_status = ttk.Label(file_row, text="No document selected", font=('Segoe UI', 9, 'italic'), foreground="#495057")
        self.lbl_file_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        api_row = ttk.Frame(config_frame)
        api_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(api_row, text="Gemini API Key:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 8))
        self.ent_api_key = ttk.Entry(api_row, show="*", width=40)
        self.ent_api_key.pack(side=tk.LEFT, padx=(0, 10))
        # Pre-fill from environment variable if available
        env_key = os.getenv("GEMINI_API_KEY", "")
        if env_key:
            self.ent_api_key.insert(0, env_key)
        else:
            self.ent_api_key.insert(0, "YOUR_GEMINI_API_KEY_HERE")
        ttk.Label(api_row, text="(Pre-filled - editable)", font=('Segoe UI', 8), foreground="#6c757d").pack(side=tk.LEFT)

        notes_row = ttk.Frame(config_frame)
        notes_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(notes_row, text="Patient Notes / Symptoms:").pack(anchor="w", pady=(0, 4))
        self.txt_patient_notes = tk.Text(notes_row, height=3, font=('Segoe UI', 9), bg="#ffffff", relief="solid", bd=1)
        self.txt_patient_notes.pack(fill=tk.X)

        btn_row = ttk.Frame(config_frame)
        btn_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(btn_row, text="⚡ Generate AI Summary & Key Points", command=self.start_analysis).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="📄 Generate PDF Report", command=self.on_generate_report).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="🧹 Clear", command=self.clear_fields).pack(side=tk.LEFT)

        # Summary Output
        output_frame = ttk.LabelFrame(parent, text=" 📊 AI Summary & Key Points ", padding=12)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.txt_summary_output = scrolledtext.ScrolledText(output_frame, font=('Segoe UI', 9), bg="#ffffff", fg="#212529", relief="solid", bd=1, wrap="word", padx=10, pady=10)
        self.txt_summary_output.grid(row=0, column=0, sticky="nsew")

        # Chat Bot Section
        chat_frame = ttk.LabelFrame(parent, text=" 💬 Chat with Report (Ask Questions) ", padding=12)
        chat_frame.pack(fill=tk.BOTH, expand=True)
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)

        self.txt_chat_history = scrolledtext.ScrolledText(chat_frame, font=('Segoe UI', 9), bg="#f8f9fa", fg="#212529", relief="solid", bd=1, wrap="word", padx=10, pady=10, state="disabled")
        self.txt_chat_history.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 8))

        chat_input_row = ttk.Frame(chat_frame)
        chat_input_row.grid(row=1, column=0, columnspan=2, sticky="ew")
        chat_input_row.columnconfigure(0, weight=1)
        self.ent_chat_input = ttk.Entry(chat_input_row, font=('Segoe UI', 10))
        self.ent_chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.ent_chat_input.bind('<Return>', lambda e: self.send_chat_message())
        ttk.Button(chat_input_row, text="📤 Send", command=self.send_chat_message).grid(row=0, column=1)

        export_row = ttk.Frame(output_frame)
        export_row.grid(row=1, column=0, sticky="e", pady=(8, 0))
        ttk.Button(export_row, text="💾 Save Summary", command=self.save_summary).pack(side=tk.RIGHT)

    def _build_right_panel(self, parent):
        # PDF Viewer
        viewer_frame = ttk.LabelFrame(parent, text=" 📄 PDF Preview ", padding=10)
        viewer_frame.pack(fill=tk.BOTH, expand=True)
        viewer_frame.rowconfigure(1, weight=1)
        viewer_frame.columnconfigure(0, weight=1)

        # PDF Controls
        pdf_ctrl = ttk.Frame(viewer_frame)
        pdf_ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.lbl_pdf_status = ttk.Label(pdf_ctrl, text="No PDF loaded", font=('Segoe UI', 9, 'italic'), foreground="#6c757d")
        self.lbl_pdf_status.pack(side=tk.LEFT)
        ttk.Button(pdf_ctrl, text="⏮ Prev", command=self.prev_pdf_page, width=8).pack(side=tk.RIGHT, padx=2)
        ttk.Button(pdf_ctrl, text="Next ⏭", command=self.next_pdf_page, width=8).pack(side=tk.RIGHT, padx=2)
        self.lbl_page_num = ttk.Label(pdf_ctrl, text="Page 0/0", font=('Segoe UI', 9))
        self.lbl_page_num.pack(side=tk.RIGHT, padx=8)

        # Canvas for PDF rendering
        self.pdf_canvas = tk.Canvas(viewer_frame, bg="#e8e8e8", highlightthickness=1, highlightbackground="#ccc")
        self.pdf_canvas.grid(row=1, column=0, sticky="nsew")

        # Scrollbars for PDF
        pdf_v_scroll = ttk.Scrollbar(viewer_frame, orient=tk.VERTICAL, command=self.pdf_canvas.yview)
        pdf_v_scroll.grid(row=1, column=1, sticky="ns")
        pdf_h_scroll = ttk.Scrollbar(viewer_frame, orient=tk.HORIZONTAL, command=self.pdf_canvas.xview)
        pdf_h_scroll.grid(row=2, column=0, sticky="ew")
        self.pdf_canvas.configure(yscrollcommand=pdf_v_scroll.set, xscrollcommand=pdf_h_scroll.set)

    def _setup_patient_info_sync(self):
        if self.patient_info:
            self.patient_info.add_callback(self._on_patient_info_changed)
            self.after(100, self._sync_fields_from_model)

    def _on_patient_info_changed(self, name, age, gender):
        self.ent_patient_name.delete(0, tk.END)
        self.ent_patient_name.insert(0, name)
        self.ent_patient_age.delete(0, tk.END)
        self.ent_patient_age.insert(0, age)
        self.combo_gender.set(gender)

    def _sync_fields_from_model(self):
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
            self._load_pdf_preview(path)

    def _load_pdf_preview(self, path):
        if not PYMUPDF_AVAILABLE:
            self.lbl_pdf_status.configure(text="PyMuPDF not installed - cannot preview PDF", foreground="#dc3545")
            return
        if not path.lower().endswith('.pdf'):
            self.lbl_pdf_status.configure(text="Preview only for PDF files", foreground="#6c757d")
            self.pdf_canvas.delete("all")
            return
        try:
            if self.pdf_doc:
                self.pdf_doc.close()
            self.pdf_doc = fitz.open(path)
            self.current_pdf_page = 0
            self._render_pdf_page()
        except Exception as e:
            self.lbl_pdf_status.configure(text=f"Error loading PDF: {e}", foreground="#dc3545")

    def _render_pdf_page(self):
        if not self.pdf_doc:
            return
        try:
            page = self.pdf_doc[self.current_pdf_page]
            zoom = 1.5
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            img = tk.PhotoImage(data=img_data)
            self.pdf_canvas.delete("all")
            self.pdf_canvas.create_image(0, 0, anchor="nw", image=img)
            self.pdf_canvas.image = img
            self.pdf_canvas.configure(scrollregion=self.pdf_canvas.bbox("all"))
            self.lbl_page_num.configure(text=f"Page {self.current_pdf_page + 1}/{len(self.pdf_doc)}")
            self.lbl_pdf_status.configure(text=f"Loaded: {os.path.basename(self.current_report_path)}", foreground="#198754")
        except Exception as e:
            self.lbl_pdf_status.configure(text=f"Render error: {e}", foreground="#dc3545")

    def prev_pdf_page(self):
        if self.pdf_doc and self.current_pdf_page > 0:
            self.current_pdf_page -= 1
            self._render_pdf_page()

    def next_pdf_page(self):
        if self.pdf_doc and self.current_pdf_page < len(self.pdf_doc) - 1:
            self.current_pdf_page += 1
            self._render_pdf_page()

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

    def display_summary_result(self, result):
        # Handle both old format (string) and new format (tuple)
        if isinstance(result, tuple):
            summary_text, extracted_text = result
            self.extracted_text = extracted_text
        else:
            summary_text = result
            self.extracted_text = ""
        
        self.txt_summary_output.delete(1.0, tk.END)
        self.txt_summary_output.insert(tk.END, summary_text)
        self.current_summary = summary_text
        self._clear_chat()
        self._append_chat("System", "Report analyzed. You can now ask questions about this document.")

    def save_summary(self):
        content = self.txt_summary_output.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No summary content to save.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text File", "*.txt"), ("Markdown File", "*.md")])
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
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Report", "*.pdf")], initialfile=f"MediVision_ReportAnalysis_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        if not save_path:
            return
        try:
            generate_report_analysis_report(patient_name=patient_name, patient_age=patient_age, patient_gender=patient_gender, summary_text=self.current_summary, output_path=save_path)
            messagebox.showinfo("Success", f"Professional Medical Report Analysis PDF generated successfully!\n\nSaved to: {save_path}")
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {str(e)}")

    def clear_fields(self):
        self.current_report_path = None
        self.current_summary = None
        self.extracted_text = ""
        if self.pdf_doc:
            self.pdf_doc.close()
            self.pdf_doc = None
        self.lbl_file_status.configure(text="No document selected", foreground="#495057")
        self.lbl_pdf_status.configure(text="No PDF loaded", foreground="#6c757d")
        self.pdf_canvas.delete("all")
        self.lbl_page_num.configure(text="Page 0/0")
        self.txt_patient_notes.delete(1.0, tk.END)
        self.txt_summary_output.delete(1.0, tk.END)
        self._clear_chat()

    # --- Chat Bot Methods ---
    def _clear_chat(self):
        self.txt_chat_history.configure(state="normal")
        self.txt_chat_history.delete(1.0, tk.END)
        self.txt_chat_history.configure(state="disabled")

    def _append_chat(self, sender, message):
        self.txt_chat_history.configure(state="normal")
        self.txt_chat_history.insert(tk.END, f"{sender}: ", "sender")
        self.txt_chat_history.insert(tk.END, f"{message}\n\n")
        self.txt_chat_history.tag_configure("sender", font=('Segoe UI', 9, 'bold'), foreground="#0d6efd")
        self.txt_chat_history.see(tk.END)
        self.txt_chat_history.configure(state="disabled")

    def send_chat_message(self):
        question = self.ent_chat_input.get().strip()
        if not question:
            return
        if not self.current_report_path:
            messagebox.showwarning("Warning", "Please upload and analyze a report first.")
            return
        self.ent_chat_input.delete(0, tk.END)
        self._append_chat("You", question)
        self._append_chat("Bot", "Thinking...")
        threading.Thread(target=self._get_bot_response, args=(question,), daemon=True).start()

    def _get_bot_response(self, question):
        try:
            answer = self._query_gemini_chat(question)
            self.after(0, lambda: self._replace_last_bot_message(answer))
        except Exception as e:
            self.after(0, lambda: self._replace_last_bot_message(f"Error: {e}"))

    def _replace_last_bot_message(self, answer):
        self.txt_chat_history.configure(state="normal")
        content = self.txt_chat_history.get(1.0, tk.END)
        lines = content.strip().split('\n')
        if lines and lines[-1].startswith("Bot: "):
            self.txt_chat_history.delete("end-2l", "end-1l")
        self.txt_chat_history.insert(tk.END, f"Bot: {answer}\n\n")
        self.txt_chat_history.see(tk.END)
        self.txt_chat_history.configure(state="disabled")

    def _query_gemini_chat(self, question):
        api_key = self.ent_api_key.get().strip()
        if not api_key:
            return "Please enter a Gemini API key to use the chat feature."

        context = self.extracted_text or self.current_summary or "No document context available."
        prompt = (
            "You are a medical AI assistant. Answer the user's question based ONLY on the provided medical report context.\n"
            "If the answer is not in the context, say you cannot find it in the report.\n"
            "Be concise and accurate.\n\n"
            f"MEDICAL REPORT CONTEXT:\n{context[:8000]}\n\n"
            f"USER QUESTION: {question}"
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            candidates = res_data.get('candidates', [])
            if candidates:
                return candidates[0]['content']['parts'][0]['text']
            return "No response from AI."
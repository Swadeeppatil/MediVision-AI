"""
App Entry Point Wrapper (Refactored into modular components).
All business logic, UI components, image processing, and AI report analysis 
have been separated into distinct modules:
  - main.py
  - ui_main.py
  - bone_detection_tab.py
  - report_analysis_tab.py
  - image_processing.py
  - report_analyzer.py
  - model_handler.py
  - database.py
"""

from main import MedicalDiagnosticApp

if __name__ == "__main__":
    app = MedicalDiagnosticApp()
    app.run()

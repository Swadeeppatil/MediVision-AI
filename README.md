# MediVision AI - Advanced Multi-Modality Medical Diagnostic Workstation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive AI-powered medical diagnostic workstation for radiology analysis, featuring bone fracture detection, MRI/CT anomaly analysis, medical report summarization, symptom advisory, and professional PDF report generation.

## 🏥 Features

### 1. **Bone Detection & Thermal Imaging (X-Ray Analysis)**
- Upload X-ray images (JPG, PNG, BMP, DICOM)
- AI-powered fracture classification using DenseNet169
- 4 fracture types: Transverse, Oblique, Compound, Stress
- Thermal heatmap generation (JET colormap)
- Fracture region highlighting with crosshair markers
- Contrast enhancement (CLAHE)
- Patient information management

### 2. **MRI / CT Scan Analysis**
- Support for both MRI and CT scans
- Anomaly detection: Mass Lesion, Edema, Herniation, Normal
- Thermal intensity heatmap visualization
- Scan enhancement and analysis
- Professional diagnostic output

### 3. **Medical Report AI Summarizer**
- Upload PDF, TXT, or image-based medical reports
- AI-powered analysis using Google Gemini API
- Structured clinical summaries with:
  - Executive Summary
  - Critical Findings
  - Health Warnings & Alert Risks
  - Recommended Actions & Specialist Advice
- Offline fallback analysis
- PDF report generation

### 4. **Symptom & Medicine Advisor**
- Natural language symptom input
- Medicine suggestions with dosage information
- FDA drug database integration
- Specialist recommendations
- Quick-select common symptoms

### 5. **Professional PDF Report Generation**
- **Individual reports** for each modality
- **Master Comprehensive Report** combining all analyses
- Hospital-style formatting with:
  - Patient demographics
  - Diagnostic results with confidence scores
  - Clinical descriptions & treatment plans
  - All 3 images: Original, Highlighted, Thermal
  - Medical disclaimer
- Downloadable PDF format

### 6. **Shared Patient Information**
- Auto-sync patient name, age, gender across all tabs
- Enter once, appears everywhere
- Real-time synchronization

### 7. **Database & History**
- SQLite database for scan history
- Recent scans tracking with patient details
- Persistent storage

---

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster inference)
- **Storage**: 2 GB free space

### Python Dependencies
```
# Core dependencies
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
tensorflow>=2.15.0

# Training dependencies (PyTorch)
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
pytorch-lightning>=2.1.0
timm>=0.9.12
albumentations>=1.3.0
scikit-learn>=1.3.0
pandas>=2.1.0
tqdm>=4.66.0

# ONNX export
onnx>=1.15.0
onnxruntime>=1.16.0
onnxruntime-gpu>=1.16.0; sys_platform != 'darwin'

# MURA dataset handling
requests>=2.31.0
gdown>=4.7.0

# Utilities
pyyaml>=6.0.1
matplotlib>=3.8.0
seaborn>=0.13.0

# PDF Report Generation
fpdf2>=2.7.0

# PDF Text Extraction (for report analyzer)
PyMuPDF>=1.23.0
```

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Swadeeppatil/MediVision-AI.git
cd MediVision-AI
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install PyMuPDF for PDF Text Extraction
```bash
pip install pymupdf
```

### 5. (Optional) Set Gemini API Key for AI Summaries
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Windows CMD
set GEMINI_API_KEY=your-api-key-here

# macOS/Linux
export GEMINI_API_KEY="your-api-key-here"
```

Or enter it directly in the **Medical Report Summarizer** tab.

---

## ▶️ Running the Application

### Standard Run
```bash
python app.py
```

### Alternative Entry Points
```bash
# Direct main
python main.py

# With specific Python version
python3 app.py
```

---

## 📖 User Guide

### Getting Started
1. Launch the application: `python app.py`
2. The main window opens with 4 tabs + Master Report button

### Tab 1: Bone Detection & Thermal Imaging
1. **Enter Patient Info**: Name, Age, Gender (auto-syncs to other tabs)
2. **Upload X-Ray**: Click "📂 Upload X-Ray/MRI" or load sample
3. **Enhance** (optional): Click "✨ Enhance Contrast"
4. **Detect Fracture**: Click "🔬 Detect Fracture" → AI analyzes
5. **Thermal View**: Click "🔥 Thermal View" for heatmap
6. **Generate Report**: Click "📄 Generate PDF Report"

### Tab 2: MRI / CT Scan Analysis
1. Select **Scan Type**: MRI or CT
2. **Upload Scan**: Click "📂 Upload MRI/CT Image"
3. **Analyze**: Click "🔬 Analyze Anomaly"
4. **Thermal View**: Click "🔥 Thermal View"
5. **Generate Report**: Click "📄 Generate PDF Report"

### Tab 3: Medical Report Summarizer
1. **Enter Patient Info** (auto-filled from other tabs)
2. **Upload Report**: Click "📁 Choose Medical File" (PDF/TXT/Image)
3. **Add API Key**: Enter Gemini API key (or set env variable)
4. **Add Notes**: Patient symptoms/notes (optional)
5. **Generate Summary**: Click "⚡ Generate AI Summary & Key Info"
6. **Generate PDF**: Click "📄 Generate PDF Report"

### Tab 4: Symptom & Medicine Advisor
1. **Enter Patient Info** (auto-filled)
2. **Describe Symptoms**: Type or use quick-select buttons
3. **Get Advice**: Click "⚡ Get Medicine Suggestions & FDA Info"
4. **Generate PDF**: Click "📄 Generate PDF Report"

### Master Report (Header Button)
- Click **"📋 Generate Master Report"** in top-right
- Combines ALL analyses from all tabs into one comprehensive PDF
- Includes Table of Contents, all images, and findings

---

## 🔧 Configuration

### config.yaml
```yaml
model:
  name: "densenet169"
  input_size: [224, 224]
  num_classes: 4
  dropout: 0.5

training:
  batch_size: 32
  epochs: 50
  learning_rate: 1e-4
  weight_decay: 1e-5

data:
  train_split: 0.8
  val_split: 0.2
  augment: true
```

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI summaries | No (fallback available) |

---

## 🏗️ Project Structure

```
MediVision-AI/
├── app.py                      # Entry point
├── main.py                     # Main application class
├── ui_main.py                  # Main UI with tabs
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── Core Modules/
│   ├── database.py             # SQLite database manager
│   ├── model_handler.py        # DenseNet169 model wrapper
│   ├── image_processing.py     # CV2 image processing
│   ├── report_analyzer.py      # Medical report AI analyzer
│   ├── report_generator.py     # PDF report generation
│   ├── symptom_advisor.py      # Symptom analysis engine
│   └── custom_model_handler.py # Custom model utilities
│
├── UI Tabs/
│   ├── bone_detection_tab.py   # X-Ray fracture detection
│   ├── mri_ct_scan_tab.py      # MRI/CT anomaly analysis
│   ├── report_analysis_tab.py  # Medical report summarizer
│   └── symptom_advisor_tab.py  # Symptom & medicine advisor
│
├── Training/
│   ├── train_fracture_model.py # Fracture model training
│   ├── finetune_densenet.py    # DenseNet fine-tuning
│   ├── export_onnx.py          # ONNX export
│   ├── download_mura.py        # MURA dataset downloader
│   ├── create_synthetic_data.py# Synthetic data generator
│   └── run_pipeline.py         # Training pipeline
│
├── Data/
│   ├── test_images/            # Sample test images
│   └── fracture_classification/# Training data
│
├── Models/
│   └── logs/                   # Training logs
│
├── fracture_scans.db           # SQLite database (auto-created)
└── __pycache__/                # Python cache
```

---

## 🤖 AI Models

### Fracture Classification (DenseNet169)
- **Architecture**: DenseNet169 + GlobalAveragePooling + Dense(1024) + Dropout(0.5) + Dense(4)
- **Input**: 224x224 RGB images
- **Classes**: Transverse, Oblique, Compound, Stress
- **Preprocessing**: ImageNet normalization
- **Training**: Transfer learning from ImageNet weights

### MRI/CT Anomaly Detection
- Uses same DenseNet169 backbone
- Maps to: Mass Lesion, Edema, Herniation, Normal
- Confidence-based classification

### Medical Report Analysis
- **Primary**: Google Gemini 1.5 Flash (via REST API)
- **Fallback**: Local rule-based analysis
- **Output**: Structured markdown sections

---

## 📊 Database Schema

```sql
CREATE TABLE scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    patient_age INTEGER,
    patient_gender TEXT,
    fracture_type TEXT,
    confidence REAL,
    severity TEXT,
    timestamp DATETIME,
    image_path TEXT,
    highlighted_path TEXT,
    thermal_path TEXT
);
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. TensorFlow GPU not detected**
```bash
pip install tensorflow[and-cuda]
# Or for CPU only (default)
pip install tensorflow-cpu
```

**2. PyMuPDF import error**
```bash
pip install pymupdf --force-reinstall
```

**3. Tkinter not found (Linux)**
```bash
sudo apt-get install python3-tk
```

**4. Port 504 / API timeout (Gemini)**
- Check internet connection
- Verify API key is valid
- Try again - may be temporary rate limiting

**5. Module import errors**
```bash
pip install -r requirements.txt --force-reinstall
```

**6. Database locked**
- Close other instances of the app
- Delete `fracture_scans.db` to reset

### Performance Tips
- Use GPU for faster inference
- Resize large images before upload
- Close unused tabs to free memory

---

## 🔐 Security & Privacy

- **Local Processing**: All image analysis runs locally
- **API Keys**: Stored only in memory, never logged
- **Patient Data**: Stored locally in SQLite
- **No Cloud Upload**: Images never leave your machine
- **PDF Reports**: Generated locally

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit Pull Request

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Swadeeppatil/MediVision-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Swadeeppatil/MediVision-AI/discussions)

---

## ⚠️ Medical Disclaimer

> **This software is for diagnostic assistance only and does not constitute medical advice. All AI-generated findings must be verified by qualified medical professionals. The developers assume no liability for clinical decisions based on this software.**

---

## 🙏 Acknowledgments

- **DenseNet169**: Huang et al. (CVPR 2017)
- **MURA Dataset**: Stanford ML Group
- **TensorFlow/Keras**: Google Brain Team
- **PyTorch Lightning**: William Falcon et al.
- **fpdf2**: PyFPDF contributors
- **PyMuPDF**: Artifex Software
- **Gemini API**: Google AI

---

**Made with ❤️ for Medical AI Research**
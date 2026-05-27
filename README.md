# Multimodal Emotion Recognition on TESS

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning system that recognizes human emotions from speech audio, text transcripts, and their combination (multimodal late fusion) using the **Toronto Emotional Speech Set (TESS)**.

---

## ⚡ Quick Start / Execution Hint
If you are looking to execute the code immediately:
1. Go to the **[Setup & Installation Instructions](#-setup--installation-instructions)** to configure your Python environment.
2. Download and prepare the dataset as outlined in **[Dataset Acquisition](#-dataset-acquisition)**.
3. Run the scripts using the commands in **[Running the Pipelines](#-running-the-pipelines)** (using `run_all.sh` on Linux/macOS or the PowerShell commands on Windows).

---


## 🌟 Key Features
- **Speech Pipeline:** Uses raw audio resampled to 16 kHz $\rightarrow$ Mel-spectrogram extraction $\rightarrow$ Bidirectional LSTM (BiLSTM) sequence classifier.
- **Text Pipeline:** Uses a frozen pre-trained **BERT (bert-base-uncased)** transformer model to extract contextual CLS token embeddings.
- **Late Fusion Pipeline:** Concatenates speech temporal representations (256-d) and text contextual embeddings (768-d) into a unified classifier head.
- **Visualizations:** Automatically generates **t-SNE cluster plots** and **Confusion Matrices** for all three variants in the results folder.

---

## 📊 Performance Summary
Evaluated on an 80/20 train/test split of the TESS dataset:

| Model Variant | Input Modality | Overall Accuracy |
| :--- | :--- | :---: |
| **Speech-Only** | Mel-Spectrogram + BiLSTM | **97.32%** |
| **Text-Only** | Frozen BERT Embedding | **13.48%** *(Random Guessing Baseline)* |
| **Multimodal (Fusion)** | Speech (BiLSTM) + Text (BERT) | **99.38%** |

*Note: The Text-Only pipeline performs poorly because the TESS transcripts strictly follow the carrier phrase pattern "Say the word [word]" which lacks any emotional semantics. However, in the **Fusion** model, the target word identity provides a strong context to disambiguate acoustically overlapping sounds, boosting accuracy.*

---

## 🚀 Setup & Installation Instructions

Follow these steps to set up and run this project locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/aaronroy123/speech_analysis_final.git
cd speech_analysis_final
```

### 2. Configure Your Python Environment
It is recommended to use Python 3.12. Create a virtual environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (CMD):
.\.venv\Scripts\activate.bat

# Install package dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Acquisition

This project uses the **Toronto emotional speech set (TESS)** dataset.

1. Download the dataset from [Kaggle: TESS Toronto emotional speech set data](https://www.kaggle.com/datasets/ejlocurto/tess-toronto-emotional-speech-set-data).
2. Extract the downloaded zip file.
3. Place the dataset folder directly into the project's `data/` directory. Your folder structure must look like this:
   ```text
   speech_analysis_final/
   ├── data/
   │   └── TESS Toronto emotional speech set data/
   │       ├── OAF_angry/
   │       ├── OAF_disgust/
   │       └── ... (remaining folders containing .wav files)
   ```

---

## 💻 Running the Pipelines

You can train and evaluate all models using the provided runner script, or execute the pipelines individually.

### Run All Pipelines Sequentially
Execute the unified runner script (which trains and evaluates the Speech, Text, and Fusion pipelines back-to-back):

**On Linux/macOS:**
```bash
chmod +x run_all.sh
./run_all.sh
```

**On Windows (PowerShell):**
```powershell
# Make sure your virtual environment is active, then run:
python models/speech_pipeline/train.py --data_dir "data/TESS Toronto emotional speech set data"
python models/speech_pipeline/test.py --data_dir "data/TESS Toronto emotional speech set data"

python models/text_pipeline/train.py --data_dir "data/TESS Toronto emotional speech set data"
python models/text_pipeline/test.py --data_dir "data/TESS Toronto emotional speech set data"

python models/fusion_pipeline/train.py --data_dir "data/TESS Toronto emotional speech set data"
python models/fusion_pipeline/test.py --data_dir "data/TESS Toronto emotional speech set data"
```

### Output Files & Artifacts
Once execution finishes:
- Trained model weights are saved under `models/<pipeline_name>/weights/`.
- Accuracy tables are updated under `Results/accuracy_tables.md`.
- Scatter plots showing the t-SNE clustering and confusion matrices are generated under `Results/plots/`.

---

## 🛠️ Project Structure
```text
speech_analysis_final/
├── data/                           # Contains raw TESS audio dataset files (Git ignored)
├── models/
│   ├── dataset.py                  # PyTorch Dataset loader (resampling, spectrograms, tokenization)
│   ├── speech_pipeline/            # Speech training and test evaluation scripts
│   ├── text_pipeline/              # Text training and test evaluation scripts
│   └── fusion_pipeline/            # Late fusion model training and evaluation scripts
├── Results/
│   ├── plots/                      # Generated t-SNE scatter plots and confusion matrices
│   └── accuracy_tables.md          # Log of overall model accuracies
├── README.md                       # Setup and project guide
├── Report.md                       # Comprehensive analysis report (Markdown)
├── report.tex                      # Publication-ready LaTeX source file for Overleaf
└── requirements.txt                # Python package list
```

---

## 📄 LaTeX Compilation (Overleaf)
A fully formatted LaTeX source code for the research report has been generated at [report.tex](report.tex). 
To open it in Overleaf:
1. Create a new blank project.
2. Copy the contents of [report.tex](report.tex) into your `main.tex`.
3. Upload the generated plots from `Results/plots/` into your Overleaf workspace to compile the figures.

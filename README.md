# Multimodal Emotion Recognition System

This repository contains the code and report for a Multimodal Emotion Recognition system built on the Toronto emotional speech set (TESS) dataset.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd Speech
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install torchcodec # for robust audio loading
   ```

3. **Provide the Dataset:**
   Place the TESS dataset folder into `data/TESS Toronto emotional speech set data`.

4. **Run Training and Evaluation:**
   Execute the unified runner script:
   ```bash
   chmod +x run_all.sh
   ./run_all.sh
   ```
   This script trains the Speech-Only, Text-Only, and Fusion pipelines sequentially. It evaluates each model and generates confusion matrices and t-SNE scatter plots in the `Results/plots/` directory.

## Deliverables

- `Report.md`: Contains architecture decisions, experiments, and analysis of failure cases.
- `Results/accuracy_tables.md`: Final performance breakdown for all modalities.
- `models/`: Contains the architecture and pipelines for speech, text, and fusion.

## Pushing to GitHub

To push this local repository to your public GitHub account:
1. Go to github.com and create a new public repository (e.g., `multimodal-emotion-recognition`).
2. Run the following commands in your terminal:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git branch -M main
   git push -u origin main
   ```

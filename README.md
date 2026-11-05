# Citrus Disease Detector

An AI-powered Streamlit application for detecting common citrus fruit conditions from images. The application uses a YOLO-based computer vision model to identify:

- Citrus Canker
- Citrus Greening (HLB)
- Melanose
- Fresh / healthy fruit

## Live Demo

Try the deployed application here:

**[Citrus Disease Detector](https://citrusdiseasedetector.streamlit.app/)**

## Features

- Upload JPG, JPEG, or PNG images for analysis.
- Try built-in sample images without uploading a file.
- Adjust the confidence threshold used by the detector.
- View the uploaded image with detected regions annotated.
- Receive a diagnostic status for detected conditions.
- Generate an agricultural treatment plan with Google Gemini when an API key is configured.

## Project Structure

```text
.
├── app.py                    # Streamlit application
├── best.pt                   # Trained YOLO model weights
├── requirements.txt          # Python dependencies
├── images/                   # Demo sample images
├── results/                  # Training outputs and evaluation artifacts
└── args.yaml                 # Model training configuration
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Favy-Jay/orange_disease_dectector.git
cd orange_disease_dectector
```

### 2. Install dependencies

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

### 3. Start the application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

## Optional Gemini Configuration

The detector works without Gemini, but treatment-plan generation requires a Google Gemini API key. Create `.streamlit/secrets.toml` with:

```toml
GEMINI_API_KEY = "your-api-key"
```

Do not commit this file or expose the API key publicly.

## How to Use

1. Open the live demo or start the application locally.
2. Upload a citrus image, or select a built-in sample.
3. Adjust the confidence threshold if needed.
4. Review the annotated scan and diagnostic report.
5. If Gemini is configured, expand the treatment plan for detected diseases.

## Model and Training

The application loads the trained weights from `best.pt`. The training configuration is recorded in `args.yaml`; the model was trained for object detection at 640px image size with 150 epochs. The files in `results/` contain training visualizations and evaluation outputs.

## Disclaimer

This tool is intended for research and educational use. Model predictions should be verified by a qualified agricultural professional before making treatment or crop-management decisions.

## License

No license has currently been specified for this repository.

# 🦴 Bone Fracture Detection AI

An AI-powered web app that analyzes bone **X-ray images** and predicts whether a
**fracture** is present, along with a confidence score. Built for a school project
with a hard constraint: **must run on a low-spec laptop with no TensorFlow / Keras /
PyTorch**.

📖 New to this project? See **[SETUP_AND_SHOWCASE_GUIDE.md](SETUP_AND_SHOWCASE_GUIDE.md)**
for a beginner-friendly setup walkthrough plus tips for presenting this in school.

## How it stays lightweight

The heavy AI runs **remotely** on the Hugging Face Inference API, so the laptop only
sends an image and reads back a result. If there is no internet or no API token, the
app automatically falls back to a **simple OpenCV image analyzer** so it always works
for the demo.

| Component        | Technology              | Runs where |
|------------------|-------------------------|------------|
| UI + backend     | Streamlit               | Laptop (light) |
| Image processing | Pillow + OpenCV + NumPy | Laptop (light) |
| AI inference     | Hugging Face API        | Hugging Face servers |
| Offline fallback | OpenCV edge heuristics  | Laptop (light) |

## Project structure

```
school-project/
├── app.py              # Streamlit dashboard (frontend + backend)
├── model.py            # AI inference: Hugging Face API + offline fallback
├── utils.py            # Image loading, enhancement, edge metrics
├── requirements.txt    # Lightweight dependencies only
├── .streamlit/
│   └── config.toml     # Dashboard theme
└── README.md
```

## Step-by-step setup (low-spec laptop friendly)

### 1. Install Python
Use Python 3.10 or newer. Check with:
```bash
python --version
```

### 2. Create a virtual environment (keeps things clean)
**macOS / Linux**
```bash
cd school-project
python -m venv venv
source venv/bin/activate
```
**Windows (PowerShell)**
```powershell
cd school-project
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install the lightweight dependencies
```bash
pip install -r requirements.txt
```
This installs only small libraries (no multi-GB ML frameworks).

### 4. (Recommended) Add a free Hugging Face token for real AI
1. Create a free account at https://huggingface.co
2. Make a token at https://huggingface.co/settings/tokens (role: **Read**)
3. Set it as an environment variable:

**macOS / Linux**
```bash
export HF_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```
**Windows (PowerShell)**
```powershell
$env:HF_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```

> ⏳ The first API call may take ~20 seconds while the model "wakes up" on
> Hugging Face. Just click **Scan Image** again if it times out the first time.

> 💡 No token? No problem. The app runs in **offline demo mode** automatically.

### 5. Run the app
```bash
streamlit run app.py
```
Your browser opens at http://localhost:8501.

### 6. Use it
1. Upload a bone X-ray (`.jpg`, `.jpeg`, `.png`).
2. Click **🔍 Scan Image**.
3. Read the prediction + confidence score.

## Swapping in a different model

Open `model.py` and change one line:
```python
HF_MODEL_ID = "Heem2/bone-fracture-detection-using-xray"
```
Any Hugging Face **image-classification** model id works. You can also plug in a
**Google Teachable Machine / Edge Impulse** model by replacing `predict_with_huggingface`
with a call to that service's REST endpoint - the return format just needs to be a
list of `{"label": ..., "score": ...}` items.

## ⚠️ Important disclaimer
This is an **educational school project**, not a medical device. It must **not** be
used for real diagnosis. Always consult a qualified radiologist.

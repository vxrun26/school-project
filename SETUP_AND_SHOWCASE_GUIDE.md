# 🦴 Setup & School Showcase Guide

This guide walks you through installing, running, and **presenting** the
Bone Fracture Detection AI project for a school submission or demo day.
It's written for someone doing this for the first time.

---

## Part 1: One-time setup

### Step 1 — Check Python is installed
Open a terminal and run:
```bash
python --version
```
You need **Python 3.10 or newer**. If it's missing, install it from
https://www.python.org/downloads/.

### Step 2 — Open the project folder
```bash
cd school-project
```

### Step 3 — Create and activate a virtual environment
A virtual environment keeps this project's libraries separate from the rest
of your computer.

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You'll know it worked when you see `(venv)` at the start of your terminal line.

### Step 4 — Install the project's libraries
```bash
pip install -r requirements.txt
```
This only installs small, fast libraries (Streamlit, Pillow, OpenCV, NumPy,
requests) — nothing heavy like TensorFlow.

### Step 5 — (Optional but recommended) Turn on real AI predictions
Without this step the app still works, using a simple offline analyzer.
With it, predictions come from a real Hugging Face AI model.

1. Go to https://huggingface.co and create a free account.
2. Go to https://huggingface.co/settings/tokens and create a new token
   (role: **Read**).
3. Copy the token, then set it in your terminal:

   **macOS / Linux**
   ```bash
   export HF_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
   ```
   **Windows (PowerShell)**
   ```powershell
   $env:HF_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
   ```

   ⚠️ This only lasts for the current terminal session. If you close the
   terminal, you'll need to set it again before running the app.

### Step 6 — Run the app
```bash
streamlit run app.py
```
A browser tab should open automatically at **http://localhost:8501**.
If not, copy that link into your browser manually.

---

## Part 2: Using the app

1. Click **Browse files** and upload a bone X-ray image (`.jpg`, `.jpeg`, or `.png`).
2. You'll see the **original** image next to a **contrast-enhanced** version.
3. Click the **🔍 Scan Image** button.
4. Read the result:
   - 🚨 **Fracture Detected** or ✅ **Normal / No Fracture**
   - A **confidence score** (how sure the AI is)
5. Expand **"See raw model output"** if you want to show the technical detail
   behind the prediction.

> 💡 Don't have X-ray images handy? Search "sample bone x-ray fracture image"
> on a site like Radiopaedia or Wikimedia Commons for freely usable images,
> or ask your teacher if the class has any provided sample images.

---

## Part 3: Presenting this in school (showcase tips)

### Suggested demo flow (5-7 minutes)
1. **Open with the problem (30 sec):** "Radiologists have to review lots of
   X-rays, and delays or human fatigue can slow down diagnosis. AI can help
   by doing a fast first-pass check."
2. **Show the dashboard (30 sec):** Point out the clean UI and the sidebar
   explaining how AI helps in radiology.
3. **Live demo (2-3 min):**
   - Upload a fractured bone X-ray → show "Fracture Detected" + confidence.
   - Upload a normal X-ray → show "Normal / No Fracture" + confidence.
   - Point out the contrast-enhanced image and mention it's a classic
     image-processing technique (CLAHE) used in real radiology software.
4. **Explain the "under the hood" architecture (1-2 min):**
   - The heavy AI model runs on Hugging Face's servers, not the laptop —
     that's why this works even on a low-spec machine.
   - If there's no internet, the app still works using a lightweight,
     offline image-analysis fallback (mention this shows you designed for
     reliability).
5. **Close with the disclaimer (30 sec):** "This is a learning project, not
   a medical device — a real diagnosis always needs a qualified doctor."

### Talking points if the teacher asks questions
- **"Why didn't you train your own model?"** → Training deep learning
  models needs GPUs and large labeled datasets; instead we used a
  pre-trained model via an API, which is a real-world pattern many
  production apps use (calling a hosted AI service instead of hosting your
  own).
- **"What happens without internet?"** → The app automatically switches to
  an offline OpenCV-based heuristic, so the demo never fails to show a
  result.
- **"Is this accurate enough for real doctors?"** → No — it's for
  education only. Explain the difference between a demo/heuristic and a
  clinically validated, regulated medical device.

### Before you present — a quick checklist
- [ ] Virtual environment activated (`(venv)` visible in terminal)
- [ ] `streamlit run app.py` tested at least once beforehand
- [ ] A few sample X-ray images ready on your desktop for quick upload
- [ ] Internet connection checked (for the online AI mode) — but you're
      covered either way thanks to the offline fallback
- [ ] Browser zoom/window sized so the dashboard is fully visible on the
      projector or shared screen

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Make sure the virtual environment is activated (Step 3), then re-run Step 4. |
| App opens but upload does nothing | Confirm the file is `.jpg`, `.jpeg`, or `.png`. |
| First AI scan takes ~20 seconds or times out | Normal — the Hugging Face model is "waking up." Click **Scan Image** again. |
| Predictions seem random with no token set | Expected — that's the offline heuristic mode. Add an `HF_API_TOKEN` for real AI predictions (Step 5). |
| `ModuleNotFoundError` for cv2/PIL/streamlit | Run `pip install -r requirements.txt` again inside the activated venv. |

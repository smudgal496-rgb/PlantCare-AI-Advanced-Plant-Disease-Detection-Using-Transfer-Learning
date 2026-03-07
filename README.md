## PlantCare AI (Frontend + Flask Backend)

### What you have
- **Frontend**: `index.html`, `about.html`, `upload.html`, `result.html`, `styles.css`
- **Backend (Flask)**: `app.py`
  - `POST /predict`: accepts an uploaded image and returns JSON with `prediction` + `image_url`
  - `GET /result`: optional server-rendered result page

### Run locally (Windows)
From `d:\Love Babbar`:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:
- Home: `http://127.0.0.1:5000/`
- Upload: `http://127.0.0.1:5000/upload`

### Using your real model
- **Model logic** lives in **`model_loader.py`**:
  - **`load_model()`** – called at startup. Put your Keras/TF or PyTorch load here (see comments in the file).
  - **`predict_image(filepath)`** – must return a dict with: `mode` ("healthy" | "disease"), `plant_type`, `condition`, `confidence`, `actions` (list of strings), and for disease also `severity`.
- **Where to put the model file**: create a `models/` folder and place your saved model there (e.g. `models/plant_model.h5`). Set `MODEL_PATH` in `model_loader.py` to match.
- **Labels**: in `model_loader.py`, set **`CLASS_LABELS`** (index → `(plant_type, condition)`) and **`RECOMMENDED_ACTIONS`** (condition → list of tips) to match your dataset.
- If no model file is present or loading fails, the app falls back to a **demo** predictor so the site still runs.
- For Keras/TF add `tensorflow` to `requirements.txt`; for PyTorch add `torch` and `torchvision`.


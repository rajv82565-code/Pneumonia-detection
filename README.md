## Pneumonia Detection (Streamlit)

### Files you should keep

- `pneumonia detection.ipynb` – trains the CNN and **saves `cnn_model.pkl`**
- `cnn_model.pkl` – pickled payload containing model JSON + weights + metadata
- `app.py` – Streamlit UI that loads `cnn_model.pkl`
- `requirements.txt` – Python dependencies

### How to generate `cnn_model.pkl`

In `pneumonia detection.ipynb`:

1. Make sure the `chest_xray/` dataset is available in the project folder.
2. Run the main training cell. At the end it will create:
   - `cnn_model.pkl`

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Uploading to GitHub

`cnn_model.pkl` is larger than 25MB, so upload it using **git push**, not the GitHub web upload.
Only keep the small code files and `cnn_model.pkl` in the repository.


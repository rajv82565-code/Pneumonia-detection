## Pneumonia Detection (Streamlit)

### What to upload to GitHub

GitHub blocks large files via the web UI (25MB). To avoid this, **export the model as TFLite** and upload the smaller file:

- `cnn_model.tflite` (recommended, small)
- `app.py`
- `requirements.txt`

### Export the smaller model

In `pneumonia detection.ipynb`, run the training cell. It will create:

- `cnn_model.keras`
- `cnn_model.tflite` (float16 quantized; usually much smaller)

Upload **`cnn_model.tflite`** to GitHub.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Cloud

- Push the repo to GitHub (include `cnn_model.tflite`)
- On Streamlit Cloud, choose:
  - **Main file path**: `app.py`




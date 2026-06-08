---
title: Employee KPI Predictor
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Employee KPI Predictor (Gradio)

A simple interactive demo: predict whether an employee will meet **more than 80%**
of their KPIs, from their profile and performance history.

- **`app.py`** trains a RandomForest from `data/employees.csv` at startup, then serves
  a Gradio UI — no MLflow or API server needed.
- Deployed as a **Gradio SDK** Space (Hugging Face installs Gradio from `sdk_version`).

## Run locally

```bash
pip install gradio -r requirements.txt
python app.py        # → http://127.0.0.1:7860
```

## Files

| File                 | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| `app.py`             | trains the model + builds the Gradio interface |
| `requirements.txt`   | model deps (Gradio comes from `sdk_version`)   |
| `data/employees.csv` | training data, read at startup                 |

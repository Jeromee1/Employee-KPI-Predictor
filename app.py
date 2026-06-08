"""Employee KPI Predictor — a simple Gradio app for Hugging Face Spaces.

Self-contained: it trains a model ONCE from data/employees.csv at startup, then
serves an interactive UI that predicts whether an employee will meet >80% of
their KPIs. No MLflow / FastAPI needed — perfect for a Gradio SDK Space.

Run locally:   python app.py   →  http://127.0.0.1:7860
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent / "data" / "employees.csv"
TARGET = "KPIs_met_more_than_80"
CATEGORICAL = ["department", "region", "education", "gender", "recruitment_channel"]
NUMERIC = [
    "no_of_trainings", "age", "previous_year_rating",
    "length_of_service", "awards_won", "avg_training_score",
]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the messy raw categories so the UI and model agree."""
    df = df.copy()
    df["gender"] = df["gender"].str.lower().replace({"female": "f", "male": "m"})
    df["education"] = df["education"].replace({"BACHELOR": "Bachelors"})
    return df


def train():
    """Load data, build a pipeline, fit it. Returns (pipeline, accuracy, options)."""
    df = _clean(pd.read_csv(DATA_PATH))
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET].astype(int)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)
    acc = pipe.score(X_te, y_te)

    # Dropdown choices, derived from the cleaned data so they always match training.
    options = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}
    return pipe, acc, options


MODEL, ACCURACY, OPTIONS = train()


def predict(department, region, education, gender, recruitment_channel,
            no_of_trainings, age, previous_year_rating,
            length_of_service, awards_won, avg_training_score):
    """Score one employee and return label-confidences + a verdict string."""
    row = pd.DataFrame([{
        "department": department, "region": region, "education": education,
        "gender": gender, "recruitment_channel": recruitment_channel,
        "no_of_trainings": no_of_trainings, "age": age,
        "previous_year_rating": previous_year_rating,
        "length_of_service": length_of_service, "awards_won": int(awards_won),
        "avg_training_score": avg_training_score,
    }])
    proba = float(MODEL.predict_proba(row)[0][1])  # P(meets KPI)
    verdict = "✅ Likely to MEET >80% of KPIs" if proba >= 0.5 else "⚠️ At risk of NOT meeting KPIs"
    return {"Meets >80% KPI": proba, "Does not": 1 - proba}, f"### {verdict}\nConfidence: **{proba:.0%}**"


with gr.Blocks(title="Employee KPI Predictor") as demo:
    gr.Markdown(
        "# 📊 Employee KPI Predictor\n"
        "Predict whether an employee will meet **more than 80%** of their KPIs.\n"
        f"*Model: RandomForest trained on {DATA_PATH.name} · held-out accuracy "
        f"**{ACCURACY:.1%}***"
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Profile")
            department = gr.Dropdown(OPTIONS["department"], label="Department", value=OPTIONS["department"][0])
            region = gr.Dropdown(OPTIONS["region"], label="Region", value="region_2")
            education = gr.Dropdown(OPTIONS["education"], label="Education", value="Bachelors")
            gender = gr.Radio(OPTIONS["gender"], label="Gender", value="m")
            recruitment_channel = gr.Dropdown(OPTIONS["recruitment_channel"], label="Recruitment channel", value="sourcing")
        with gr.Column():
            gr.Markdown("#### Performance & tenure")
            no_of_trainings = gr.Slider(1, 10, value=1, step=1, label="No. of trainings")
            age = gr.Slider(18, 65, value=30, step=1, label="Age")
            previous_year_rating = gr.Slider(1, 5, value=3, step=1, label="Previous year rating")
            length_of_service = gr.Slider(1, 40, value=5, step=1, label="Length of service (years)")
            avg_training_score = gr.Slider(39, 99, value=70, step=1, label="Avg training score")
            awards_won = gr.Radio([0, 1], label="Awards won?", value=0)

    btn = gr.Button("Predict", variant="primary")
    with gr.Row():
        out_label = gr.Label(label="Probability", num_top_classes=2)
        out_verdict = gr.Markdown()

    inputs = [department, region, education, gender, recruitment_channel,
              no_of_trainings, age, previous_year_rating,
              length_of_service, awards_won, avg_training_score]
    btn.click(predict, inputs=inputs, outputs=[out_label, out_verdict])

    gr.Examples(
        examples=[
            ["Technology", "region_26", "Bachelors", "m", "sourcing", 1, 30, 3, 5, 0, 77],
            ["Sales & Marketing", "region_2", "Masters & above", "f", "referred", 3, 45, 5, 12, 1, 92],
        ],
        inputs=inputs,
    )


if __name__ == "__main__":
    demo.launch()

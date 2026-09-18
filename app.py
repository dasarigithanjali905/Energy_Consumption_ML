import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    PROJECT_TITLE,
    ensure_directories,
)
from hf_api import generate_ai_response
from predict import predict_ticket, predict_from_dataframe
from train import train_and_save
from utils import load_dataset


ensure_directories()


st.set_page_config(
    page_title=PROJECT_TITLE,
    layout="wide",
    page_icon="⚡",
)


def load_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def ensure_trained():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
        train_and_save(DATA_PATH)


load_css()
ensure_trained()


def dashboard_page():
    st.markdown('<h1 class="main-title">Energy Consumption Dashboard</h1>', unsafe_allow_html=True)

    try:
        df = load_dataset(DATA_PATH)
        dataset_size = len(df)
    except Exception:
        df = None
        dataset_size = 0

    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Dataset Size", f"{dataset_size:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    best_accuracy = None
    if metrics:
        best_model = metrics.get("best_model")
        accuracies = metrics.get("accuracies", {})
        best_accuracy = accuracies.get(best_model)

    with col2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        if best_accuracy is not None:
            st.metric("Best Model Accuracy", f"{best_accuracy:.3f}")
        else:
            st.metric("Best Model Accuracy", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Models Trained", "3")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    if df is not None:
        target_col = metrics.get("target_column") if metrics else None
        if target_col and target_col in df.columns:
            class_counts = df[target_col].value_counts().reset_index()
            class_counts.columns = ["class", "count"]
            fig = px.bar(
                class_counts,
                x="class",
                y="count",
                title="Class Distribution",
                color="class",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Target column not available for class distribution chart.")
    else:
        st.warning("Dataset not available. Please upload data in Admin Panel.")


def ticket_classifier_page():
    st.markdown('<h2 class="main-title">Ticket Classifier</h2>', unsafe_allow_html=True)
    st.markdown("Provide ticket text to classify into categories.")

    ticket_text = st.text_area("Ticket Text", height=200)
    if st.button("Classify Ticket"):
        if not ticket_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Classifying ticket..."):
                try:
                    result = predict_ticket(ticket_text)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Category (encoded)", str(result["prediction"]))
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.3f}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Prediction error: {str(e)}")


def ai_response_page():
    st.markdown('<h2 class="main-title">AI Response Generator</h2>', unsafe_allow_html=True)
    st.markdown("Generate professional responses to customer tickets using an LLM.")

    ticket_text = st.text_area("Ticket Text", height=200)

    if st.button("Generate Response"):
        if not ticket_text.strip():
            st.warning("Please enter ticket text to generate a response.")
        else:
            with st.spinner("Generating AI response..."):
                response = generate_ai_response(ticket_text)
                st.markdown("#### LLM Response")
                st.write(response)


def model_analytics_page():
    st.markdown('<h2 class="main-title">Model Analytics</h2>', unsafe_allow_html=True)

    if not os.path.exists(METRICS_PATH):
        st.warning("Metrics file not found. Train a model first.")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    accuracies = metrics.get("accuracies", {})
    if accuracies:
        acc_df = pd.DataFrame(
            [{"model": name, "accuracy": acc} for name, acc in accuracies.items()]
        )
        fig_acc = px.bar(
            acc_df,
            x="model",
            y="accuracy",
            title="Accuracy Comparison",
            color="model",
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    best_model_name = metrics.get("best_model")
    conf_matrices = metrics.get("confusion_matrices", {})
    if best_model_name in conf_matrices:
        matrix = np.array(conf_matrices[best_model_name])
        fig_cm = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix - {best_model_name}",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    feature_importance = metrics.get("feature_importance")
    if feature_importance:
        fi_values = feature_importance.get("values", [])
        indices = feature_importance.get("indices", [])
        if fi_values:
            fi_df = pd.DataFrame(
                {"feature_index": indices, "importance": fi_values}
            ).sort_values("importance", ascending=False).head(20)
            fig_fi = px.bar(
                fi_df,
                x="feature_index",
                y="importance",
                title="Top Feature Importances (indices)",
            )
            st.plotly_chart(fig_fi, use_container_width=True)


def model_prediction_page():
    st.markdown('<h2 class="main-title">Model Prediction + LLM Insight</h2>', unsafe_allow_html=True)
    st.markdown("Provide feature values to get a model prediction and an AI-generated explanation.")

    try:
        df_sample = load_dataset(DATA_PATH).head(1)
    except Exception:
        st.warning("Dataset not available. Please upload data in Admin Panel.")
        return

    feature_cols = [c for c in df_sample.columns if c != "EnergyConsumption"]
    with st.form("prediction_form"):
        inputs = {}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df_sample[col]):
                default_val = float(df_sample[col].iloc[0]) if not pd.isna(df_sample[col].iloc[0]) else 0.0
                inputs[col] = st.number_input(col, value=default_val)
            else:
                default_str = str(df_sample[col].iloc[0]) if not pd.isna(df_sample[col].iloc[0]) else ""
                inputs[col] = st.text_input(col, value=default_str)
        submitted = st.form_submit_button("Predict & Generate Explanation")

    if submitted:
        with st.spinner("Running model prediction and generating explanation..."):
            try:
                df_input = pd.DataFrame([inputs])
                preds = predict_from_dataframe(df_input)
                prediction_value = float(preds[0])

                st.subheader("Model Prediction")
                st.metric("Predicted Energy Consumption", f"{prediction_value:.3f}")

                feature_text = ", ".join(f"{k}={v}" for k, v in inputs.items())
                llm_prompt = (
                    "Given the following building conditions, explain the predicted energy consumption "
                    "in a concise, professional way.\n\n"
                    f"Conditions: {feature_text}\n"
                    f"Predicted Energy Consumption: {prediction_value:.3f}"
                )
                llm_response = generate_ai_response(llm_prompt)

                st.subheader("AI Explanation")
                st.write(llm_response)
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction or LLM generation failed: {str(e)}")


def admin_panel_page():
    st.markdown('<h2 class="main-title">Admin Panel</h2>', unsafe_allow_html=True)
    st.markdown("Upload new data and retrain the model.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
            st.success("Dataset uploaded successfully.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to read uploaded CSV: {str(e)}")

    if st.button("Retrain Model"):
        with st.spinner("Retraining model..."):
            try:
                train_and_save(DATA_PATH)
                st.success("Model retrained successfully.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Retraining failed: {str(e)}")


def sidebar_navigation():
    st.sidebar.markdown('<p class="sidebar-title">Navigation</p>', unsafe_allow_html=True)
    return st.sidebar.radio(
        "Go to",
        (
            "Dashboard",
            "Model Prediction",
            "Ticket Classifier",
            "AI Response Generator",
            "Model Analytics",
            "Admin Panel",
        ),
    )


def main():
    page = sidebar_navigation()
    if page == "Dashboard":
        dashboard_page()
    elif page == "Model Prediction":
        model_prediction_page()
    elif page == "Ticket Classifier":
        ticket_classifier_page()
    elif page == "AI Response Generator":
        ai_response_page()
    elif page == "Model Analytics":
        model_analytics_page()
    elif page == "Admin Panel":
        admin_panel_page()


if __name__ == "__main__":
    main()


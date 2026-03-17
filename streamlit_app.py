import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import random

# Page configuration
st.set_page_config(
    page_title="Jargomatic",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }

    /* App canvas */
    html, body, .stApp, [data-testid='stAppViewContainer'] > .main, .block-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Hide Streamlit chrome (optional) */
    header, footer { visibility: hidden !important; height: 0 !important; }

    .block-container { padding-top: 3rem !important; padding-left: 36px !important; padding-right: 36px !important; }

    .app-title { text-align: center; font-size: 3.2rem; margin: 0; color: #111 !important; font-weight: 800; }
    .app-subtitle { text-align: center; color: #666; margin-top: 4px; margin-bottom: 18px; }

    textarea, .stTextArea textarea, .stTextInput input, input[type="text"], .stNumberInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e6e6e6 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: none !important;
        font-size: 15px !important;
    }

    /* --- FIX 1: Make disabled text area readable --- */
    textarea[disabled], input[disabled] {
        color: #000000 !important; /* Force text to black */
        -webkit-text-fill-color: #000000 !important; /* For Safari/Chrome */
        opacity: 1 !important; /* Override default disabled opacity */
        background-color: #f5f5f5 !important; /* Light gray to show it's disabled */
    }

    textarea::placeholder, input::placeholder { color: #999 !important; }

    .stButton>button, .stDownloadButton>button {
        background: #111 !important;
        color: #fff !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        font-weight: 600 !important;
        border: none !important;
    }

    /* Expander Container & Content (The white box) */
    .stExpander, .streamlit-expanderContent {
        background-color: #ffffff !important;
        border: 1px solid #f0f0f0 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        color: #000000 !important;
    }

    /* --- FIX 2: Force expander header to be white/black --- */
    .streamlit-expanderHeader, /* Legacy Streamlit */
    [data-testid="stExpander"] summary /* Modern Streamlit */
    {
        background: #ffffff !important; /* Use background property */
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px;
        box-shadow: none !important;
    }

    /* Force hover/active/focus states to also be white/black */
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:active,
    [data-testid="stExpander"] summary:focus {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Defensive: reduce visual surprises */
    * { box-shadow: none !important; background-image: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model(model_path: str = "models/t5_jargon_v1"):
    """Load the fine-tuned T5 model + tokenizer. Returns (model, tokenizer, device) or (None, None, None) if not found."""
    try:
        if not os.path.exists(model_path):
            return None, None, None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        # We don't call st.* functions inside cached function (keeps cache deterministic),
        # the caller will show the error UI instead.
        return None, None, None


def translate_text(text: str, model, tokenizer, device, max_length: int = 128):
    """Translate the provided text to 'corporate jargon' using the loaded model."""
    if not text.strip():
        return ""

    prefix = "Translate to corporate jargon:"
    input_text = f"{prefix} {text.strip()}"

    # Tokenize safely with truncation/padding and move tensors to device
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            do_sample=False,
        )

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated


def clear_input():
    st.session_state["input_text"] = ""


def main():
    st.markdown("<div class='app-title'>Jargomatic</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Transform plain English into enterprise-grade corporate jargons</div>", unsafe_allow_html=True)

    st.divider()

    model, tokenizer, device = load_model()

    if model is None:
        st.error("⚠️ Model not found. Please train or place the fine-tuned model in `models/t5_jargon_v1`.")
        with st.expander("How to fix"):
            st.markdown("""
            1. Run your training pipeline and confirm `models/t5_jargon_v1` exists.
            2. The directory must contain the model checkpoint and tokenizer files (pytorch_model.bin, config.json, tokenizer files).
            3. Re-run this app.
            """)
        return

    st.success(f"✅ Model loaded on **{device.type.upper()}**")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📝 Input")
        input_text = st.text_area(
            "Enter your plain English text:",
            key="input_text",
            height=220,
            placeholder="Type something simple like 'We need to improve our sales'...",
            label_visibility="collapsed",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            translate_button = st.button("🚀 Translate to Jargon", use_container_width=True)
        with c2:
            clear_button = st.button(
                "🧽 Clear", 
                use_container_width=True,
                on_click=clear_input
            )

    with col2:
        st.markdown("### 💼 Corporate Translation")
        output_area = st.empty()

    if translate_button:
        if not input_text or not input_text.strip():
            st.warning("⚠️ Please enter some text to translate!")
        else:
            spinner_messages = [
                "Operationalizing synergies...",
                "Recalibrating value proposition...",
                "Leveraging core competencies...",
                "Driving stakeholder alignment...",
                "Optimizing strategic frameworks...",
                "Deploying best practices...",
            ]
            spinner_message = random.choice(spinner_messages)

            with st.spinner(spinner_message):
                try:
                    translated_output = translate_text(input_text, model, tokenizer, device)
                except Exception as e:
                    translated_output = f"Error during generation: {e}"

            output_area.text_area("Translation:", value=translated_output, height=220, label_visibility="collapsed")
    else:
        output_area.text_area(
            "Translation:",
            value="Your corporate jargon will appear here...",
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()

    with st.expander("ℹ️ About this App"):
        st.markdown(
            """
        **About Jargomatic**

        This Streamlit app serves as the interactive demo for a complete MLOps project.

        * **Core Model:** The translation is performed by a fine-tuned `google/flan-t5-small` model. It's a custom 8-layer, 6-head configuration loaded locally from the `models/t5_jargon_v1` directory.

        * **MLOps Pipeline:** This app is the final step. It's supported by a full pipeline built with **Dagster** that automates the entire model lifecycle:
            * **Data Generation:** A `weak_labeler` asset creates new training data.
            * **Data Versioning:** All datasets (like `training_data.jsonl`) are versioned using **DVC**.
            * **Training & QA:** The pipeline includes assets for `model_trainer` and a `model_qa_gate` to ensure quality.
            * **Monitoring:** It features a `jargon_drift_detector` and a `jargon_candidate_sensor` to find new jargon and monitor for data drift.
            * **Registry:** **MLflow** is used for experiment tracking and model registration.
        """
        )


if __name__ == "__main__":
    main()
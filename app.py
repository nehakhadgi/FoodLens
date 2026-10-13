import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="FoodLens", layout="centered")
st.title("FoodLens 🥑")
st.caption("Results are estimates, not medical advice.")

@st.cache_resource
def load_model_and_tokenizer():
    MODEL_ID = "vikhyatk/moondream2"
    revision = "2024-08-26"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    
    # Load the model directly without manually modifying config attributes
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        revision=revision
    )
    model.eval()
    return model, tokenizer

try:
    with st.spinner("Loading lightweight Vision Model... This might take a few moments."):
        model, tokenizer = load_model_and_tokenizer()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

def run_model(image, prompt):
    if image is None:
        return "Please upload a meal photo."
    try:
        with torch.inference_mode():
            enc_image = model.encode_image(image)
            response = model.answer_question(enc_image, prompt, tokenizer)
        return response
    except Exception as e:
        return f"Error during analysis: {e}"

def build_meal_prompt(preferences, portion):
    preferences = preferences.strip() or "No preferences supplied."
    portion = portion.strip() or "No portion context supplied."
    return f""""
Analyze this meal photo as a cautious meal assistant.
Dietary preferences: {preferences}
Portion context: {portion}

Provide an analysis highlighting:
1. Visible foods and ingredients.
2. Potential preference matches and warnings.
3. Rough nutrition ranges.
4. Limitations of visual-only analysis.
""".strip()

with st.form("meal_form"):
    uploaded_file = st.file_uploader("Upload a meal photo", type=["png", "jpg", "jpeg"])
    preferences = st.text_input("Dietary preferences or restrictions", placeholder="e.g., vegetarian, nut-free")
    portion = st.text_input("Portion context", placeholder="e.g., one large bowl, small plate")
    submit_button = st.form_submit_button("Analyze Meal")

if submit_button:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Meal", use_container_width=True)
        with st.spinner("Analyzing your meal..."):
            prompt = build_meal_prompt(preferences, portion)
            analysis = run_model(image, prompt)
            st.subheader("FoodLens Analysis")
            st.markdown(analysis)
    else:
        st.warning("Please upload an image first!")

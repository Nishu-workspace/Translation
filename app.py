import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Streamlit page configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="Language Translation", page_icon="🌍", layout="wide")

# Path to the model
model_path = "D:/Projects/Translation/tf_model"

# Cache the model and tokenizer loading
@st.cache_resource
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(path)
    return tokenizer, model

# Load model and tokenizer
try:
    st.info("Loading model and tokenizer. This may take a few moments...")
    tokenizer, model = load_model_and_tokenizer(model_path)
    st.success("Model and tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Title of the app
st.title("Language Translation with T5 Model")
st.write("Enter text below and select the translation direction.")

# Input text from the user
input_text = st.text_area("Enter text for translation:")

# Dropdown menu for selecting translation direction
translation_direction = st.selectbox(
    "Choose translation direction:",
    ("English to Hindi", "Hindi to English")
)

# Define function to translate text
def translate_text(input_text, translation_direction):
    try:
        # Determine the translation task based on the selected direction
        if translation_direction == "English to Hindi":
            task = "translate English to Hindi: "
        else:
            task = "translate Hindi to English: "
        
        # Prepare input for the model
        input_with_task = task + input_text
        inputs = tokenizer(input_text, return_tensors="tf", max_length=128, truncation=True)
        
        # Generate output
        outputs = model.generate(**inputs)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    except Exception as e:
        return f"Error during translation: {e}"

# Button to trigger translation
if st.button("Translate"):
    if input_text.strip():
        with st.spinner("Translating..."):
            translated_text = translate_text(input_text, translation_direction)
        if "Error during translation" in translated_text:
            st.error(translated_text)
        else:
            st.success(f"Translated Text:\n\n{translated_text}")
    else:
        st.warning("Please enter some text to translate.")

# Footer
st.markdown("---")
st.markdown(
    "Created with ❤️ using [Streamlit](https://streamlit.io/) and [Transformers](https://huggingface.co/transformers)."
)

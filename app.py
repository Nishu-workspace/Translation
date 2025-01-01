import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
model_path = "D:/Projects/Translation/tf_model"
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('tf_model')
model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model")

# Streamlit page configuration
st.set_page_config(page_title="Language Translation", page_icon="🌍", layout="wide")

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
    # Determine the translation task based on the selected direction
    if translation_direction == "English to Hindi":
        task = "translate English to Hindi: "
    else:
        task = "translate Hindi to English: "
    
    # Prepare input for the model
    input_with_task = task + input_text
    inputs = tokenizer(input_with_task, return_tensors="pt", max_length=128, truncation=True)
    
    # Generate output
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result

# Button to trigger translation
if st.button("Translate"):
    if input_text.strip():
        with st.spinner("Translating..."):
            translated_text = translate_text(input_text, translation_direction)
        st.success(f"Translated Text: {translated_text}")
    else:
        st.warning("Please enter some text to translate.")

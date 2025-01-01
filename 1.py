import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
model_path = "D:/Projects/Translation/tf_model"
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('tf_model')
model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model")
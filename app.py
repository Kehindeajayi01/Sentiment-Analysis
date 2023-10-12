import streamlit as st
import torch
import numpy as np
from simpletransformers.model import TransformerModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import onnxruntime

st.title("Sentiment Analysis App with RoBERTa")
# Create a text input widget for user input
user_input = st.text_area(" ##### Enter some text and we will predict its sentiment", "")

# function to convert the tensor to a numpy array
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# load the tokenizer used for RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# process the user input
input_ids = torch.tensor(tokenizer.encode(user_input, add_special_tokens=True)).unsqueeze(0)  # Batch size 1

# load the model for inference using onnx
ort_session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")

sent_label = ["Negative", "Positive"]
# Create a button to trigger sentiment analysis
if st.button('Analyze Sentiment'):

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
    ort_out = ort_session.run(None, ort_inputs)
   # Make predictions
    pred = np.argmax(ort_out)

    st.write(f" ### Sentiment: {sent_label[pred]}")


import streamlit as st
from transformers import AutoTokenizer, BertForSequenceClassification, BartForConditionalGeneration
import torch
import numpy as np

if "kr_dir" not in st.session_state:
  st.session_state["kr_dir"] = "/home/bill/Desktop/kr/"
if "classifier_tokenizer" not in st.session_state:
  st.session_state["classifier_tokenizer"] = AutoTokenizer.from_pretrained("klue/bert-base")
if "classifier" not in st.session_state:
  st.session_state["classifier"] = BertForSequenceClassification.from_pretrained(st.session_state["kr_dir"] + "kr-honorifics-classification/trained_model/")
if "transferer_tokenizer" not in st.session_state:
  st.session_state["transferer_tokenizer"] = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
if "transferer" not in st.session_state:
  st.session_state["transferer"] = BartForConditionalGeneration.from_pretrained(st.session_state["kr_dir"] + "kr-honorifics-style-transfer/trained_model/")
st.title("Korean Honorifics App")
sentence = st.text_input("Input Korean sentence:")
if sentence != "":
  classifier_inputs = st.session_state["classifier_tokenizer"](sentence, padding=True, return_tensors="pt")
  logits = st.session_state["classifier"](**classifier_inputs).logits
  predicted_probabilities = torch.nn.functional.softmax(logits, dim=-1)
  is_honorific = np.argmax(predicted_probabilities.detach().numpy(), axis=1)[0]
  if is_honorific == 1:
    st.success(sentence + " is honorific!")
  else:
    tokenizer = st.session_state["transferer_tokenizer"]
    transferer_input = tokenizer(sentence, padding=True, truncation=True, max_length=128)
    transferer_input["input_ids"] = [tokenizer.bos_token_id] + transferer_input["input_ids"] + [tokenizer.eos_token_id]
    transferer_output = st.session_state["transferer"].generate(torch.tensor([transferer_input["input_ids"]]))[0]
    honorific_sentence = tokenizer.decode(transferer_output, skip_special_tokens=True)
    st.error("Honorific sentence: " + honorific_sentence)

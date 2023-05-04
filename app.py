from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from credentials import API_KEY
# import gradio as gr
import streamlit as st
import sys
import os

# streamlit run app.py to run the application

os.environ["OPENAI_API_KEY"] = API_KEY

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="default") # default, compact, and tree_summarize
    return response.response

if not os.path.exists('index.json'):
    construct_index("docs")

st.title("Max's Custom-trained AI Chatbot")
input_text = st.text_area("Enter your text", height=200)
submit_button = st.button('Submit')

if submit_button:
    response = chatbot(input_text)
    st.write(response)
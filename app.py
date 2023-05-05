from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from credentials import API_KEY, USERNAMES
# import gradio as gr
import streamlit as st
import sys
import os
import time

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


def add_document(input_text):
    input_text = input_text.replace("/add-doc", "").strip()
    input_text = input_text.split(" ", 2)
    document_user = input_text[0]
    document_name = input_text[1]

    if document_user not in USERNAMES:
        return "You are not authorized to add documents!"
    else:
        document_content = input_text[2]
        with open(f"docs/{document_name}.txt", "w") as document:
            document.write(document_content)
        return "Document added successfully!"


def delete_document(input_text):
    input_text = input_text.replace("/delete-doc", "").strip()
    input_text = input_text.split(" ", 1)
    document_user = input_text[0]
    document_name = input_text[1]

    if document_user not in USERNAMES:
        return "You are not authorized to delete documents!"
    else:
        os.remove(f"docs/{document_name}.txt")
        return "Document deleted successfully!"


def update_document_list():
    document_list = []
    for document in os.listdir("docs"):
        document_list.append(document)
    return document_list

# only construct the index if one doesn't already exist
if not os.path.exists('index.json'):
    construct_index("docs")



# Use the streamlit library to create a simple web app
st.title("Max's Custom-trained AI Chatbot")
#display the list of document names
st.write("Trained on the following Documents:")
container = st.empty()
text_placeholder = container.markdown(update_document_list())

input_text = st.text_area("Enter your text", height=200)
submit_button = st.button('Submit')

if submit_button:
    # get the current length of the docs folder
    documentCount = len(os.listdir("docs"))

    if input_text.startswith("/add-doc"):
        message = add_document(input_text)
        st.write(message)        

    elif input_text.startswith("/delete-doc"):
        message = delete_document(input_text)
        st.write(message)

    # If the user adds or deletes a new document, re-construct the index
    if len(os.listdir("docs")) != documentCount:
        construct_index("docs")        
        time.sleep(2)
        text_placeholder.markdown(update_document_list())
    else:
        response = chatbot(input_text)
        st.write(response)    

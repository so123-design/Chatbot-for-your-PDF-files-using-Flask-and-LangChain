# Chatbot for your PDF files using Flask and LangChain

## Project Overview
This project demonstrates how to build a chatbot that interacts with users and answers questions related to the content of a specific PDF document. Using **Flask** for the backend and **LangChain** for working with large language models (LLMs), the chatbot can comprehend the contents of a PDF file, analyze it, and provide responses based on user queries.

## Introduction
Chatbots powered by large language models have transformed how we interact with digital systems. However, most chatbots rely on generalized knowledge, limiting their effectiveness for document-specific inquiries. This project addresses that limitation by integrating document-based comprehension into a chatbot. Users can upload a PDF file, and the chatbot will analyze its content, allowing for interactive and meaningful conversations specific to the document.

![Alt text]([Chatbot PDF 2.PNG](https://github.com/so123-design/Chatbot-for-your-PDF-files-using-Flask-and-LangChain/blob/c67aa5e1fe48039fe70ca9604ee07a5eb463fb42/Chatbot%20PDF%20%202.PNG))

## Technologies Used
- **Flask** – A lightweight web framework for building the chatbot’s backend.
- **LangChain** – A framework for building applications powered by large language models (LLMs).
- **IBM Watsonx** – A powerful AI service used for language model integration.
- **Hugging Face Transformers** – Used for text embeddings and understanding document content.
- **PyPDFLoader** – A library for extracting text from PDF documents.
- **Chroma** – A vector store used to efficiently retrieve relevant document chunks.

## Features
- Upload and process PDF files.
- Store and analyze document content using embeddings.
- Respond to user queries based on document content.
- Maintain chat history for contextual responses.

## Project Structure
The project consists of two main components:
1. **Backend (Flask Server)**: Handles user queries and PDF processing.
2. **Worker Module**: Manages language model interactions and vector database operations.

## How It Works
1. **User Uploads a PDF** – The user selects a PDF document, and the backend processes it using PyPDFLoader.
2. **Text Extraction & Embeddings** – The document’s text is extracted, split into chunks, and embedded using Hugging Face models.
3. **Storage in Vector Database** – The processed document is stored in Chroma, allowing efficient retrieval of relevant content.
4. **User Queries** – The user submits a question related to the document.
5. **Response Generation** – LangChain retrieves relevant document sections and generates a response using Watsonx LLM.
6. **Chat History Maintenance** – Past interactions are stored to provide contextual responses in ongoing conversations.

## Code Breakdown
### `server.py`
Handles incoming HTTP requests using Flask. It provides routes for:
- Rendering the frontend UI.
- Processing user messages and document uploads.
- Returning responses in JSON format.

```python
import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker  # Import the worker module

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render the index.html template

@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage']
    print('user_message', user_message)

    bot_response = worker.process_prompt(user_message)

    return jsonify({"botResponse": bot_response}), 200

@app.route('/process-document', methods=['POST'])
def process_document_route():
    if 'file' not in request.files:
        return jsonify({"botResponse": "It seems like the file was not uploaded correctly. Try again."}), 400

    file = request.files['file']
    file_path = file.filename
    file.save(file_path)

    worker.process_document(file_path)

    return jsonify({"botResponse": "PDF processed! You can now ask questions about its content."}), 200

if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
```


### `worker.py`
Responsible for:
- Loading and processing PDF documents.
- Generating text embeddings.
- Storing and retrieving relevant document sections.
- Querying the chatbot model and maintaining chat history.

```python
import os
import torch
import logging
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxLLM

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

def init_llm():
    global llm_hub, embeddings
    logger.info("Initializing WatsonxLLM and embeddings...")
    MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
    WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
    PROJECT_ID = "skills-network"
    model_parameters = {"max_new_tokens": 256, "temperature": 0.1}
    
    llm_hub = WatsonxLLM(model_id=MODEL_ID, url=WATSONX_URL, project_id=PROJECT_ID, params=model_parameters)
    
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )

def process_document(document_path):
    global conversation_retrieval_chain
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    
    db = Chroma.from_documents(texts, embedding=embeddings)
    
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
    )

def process_prompt(prompt):
    global conversation_retrieval_chain, chat_history
    output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    chat_history.append((prompt, answer))
    return answer

init_llm()
```


## Use Cases
- **Research Assistance** – Quickly extract insights from academic papers.
- **Legal Document Analysis** – Answer questions from lengthy legal documents.
- **Business Reports** – Analyze corporate reports and financial documents.

## Conclusion
This project showcases how AI-powered chatbots can be enhanced by document-based knowledge. By leveraging **Flask, LangChain, and IBM Watsonx**, we enable document-aware conversational AI that can provide meaningful insights from user-uploaded PDFs.



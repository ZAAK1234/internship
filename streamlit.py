import streamlit as st
import io
import PyPDF2
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import BaseLLM
from typing import Optional, List, Mapping, Any
from langchain.schema import Generation, LLMResult

# Define the local model directories
sentence_transformer_model_dir = "all-MiniLM-L6-v2"
summarization_model_dir = 'facebook/bart-large-cnn'  # Corrected model identifier

# --- Helper Functions ---

def get_file_text(uploaded_files):
    text = ""
    for filename, file_content in uploaded_files.items():
        if filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(file_content))
            for paragraph in doc.paragraphs:
                text += paragraph.text
        elif filename.endswith(".txt"):
            text += file_content.decode('utf-8')  # Directly decode bytes to string
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    
    print("Types of objects in text_chunks:", [type(t) for t in text_chunks])  # Check types before embedding
    model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2') # Define model before using it
    embeddings = model.client.encode([t for t in text_chunks]) # Assuming text_chunks are strings
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=model)
    print("Vector store created successfully.")
    return vector_store

from typing import Union

class SummarizationLLM(BaseLLM):
    model_dir: str
    #tokenizer: Union[AutoTokenizer, None] = None
    #model: Union[AutoModelForSeq2SeqLM, None] = None

    def __init__(self, model_dir: str):
        super().__init__(model_dir=model_dir)
        self.model_dir = model_dir
        self.initialize_model()  

    def initialize_model(self):
        if self.tokenizer is None and self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        self.initialize_model()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_dir": self.model_dir}

    # Remove the duplicate _generate method
    # def _generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    #     return self._call(prompt, stop)

    @property
    def _llm_type(self) -> str:
        return "custom_summarization_llm"

    def _generate(self, prompt: str, stop: Optional[List[str]] = None) -> LLMResult:
        self.initialize_model()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Add the missing closing parenthesis here
        return LLMResult(generations=[[Generation(text=summary)]])


print(isinstance(SummarizationLLM, type))

def get_conversation_chain(vectorstore):
    
    summarizer = SummarizationLLM(model_dir=summarization_model_dir)  # Replace with your SummarizationLLM
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=summarizer,  # Use your SummarizationLLM here
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- Streamlit App ---
def main():  # Wrap your Streamlit app code in a function
    st.title("Chat with Your Documents")
    # File Upload
    uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    # Process Files and Create Chatbot
    if uploaded_files:
    # Process files
        processed_files = {file.name: file.read() for file in uploaded_files}
        raw_text = get_file_text(processed_files)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vector_store(text_chunks)

    # Create conversation chain
        if vectorstore:
            conversation_chain = get_conversation_chain(vectorstore)

    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # User Input
    user_question = st.text_input("Ask a question:")

    # Handle User Input
    if user_question:
        if "conversation_chain" in locals():
            response = conversation_chain({"question": user_question})
            st.session_state["chat_history"].append((user_question, response['answer']))
        else:
            st.write("Please upload and process documents before asking questions.")

    # Display Chat History
    for user_msg, bot_msg in st.session_state["chat_history"]:
        st.write(f"**User:** {user_msg}")
        st.write(f"**Bot:** {bot_msg}")

if __name__=="__main__":
    main()




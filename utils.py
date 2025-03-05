import os
from typing import List, Dict
import PyPDF2
from io import BytesIO
import tiktoken
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_text(text: str) -> List[str]:
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks: List[str]) -> FAISS:
    """Create FAISS vector store from text chunks."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

def create_conversation_chain(vector_store: FAISS) -> RetrievalQA:
    """Create conversation chain with memory."""
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    return conversation_chain

def get_conversation_response(
    conversation_chain: RetrievalQA,
    question: str
) -> Dict:
    """Get response from conversation chain."""
    response = conversation_chain({"query": question})
    return {
        "answer": response["result"],
        "source_documents": response["source_documents"]
    }
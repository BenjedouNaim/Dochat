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
from langchain.chains import RunnableSequence
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
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

def create_conversation_chain(vector_store: FAISS) -> RunnableSequence:
    """Create conversation chain."""
    llm = ChatOpenAI(temperature=0.7)

    # Create the question answering prompt
    prompt = PromptTemplate.from_template(
        """Answer the following question based on the provided context:

        Context: {context}

        Question: {question}

        Answer:"""
    )

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    retrieval_chain = vector_store.as_retriever() | document_chain

    return retrieval_chain

def get_conversation_response(
    conversation_chain: RunnableSequence,
    question: str
) -> Dict:
    """Get response from conversation chain."""
    response = conversation_chain.invoke({
        "question": question
    })
    return {
        "answer": response,
        "source_documents": []  # Updated to match new chain response format
    }
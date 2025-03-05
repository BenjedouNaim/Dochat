import os
from typing import List, Dict
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

def create_conversation_chain(vector_store: FAISS):
    """Create conversation chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3  # Lower temperature for more focused responses
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    return chain, vector_store

def get_conversation_response(
    conversation_chain,
    question: str
) -> Dict:
    """Get response from conversation chain."""
    chain, vector_store = conversation_chain
    docs = vector_store.similarity_search(question)
    response = chain({"input_documents": docs, "question": question})

    return {
        "answer": response["output_text"],
        "source_documents": docs
    }
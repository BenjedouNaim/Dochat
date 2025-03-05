import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils import (
    extract_text_from_pdf,
    process_text,
    create_vector_store,
    create_conversation_chain,
    get_conversation_response
)

# Set the API key in the environment from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📚",
    layout="wide"
)

# Load custom CSS (optional, retained for non-chat styling)
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for conversation controls
st.header("Conversation Controls")
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    chat_data = [
        {
            'Timestamp': msg['timestamp'],
            'Role': msg['role'].capitalize(),
            'Message': msg['content']
        }
        for msg in st.session_state.chat_history
    ]
    chat_df = pd.DataFrame(chat_data)
    csv = chat_df.to_csv(index=False)
    st.download_button(
        label="💾 Export Conversation",
        data=csv,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        help="Download your conversation history as a CSV file"
    )
else:
    st.info("No conversation history yet.")

# Main header
st.markdown(
    """
    <div class='main-header'>
        <h1>📚 Document Chat Assistant</h1>
        <p>Upload your documents and start an intelligent conversation about their content.
        The AI assistant will help you extract insights and answer questions.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# File upload section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("### 📄 Upload Your Documents")
st.markdown(
    "Upload PDF or text files to begin. Multiple files are supported.",
    help="The assistant will process your documents and allow you to ask questions."
)
uploaded_files = st.file_uploader(
    "Drag and drop your files here",
    type=["pdf", "txt"],
    accept_multiple_files=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Process uploaded files
if uploaded_files:
    with st.spinner("📚 Processing your documents..."):
        try:
            all_text = ""
            for file in uploaded_files:
                if file.type == "application/pdf":
                    text = extract_text_from_pdf(file.read())
                else:  # txt file
                    text = file.read().decode()
                all_text += text + "\n\n"

            text_chunks = process_text(all_text)
            st.session_state.vector_store = create_vector_store(text_chunks)
            st.session_state.conversation = create_conversation_chain(st.session_state.vector_store)
            st.success("✅ Documents processed successfully! You can now start asking questions.")

            with st.expander("📝 View Processed Documents", expanded=False):
                st.markdown("### Document Content Preview")
                preview_text = all_text[:500] + "..." if len(all_text) > 500 else all_text
                st.markdown(preview_text)
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")

# Chat interface
if st.session_state.conversation is not None:
    st.markdown("### 💬 Chat with Your Documents")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about your documents:"):
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = get_conversation_response(st.session_state.conversation, prompt)
                st.markdown(response["answer"])
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    # Example questions
    st.markdown(
        """
        **Example questions:**
        - What is the main topic of the document?
        - Summarize the key points.
        - Are there any mentions of [specific term]?
        """
    )

# Info message if no documents
else:
    st.info("👋 Please upload your documents to start the conversation!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        This AI assistant helps you chat with your documents. 
        Simply upload PDFs or text files to get started!
    </div>
    """,
    unsafe_allow_html=True
)
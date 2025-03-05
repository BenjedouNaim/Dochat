import streamlit as st
from utils import (
    extract_text_from_pdf,
    process_text,
    create_vector_store,
    create_conversation_chain,
    get_conversation_response
)

# Page configuration
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="💬",
    layout="wide"
)

# Load custom CSS
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

# Main title
st.title("📚 Document Chat Assistant")

# File upload section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload your documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)
st.markdown("</div>", unsafe_allow_html=True)

# Process uploaded files
if uploaded_files:
    with st.spinner("Processing documents..."):
        all_text = ""
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file.read())
            else:  # txt file
                text = file.read().decode()
            all_text += text + "\n\n"

        # Create text chunks and vector store
        text_chunks = process_text(all_text)
        st.session_state.vector_store = create_vector_store(text_chunks)
        st.session_state.conversation = create_conversation_chain(
            st.session_state.vector_store
        )

        # Display processed document content
        with st.expander("View Processed Documents", expanded=False):
            st.markdown("### Document Content")
            st.markdown(
                f"<div class='document-content'>{all_text}</div>",
                unsafe_allow_html=True
            )

        st.success("Documents processed successfully!")

# Chat interface
if st.session_state.conversation is not None:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f"<div class='chat-message user-message'>"
                f"<strong>You:</strong><br>{message['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='chat-message assistant-message'>"
                f"<strong>Assistant:</strong><br>{message['content']}</div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Function to handle sending messages
    def send_message():
        if st.session_state.user_question:
            with st.spinner("Thinking..."):
                # Get response from conversation chain
                response = get_conversation_response(
                    st.session_state.conversation,
                    st.session_state.user_question
                )

                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": st.session_state.user_question},
                    {"role": "assistant", "content": response["answer"]}
                ])
                # Clear the input
                st.session_state.user_question = ""

    # User input
    st.text_input(
        "Ask a question about your documents:",
        key="user_question",
        on_change=send_message
    )

else:
    st.info("Please upload documents to start the conversation.")

# Footer
st.markdown("---")
st.markdown(
    "This AI assistant helps you chat with your documents. "
    "Upload PDFs or text files to get started!"
)
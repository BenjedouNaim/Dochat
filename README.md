

Here's a formatted README.md file based on the project details:

```markdown
# Document Chat Assistant 📚

A Streamlit-powered AI application that allows users to upload documents (PDF/text) and engage in intelligent conversations with their content. Built using LangChain, FAISS, and Google GenAI for contextual question-answering.

## Features

- **Document Upload**  
  Supports PDF and text files. Multiple files can be uploaded at once.

- **Contextual Q&A**  
  Ask questions about your documents and get AI-powered answers based on their content.

- **Conversation History**  
  - Clear chat history with one click  
  - Export chat logs as CSV files  
  - Timestamped messages for reference  

- **Preview & Processing**  
  View a preview of processed documents and track loading progress with spinner indicators.

- **Example Use Cases**  
  ```text
  - "What is the main topic of the document?"
  - "Summarize the key points."
  - "Are there any mentions of [specific term]?"
  ```

## Directory Structure

```
benjedounaim-dochat/
├── README.md         - This file
├── app.py            - Main Streamlit application
├── pyproject.toml    - Python project dependencies
├── utils.py          - Helper functions for text processing
├── uv.lock           - Poetry lock file (dependency versions)
├── .env.example      - Environment variable template
├── static/           - CSS styles and assets
│   └── styles.css    - Custom UI styling
└── .streamlit/       - App configuration
```

## Tech Stack

- **Frontend**: Streamlit (Python)
- **AI Models**: 
  - Google GenAI Gemini (for responses)
  - Google Embedding Model (for document encoding)
- **Vector Database**: FAISS (for text chunk storage)
- **Text Processing**: LangChain, PyPDF2
- **Dependency Management**: Poetry

## Getting Started

### Prerequisites
1. Python 3.11+
2. Google GenAI API Key (free tier available)
3. Poetry (for dependency management)

### Installation
```bash
# Clone repository
git clone https://github.com/BenjedouNaim/Dochat.git
cd benjedounaim-dochat

# Install dependencies
poetry install

# Create .env file
cp .env.example .env
echo "GOOGLE_API_KEY=your-api-key" >> .env
```

### Running the App
```bash
poetry run streamlit run app.py
```

## Usage

1. **Upload Documents**  
   Drag & drop PDF/text files into the uploader section

2. **Ask Questions**  
   Type your query in the chat input field after documents are processed

3. **Manage Conversations**  
   - Clear history via sidebar button  
   - Export chat logs as CSV files  

## Configuration

### Environment Variables
Set your Google API key in `.env`:
```dotenv
GOOGLE_API_KEY=your-google-api-key-here
```

### Customization
- Modify `static/styles.css` for UI theming
- Adjust vector store parameters in `utils.py` (chunk size, overlap)
- Change models in `utils.py` (Gemini model name, embedding model)

## License

MIT License - feel free to modify and distribute. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io)
- [LangChain](https://python.langchain.com)
- [Google GenAI](https://developers.generativeai.google)
```

You can copy this text directly into a `README.md` file. It includes all key project details while maintaining a clean and professional format. Let me know if you need adjustments!

# Document Chat Assistant

Document Chat Assistant is a Streamlit-based application that allows you to upload your PDF or text documents and interactively chat with an AI assistant about their contents. This tool processes your documents, extracts key insights, and provides context-aware answers to your queries.

## Features

- **Document Upload:** Supports both PDF and text files.
- **Intelligent Processing:** Extracts text from documents, splits it into manageable chunks, and creates a vector store for efficient search.
- **AI Conversation:** Utilizes Google Generative AI and LangChain to provide relevant responses based on your document content.
- **Interactive Chat:** Engage in a dynamic conversation with your document. View and export your chat history.
- **User-friendly Interface:** Clean, responsive design with custom CSS styling.

## Directory Structure

```
benjedounaim-dochat/
├───README.md            # Project documentation
├───app.py            # Main Streamlit application
├───pyproject.toml            # Project configuration and dependencies
├───utils.py            # Utility functions for document processing and AI conversation
├───uv.lock            # Dependency lock file
├───.env.example            # Example file for environment variables
├───static/
  ────styles.css            # Custom CSS for the app
└───.streamlit/
          config.toml            # Streamlit configuration file
```

## Installation

1. **Clone the Repository:**
  ```bash
git clone https://github.com/BenjedouNaim/Dochat.git
cd Benjedounaim-dochat
```

2. **Set Up a Virtual Environment:**
  ```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**
  The project uses a `pyproject.toml` file to manage dependencies. You can install the required packages using:
  ```bash
pip install -r requirements.txt
```
  *Alternatively, if you prefer Poetry:*
  ```bash
poetry install
```

4. **Configure Environment Variables:**
  - Copy the example environment file and update it with your credentials:
  ```bash
cp .env.example .env
```
  - Open the .env file and set your `GOOGLE_API_KEY`.

## Usage

1. **Start the Application:**
  ```bash
streamlit run app.py
```

2. **Interact with the App:**
  - **Upload Documents:** Drag and drop your PDF or text files in the upload section.
  - **Chat with the AI:** Ask questions about your document’s content in the chat interface.
  - **Manage Conversation:** Clear your chat history or export it as a CSV file for your records.

## File Overview

- **app.py:** Initializes the Streamlit app, handles file uploads, manages session state, and sets up the chat interface.
- **utils.py:** Contains functions to extract text from PDFs, process and chunk text, create a vector store using FAISS, and set up the conversation chain with the AI.
- **pyproject.toml:** Lists project metadata and dependencies.
- **static/styles.css:** Provides custom styling for the application.
- **.streamlit/config.toml:** Configures Streamlit settings such as server address, port, and theme.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## License

[Include your project license here, e.g., MIT License.]

## Repository

For more details and to view the source code, visit the [GitHub Repository](https://github.com/BenjedouNaim/Dochat).

---

*This project leverages modern AI and document processing techniques to provide an interactive and insightful chat experience with your documents.*
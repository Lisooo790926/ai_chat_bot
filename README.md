# RAG Bot with YouTube Summarization

A powerful RAG (Retrieval-Augmented Generation) bot that combines document processing, YouTube video summarization, and multilingual support. Built with Streamlit, LangChain, and Qdrant.

## Features

- **Chat Interface**
  - Interactive chat with RAG-powered responses
  - Support for both Azure OpenAI and Google Gemini models
  - Real-time streaming responses
  - Context-aware responses using vector search

- **Document Processing**
  - Upload and process markdown and text files
  - Automatic text chunking and embedding
  - Integration with Qdrant vector database
  - Support for multiple document collections

- **YouTube Video Summarization**
  - Process single videos or entire channels
  - Multiple summary formats:
    - Detailed summaries
    - Concise summaries
    - Bullet point summaries
  - Automatic transcript extraction
  - Save summaries as markdown files
  - Option to add summaries to RAG database

- **Multilingual Support**
  - English and Traditional Chinese interfaces
  - Automatic language detection
  - Localized error messages and UI elements

## Project Structure

```
.
├── app.py                 # Main application entry point
├── ragbot.py             # Core RAG bot implementation
├── ui/                   # UI components
│   ├── __init__.py
│   ├── chat_tab.py      # Chat interface
│   ├── upload_tab.py    # Document upload interface
│   └── youtube_tab.py   # YouTube processing interface
├── utils/               # Utility functions
│   ├── __init__.py
│   └── language.py     # Language utilities
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file with the following variables:
   ```
   AZURE_OPENAI_API_KEY=your_azure_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   GOOGLE_API_KEY=your_google_key
   QDRANT_URL=your_qdrant_url
   ```

## Execution

### Start the Application

1. Run the main application:
   ```bash
   streamlit run app.py
   ```

2. Run with specific port (if default 8501 is in use):
   ```bash
   streamlit run app.py --server.port 8502
   ```

3. Run with debug mode:
   ```bash
   streamlit run app.py --logger.level=debug
   ```

### Development Commands

1. Run tests (if implemented):
   ```bash
   python -m pytest
   ```

2. Format code:
   ```bash
   black .
   ```

3. Check code style:
   ```bash
   flake8
   ```

4. Type checking:
   ```bash
   mypy .
   ```

### Docker Support (Optional)

1. Build the Docker image:
   ```bash
   docker build -t rag-bot .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 rag-bot
   ```

## Usage

1. **Chat Interface**
   - Select your preferred AI provider (Azure OpenAI or Google Gemini)
   - Type your question in the chat input
   - View streaming responses with relevant context

2. **Document Upload**
   - Upload markdown or text files
   - Select the target collection
   - Process and embed documents
   - View processing status and results

3. **YouTube Summarization**
   - Enter a YouTube URL (single video or channel)
   - Choose summary format
   - Process video(s) and view results
   - Save summaries or add to RAG database

4. **Language Selection**
   - Use the language selector in the sidebar
   - Switch between English and Traditional Chinese
   - All UI elements and messages will update accordingly

## Dependencies

- Python 3.8+
- Streamlit
- LangChain
- Qdrant
- Azure OpenAI
- Google Generative AI
- YouTube Transcript API
- python-dotenv

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

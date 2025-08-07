# RAG-based Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) chatbot system that allows you to chat with your documents using AI. Upload PDF documents and ask questions to get intelligent responses with source citations.

## ✨ Features

- **Document Processing**: Automatic PDF ingestion with intelligent chunking and embedding generation
- **Vector Search**: Fast similarity search using FAISS vector database
- **LLM Integration**: OpenAI GPT integration for intelligent responses
- **Source Citations**: Automatic source references with page numbers and confidence scores
- **Web API**: RESTful FastAPI backend for document upload and chat functionality
- **Streamlit Frontend**: Professional web interface with real-time chat and document management
- **CLI Interface**: Interactive command-line chat interface
- **Multiple Templates**: Specialized prompts for Q&A, summaries, explanations, and comparisons
- **Demo Mode**: Try the system without OpenAI API key using demo responses
- **Real-time Processing**: Instant document upload and embedding creation

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
git clone <repository-url>
cd RAG-based-chatbot

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your OpenAI API key
# OPENAI_API_KEY=your_api_key_here
# Note: You can use Demo Mode without an API key
```

### 3. Add Sample Documents (Optional)

The system comes with sample AI documents. You can add your own PDFs to `data/documents/`:

```bash
# Your PDF files should be placed in:
data/documents/
```

### 4. Start the System

#### Backend API Server
```bash
# Start the FastAPI backend server
python main_app.py

# The API will be available at http://localhost:8000
# API documentation: http://localhost:8000/docs
```

#### Frontend Web Interface
```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Start the Streamlit web interface
python run.py

# The web interface will open automatically in your browser
# Default URL: http://localhost:8501
```

## 📖 Usage Guide

### Web Interface (Streamlit)

1. **Open the web interface** at `http://localhost:8501`
2. **Check API status** in the sidebar Control Panel
3. **Upload documents** via the sidebar Document Management section
4. **Ask questions** in the main chat interface
5. **View sources** by expanding source references under AI responses
6. **Try Demo Mode** if you don't have an OpenAI API key

### Command Line Interface

```bash
# Interactive CLI chat (requires API server to be running)
python main_app.py chat

# Available commands in CLI:
# - help: Show available commands
# - history: Show session history  
# - clear: Clear session history
# - stats: Show pipeline statistics
# - quit/exit/bye: End session
```

### Document Upload & Processing

#### Via Web Interface (Recommended)
1. Go to the sidebar → Document Management
2. Choose PDF file using file uploader
3. Click "🚀 Upload & Process"
4. Wait for processing completion (shows chunk count)

#### Via API
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf" \
  -F "process_immediately=true"
```

### Chat API

```bash
POST /chat
{
  "query": "What is artificial intelligence?",
  "max_sources": 3,
  "template_type": "qa",
  "include_sources": true
}
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   OpenAI GPT    │
│   Frontend      │ -> │   Backend       │ -> │   LLM Service   │
│   (Port 8501)   │    │   (Port 8000)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        |                       |                       |
        v                       v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Document        │    │   FAISS Vector  │    │   RAG Pipeline  │
│ Upload & Chat   │    │   Database      │    │   Orchestrator  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Features

- **Automatic Embedding Creation**: Documents are automatically processed and embedded when uploaded
- **Real-time Chat**: Professional chat interface with typing indicators and timestamps
- **Source References**: Every AI response includes clickable source references with confidence scores
- **System Monitoring**: Real-time API status, component health, and index statistics
- **Demo Mode**: Try the system without OpenAI API key

## 📖 Usage Guide

### Command Line Interface

#### Document Ingestion
```bash
# Basic ingestion
python main_app.py ingest

# Custom settings
python main_app.py ingest --source-dir /path/to/docs --store-type chromadb --chunk-size 800

# Rebuild index from scratch
python main_app.py ingest --rebuild
```

#### Interactive Chat
```bash
# Start chat session
python main_app.py chat
```

Available chat commands:
- `help` - Show available commands
- `history` - Show session history
- `clear` - Clear session history
- `stats` - Show pipeline statistics
- `quit/exit/bye` - End session

#### Web API Server
```bash
# Start API server
python main_app.py api

# Custom host/port
python main_app.py api --host 127.0.0.1 --port 8080 --reload
```

#### Testing
```bash
# Run default tests
python main_app.py test

# Test specific query
python main_app.py test --query "What is artificial intelligence?"
```

### Web API Endpoints

#### Chat Endpoint
```bash
POST /chat
{
  "query": "What is machine learning?",
  "max_sources": 3,
  "template_type": "explanation",
  "include_sources": true
}
```

#### Document Upload
```bash
POST /upload
Content-Type: multipart/form-data
- file: PDF file
- process_immediately: true/false
```

#### System Status
```bash
GET /status
GET /health
GET /config
```

#### Document Management
```bash
GET /documents          # List uploaded documents
DELETE /documents/{filename}  # Delete document
POST /index/rebuild     # Rebuild vector index
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │   Vector Store  │    │   LLM Handler   │
│   (PDF files)   │ -> │   (FAISS/       │ -> │   (OpenAI GPT)  │
│                 │    │    ChromaDB)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        |                       |                       |
        v                       v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Document        │    │   Retriever     │    │   RAG Pipeline  │
│ Processor       │    │   System        │    │   Orchestrator  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Modules

- **document_processor.py**: PDF parsing, text chunking, and embedding generation using SentenceTransformers
- **vector_store.py**: FAISS vector database implementation for efficient similarity search
- **retriever.py**: Query processing and document retrieval with ranking
- **llm_handler.py**: OpenAI GPT integration and response generation with templates
- **api.py**: FastAPI backend server with document upload and chat endpoints
- **main_app.py**: Backend server entry point and CLI interface
- **frontend/app.py**: Streamlit web interface with professional chat UI
- **frontend/config.py**: Frontend configuration and API client
- **frontend/run.py**: Frontend server startup script

### Data Flow

1. **Document Upload**:
   - PDFs uploaded via Streamlit interface or API endpoint
   - PyMuPDF extracts text and creates intelligent chunks
   - SentenceTransformers generates embeddings automatically
   - FAISS stores embeddings for fast similarity search

2. **Query Processing**:
   - User query entered in Streamlit chat interface
   - Query is embedded and sent to backend API
   - FAISS performs similarity search to find relevant chunks
   - Top-k most relevant documents retrieved with confidence scores

3. **Response Generation**:
   - Retrieved context formatted with specialized templates
   - OpenAI GPT generates response based on context and template
   - Sources are cited with page numbers and confidence scores
   - Response displayed in chat with expandable source references

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (optional for demo mode) | None |
| `OPENAI_MODEL` | GPT model to use | gpt-3.5-turbo |
| `OPENAI_TEMPERATURE` | Response creativity (0-1) | 0.7 |
| `OPENAI_MAX_TOKENS` | Max response length | 1000 |
| `INDEX_PATH` | FAISS index storage path | data/vector_index |
| `CHUNK_SIZE` | Document chunk size | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap size | 200 |
| `DEFAULT_K` | Default retrieval count | 5 |

### Frontend Configuration

The Streamlit frontend includes:
- **Professional Theme**: Custom CSS with gradient backgrounds and modern styling
- **Chat Interface**: Real-time chat with message history and typing indicators
- **Document Management**: Upload interface with progress tracking
- **System Monitoring**: API status, health checks, and component statistics
- **Source Display**: Expandable source references with page numbers and scores

### Backend API Endpoints

#### Chat Endpoint
```bash
POST /chat
{
  "query": "What is machine learning?",
  "max_sources": 3,
  "template_type": "qa",
  "include_sources": true
}
```

#### Document Upload
```bash
POST /upload
Content-Type: multipart/form-data
- file: PDF file
- process_immediately: true (auto-creates embeddings)
```

#### System Status
```bash
GET /status     # System health and statistics
GET /health     # Simple health check
GET /documents  # List uploaded documents
```

## 🧪 Testing & Development

### Manual Testing
```bash
# Start backend
python main_app.py

# Start frontend (new terminal)
cd frontend
python run.py

# Test the complete flow:
# 1. Upload a PDF via web interface
# 2. Ask questions in chat
# 3. Verify source citations work
```

### Demo Mode
If you don't have an OpenAI API key:
1. Start both backend and frontend
2. Upload documents (embeddings still work)
3. Ask questions (get demo responses)
4. Test all features except actual LLM responses

## 📊 Performance Features

### Optimizations Implemented

1. **Vector Search**:
   - FAISS for high-performance similarity search
   - Optimized chunk size (1000 tokens) for balance of context and precision
   - Confidence scoring for source ranking

2. **Frontend Performance**:
   - Streamlit session state for chat history persistence
   - Efficient API calls with proper error handling
   - Real-time status updates without polling

3. **Backend Efficiency**:
   - FastAPI async endpoints for concurrent request handling
   - Automatic embedding generation on upload
   - Memory-efficient document processing

## 🔧 Troubleshooting

### Common Setup Issues

**Backend not starting**
```bash
# Ensure you're in project root and venv is activated
cd RAG-based-chatbot
venv\Scripts\activate  # Windows
python main_app.py
```

**Frontend not connecting to backend**
```bash
# Check if backend is running on port 8000
# Check frontend/config.py for correct API_BASE_URL
curl http://localhost:8000/health
```

**Document upload failing**
```bash
# Check file permissions and disk space
# Ensure data/uploads directory exists
# Verify PDF file is not corrupted
```

**Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
# Check if you're in the correct directory
```

### Debug Mode

Enable debug logging:
```bash
# Set environment variable
set LOG_LEVEL=DEBUG  # Windows
export LOG_LEVEL=DEBUG  # macOS/Linux

# Start backend with debug
python main_app.py
```

### API Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# System status with statistics
curl http://localhost:8000/status

# Frontend should show API status in sidebar
```

## 📁 Project Structure

```
RAG-based-chatbot/
├── src/                      # Core backend modules
│   ├── __init__.py          # Package initialization with exports
│   ├── document_processor.py # PDF processing and embedding generation
│   ├── vector_store.py      # FAISS vector database implementation
│   ├── retriever.py         # Document retrieval and ranking
│   ├── llm_handler.py       # OpenAI integration and templates
│   └── api.py              # FastAPI backend server
├── frontend/                # Streamlit web interface
│   ├── app.py              # Main Streamlit application
│   ├── config.py           # Frontend configuration and API client
│   └── run.py              # Frontend startup script
├── data/                   # Data directories
│   ├── documents/          # Sample AI documents (included)
│   ├── uploads/           # API uploaded documents
│   └── vector_index/      # FAISS database files
├── tests/                 # Test files (future expansion)
├── main_app.py           # Backend server entry point
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
└── README.md           # Project documentation
```

### Component Overview

#### Backend Components (`src/`)
- **FastAPI Server**: RESTful API with automatic OpenAPI documentation
- **Document Processing**: PyMuPDF for PDF parsing, intelligent text chunking
- **Vector Database**: FAISS for high-performance similarity search
- **LLM Integration**: OpenAI GPT with specialized prompt templates
- **RAG Pipeline**: Complete retrieval-augmented generation workflow

#### Frontend Components (`frontend/`)
- **Streamlit Interface**: Professional web UI with real-time chat
- **API Client**: Seamless backend communication with error handling
- **Document Management**: Upload interface with progress tracking
- **System Monitoring**: Real-time status and health indicators

#### Sample Data (`data/documents/`)
Pre-included AI knowledge documents:
- Introduction to Artificial Intelligence
- Machine Learning Fundamentals  
- Deep Learning and Neural Networks
- Natural Language Processing
- Computer Vision and Image Understanding
- AI in Healthcare and Medicine
- AI in Finance and Banking
- Ethics and Challenges in AI
- The History of Artificial Intelligence
- The Future of AI: Trends and Predictions

## 🚀 Quick Start Commands

### Complete Setup (First Time)
```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start backend server
python main_app.py

# 4. In new terminal, start frontend
cd frontend
python run.py
```

### Daily Development Workflow
```bash
# Terminal 1: Backend
venv\Scripts\activate
python main_app.py

# Terminal 2: Frontend  
cd frontend
python run.py
```

## 🎯 Usage Examples

### Example Queries to Try

With the included AI documents, you can ask:

**General AI Questions:**
- "What is artificial intelligence and how does it work?"
- "Explain the difference between machine learning and deep learning"
- "What are the main applications of AI in healthcare?"

**Technical Deep Dives:**
- "How do neural networks learn complex patterns?"
- "What are the key algorithms used in machine learning?"
- "Explain how computer vision processes images"

**Practical Applications:**
- "How is AI being used in banking and finance?"
- "What are the ethical challenges in AI development?"
- "What does the future hold for AI technology?"

**Comparative Analysis:**
- "Compare supervised vs unsupervised learning"
- "What are the pros and cons of AI in healthcare?"
- "How has AI evolved from Turing to modern systems?"

### Source Citations

Every response includes:
- **Document Source**: Which PDF the information came from
- **Page Reference**: Specific page number in the source
- **Confidence Score**: How relevant the source is (0.0-1.0)
- **Context Preview**: Snippet of the actual source text

## 🔄 Development & Customization

### Adding Your Own Documents
1. Place PDF files in `data/documents/` or use the web upload
2. Documents are automatically processed when uploaded via frontend
3. Embeddings are created immediately for instant searchability

### Customizing Templates
Edit `src/llm_handler.py` to modify prompt templates:
- **Q&A Template**: Direct question answering
- **Summary Template**: Document summarization  
- **Explanation Template**: Detailed explanations
- **Comparison Template**: Comparative analysis

### Frontend Customization
Modify `frontend/app.py` for UI changes:
- CSS styling in the `st.markdown` sections
- Chat interface layout and behavior
- Sidebar components and controls

## 🔧 Troubleshooting

### Common Issues

**ImportError: No module named 'src'**
```bash
# Ensure you're in the project root directory
cd /path/to/RAG-based-chatbot
python main_app.py
```

**OpenAI API Key Error**
```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key"
# Or add to .env file
```

**Vector Index Not Found**
```bash
# Run document ingestion first
python main_app.py ingest
```

**Memory Issues with Large Documents**
```bash
# Reduce chunk size
python main_app.py ingest --chunk-size 500
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main_app.py chat
```

### API Debugging

Check API status:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

## 🤝 Contributing & Development

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd RAG-based-chatbot

# Setup virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Development workflow
# Terminal 1: Backend development
python main_app.py

# Terminal 2: Frontend development  
cd frontend
python run.py
```

### Code Organization

#### Backend (`src/` package)
- Modular design with clear separation of concerns
- FastAPI for modern async web framework
- Pydantic v2 for data validation and serialization
- Comprehensive error handling and logging

#### Frontend (`frontend/` package)  
- Streamlit for rapid web interface development
- Professional UI with custom CSS styling
- Real-time API communication and status monitoring
- Session state management for chat persistence

### Adding New Features

#### New Document Types
1. Extend `DocumentProcessor` in `src/document_processor.py`
2. Add new parsers for different file formats
3. Update upload validation in `src/api.py`

#### New LLM Providers
1. Create new handler in `src/llm_handler.py`  
2. Implement provider-specific API calls
3. Add configuration options

#### Frontend Enhancements
1. Modify `frontend/app.py` for new UI components
2. Update `frontend/config.py` for new settings
3. Add new pages or sidebar sections

## 🚀 Deployment Considerations

### Production Deployment

#### Backend Deployment
```bash
# Use production ASGI server
pip install uvicorn[standard]
uvicorn main_app:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Frontend Deployment  
```bash
# Streamlit for production
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

#### Environment Variables for Production
```bash
OPENAI_API_KEY=your_production_key
OPENAI_MODEL=gpt-3.5-turbo
LOG_LEVEL=INFO
CHUNK_SIZE=1000
DEFAULT_K=5
```

### Security Considerations
- Store OpenAI API keys securely using environment variables
- Implement rate limiting for API endpoints
- Validate and sanitize uploaded documents
- Use HTTPS in production environments

### Scalability Options
- **Vector Database**: Consider Pinecone or Weaviate for cloud-scale
- **LLM Caching**: Implement Redis for response caching
- **Load Balancing**: Use nginx for frontend/backend load balancing
- **Document Storage**: Use cloud storage (S3, Azure Blob) for large datasets

## 📜 License & Attribution

### License
This project is licensed under the MIT License. See LICENSE file for details.

### Key Dependencies & Attribution

**Core Framework & Processing:**
- **FastAPI** - Modern async web framework for the backend API
- **Streamlit** - Rapid web application framework for the frontend
- **PyMuPDF (fitz)** - Efficient PDF parsing and text extraction
- **SentenceTransformers** - State-of-the-art text embedding models

**AI & Vector Search:**
- **OpenAI GPT** - Large language model for response generation
- **FAISS** - Facebook AI Similarity Search for vector operations
- **Transformers** - Hugging Face transformers library

**Data & Utilities:**
- **Pydantic v2** - Data validation and settings management
- **python-dotenv** - Environment variable management
- **Uvicorn** - Lightning-fast ASGI server

### Sample Documents
The included AI documents in `data/documents/` are educational resources covering:
- Fundamental AI concepts and history
- Machine learning and deep learning principles  
- AI applications across industries
- Ethical considerations and future trends

## 📞 Support & Resources

### Getting Help

1. **Setup Issues**: Check the [Troubleshooting](#-troubleshooting) section
2. **API Documentation**: Visit `http://localhost:8000/docs` when backend is running
3. **Frontend Issues**: Check browser console and Streamlit logs
4. **Performance**: Review the [Configuration](#-configuration) section

### Useful Commands Reference

```bash
# Project setup
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# Start development servers
python main_app.py                    # Backend on :8000
cd frontend && python run.py          # Frontend on :8501

# Health checks
curl http://localhost:8000/health      # Backend health
curl http://localhost:8000/status     # System status

# Environment check
python -c "import src; print('Package imports OK')"
```

### Resources for Extension

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **FAISS Documentation**: https://faiss.ai/
- **OpenAI API Documentation**: https://platform.openai.com/docs/
- **SentenceTransformers**: https://www.sbert.net/

---

## 🎉 Ready to Use!

Your RAG-based chatbot is ready for:
- ✅ **Immediate Use**: Pre-loaded with AI knowledge documents
- ✅ **Document Upload**: Add your own PDFs via web interface  
- ✅ **Professional Chat**: Modern UI with source citations
- ✅ **Demo Mode**: Try without OpenAI API key
- ✅ **Development**: Clean, modular codebase for customization

**Quick Start**: Activate venv → `python main_app.py` → `cd frontend && python run.py` → Open browser!

Built with ❤️ for the AI community

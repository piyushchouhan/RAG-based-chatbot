# RAG-based Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) chatbot system that allows you to chat with your documents using AI. Upload PDF documents and ask questions to get intelligent responses with source citations.

## ✨ Features

- **Document Processing**: Automatic PDF ingestion with intelligent chunking
- **Vector Search**: Fast similarity search using FAISS or ChromaDB
- **LLM Integration**: OpenAI GPT integration for intelligent responses
- **Source Citations**: Automatic source references in responses
- **Web API**: RESTful API for integration with other applications
- **CLI Interface**: Interactive command-line chat interface
- **Multiple Templates**: Specialized prompts for different query types
- **Background Processing**: Asynchronous document processing
- **Error Handling**: Robust error handling and retry logic

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd RAG-based-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your OpenAI API key
# OPENAI_API_KEY=your_api_key_here
```

### 3. Add Documents

Place your PDF documents in the `data/documents/` directory:

```bash
mkdir -p data/documents
# Copy your PDF files to data/documents/
```

### 4. Process Documents

```bash
# Ingest documents into vector database
python main_app.py ingest
```

### 5. Start Chatting

```bash
# Interactive CLI chat
python main_app.py chat

# Or start web API server
python main_app.py api
```

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

- **document_processor.py**: PDF parsing, text chunking, embedding generation
- **vector_store.py**: Vector database implementations (FAISS/ChromaDB)
- **retriever.py**: Query processing and document retrieval
- **llm_handler.py**: LLM integration and response generation
- **api.py**: FastAPI web server and endpoints
- **main_app.py**: CLI interface and application entry point

### Data Flow

1. **Document Ingestion**:
   - PDF files are parsed and chunked
   - Text chunks are embedded using SentenceTransformers
   - Embeddings are stored in vector database

2. **Query Processing**:
   - User query is processed and embedded
   - Similar document chunks are retrieved via vector search
   - Context is formatted for LLM input

3. **Response Generation**:
   - LLM generates response based on retrieved context
   - Sources are cited and formatted
   - Response is returned with metadata

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | GPT model to use | gpt-3.5-turbo |
| `OPENAI_TEMPERATURE` | Response creativity (0-1) | 0.7 |
| `OPENAI_MAX_TOKENS` | Max response length | 1000 |
| `VECTOR_STORE_TYPE` | Vector database type | faiss |
| `INDEX_PATH` | Vector index storage path | data/vector_index |
| `CHUNK_SIZE` | Document chunk size | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap size | 200 |
| `DEFAULT_K` | Default retrieval count | 5 |

### Vector Store Options

#### FAISS (Default)
- **Pros**: Fast, memory-efficient, good for production
- **Cons**: In-memory only, requires saving/loading
- **Best for**: High-performance applications

#### ChromaDB
- **Pros**: Persistent storage, metadata support, built for AI
- **Cons**: Slower than FAISS, more resource intensive
- **Best for**: Development, complex metadata needs

### LLM Models

Supported OpenAI models:
- `gpt-3.5-turbo` (default) - Fast, cost-effective
- `gpt-4` - Higher quality, more expensive
- `gpt-4-turbo` - Latest GPT-4 variant

## 🧪 Testing

### Unit Tests
```bash
# Run document processor tests
python -m pytest tests/test_document_processor.py

# Run vector store tests
python -m pytest tests/test_vector_store.py

# Run all tests
python -m pytest tests/
```

### Integration Testing
```bash
# Test complete pipeline
python main_app.py test

# Test with sample documents
python test_processor.py
```

### API Testing
```bash
# Start API server
python main_app.py api

# Test endpoints
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "max_sources": 3}'
```

## 📊 Performance

### Benchmarks (on sample dataset)

| Component | Metric | Performance |
|-----------|--------|-------------|
| Document Processing | 10 PDFs (100 pages) | ~30 seconds |
| Vector Search | Query response | <100ms |
| LLM Response | gpt-3.5-turbo | 1-3 seconds |
| End-to-end Query | Complete RAG pipeline | 2-5 seconds |

### Optimization Tips

1. **Vector Store**:
   - Use FAISS for production deployments
   - Consider GPU acceleration for large datasets
   - Optimize chunk size for your document types

2. **LLM Usage**:
   - Use gpt-3.5-turbo for cost efficiency
   - Implement caching for repeated queries
   - Monitor token usage and costs

3. **Document Processing**:
   - Process documents in batches
   - Use background processing for uploads
   - Consider OCR for scanned documents

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

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run tests
pytest tests/
```

### Project Structure

```
RAG-based-chatbot/
├── src/                    # Core application modules
│   ├── __init__.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm_handler.py
│   └── api.py
├── tests/                  # Test files
├── data/                   # Data directories
│   ├── documents/          # Input documents
│   ├── uploads/           # API uploaded files
│   └── vector_index/      # Vector database
├── frontend/              # Web frontend (future)
├── main_app.py           # Main application entry point
├── ingest_documents.py   # Document ingestion script
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
└── README.md           # This file
```

## 📜 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **SentenceTransformers** for embedding models
- **FAISS** for efficient vector search
- **ChromaDB** for vector database capabilities
- **OpenAI** for GPT models
- **FastAPI** for web framework
- **PyMuPDF** for PDF processing

## 📞 Support

For questions, issues, or contributions:

1. Check the [troubleshooting](#troubleshooting) section
2. Search existing GitHub issues
3. Create a new issue with detailed description
4. For urgent issues, include logs and configuration

---

Built with ❤️ for the AI community

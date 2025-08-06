"""
FastAPI-based REST API for RAG Chatbot

This module provides:
- RESTful endpoints for chatbot queries
- Document upload and ingestion
- Health checks and system status
- Configuration management
- Error handling and validation
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Local imports
try:
    from .llm_handler import RAGPipeline, LLMConfig, RAGResponse
    from .retriever import RetrievalConfig
    from .document_processor import DocumentProcessor
    from .vector_store import VectorStoreFactory
except ImportError:
    from llm_handler import RAGPipeline, LLMConfig, RAGResponse
    from retriever import RetrievalConfig
    from document_processor import DocumentProcessor
    from vector_store import VectorStoreFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG-based Chatbot API",
    description="A Retrieval-Augmented Generation chatbot with document ingestion capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_pipeline: Optional[RAGPipeline] = None
document_processor: Optional[DocumentProcessor] = None


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    max_sources: int = Field(3, ge=1, le=10, description="Maximum number of sources to retrieve")
    template_type: Optional[str] = Field(None, description="Template type (default, qa, summary, explanation, comparison)")
    include_sources: bool = Field(True, description="Whether to include source citations")


class ChatResponse(BaseModel):
    """Chat response model"""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    model_used: str
    timestamp: str
    response_time: float
    confidence_score: Optional[float] = None
    
    @classmethod
    def from_rag_response(cls, rag_response: RAGResponse) -> "ChatResponse":
        """Convert RAGResponse to ChatResponse"""
        sources = []
        for source in rag_response.sources:
            sources.append({
                "filename": source.source_file,
                "page_number": source.page_number,
                "chunk_text": source.chunk_text[:200] + "..." if len(source.chunk_text) > 200 else source.chunk_text,
                "similarity_score": source.similarity_score
            })
        
        return cls(
            query=rag_response.query,
            response=rag_response.response,
            sources=sources,
            model_used=rag_response.model_used,
            timestamp=rag_response.timestamp.isoformat(),
            response_time=rag_response.response_time,
            confidence_score=rag_response.confidence_score
        )


class DocumentUploadResponse(BaseModel):
    """Document upload response model"""
    filename: str
    status: str
    message: str
    processing_time: Optional[float] = None
    chunks_created: Optional[int] = None


class SystemStatus(BaseModel):
    """System status model"""
    status: str
    components: Dict[str, str]
    index_stats: Dict[str, Any]
    configuration: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global rag_pipeline, document_processor
    
    try:
        logger.info("Starting RAG Chatbot API...")
        
        # Initialize configurations
        retrieval_config = RetrievalConfig(
            vector_store_type=os.getenv("VECTOR_STORE_TYPE", "faiss"),
            index_path=os.getenv("INDEX_PATH", "data/vector_index"),
            default_k=int(os.getenv("DEFAULT_K", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        
        llm_config = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "4000")),
            include_sources=os.getenv("INCLUDE_SOURCES", "true").lower() == "true",
            citation_style=os.getenv("CITATION_STYLE", "numbered")
        )
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(retrieval_config, llm_config)
        
        # Initialize document processor for uploads
        document_processor = DocumentProcessor()
        
        logger.info("RAG Chatbot API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start RAG pipeline: {e}")
        # Don't fail startup, but log the error


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG Chatbot API...")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get detailed system status"""
    global rag_pipeline, document_processor
    
    components = {
        "rag_pipeline": "initialized" if rag_pipeline else "not_initialized",
        "document_processor": "initialized" if document_processor else "not_initialized",
        "vector_store": "available" if rag_pipeline and rag_pipeline.retriever.vector_store else "unavailable"
    }
    
    index_stats = {}
    if rag_pipeline and rag_pipeline.retriever.vector_store:
        try:
            # Get index statistics
            if hasattr(rag_pipeline.retriever.vector_store, 'get_stats'):
                index_stats = rag_pipeline.retriever.vector_store.get_stats()
            else:
                index_stats = {"status": "available", "type": type(rag_pipeline.retriever.vector_store).__name__}
        except Exception as e:
            index_stats = {"status": "error", "error": str(e)}
    
    configuration = {
        "vector_store_type": os.getenv("VECTOR_STORE_TYPE", "faiss"),
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "index_path": os.getenv("INDEX_PATH", "data/vector_index"),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY"))
    }
    
    overall_status = "healthy" if all(status != "not_initialized" for status in components.values()) else "degraded"
    
    return SystemStatus(
        status=overall_status,
        components=components,
        index_stats=index_stats,
        configuration=configuration
    )


# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat query using the RAG pipeline
    
    Args:
        request: Chat request containing query and parameters
        
    Returns:
        Chat response with generated answer and sources
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized. Please check system configuration."
        )
    
    try:
        logger.info(f"Processing chat query: '{request.query[:50]}...'")
        
        # Process the query through RAG pipeline
        rag_response = rag_pipeline.query(
            question=request.query,
            k=request.max_sources,
            template_type=request.template_type
        )
        
        # Convert to API response format
        chat_response = ChatResponse.from_rag_response(rag_response)
        
        # Optionally remove sources if not requested
        if not request.include_sources:
            chat_response.sources = []
        
        logger.info(f"Chat query processed successfully in {rag_response.response_time:.2f}s")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Document upload endpoint
@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_immediately: bool = Form(True)
):
    """
    Upload and optionally process a document
    
    Args:
        file: Uploaded file
        process_immediately: Whether to process the document immediately
        
    Returns:
        Upload response with processing status
    """
    global document_processor, rag_pipeline
    
    if not document_processor:
        raise HTTPException(
            status_code=503,
            detail="Document processor not initialized"
        )
    
    # Validate file type
    allowed_types = ["application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only PDF files are supported."
        )
    
    try:
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded: {file.filename}")
        
        if process_immediately:
            # Process document immediately
            start_time = datetime.now()
            
            try:
                # Process the document
                chunks = document_processor.process_document(str(file_path))
                
                # Add to vector store if pipeline is available
                if rag_pipeline and rag_pipeline.retriever.vector_store:
                    rag_pipeline.retriever.vector_store.add_documents(chunks)
                    logger.info(f"Added {len(chunks)} chunks to vector store")
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return DocumentUploadResponse(
                    filename=file.filename,
                    status="processed",
                    message=f"Document processed successfully. Created {len(chunks)} chunks.",
                    processing_time=processing_time,
                    chunks_created=len(chunks)
                )
                
            except Exception as e:
                logger.error(f"Error processing document {file.filename}: {e}")
                return DocumentUploadResponse(
                    filename=file.filename,
                    status="error",
                    message=f"Error processing document: {str(e)}"
                )
        else:
            # Queue for background processing
            background_tasks.add_task(process_document_background, str(file_path))
            
            return DocumentUploadResponse(
                filename=file.filename,
                status="queued",
                message="Document uploaded and queued for processing"
            )
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


async def process_document_background(file_path: str):
    """Background task for document processing"""
    global document_processor, rag_pipeline
    
    try:
        logger.info(f"Background processing: {file_path}")
        
        # Process the document
        chunks = document_processor.process_document(file_path)
        
        # Add to vector store if available
        if rag_pipeline and rag_pipeline.retriever.vector_store:
            rag_pipeline.retriever.vector_store.add_documents(chunks)
            logger.info(f"Background processing completed: {len(chunks)} chunks added")
        
    except Exception as e:
        logger.error(f"Background processing error for {file_path}: {e}")


# Document management endpoints
@app.get("/documents")
async def list_documents():
    """List uploaded documents"""
    upload_dir = Path("data/uploads")
    
    if not upload_dir.exists():
        return {"documents": []}
    
    documents = []
    for file_path in upload_dir.glob("*.pdf"):
        stat = file_path.stat()
        documents.append({
            "filename": file_path.name,
            "size": stat.st_size,
            "uploaded": datetime.fromtimestamp(stat.st_ctime).isoformat()
        })
    
    return {"documents": documents}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete an uploaded document"""
    upload_dir = Path("data/uploads")
    file_path = upload_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        file_path.unlink()
        logger.info(f"Deleted document: {filename}")
        return {"message": f"Document {filename} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


# Index management endpoints
@app.post("/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild the vector index from all documents"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # Queue rebuild as background task
    background_tasks.add_task(rebuild_index_background)
    
    return {"message": "Index rebuild queued. Check status endpoint for progress."}


async def rebuild_index_background():
    """Background task for index rebuilding"""
    global rag_pipeline, document_processor
    
    try:
        logger.info("Starting index rebuild...")
        
        upload_dir = Path("data/uploads")
        if not upload_dir.exists():
            logger.warning("No uploads directory found")
            return
        
        # Clear existing index
        if rag_pipeline.retriever.vector_store:
            # This would need to be implemented in the vector store classes
            logger.info("Clearing existing index...")
        
        # Process all documents
        total_chunks = 0
        for file_path in upload_dir.glob("*.pdf"):
            try:
                chunks = document_processor.process_document(str(file_path))
                if rag_pipeline.retriever.vector_store:
                    rag_pipeline.retriever.vector_store.add_documents(chunks)
                total_chunks += len(chunks)
                logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
        
        logger.info(f"Index rebuild completed: {total_chunks} total chunks")
        
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")


# Configuration endpoints
@app.get("/config")
async def get_configuration():
    """Get current system configuration"""
    return {
        "vector_store_type": os.getenv("VECTOR_STORE_TYPE", "faiss"),
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
        "default_k": int(os.getenv("DEFAULT_K", "5")),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200"))
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


# Run the application
def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI application"""
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the API
    run_api(reload=True)

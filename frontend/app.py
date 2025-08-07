"""
Professional RAG Chatbot Frontend using Streamlit

This module provides:
- Professional chat interface with conversation history
- Document upload and management
- Source reference display with highlighting
- Real-time typing indicators and loading states
- Export functionality for conversations
- Responsive design optimized for professional use
"""

import streamlit as st
import requests
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import base64
from pathlib import Path

# Import local modules
from config import (
    API_BASE_URL, APP_TITLE, APP_ICON, MAX_HISTORY_LENGTH,
    THEME_CONFIG, FEATURES, ERROR_MESSAGES, SUCCESS_MESSAGES,
    get_streamlit_config
)

# Page configuration
config = get_streamlit_config()
st.set_page_config(**config)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Chat interface styling */
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        border-left: 4px solid #667eea;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
        border-left: 4px solid #f093fb;
    }
    
    /* Source reference styling */
    .source-reference {
        background: rgba(255, 255, 255, 0.95);
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        color: #333;
    }
    
    .source-header {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 5px;
        font-size: 14px;
    }
    
    .source-content {
        font-size: 13px;
        line-height: 1.4;
        color: #555;
    }
    
    .confidence-badge {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 10px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Upload section styling */
    .upload-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 15px 0;
        border: 2px dashed #667eea;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots::after {
        content: '';
        animation: dots 2s infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Configuration - use the configuration from config.py
# API_BASE_URL = "http://localhost:8000"  # old hardcoded value
MAX_HISTORY_LENGTH = 50

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_status" not in st.session_state:
    st.session_state.system_status = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "is_typing" not in st.session_state:
    st.session_state.is_typing = False

class APIClient:
    """Professional API client for RAG backend communication"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def chat_query(self, query: str, max_sources: int = 3, template_type: str = "qa", include_sources: bool = True) -> Dict[str, Any]:
        """Send chat query to RAG pipeline"""
        try:
            payload = {
                "query": query,
                "max_sources": max_sources,
                "template_type": template_type,
                "include_sources": include_sources
            }
            response = requests.post(f"{self.base_url}/chat", json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}", "detail": response.text}
        except Exception as e:
            return {"error": "Connection Error", "detail": str(e)}
    
    def upload_document(self, file_data: bytes, filename: str, process_immediately: bool = True) -> Dict[str, Any]:
        """Upload document to knowledge base"""
        try:
            files = {"file": (filename, file_data, "application/pdf")}
            data = {"process_immediately": process_immediately}
            response = requests.post(f"{self.base_url}/upload", files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Upload Error: {response.status_code}", "detail": response.text}
        except Exception as e:
            return {"error": "Upload Failed", "detail": str(e)}
    
    def list_documents(self) -> Dict[str, Any]:
        """Get list of uploaded documents"""
        try:
            response = requests.get(f"{self.base_url}/documents", timeout=10)
            return response.json() if response.status_code == 200 else {"documents": []}
        except Exception:
            return {"documents": []}
    
    def get_query_examples(self) -> Dict[str, Any]:
        """Get example queries"""
        try:
            response = requests.get(f"{self.base_url}/examples", timeout=10)
            return response.json() if response.status_code == 200 else {"examples": [], "categories": []}
        except Exception:
            return {"examples": [], "categories": []}

# Initialize API client
api_client = APIClient(API_BASE_URL)

def display_typing_indicator():
    """Display professional typing indicator"""
    with st.empty():
        for i in range(3):
            st.markdown(f"""
            <div class="assistant-message">
                <span class="loading-dots">AI Assistant is thinking</span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)
            st.empty()

def format_message_time(timestamp: str) -> str:
    """Format message timestamp"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%H:%M")
    except:
        return datetime.now().strftime("%H:%M")

def display_source_references(sources: List[Dict[str, Any]]):
    """Display source references with professional styling"""
    if not sources:
        return
        
    st.markdown("**📚 Source References:**")
    
    for i, source in enumerate(sources, 1):
        confidence = source.get('similarity_score', 0) * 100
        confidence_color = "#4CAF50" if confidence > 80 else "#FF9800" if confidence > 60 else "#F44336"
        
        st.markdown(f"""
        <div class="source-reference">
            <div class="source-header">
                📄 {source.get('filename', 'Unknown Document')} (Page {source.get('page_number', 'N/A')})
                <span class="confidence-badge" style="background-color: {confidence_color};">
                    {confidence:.1f}% match
                </span>
            </div>
            <div class="source-content">
                {source.get('chunk_text', 'No preview available')[:200]}...
            </div>
        </div>
        """, unsafe_allow_html=True)

def export_conversation():
    """Export conversation history"""
    if not st.session_state.messages:
        st.warning("No conversation to export!")
        return
    
    # Create export data
    export_data = {
        "export_date": datetime.now().isoformat(),
        "conversation_count": len(st.session_state.messages),
        "messages": st.session_state.messages
    }
    
    # Convert to JSON
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    # Create download button
    st.download_button(
        label="📥 Download Conversation",
        data=json_str,
        file_name=f"rag_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def main():
    """Main application interface"""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px; color: white;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">🤖 RAG-based AI Assistant</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">Professional Document-Powered Conversations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # System status
        health_status = api_client.health_check()
        if health_status:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Disconnected")
            st.warning(f"Please ensure the RAG API is running on {API_BASE_URL}")
        
        # System metrics
        if health_status:
            system_status = api_client.get_system_status()
            if system_status:
                st.markdown("### 📊 System Status")
                
                # Component status
                components = system_status.get('components', {})
                for component, status in components.items():
                    color = "🟢" if status in ["initialized", "available"] else "🔴"
                    st.markdown(f"{color} **{component.replace('_', ' ').title()}**: {status}")
                
                # OpenAI API Key status
                config = system_status.get('configuration', {})
                has_openai_key = config.get('has_openai_key', False)
                if has_openai_key:
                    st.markdown("🟢 **OpenAI API**: Connected")
                else:
                    st.markdown("🟡 **OpenAI API**: Not configured")
                    st.info("💡 For full AI responses, set up an OpenAI API key. You can use **🧠 Demo Mode** in the meantime!")
                
                # Index statistics
                index_stats = system_status.get('index_stats', {})
                if index_stats:
                    st.markdown("### 📈 Index Statistics")
                    for key, value in index_stats.items():
                        if isinstance(value, (int, float)):
                            st.metric(key.replace('_', ' ').title(), value)
        
        st.markdown("---")
        
        # Chat settings
        st.markdown("### ⚙️ Chat Settings")
        max_sources = st.slider("Max Sources", 1, 10, 3, help="Number of source documents to retrieve")
        template_type = st.selectbox(
            "Response Style",
            ["qa", "summary", "explanation", "comparison", "default"],
            help="Template type for response generation"
        )
        include_sources = st.checkbox("Include Sources", True, help="Show source references in responses")
        
        st.markdown("---")
        
        # Document management
        st.markdown("### 📁 Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF to expand the knowledge base"
        )
        
        if uploaded_file:
            if st.button("🚀 Upload & Process", type="primary"):
                with st.spinner("Uploading and processing document..."):
                    result = api_client.upload_document(
                        uploaded_file.read(),
                        uploaded_file.name,
                        process_immediately=True
                    )
                    
                    if "error" not in result:
                        st.success(f"✅ {result.get('message', 'Document uploaded successfully!')}")
                        if result.get('chunks_created'):
                            st.info(f"📄 Created {result['chunks_created']} text chunks")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}: {result.get('detail', '')}")
        
        # Document list
        documents_data = api_client.list_documents()
        documents = documents_data.get('documents', [])
        
        if documents:
            st.markdown("#### 📚 Knowledge Base")
            for doc in documents:
                st.markdown(f"📄 **{doc.get('filename', 'Unknown')}**")
                if doc.get('size'):
                    st.caption(f"Size: {doc['size'] / 1024 / 1024:.1f} MB")
        else:
            st.info("No documents uploaded yet")
        
        st.markdown("---")
        
        # Example queries
        examples_data = api_client.get_query_examples()
        examples = examples_data.get('examples', [])
        
        if examples:
            st.markdown("### 💡 Example Queries")
            selected_example = st.selectbox(
                "Choose an example",
                ["Select an example..."] + [f"{ex['category']}: {ex['query'][:30]}..." for ex in examples]
            )
            
            if selected_example != "Select an example...":
                example_index = int(selected_example.split(":")[0]) - 1
                if st.button("Use This Example"):
                    st.session_state.example_query = examples[example_index]['query']
        
        st.markdown("---")
        
        # Conversation management
        st.markdown("### 💬 Conversation")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.session_state.messages:
                export_conversation()
    
    # Main chat interface
    st.markdown("## 💬 Chat Interface")
    
    # Chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 You</strong> <span style="float: right; opacity: 0.7; font-size: 12px;">{format_message_time(message.get('timestamp', ''))}</span><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 AI Assistant</strong> <span style="float: right; opacity: 0.7; font-size: 12px;">{format_message_time(message.get('timestamp', ''))}</span><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources if available
                if message.get('sources'):
                    with st.expander("📚 View Sources", expanded=False):
                        display_source_references(message['sources'])
                
                # Display response metrics
                if message.get('response_time'):
                    st.caption(f"⚡ Response time: {message['response_time']:.2f}s | Model: {message.get('model_used', 'Unknown')}")
    
    # Chat input
    st.markdown("---")
    
    # Use example query if selected or clear input if needed
    initial_query = ""
    if hasattr(st.session_state, 'example_query'):
        initial_query = st.session_state.example_query
        del st.session_state.example_query
    elif hasattr(st.session_state, 'clear_input') and st.session_state.clear_input:
        initial_query = ""
        st.session_state.clear_input = False
    
    query = st.text_area(
        "💭 Ask your question:",
        value=initial_query,
        placeholder="Enter your question about the uploaded documents...",
        height=100,
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        send_button = st.button("🚀 Send", type="primary", disabled=not query.strip(), use_container_width=True)
    
    with col2:
        if st.button("🎲 Random Example", use_container_width=True) and examples:
            import random
            random_example = random.choice(examples)
            st.session_state.example_query = random_example['query']
            st.rerun()
    
    # Process query
    if send_button and query.strip():
        if not health_status:
            st.error("❌ Cannot send query: API is not available")
            return
        
        # Check if we should use demo mode
        use_demo_mode = hasattr(st.session_state, 'demo_mode') and st.session_state.demo_mode
        
        # Add user message
        user_message = {
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Show typing indicator
        typing_placeholder = st.empty()
        with typing_placeholder:
            st.markdown("""
            <div class="assistant-message">
                <strong>🤖 AI Assistant</strong><br>
                <span class="loading-dots">Thinking</span>
            </div>
            """, unsafe_allow_html=True)
        
        if use_demo_mode:
            # Demo mode - simulate response
            time.sleep(2)  # Simulate processing time
            
            demo_response = f"""
            **🧠 Demo Mode Response**
            
            I found relevant information about your query: "{query}"
            
            Based on the 343 documents in the knowledge base, here's what I can tell you:
            
            • Your question relates to topics covered in our AI and machine learning document collection
            • The vector search found several relevant passages
            • For a complete AI-powered response, please set up an OpenAI API key
            
            **📋 Quick Setup:**
            1. Get an OpenAI API key from [platform.openai.com](https://platform.openai.com)
            2. Set it as an environment variable: `OPENAI_API_KEY=your_key_here`
            3. Restart the server to enable full AI responses
            
            **🔍 Retrieved Sources:** Found relevant content in the document collection (343 documents available)
            """
            
            assistant_message = {
                "role": "assistant", 
                "content": demo_response,
                "sources": [],
                "model_used": "Demo Mode",
                "response_time": 2.0,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            
            # Reset demo mode
            if hasattr(st.session_state, 'demo_mode'):
                del st.session_state.demo_mode
        else:
            # Regular API call
            response = api_client.chat_query(
                query=query,
                max_sources=max_sources,
                template_type=template_type,
                include_sources=include_sources
            )
        
        # Clear typing indicator
        typing_placeholder.empty()
        
        if not use_demo_mode:
            if "error" not in response:
                # Add assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response.get('response', 'No response received'),
                    "sources": response.get('sources', []),
                    "model_used": response.get('model_used', 'Unknown'),
                    "response_time": response.get('response_time', 0),
                    "timestamp": response.get('timestamp', datetime.now().isoformat())
                }
                st.session_state.messages.append(assistant_message)
            else:
                # Add error message with helpful information
                error_content = f"""
                ❌ **Error**: {response['error']}
                
                {response.get('detail', '')}
                
                **💡 Tip**: If you're seeing this error due to missing OpenAI API key, try the **🧠 Demo Mode** button instead!
                """
                error_message = {
                    "role": "assistant",
                    "content": error_content,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(error_message)
        
        # Limit history length
        if len(st.session_state.messages) > MAX_HISTORY_LENGTH:
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY_LENGTH:]
        
        # Clear input and rerun
        st.session_state.clear_input = True
        st.rerun()

if __name__ == "__main__":
    main()

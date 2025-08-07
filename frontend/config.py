"""
Frontend configuration file for RAG Chatbot
"""

import os
from typing import Dict, Any

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# UI Configuration
APP_TITLE = "RAG-based AI Assistant"
APP_ICON = "🤖"
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "50"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Chat Settings
DEFAULT_MAX_SOURCES = int(os.getenv("DEFAULT_MAX_SOURCES", "3"))
DEFAULT_TEMPLATE_TYPE = os.getenv("DEFAULT_TEMPLATE_TYPE", "qa")
DEFAULT_INCLUDE_SOURCES = os.getenv("DEFAULT_INCLUDE_SOURCES", "true").lower() == "true"

# Theme Configuration
THEME_CONFIG = {
    "primary_color": "#667eea",
    "secondary_color": "#764ba2",
    "accent_color": "#f093fb",
    "background_gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "message_gradient_user": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "message_gradient_assistant": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
}

# Features Configuration
FEATURES = {
    "document_upload": True,
    "conversation_export": True,
    "source_highlighting": True,
    "typing_indicators": True,
    "example_queries": True,
    "system_monitoring": True,
    "real_time_updates": True
}

# Error Messages
ERROR_MESSAGES = {
    "api_unavailable": "🔴 API service is currently unavailable. Please check if the backend is running.",
    "upload_failed": "❌ Document upload failed. Please try again.",
    "query_failed": "❌ Failed to process your query. Please try again.",
    "invalid_file": "❌ Please upload a valid PDF file.",
    "file_too_large": f"❌ File size exceeds {MAX_FILE_SIZE_MB}MB limit.",
    "network_error": "🌐 Network connection error. Please check your internet connection."
}

# Success Messages
SUCCESS_MESSAGES = {
    "document_uploaded": "✅ Document uploaded and processed successfully!",
    "conversation_exported": "📥 Conversation exported successfully!",
    "query_processed": "✅ Query processed successfully!",
    "api_connected": "🟢 Connected to RAG API service"
}

def get_streamlit_config() -> Dict[str, Any]:
    """Get Streamlit page configuration"""
    return {
        "page_title": APP_TITLE,
        "page_icon": APP_ICON,
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "menu_items": {
            'Get Help': None,
            'Report a bug': None,
            'About': f"{APP_TITLE} - Professional document-powered conversations using RAG technology"
        }
    }

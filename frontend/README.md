# RAG Chatbot Frontend

Professional frontend interface for the RAG-based AI chatbot with document processing capabilities.

## Features

### 🎨 Professional UI Design
- **Modern Interface**: Clean, professional design with gradient themes
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Dark/Light Theme**: Professional color scheme optimized for extended use
- **Smooth Animations**: Subtle animations and transitions for better UX

### 💬 Advanced Chat Interface
- **Real-time Messaging**: Instant chat with AI assistant
- **Typing Indicators**: Visual feedback when AI is processing
- **Message History**: Persistent conversation history during session
- **Source References**: Clickable source citations with confidence scores
- **Error Handling**: Graceful error display and recovery

### 📁 Document Management
- **Drag & Drop Upload**: Intuitive file upload with visual feedback
- **Upload Progress**: Real-time upload progress tracking
- **Document List**: View all uploaded documents with metadata
- **File Validation**: Automatic PDF validation and size checking
- **Batch Upload**: Support for multiple file uploads

### ⚙️ Customizable Settings
- **Source Control**: Adjust number of sources (1-10)
- **Response Style**: Choose from multiple templates (Q&A, Summary, etc.)
- **Source Toggle**: Enable/disable source references
- **Real-time Updates**: Settings apply immediately

### 📊 System Monitoring
- **API Status**: Live connection status monitoring
- **Component Health**: Individual system component status
- **Performance Metrics**: Response times and system statistics
- **Auto-reconnection**: Automatic retry on connection failures

### 🚀 UX Enhancements
- **Example Queries**: Pre-built query examples for quick start
- **Export Functionality**: Download conversation history as JSON
- **Keyboard Shortcuts**: Enter to send, Shift+Enter for new line
- **Auto-focus**: Smart input focus management
- **Toast Notifications**: Non-intrusive status updates

## Quick Start

### Option 1: Streamlit Interface (Recommended)

1. **Install Dependencies**:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

2. **Launch Streamlit App**:
   ```bash
   python run.py
   # OR
   streamlit run app.py
   ```

3. **Access Interface**:
   - Open browser to `http://localhost:8501`
   - Interface opens automatically

### Option 2: HTML/JavaScript Interface

1. **Start a Local Server**:
   ```bash
   cd frontend
   python -m http.server 8080
   # OR
   npx serve .
   ```

2. **Access Interface**:
   - Open browser to `http://localhost:8080`
   - View `index.html`

## File Structure

```
frontend/
├── app.py                 # Main Streamlit application
├── components.py          # Reusable UI components
├── utils.py              # Utility functions
├── config.py             # Configuration settings
├── run.py                # Launch script for Streamlit
├── requirements.txt      # Python dependencies
├── index.html            # HTML interface
├── styles.css            # Professional CSS styling
├── script.js             # JavaScript functionality
└── README.md             # This file
```

## Configuration

### Environment Variables

```bash
# API Configuration
API_BASE_URL=http://localhost:8000
API_TIMEOUT=30

# UI Configuration
MAX_HISTORY_LENGTH=50
MAX_FILE_SIZE_MB=50
DEFAULT_MAX_SOURCES=3
DEFAULT_TEMPLATE_TYPE=qa
DEFAULT_INCLUDE_SOURCES=true
```

### Streamlit Configuration

The app uses custom Streamlit configuration for optimal performance:

```toml
[server]
address = "0.0.0.0"
port = 8501
headless = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## API Integration

The frontend communicates with the RAG backend through REST API endpoints:

### Core Endpoints
- `GET /health` - Health check
- `GET /status` - System status
- `POST /chat` - Send queries
- `POST /upload` - Upload documents
- `GET /documents` - List documents
- `GET /examples` - Query examples

### Request Format
```javascript
// Chat query
{
  "query": "What is artificial intelligence?",
  "max_sources": 3,
  "template_type": "qa",
  "include_sources": true
}
```

### Response Format
```javascript
// Chat response
{
  "query": "What is artificial intelligence?",
  "response": "AI response text...",
  "sources": [
    {
      "filename": "document.pdf",
      "page_number": 2,
      "similarity_score": 0.92,
      "chunk_text": "Relevant text..."
    }
  ],
  "model_used": "gpt-3.5-turbo",
  "response_time": 2.45
}
```

## Customization

### Themes and Styling

1. **Color Scheme**: Modify CSS variables in `styles.css`:
   ```css
   :root {
     --primary-color: #667eea;
     --secondary-color: #764ba2;
     --accent-color: #f093fb;
   }
   ```

2. **Layout**: Adjust grid layouts and spacing
3. **Components**: Customize individual UI components in `components.py`

### Functionality

1. **API Configuration**: Update endpoints in `config.py`
2. **Chat Features**: Modify chat logic in `app.py` or `script.js`
3. **Upload Handling**: Customize file processing in utils

## Browser Support

### Recommended Browsers
- **Chrome 90+** (Recommended)
- **Firefox 88+**
- **Safari 14+**
- **Edge 90+**

### Required Features
- ES6+ JavaScript support
- CSS Grid and Flexbox
- Fetch API
- WebRTC (for future enhancements)

## Troubleshooting

### Common Issues

1. **API Connection Failed**:
   - Ensure backend is running on `localhost:8000`
   - Check firewall settings
   - Verify API endpoint URLs

2. **Upload Failures**:
   - Check file size limits (50MB default)
   - Verify PDF file format
   - Ensure sufficient disk space

3. **Styling Issues**:
   - Clear browser cache
   - Check console for CSS errors
   - Verify font loading

### Debug Mode

Enable debug mode for additional logging:

```bash
# Streamlit
streamlit run app.py --logger.level=debug

# HTML/JS
# Open browser developer tools
# Check console for debug messages
```

## Performance Optimization

### Best Practices

1. **Image Optimization**: Use WebP format for images
2. **Code Splitting**: Lazy load components when needed
3. **Caching**: Implement service worker for offline support
4. **Compression**: Enable gzip compression on server

### Monitoring

- Monitor API response times
- Track user interaction patterns
- Analyze error rates and types
- Monitor resource usage

## Security Considerations

1. **Input Validation**: All user inputs are validated
2. **File Upload Security**: PDF validation and virus scanning
3. **XSS Prevention**: Content sanitization
4. **CORS Configuration**: Proper CORS setup for API calls

## Development

### Local Development

1. **Hot Reload**: Streamlit provides automatic reloading
2. **Debug Tools**: Browser developer tools for HTML/JS
3. **Code Formatting**: Use Black for Python, Prettier for JS
4. **Testing**: Run tests before deployment

### Contributing

1. Follow existing code style
2. Add comments for complex logic
3. Test on multiple browsers
4. Update documentation

## Deployment

### Production Deployment

1. **Streamlit Cloud**: Direct deployment from GitHub
2. **Docker**: Containerized deployment
3. **Static Hosting**: For HTML/JS version (Netlify, Vercel)
4. **CDN**: Use CDN for static assets

### Environment Setup

```bash
# Production environment variables
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
API_BASE_URL=https://your-api-domain.com
```

## License

This frontend is part of the RAG Chatbot project and follows the same MIT license.

## Support

For issues and questions:
1. Check this README
2. Review the troubleshooting section
3. Check browser console for errors
4. Contact development team

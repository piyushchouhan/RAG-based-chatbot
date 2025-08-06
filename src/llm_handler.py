"""
LLM Handler for RAG-based Chatbot

This module provides:
- OpenAI API integration with error handling
- Prompt templates for RAG responses
- Context window management
- Source citation formatting
- Response post-processing and validation
- Rate limiting and retry logic
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import openai
from openai import OpenAI

# Local imports
try:
    from .retriever import DocumentRetriever, RetrievalResult, RetrievalConfig
except ImportError:
    from retriever import DocumentRetriever, RetrievalResult, RetrievalConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM handler"""
    # OpenAI Configuration
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    max_context_length: int = 4000
    
    # Response Configuration
    include_sources: bool = True
    max_sources: int = 3
    citation_style: str = "numbered"  # "numbered", "inline", "footnote"
    
    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0
    requests_per_minute: int = 60
    
    # Quality validation
    min_response_length: int = 50
    max_response_length: int = 2000
    validate_sources: bool = True


@dataclass
class RAGResponse:
    """RAG response with metadata"""
    query: str
    response: str
    sources: List[RetrievalResult]
    model_used: str
    timestamp: datetime
    response_time: float
    token_usage: Optional[Dict[str, int]] = None
    confidence_score: Optional[float] = None


class PromptTemplateManager:
    """Manages prompt templates for different query types"""
    
    def __init__(self):
        """Initialize prompt templates"""
        self.templates = {
            'default': self._get_default_template(),
            'qa': self._get_qa_template(),
            'summary': self._get_summary_template(),
            'explanation': self._get_explanation_template(),
            'comparison': self._get_comparison_template()
        }
    
    def _get_default_template(self) -> str:
        """Default RAG prompt template"""
        return """You are a helpful AI assistant that answers questions based on the provided context. Use the context below to answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Instructions:
1. Answer the question based on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Include relevant source references when possible
4. Be concise but thorough
5. Use a professional and helpful tone

Answer:"""
    
    def _get_qa_template(self) -> str:
        """Question-answering focused template"""
        return """Based on the following documents, please provide a detailed answer to the question.

Document Context:
{context}

Question: {query}

Please provide:
1. A direct answer to the question
2. Supporting details from the context
3. Any relevant examples or explanations
4. Source references where applicable

Answer:"""
    
    def _get_summary_template(self) -> str:
        """Summary-focused template"""
        return """Please provide a comprehensive summary based on the following context.

Content to Summarize:
{context}

Query: {query}

Please create a summary that:
1. Captures the key points and main ideas
2. Is well-organized and easy to understand
3. Includes important details and examples
4. References the source materials

Summary:"""
    
    def _get_explanation_template(self) -> str:
        """Explanation-focused template"""
        return """Please provide a clear and detailed explanation based on the context below.

Context:
{context}

Topic to Explain: {query}

Please explain:
1. The fundamental concepts involved
2. How things work or why they are important
3. Any relevant examples or applications
4. Key relationships and connections

Explanation:"""
    
    def _get_comparison_template(self) -> str:
        """Comparison-focused template"""
        return """Based on the provided context, please compare and contrast the relevant topics.

Context:
{context}

Comparison Request: {query}

Please provide:
1. Key similarities between the topics
2. Important differences
3. Advantages and disadvantages where applicable
4. Practical implications or use cases

Comparison:"""
    
    def get_template(self, template_type: str = 'default') -> str:
        """Get a specific prompt template"""
        return self.templates.get(template_type, self.templates['default'])
    
    def detect_query_type(self, query: str) -> str:
        """Detect the type of query to use appropriate template"""
        query_lower = query.lower()
        
        # Check for comparison keywords
        comparison_keywords = ['compare', 'contrast', 'difference', 'versus', 'vs', 'better']
        if any(keyword in query_lower for keyword in comparison_keywords):
            return 'comparison'
        
        # Check for explanation keywords
        explanation_keywords = ['explain', 'how does', 'why does', 'what is', 'how to']
        if any(keyword in query_lower for keyword in explanation_keywords):
            return 'explanation'
        
        # Check for summary keywords
        summary_keywords = ['summarize', 'summary', 'overview', 'main points']
        if any(keyword in query_lower for keyword in summary_keywords):
            return 'summary'
        
        # Check for direct questions
        if query.strip().endswith('?'):
            return 'qa'
        
        return 'default'


class ContextManager:
    """Manages context window and chunk formatting"""
    
    def __init__(self, max_context_length: int = 4000):
        """
        Initialize context manager
        
        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length
    
    def format_context(self, retrieval_results: List[RetrievalResult]) -> str:
        """
        Format retrieval results into context string
        
        Args:
            retrieval_results: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        if not retrieval_results:
            return "No relevant context found."
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(retrieval_results):
            # Format source info
            source_info = f"Source {i+1}: {result.source_file} (Page {result.page_number})"
            
            # Format chunk content
            chunk_content = f"\n\n{source_info}\n{result.chunk_text}"
            
            # Check if adding this chunk would exceed context length
            if current_length + len(chunk_content) > self.max_context_length:
                # Try to fit partial content
                remaining_space = self.max_context_length - current_length - len(source_info) - 10
                if remaining_space > 100:  # Only add if we have meaningful space
                    truncated_text = result.chunk_text[:remaining_space] + "..."
                    chunk_content = f"\n\n{source_info}\n{truncated_text}"
                    context_parts.append(chunk_content)
                break
            
            context_parts.append(chunk_content)
            current_length += len(chunk_content)
        
        return "".join(context_parts)
    
    def optimize_context(self, context: str, query: str) -> str:
        """
        Optimize context by prioritizing relevant sentences
        
        Args:
            context: Raw context string
            query: User query for relevance scoring
            
        Returns:
            Optimized context string
        """
        if len(context) <= self.max_context_length:
            return context
        
        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        query_words = set(query.lower().split())
        
        # Score sentences by relevance to query
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            score = overlap / len(query_words) if query_words else 0
            
            scored_sentences.append((score, sentence.strip()))
        
        # Sort by relevance and reconstruct context
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        optimized_context = ""
        for score, sentence in scored_sentences:
            if len(optimized_context) + len(sentence) + 2 <= self.max_context_length:
                optimized_context += sentence + ". "
            else:
                break
        
        return optimized_context.strip()


class SourceCitationFormatter:
    """Formats source citations in different styles"""
    
    def format_citations(
        self, 
        response: str, 
        sources: List[RetrievalResult],
        style: str = "numbered"
    ) -> Tuple[str, str]:
        """
        Format response with citations
        
        Args:
            response: Generated response text
            sources: List of source documents
            style: Citation style ("numbered", "inline", "footnote")
            
        Returns:
            Tuple of (formatted_response, citations_text)
        """
        if style == "numbered":
            return self._format_numbered_citations(response, sources)
        elif style == "inline":
            return self._format_inline_citations(response, sources)
        elif style == "footnote":
            return self._format_footnote_citations(response, sources)
        else:
            return response, self._format_source_list(sources)
    
    def _format_numbered_citations(
        self, 
        response: str, 
        sources: List[RetrievalResult]
    ) -> Tuple[str, str]:
        """Format with numbered citations [1], [2], etc."""
        citations = []
        for i, source in enumerate(sources, 1):
            citation = f"[{i}] {source.source_file}, Page {source.page_number}"
            citations.append(citation)
        
        citations_text = "\n\nSources:\n" + "\n".join(citations)
        return response, citations_text
    
    def _format_inline_citations(
        self, 
        response: str, 
        sources: List[RetrievalResult]
    ) -> Tuple[str, str]:
        """Format with inline citations (Source: filename)"""
        # This is a simplified version - in practice, you'd need NLP to determine
        # where to place citations within the response text
        unique_sources = {}
        for source in sources:
            key = f"{source.source_file}_p{source.page_number}"
            if key not in unique_sources:
                unique_sources[key] = source
        
        citations = [f"({source.source_file}, p.{source.page_number})" 
                    for source in unique_sources.values()]
        
        citations_text = "\n\nSources: " + ", ".join(citations)
        return response, citations_text
    
    def _format_footnote_citations(
        self, 
        response: str, 
        sources: List[RetrievalResult]
    ) -> Tuple[str, str]:
        """Format with footnote-style citations"""
        footnotes = []
        for i, source in enumerate(sources, 1):
            footnote = f"{i}. {source.source_file}, Page {source.page_number}"
            footnotes.append(footnote)
        
        citations_text = "\n\nReferences:\n" + "\n".join(footnotes)
        return response, citations_text
    
    def _format_source_list(self, sources: List[RetrievalResult]) -> str:
        """Format simple source list"""
        source_list = []
        seen_sources = set()
        
        for source in sources:
            source_key = f"{source.source_file}_{source.page_number}"
            if source_key not in seen_sources:
                source_list.append(f"• {source.source_file} (Page {source.page_number})")
                seen_sources.add(source_key)
        
        return "\n\nSources:\n" + "\n".join(source_list)


class LLMHandler:
    """Main LLM handler class for RAG pipeline"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM handler
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self.client = None
        self.prompt_manager = PromptTemplateManager()
        self.context_manager = ContextManager(self.config.max_context_length)
        self.citation_formatter = SourceCitationFormatter()
        self.request_times = []  # For rate limiting
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            # Get API key from config or environment
            api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                logger.warning("No OpenAI API key provided. LLM functionality will be limited.")
                return
            
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're at the rate limit
        if len(self.request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    def _make_api_call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Make API call with retry logic
        
        Args:
            messages: List of message objects for the API
            
        Returns:
            API response
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide an API key.")
        
        for attempt in range(self.config.max_retries):
            try:
                self._check_rate_limit()
                
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                return response
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise
                    
            except openai.APIError as e:
                logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise
    
    def generate_response(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult],
        template_type: Optional[str] = None
    ) -> str:
        """
        Generate response using LLM
        
        Args:
            query: User query
            retrieval_results: Retrieved context chunks
            template_type: Specific template type to use
            
        Returns:
            Generated response
        """
        start_time = time.time()
        
        try:
            # Detect query type if not specified
            if template_type is None:
                template_type = self.prompt_manager.detect_query_type(query)
            
            # Get appropriate template
            template = self.prompt_manager.get_template(template_type)
            
            # Format context from retrieval results
            context = self.context_manager.format_context(retrieval_results)
            
            # Optimize context if needed
            context = self.context_manager.optimize_context(context, query)
            
            # Create the prompt
            prompt = template.format(context=context, query=query)
            
            # Prepare messages for API
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that provides accurate information based on the given context."},
                {"role": "user", "content": prompt}
            ]
            
            # Make API call
            response = self._make_api_call(messages)
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            response_time = time.time() - start_time
            logger.info(f"Generated response in {response_time:.2f} seconds")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def validate_response(self, response: str, sources: List[RetrievalResult]) -> bool:
        """
        Validate response quality
        
        Args:
            response: Generated response
            sources: Source documents
            
        Returns:
            True if response passes validation
        """
        # Check response length
        if len(response) < self.config.min_response_length:
            logger.warning("Response too short")
            return False
        
        if len(response) > self.config.max_response_length:
            logger.warning("Response too long")
            return False
        
        # Check if response is meaningful (not just error messages)
        error_indicators = ["error", "sorry", "apologize", "cannot", "unable"]
        if all(indicator in response.lower() for indicator in error_indicators):
            logger.warning("Response appears to be an error message")
            return False
        
        return True
    
    def create_rag_response(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult],
        template_type: Optional[str] = None
    ) -> RAGResponse:
        """
        Create complete RAG response with citations
        
        Args:
            query: User query
            retrieval_results: Retrieved context chunks
            template_type: Specific template type to use
            
        Returns:
            Complete RAG response object
        """
        start_time = time.time()
        
        # Generate response
        response_text = self.generate_response(query, retrieval_results, template_type)
        
        # Format citations if enabled
        if self.config.include_sources and retrieval_results:
            limited_sources = retrieval_results[:self.config.max_sources]
            response_text, citations = self.citation_formatter.format_citations(
                response_text, limited_sources, self.config.citation_style
            )
            response_text += citations
        
        # Validate response
        is_valid = self.validate_response(response_text, retrieval_results)
        if not is_valid:
            logger.warning("Generated response failed validation")
        
        # Create response object
        rag_response = RAGResponse(
            query=query,
            response=response_text,
            sources=retrieval_results[:self.config.max_sources],
            model_used=self.config.model,
            timestamp=datetime.now(),
            response_time=time.time() - start_time,
            confidence_score=0.8 if is_valid else 0.3  # Simple confidence scoring
        )
        
        return rag_response


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation"""
    
    def __init__(
        self, 
        retrieval_config: Optional[RetrievalConfig] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retrieval_config: Configuration for retrieval system
            llm_config: Configuration for LLM handler
        """
        self.retriever = DocumentRetriever(retrieval_config)
        self.llm_handler = LLMHandler(llm_config)
        
        # Load retrieval index
        try:
            self.retriever.load_index()
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load retrieval index: {e}")
            raise
    
    def query(
        self, 
        question: str, 
        k: int = 5,
        template_type: Optional[str] = None
    ) -> RAGResponse:
        """
        Process a complete RAG query
        
        Args:
            question: User question
            k: Number of documents to retrieve
            template_type: Specific template type to use
            
        Returns:
            Complete RAG response
        """
        logger.info(f"Processing RAG query: '{question[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant documents
            retrieval_results = self.retriever.retrieve(question, k=k)
            
            if not retrieval_results:
                logger.warning("No relevant documents found")
                return RAGResponse(
                    query=question,
                    response="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    model_used=self.llm_handler.config.model,
                    timestamp=datetime.now(),
                    response_time=0.0,
                    confidence_score=0.0
                )
            
            # Step 2: Generate response
            rag_response = self.llm_handler.create_rag_response(
                question, retrieval_results, template_type
            )
            
            logger.info(f"RAG query completed in {rag_response.response_time:.2f} seconds")
            return rag_response
            
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            return RAGResponse(
                query=question,
                response=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                model_used=self.llm_handler.config.model,
                timestamp=datetime.now(),
                response_time=0.0,
                confidence_score=0.0
            )


def test_llm_integration():
    """Test the complete LLM integration"""
    try:
        # Test configuration
        llm_config = LLMConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
            include_sources=True,
            citation_style="numbered"
        )
        
        retrieval_config = RetrievalConfig(
            vector_store_type="faiss",
            index_path="data/vector_index",
            default_k=3
        )
        
        # Initialize RAG pipeline
        print("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(retrieval_config, llm_config)
        
        # Test queries
        test_queries = [
            "What is artificial intelligence and how does it work?",
            "How is AI being used in healthcare applications?",
            "Compare machine learning and deep learning approaches",
            "Explain the main challenges in AI development"
        ]
        
        print("\nTesting RAG pipeline with sample queries...")
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            # Process query
            response = rag_pipeline.query(query, k=3)
            
            print(f"Response: {response.response}")
            print(f"Model: {response.model_used}")
            print(f"Response Time: {response.response_time:.2f}s")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Sources Used: {len(response.sources)}")
        
        print("\nLLM integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"LLM integration test failed: {e}")
        return False


if __name__ == "__main__":
    test_llm_integration()

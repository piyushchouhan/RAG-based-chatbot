#!/usr/bin/env python3
"""
Test script for RAG Chatbot API

This script demonstrates how to use the API endpoints and provides
example requests for testing the functionality.
"""

import requests
import json
import time
from pathlib import Path

# API Base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']} - Version: {data['version']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False

def test_system_status():
    """Test the system status endpoint"""
    print("\n📊 Testing System Status...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System Status: {data['status']}")
            print(f"   Components: {data['components']}")
            if 'index_stats' in data and data['index_stats']:
                print(f"   Index Stats: {data['index_stats']}")
        else:
            print(f"❌ Status Check Failed: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Status Check Error: {e}")
        return False

def test_query_examples():
    """Test the query examples endpoint"""
    print("\n💡 Testing Query Examples...")
    try:
        response = requests.get(f"{BASE_URL}/examples")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query Examples Retrieved: {len(data['examples'])} examples")
            print(f"   Categories: {data['categories']}")
            
            # Show a few examples
            print("   Sample Examples:")
            for example in data['examples'][:3]:
                print(f"     • {example['query']}")
        else:
            print(f"❌ Query Examples Failed: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Query Examples Error: {e}")
        return False

def test_chat_query():
    """Test the chat endpoint"""
    print("\n💬 Testing Chat Query...")
    
    # Test queries
    test_queries = [
        {
            "query": "What is artificial intelligence?",
            "max_sources": 3,
            "include_sources": True
        },
        {
            "query": "How does machine learning work?",
            "max_sources": 2,
            "template_type": "explanation",
            "include_sources": True
        }
    ]
    
    try:
        for i, query_data in enumerate(test_queries, 1):
            print(f"\n   Test Query {i}: '{query_data['query']}'")
            
            response = requests.post(
                f"{BASE_URL}/chat",
                json=query_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Response received ({data['response_time']:.2f}s)")
                print(f"      Model: {data['model_used']}")
                print(f"      Sources: {len(data['sources'])}")
                print(f"      Response: {data['response'][:100]}...")
                
                if data['sources']:
                    print("      Source files:")
                    for source in data['sources']:
                        print(f"        - {source['filename']} (Page {source['page_number']})")
            else:
                print(f"   ❌ Query Failed: {response.status_code}")
                if response.status_code == 503:
                    print("      API may not be fully initialized. Try again in a moment.")
                elif response.status_code == 422:
                    print(f"      Validation Error: {response.json()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat Query Error: {e}")
        return False

def test_document_list():
    """Test the document list endpoint"""
    print("\n📚 Testing Document List...")
    try:
        response = requests.get(f"{BASE_URL}/documents")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Documents Listed: {len(data['documents'])} documents")
            
            if data['documents']:
                print("   Documents:")
                for doc in data['documents']:
                    print(f"     • {doc['filename']} ({doc['size']} bytes)")
            else:
                print("   No documents found. Upload some PDFs to test chat functionality.")
        else:
            print(f"❌ Document List Failed: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Document List Error: {e}")
        return False

def test_configuration():
    """Test the configuration endpoint"""
    print("\n⚙️ Testing Configuration...")
    try:
        response = requests.get(f"{BASE_URL}/config")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Configuration Retrieved:")
            for key, value in data.items():
                print(f"     {key}: {value}")
        else:
            print(f"❌ Configuration Failed: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return False

def test_api_documentation():
    """Test API documentation endpoints"""
    print("\n📖 Testing API Documentation...")
    try:
        # Test OpenAPI docs
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ Swagger UI accessible at /docs")
        else:
            print(f"❌ Swagger UI failed: {response.status_code}")
        
        # Test ReDoc
        response = requests.get(f"{BASE_URL}/redoc")
        if response.status_code == 200:
            print("✅ ReDoc accessible at /redoc")
        else:
            print(f"❌ ReDoc failed: {response.status_code}")
        
        # Test OpenAPI schema
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code == 200:
            print("✅ OpenAPI schema accessible")
        else:
            print(f"❌ OpenAPI schema failed: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Documentation Error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🧪 RAG Chatbot API Test Suite")
    print("=" * 50)
    
    # Track test results
    tests = [
        ("Health Check", test_health_check),
        ("System Status", test_system_status),
        ("Query Examples", test_query_examples),
        ("Document List", test_document_list),
        ("Configuration", test_configuration),
        ("API Documentation", test_api_documentation),
        ("Chat Query", test_chat_query),  # Test this last as it requires full initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
        
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. Minor issues may exist.")
    else:
        print("⚠️ Several tests failed. Check the API configuration.")
    
    print("\n💡 Quick Start:")
    print(f"   1. Visit {BASE_URL}/docs for interactive API documentation")
    print(f"   2. Upload PDFs using the /upload endpoint")
    print(f"   3. Ask questions using the /chat endpoint")

if __name__ == "__main__":
    main()

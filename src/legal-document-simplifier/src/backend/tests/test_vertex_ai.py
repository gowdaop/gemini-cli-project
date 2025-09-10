#!/usr/bin/env python3
"""
Test script for Vertex AI integration
Run this to verify your setup before deploying
"""

import os
import asyncio
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_vertex_ai():
    """Test Vertex AI API access"""
    print("🧪 Testing Vertex AI Integration...")
    
    try:
        import google.genai as genai
        print("✅ Google GenAI SDK imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Google GenAI: {e}")
        return False
    
    # Test 1: Check environment variables
    print("\n📋 Checking Environment Variables:")
    api_key = os.getenv('GOOGLE_API_KEY')
    creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    project_id = os.getenv('GCP_PROJECT_ID')
    
    print(f"GOOGLE_API_KEY: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {'✅ Set' if creds else '❌ Missing'}")
    print(f"GCP_PROJECT_ID: {'✅ Set' if project_id else '❌ Missing'}")
    
    if not api_key and not creds:
        print("❌ No authentication method found!")
        return False
    
    # Test 2: Initialize client
    print("\n🔧 Initializing Vertex AI Client:")
    try:
        if api_key:
            client = genai.Client(api_key=api_key)
            print("✅ Client initialized with API key")
        elif creds:
            # For service account authentication, use vertexai=True
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location="us"
            )
            print(f"✅ Client initialized with service account")
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False
    
    # Test 3: Generate text
# Test 3: Generate text
    print("\n🤖 Testing Text Generation:")
    try:
        # ✅ NEW SDK SYNTAX - Replace the old config approach
        from google.genai import types
        
        config = types.GenerateContentConfig(
            max_output_tokens=100,
            temperature=0.1
        )
        
        response = client.models.generate_content(
        model="gemini-2.0-flash-exp",  # ✅ Available for new projects
            contents=["Explain what an indemnification clause means in simple terms."],
            config=config
        )
        
        if response and hasattr(response, 'text'):
            print("✅ Text generation successful!")
            print(f"📝 Response: {response.text[:200]}...")
            return True
        else:
            print("❌ No text in response")
            return False
            
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False


async def test_milvus_connection():
    """Test Milvus connection"""
    print("\n🗄️ Testing Milvus Connection:")
    try:
        from pymilvus import connections, utility, Collection
        
        # CHANGE: Use 'milvus' instead of 'localhost' when running inside container
        # Check if we're inside container or running locally
        import os
        milvus_host = "milvus" if os.path.exists("/.dockerenv") else "localhost"
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=milvus_host,  # This will use 'milvus' inside container
            port="19530"
        )
        
        # Test collection access
        if utility.has_collection("data2"):
            collection = Collection("data2")
            print(f"✅ Connected to Milvus collection 'data2' on {milvus_host}")
            print(f"📊 Collection stats: {collection.num_entities} entities")
            return True
        else:
            print("❌ Collection 'data2' not found")
            return False
            
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting API Tests...\n")
    
    vertex_ok = await test_vertex_ai()
    milvus_ok = await test_milvus_connection()
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print("="*50)
    print(f"Vertex AI: {'✅ PASS' if vertex_ok else '❌ FAIL'}")
    print(f"Milvus:    {'✅ PASS' if milvus_ok else '❌ FAIL'}")
    
    if vertex_ok and milvus_ok:
        print("\n🎉 All tests passed! Your setup is ready.")
        return True
    else:
        print("\n⚠️  Some tests failed. Fix issues before deploying.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

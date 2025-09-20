#!/usr/bin/env python3
"""
Debug script to check the OCR structure from PDF uploads
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"
API_KEY = "legal-doc-analyzer-2025-secure-key-f47d4a2c"

def debug_ocr_structure():
    """Debug the OCR structure from PDF upload"""
    print("🔍 Debugging OCR Structure...")
    
    # Check if we have a sample PDF
    sample_pdf_path = "src/legal-document-simplifier/src/backend/tests/sample.pdf"
    if not os.path.exists(sample_pdf_path):
        print(f"❌ Sample PDF not found at {sample_pdf_path}")
        return
    
    # Upload PDF
    print("Uploading sample PDF...")
    with open(sample_pdf_path, 'rb') as f:
        files = {'file': ('sample.pdf', f, 'application/pdf')}
        response = requests.post(
            f"{BASE_URL}/upload/",
            headers={"X-API-Key": API_KEY},
            files=files
        )
    
    if response.status_code != 200:
        print(f"❌ PDF upload failed: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    upload_result = response.json()
    print("✅ PDF uploaded successfully")
    
    # Debug OCR structure
    ocr_data = upload_result.get("ocr", {})
    print(f"\n📄 OCR Structure:")
    print(f"Full text length: {len(ocr_data.get('full_text', ''))}")
    print(f"Full text preview: {ocr_data.get('full_text', '')[:200]}...")
    
    blocks = ocr_data.get("blocks", [])
    print(f"\n📦 Blocks count: {len(blocks)}")
    
    for i, block in enumerate(blocks[:3]):  # Show first 3 blocks
        print(f"\nBlock {i}:")
        print(f"  Type: {type(block)}")
        if isinstance(block, dict):
            print(f"  Keys: {list(block.keys())}")
            print(f"  Text: {block.get('text', 'NO TEXT')[:100]}...")
            print(f"  Span: {block.get('span', 'NO SPAN')}")
        else:
            print(f"  Object: {block}")
    
    # Test the OCR data in a chat request
    print(f"\n🧪 Testing OCR data in chat request...")
    
    # Create conversation
    response = requests.post(f"{BASE_URL}/chat/new", headers={"X-API-Key": API_KEY})
    if response.status_code != 200:
        print(f"❌ Failed to create conversation: {response.status_code}")
        return
    
    conversation_id = response.json()["conversation_id"]
    print(f"✅ Created conversation: {conversation_id}")
    
    # Test with OCR data
    chat_data = {
        "question": "What are the termination requirements in this document?",
        "conversation_id": conversation_id,
        "ocr": ocr_data
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json=chat_data
    )
    
    if response.status_code != 200:
        print(f"❌ Chat request failed: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    result = response.json()
    print("✅ Chat response received")
    
    # Check evidence
    evidence = result.get("evidence", [])
    print(f"Evidence count: {len(evidence)}")
    
    # Look for current document evidence
    current_document_evidence = [e for e in evidence if e.get("doc_type") == "current_document"]
    print(f"Current document evidence: {len(current_document_evidence)}")
    
    if current_document_evidence:
        print("✅ Current document evidence found!")
        for i, e in enumerate(current_document_evidence[:2]):
            print(f"  Evidence {i}: {e['content'][:100]}...")
    else:
        print("❌ No current document evidence found")
        print("Available evidence types:", [e.get("doc_type") for e in evidence])

if __name__ == "__main__":
    debug_ocr_structure()

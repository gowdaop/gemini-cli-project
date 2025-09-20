#!/usr/bin/env python3
"""
Debug test to see why OCR routing isn't working in the PDF test
"""

import requests
import json
import os

BASE_URL = "http://localhost:8000"
API_KEY = "legal-doc-analyzer-2025-secure-key-f47d4a2c"

def test_ocr_routing_debug():
    """Debug OCR routing with detailed logging"""
    print("🔍 Debugging OCR Routing...")
    
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
        return
    
    upload_result = response.json()
    ocr_data = upload_result.get("ocr", {})
    print("✅ PDF uploaded successfully")
    print(f"OCR blocks count: {len(ocr_data.get('blocks', []))}")
    
    # Create conversation
    response = requests.post(f"{BASE_URL}/chat/new", headers={"X-API-Key": API_KEY})
    if response.status_code != 200:
        print(f"❌ Failed to create conversation: {response.status_code}")
        return
    
    conversation_id = response.json()["conversation_id"]
    print(f"✅ Created conversation: {conversation_id}")
    
    # Test 1: Just OCR (no analysis/selection)
    print("\n🧪 Test 1: Just OCR (no analysis/selection)")
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
    
    if response.status_code == 200:
        result = response.json()
        evidence = result.get("evidence", [])
        current_doc_evidence = [e for e in evidence if e.get("doc_type") == "current_document"]
        print(f"✅ OCR-only test: {len(current_doc_evidence)} current document evidence items")
    else:
        print(f"❌ OCR-only test failed: {response.status_code}")
    
    # Test 2: OCR + Analysis (no selection)
    print("\n🧪 Test 2: OCR + Analysis (no selection)")
    
    # Store analysis
    analysis_data = {
        "clauses": [
            {
                "id": "test_clause_1",
                "tag": "termination",
                "text": "Either party may terminate this Agreement by giving 30 days' written notice.",
                "span": {"page": 1, "start_line": 7, "end_line": 7}
            }
        ],
        "risks": [
            {
                "clause_id": "test_clause_1",
                "level": "red",
                "score": 0.8,
                "rationale": "High risk due to short notice period",
                "supporting_context": []
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/store-analysis/{conversation_id}",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json=analysis_data
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to store analysis: {response.status_code}")
        return
    
    print("✅ Analysis stored")
    
    # Test with OCR + Analysis (no selection)
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
    
    if response.status_code == 200:
        result = response.json()
        evidence = result.get("evidence", [])
        current_doc_evidence = [e for e in evidence if e.get("doc_type") == "current_document"]
        print(f"✅ OCR+Analysis test: {len(current_doc_evidence)} current document evidence items")
    else:
        print(f"❌ OCR+Analysis test failed: {response.status_code}")
    
    # Test 3: OCR + Analysis + Selection
    print("\n🧪 Test 3: OCR + Analysis + Selection")
    chat_data = {
        "question": "What are the termination requirements in this document?",
        "conversation_id": conversation_id,
        "selected_clause_id": "test_clause_1",
        "selected_risk_level": "red",
        "ocr": ocr_data
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json=chat_data
    )
    
    if response.status_code == 200:
        result = response.json()
        evidence = result.get("evidence", [])
        
        # Count different evidence types
        evidence_types = {}
        for e in evidence:
            doc_type = e.get("doc_type", "unknown")
            evidence_types[doc_type] = evidence_types.get(doc_type, 0) + 1
        
        print(f"✅ OCR+Analysis+Selection test:")
        print(f"   Evidence types: {evidence_types}")
        
        current_doc_evidence = [e for e in evidence if e.get("doc_type") == "current_document"]
        selected_clause_evidence = [e for e in evidence if e.get("doc_type") == "selected_clause"]
        
        print(f"   Current document evidence: {len(current_doc_evidence)}")
        print(f"   Selected clause evidence: {len(selected_clause_evidence)}")
        
        if current_doc_evidence:
            print("   ✅ Current document evidence found!")
        else:
            print("   ⚠️  No current document evidence (expected when clause is selected - selected clause takes priority)")
        
        if selected_clause_evidence:
            print("   ✅ Selected clause evidence found!")
        else:
            print("   ❌ No selected clause evidence found")
            
    else:
        print(f"❌ OCR+Analysis+Selection test failed: {response.status_code}")

if __name__ == "__main__":
    test_ocr_routing_debug()

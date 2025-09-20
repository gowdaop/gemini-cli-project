#!/usr/bin/env python3
"""
Test script to verify clause selection, risk selection, and whole PDF context checks.
"""

import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"
API_KEY = "legal-doc-analyzer-2025-secure-key-f47d4a2c"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# --- Test Data ---
ANALYSIS_DATA = {
    "clauses": [
        {
            "id": "pdf_clause_1",
            "tag": "termination",
            "text": "Either party may terminate this Agreement by giving 30 days' written notice.",
            "span": {"page": 1, "start_line": 7, "end_line": 7}
        },
        {
            "id": "pdf_clause_2", 
            "tag": "liability",
            "text": "The Company shall not be liable for any indirect, special, or consequential damages.",
            "span": {"page": 1, "start_line": 5, "end_line": 5}
        }
    ],
    "risks": [
        {
            "clause_id": "pdf_clause_1",
            "level": "orange",
            "score": 0.7,
            "rationale": "The 30-day notice period for termination might be considered short for long-term agreements.",
            "supporting_context": []
        },
        {
            "clause_id": "pdf_clause_2",
            "level": "red", 
            "score": 0.9,
            "rationale": "A broad limitation of liability can expose the other party to significant uncovered losses.",
            "supporting_context": []
        }
    ]
}

def setup_conversation_with_pdf():
    """Helper to set up a conversation and upload a PDF with analysis data."""
    print("\n--- SETUP: Conversation and PDF Upload ---")
    
    # 1. Create Conversation
    response = requests.post(f"{BASE_URL}/chat/new", headers=HEADERS)
    response.raise_for_status()
    conversation_id = response.json()["conversation_id"]
    print(f"PASS: Conversation created: {conversation_id}")

    # 2. Upload PDF to get OCR data
    sample_pdf_path = "src/legal-document-simplifier/src/backend/tests/sample.pdf"
    if not os.path.exists(sample_pdf_path):
        raise FileNotFoundError(f"Sample PDF not found at {sample_pdf_path}")
    
    with open(sample_pdf_path, 'rb') as f:
        files = {'file': ('sample.pdf', f, 'application/pdf')}
        response = requests.post(f"{BASE_URL}/upload/", headers={"X-API-Key": API_KEY}, files=files)
    
    response.raise_for_status()
    ocr_data = response.json().get("ocr")
    print("PASS: PDF uploaded and OCR data received.")

    # 3. Store analysis data
    response = requests.post(
        f"{BASE_URL}/chat/store-analysis/{conversation_id}",
        headers=HEADERS,
        json=ANALYSIS_DATA
    )
    response.raise_for_status()
    print("PASS: Analysis data stored in conversation.")
    
    return conversation_id, ocr_data

def test_clause_selection(conversation_id, ocr_data):
    """Test asking a question about a specifically selected clause."""
    print("\n--- TEST: Clause Selection ---")
    chat_data = {
        "question": "What are the termination requirements?",
        "conversation_id": conversation_id,
        "selected_clause_id": "pdf_clause_1",
        "ocr": ocr_data
    }
    
    response = requests.post(f"{BASE_URL}/chat/", headers=HEADERS, json=chat_data)
    response.raise_for_status()
    result = response.json()
    
    evidence = result.get("evidence", [])
    answer = result.get("answer", "").lower()
    
    assert any(e.get("doc_type") == "selected_clause" for e in evidence), "FAIL: 'selected_clause' evidence is missing."
    print("PASS: 'selected_clause' evidence found.")
    
    assert "30 days" in answer and "terminate" in answer, "FAIL: Answer is not relevant to the selected clause."
    print("PASS: Answer is relevant to the selected clause.")
    print("--- RESULT: Clause Selection PASSED ---")

def test_risk_level_selection(conversation_id, ocr_data):
    """Test asking a question filtered by a specific risk level."""
    print("\n--- TEST: Risk Level Selection ---")
    chat_data = {
        "question": "Summarize the high-risk items in this document.",
        "conversation_id": conversation_id,
        "selected_risk_level": "red",
        "ocr": ocr_data
    }
    
    response = requests.post(f"{BASE_URL}/chat/", headers=HEADERS, json=chat_data)
    response.raise_for_status()
    result = response.json()
    
    evidence = result.get("evidence", [])
    answer = result.get("answer", "").lower()

    assert any(e.get("doc_type") == "related_risk" for e in evidence), "FAIL: 'related_risk' evidence is missing."
    print("PASS: 'related_risk' evidence found.")
    assert any(e.get("doc_type") == "related_clause" for e in evidence), "FAIL: 'related_clause' evidence is missing."
    print("PASS: 'related_clause' evidence found.")

    assert "liability" in answer and "damages" in answer, "FAIL: Answer is not relevant to the selected risk level."
    print("PASS: Answer is relevant to the selected risk level.")
    print("--- RESULT: Risk Level Selection PASSED ---")

def test_whole_pdf_context(conversation_id, ocr_data):
    """Test a general question with the whole PDF as context (no selections)."""
    print("\n--- TEST: Whole PDF Context ---")
    chat_data = {
        "question": "What is the main purpose of this document?",
        "conversation_id": conversation_id,
        "ocr": ocr_data
    }
    
    response = requests.post(f"{BASE_URL}/chat/", headers=HEADERS, json=chat_data)
    response.raise_for_status()
    result = response.json()
    
    evidence = result.get("evidence", [])
    answer = result.get("answer", "").lower()

    assert any(e.get("doc_type") == "current_document" for e in evidence), "FAIL: 'current_document' evidence is missing."
    print("PASS: 'current_document' evidence found.")

    assert not any(e.get("doc_type") == "web_search_result" for e in evidence), "FAIL: Web fallback should be disabled when OCR context is present."
    print("PASS: Web fallback was correctly disabled.")
    
    assert "agreement" in answer or "contract" in answer or "terms" in answer or "deal" in answer or "party" in answer or "parties" in answer, "FAIL: General answer about the document seems irrelevant."
    print("PASS: General question received a relevant answer.")
    print("--- RESULT: Whole PDF Context PASSED ---")

def main():
    """Main test runner."""
    print(">>> Starting PDF Context and Selection Test Suite...")
    time.sleep(2)
    
    tests = [
        test_clause_selection,
        test_risk_level_selection,
        test_whole_pdf_context
    ]
    passed_count = 0
    failed_count = 0

    try:
        conversation_id, ocr_data = setup_conversation_with_pdf()
        
        for test_func in tests:
            try:
                test_func(conversation_id, ocr_data)
                passed_count += 1
            except Exception as e:
                print(f"FAIL: {test_func.__name__} failed with error: {e}")
                failed_count += 1

    except Exception as e:
        print(f"\nCRITICAL FAILURE during setup: {e}")
        failed_count = len(tests)

    print("\n" + "="*50)
    print("    Test Summary:")
    print(f"      Passed: {passed_count}")
    print(f"      Failed: {failed_count}")
    print("="*50)

    if failed_count > 0:
        print("\n>>> Some tests failed.")
    else:
        print("\n>>> All tests passed successfully!")

if __name__ == "__main__":
    main()

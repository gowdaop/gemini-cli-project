#!/usr/bin/env python3
"""
Test to check if conversation data is being passed correctly.
"""

import requests
import json

# Test configuration
BASE_URL = "http://localhost:8000"
API_KEY = "legal-doc-analyzer-2025-secure-key-f47d4a2c"

def make_request(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make HTTP request to the API"""
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    
    try:
        if method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=30)
        else:
            response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Request failed: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        print(f"❌ Request error: {e}")
        return {}

def test_conversation_data():
    """Test if conversation data is being passed correctly"""
    print("🧪 Testing Conversation Data...")
    
    # Step 1: Create conversation
    conv_response = make_request("/chat/new", "POST")
    conv_id = conv_response["conversation_id"]
    print(f"Created conversation: {conv_id}")
    
    # Step 2: Store analysis
    sample_clauses = [
        {
            "id": "test_clause_1",
            "tag": "termination",
            "text": "Either party may terminate this Agreement by giving 30 days' written notice.",
            "span": {"page": 1, "start_line": 7, "end_line": 7}
        }
    ]
    sample_risks = [
        {
            "clause_id": "test_clause_1",
            "level": "yellow",
            "score": 0.6,
            "rationale": "Moderate termination risk",
            "supporting_context": []
        }
    ]
    
    print("Storing analysis...")
    store_response = make_request(f"/chat/store-analysis/{conv_id}", "POST", {
        "clauses": sample_clauses,
        "risks": sample_risks
    })
    print(f"Store response: {store_response}")
    
    # Step 3: Check conversation data
    print("Checking conversation data...")
    history_response = make_request(f"/chat/conversations/{conv_id}")
    
    # Check if analysis data is present
    has_analysis_clauses = bool(history_response.get('analysis_clauses'))
    has_analysis_risks = bool(history_response.get('analysis_risks'))
    has_analysis_data = history_response.get('analytics', {}).get('has_analysis_data', False)
    
    print(f"Has analysis clauses: {has_analysis_clauses}")
    print(f"Has analysis risks: {has_analysis_risks}")
    print(f"Has analysis data: {has_analysis_data}")
    
    if has_analysis_clauses:
        print(f"Analysis clauses: {list(history_response.get('analysis_clauses', {}).keys())}")
    if has_analysis_risks:
        print(f"Analysis risks: {list(history_response.get('analysis_risks', {}).keys())}")
    
    # Step 4: Test chat request
    print("Testing chat request...")
    chat_request = {
        "question": "What is the selected termination clause?",
        "conversation_id": conv_id,
        "selected_clause_id": "test_clause_1"
    }
    
    chat_response = make_request("/chat/", "POST", chat_request)
    
    # Check if the response includes the selected clause
    answer = chat_response.get('answer', '')
    print(f"Answer length: {len(answer)}")
    print(f"Answer preview: {answer[:200]}...")
    
    # Check if the answer mentions the specific clause
    if "30 days" in answer and "terminate" in answer:
        print("✅ Answer correctly references the selected clause!")
        return True
    else:
        print("❌ Answer does not reference the selected clause")
        return False

if __name__ == "__main__":
    test_conversation_data()

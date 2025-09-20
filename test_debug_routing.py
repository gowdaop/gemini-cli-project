#!/usr/bin/env python3
"""
Simple test to see debug output from the routing logic.
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

def test_debug_routing():
    """Test routing with debug output"""
    print("🧪 Testing Routing with Debug Output...")
    
    # Step 1: Create conversation
    conv_response = make_request("/chat/new", "POST")
    if not conv_response.get("conversation_id"):
        print("❌ Failed to create conversation")
        return False
    
    conv_id = conv_response["conversation_id"]
    print(f"✅ Created conversation: {conv_id}")
    
    # Step 2: Store analysis
    sample_clauses = [
        {
            "id": "debug_clause_1",
            "tag": "termination",
            "text": "Either party may terminate this Agreement by giving 30 days' written notice.",
            "span": {"page": 1, "start_line": 7, "end_line": 7}
        }
    ]
    sample_risks = [
        {
            "clause_id": "debug_clause_1",
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
    print(f"✅ Analysis stored: {store_response}")
    
    # Step 3: Test with selected clause
    print("Testing with selected clause...")
    chat_request = {
        "question": "What is the selected termination clause?",
        "conversation_id": conv_id,
        "selected_clause_id": "debug_clause_1"
    }
    
    print("Sending chat request...")
    chat_response = make_request("/chat/", "POST", chat_request)
    
    if chat_response.get("answer"):
        print("✅ Chat response received")
        
        # Check evidence
        evidence = chat_response.get('evidence', [])
        print(f"Evidence count: {len(evidence)}")
        
        # Check evidence types
        evidence_types = [ev.get('doc_type', 'unknown') for ev in evidence]
        print(f"Evidence types: {evidence_types}")
        
        # Check for selected clause evidence
        selected_evidence = [ev for ev in evidence if ev.get('doc_type') == 'selected_clause']
        print(f"Selected clause evidence count: {len(selected_evidence)}")
        
        if selected_evidence:
            print("✅ Selected clause evidence found!")
            return True
        else:
            print("❌ No selected clause evidence found")
            return False
    else:
        print("❌ No chat response received")
        return False

if __name__ == "__main__":
    test_debug_routing()

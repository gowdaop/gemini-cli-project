#!/usr/bin/env python3
"""
Simple test to verify routing logic is working.
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

def test_simple_routing():
    """Test simple routing logic"""
    print("🧪 Testing Simple Routing Logic...")
    
    # Step 1: Create conversation
    conv_response = make_request("/chat/new", "POST")
    conv_id = conv_response["conversation_id"]
    print(f"Created conversation: {conv_id}")
    
    # Step 2: Store analysis
    sample_clauses = [
        {
            "id": "simple_clause_1",
            "tag": "termination",
            "text": "Either party may terminate this Agreement by giving 30 days' written notice.",
            "span": {"page": 1, "start_line": 7, "end_line": 7}
        }
    ]
    sample_risks = [
        {
            "clause_id": "simple_clause_1",
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
    
    # Step 3: Test with selected clause
    print("Testing with selected clause...")
    chat_request = {
        "question": "What is the selected termination clause?",
        "conversation_id": conv_id,
        "selected_clause_id": "simple_clause_1"
    }
    
    chat_response = make_request("/chat/", "POST", chat_request)
    print(f"Response received: {bool(chat_response.get('answer'))}")
    
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
        print(f"Content: {selected_evidence[0].get('content', '')[:200]}...")
        return True
    else:
        print("❌ No selected clause evidence found")
        
        # Check if we have any evidence at all
        if evidence:
            print("But we do have other evidence:")
            for i, ev in enumerate(evidence[:3]):  # Show first 3
                print(f"  {i+1}. {ev.get('doc_type', 'unknown')}: {ev.get('content', '')[:100]}...")
        
        return False

if __name__ == "__main__":
    test_simple_routing()

#!/usr/bin/env python3
"""
Test script to verify the routing logic is working correctly.
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

def test_routing_with_analysis():
    """Test routing with analysis data"""
    print("🧪 Testing Routing Logic with Analysis Data...")
    
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
    
    # Step 3: Check conversation has analysis data
    print("Checking conversation data...")
    history_response = make_request(f"/chat/conversations/{conv_id}")
    has_analysis = history_response.get("analytics", {}).get("has_analysis_data", False)
    print(f"Has analysis data: {has_analysis}")
    
    if not has_analysis:
        print("❌ Analysis data not found in conversation!")
        return False
    
    # Step 4: Test with selected clause
    print("Testing with selected clause...")
    chat_request = {
        "question": "Test question",
        "conversation_id": conv_id,
        "selected_clause_id": "test_clause_1"
    }
    
    chat_response = make_request("/chat/", "POST", chat_request)
    evidence_types = [ev.get('doc_type', 'unknown') for ev in chat_response.get('evidence', [])]
    print(f"Evidence types: {evidence_types}")
    
    # Check if we have selected clause evidence
    has_selected_evidence = any(ev.get('doc_type') == 'selected_clause' for ev in chat_response.get('evidence', []))
    print(f"Has selected clause evidence: {has_selected_evidence}")
    
    if has_selected_evidence:
        print("✅ Routing logic is working correctly!")
        return True
    else:
        print("❌ Routing logic is not working - no selected clause evidence found")
        return False

def test_routing_without_analysis():
    """Test routing without analysis data"""
    print("\n🧪 Testing Routing Logic without Analysis Data...")
    
    # Step 1: Create conversation
    conv_response = make_request("/chat/new", "POST")
    conv_id = conv_response["conversation_id"]
    print(f"Created conversation: {conv_id}")
    
    # Step 2: Test without analysis data
    print("Testing without analysis data...")
    chat_request = {
        "question": "What is indemnification?",
        "conversation_id": conv_id
    }
    
    chat_response = make_request("/chat/", "POST", chat_request)
    evidence_types = [ev.get('doc_type', 'unknown') for ev in chat_response.get('evidence', [])]
    print(f"Evidence types: {evidence_types}")
    
    # Should have generic legal document evidence
    has_legal_evidence = any(ev.get('doc_type') == 'legal_document' for ev in chat_response.get('evidence', []))
    print(f"Has legal document evidence: {has_legal_evidence}")
    
    if has_legal_evidence:
        print("✅ Generic routing is working correctly!")
        return True
    else:
        print("❌ Generic routing is not working")
        return False

def main():
    """Run routing tests"""
    print("🚀 Starting Routing Logic Tests")
    print("=" * 50)
    
    # Check if server is running
    health_response = make_request("/chat/health")
    if not health_response:
        print("❌ Server is not running or not accessible")
        return
    
    print("✅ Server is running")
    
    # Run tests
    tests = [
        test_routing_with_analysis,
        test_routing_without_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All routing tests passed!")
    else:
        print("⚠️  Some routing tests failed.")

if __name__ == "__main__":
    main()

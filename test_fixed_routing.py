#!/usr/bin/env python3
"""
Test script to verify the fixed routing logic for clause selection
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
API_KEY = "legal-doc-analyzer-2025-secure-key-f47d4a2c"

def test_clause_selection():
    """Test that clause selection now works correctly"""
    print("🧪 Testing Fixed Clause Selection Routing...")
    
    # Create conversation
    print("Creating conversation...")
    response = requests.post(f"{BASE_URL}/chat/new", headers={"X-API-Key": API_KEY})
    if response.status_code != 200:
        print(f"❌ Failed to create conversation: {response.status_code}")
        return False
    
    conversation_id = response.json()["conversation_id"]
    print(f"✅ Created conversation: {conversation_id}")
    
    # Store analysis with a termination clause
    print("Storing analysis...")
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
        return False
    
    print("✅ Analysis stored successfully")
    
    # Test with selected clause
    print("Testing with selected clause...")
    chat_data = {
        "question": "What is the selected termination clause?",
        "conversation_id": conversation_id,
        "selected_clause_id": "test_clause_1",
        "selected_risk_level": "red"
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json=chat_data
    )
    
    if response.status_code != 200:
        print(f"❌ Chat request failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    result = response.json()
    print("✅ Chat response received")
    
    # Check evidence
    evidence = result.get("evidence", [])
    print(f"Evidence count: {len(evidence)}")
    
    # Look for selected clause evidence
    selected_clause_evidence = [e for e in evidence if e.get("doc_type") == "selected_clause"]
    risk_evidence = [e for e in evidence if e.get("doc_type") == "risk_assessment"]
    
    print(f"Selected clause evidence count: {len(selected_clause_evidence)}")
    print(f"Risk evidence count: {len(risk_evidence)}")
    
    if selected_clause_evidence:
        print("✅ Selected clause evidence found!")
        print(f"Content preview: {selected_clause_evidence[0]['content'][:100]}...")
    else:
        print("❌ No selected clause evidence found")
        print("Available evidence types:", [e.get("doc_type") for e in evidence])
    
    if risk_evidence:
        print("✅ Risk evidence found!")
    else:
        print("❌ No risk evidence found")
    
    # Check answer quality
    answer = result.get("answer", "")
    if "30 days" in answer and "termination" in answer.lower():
        print("✅ Answer references the specific clause content")
    else:
        print("❌ Answer doesn't reference the specific clause")
        print(f"Answer preview: {answer[:200]}...")
    
    return len(selected_clause_evidence) > 0 and "30 days" in answer

if __name__ == "__main__":
    print("🚀 Starting Fixed Routing Test...")
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    success = test_clause_selection()
    
    if success:
        print("\n🎉 SUCCESS: Clause selection routing is working correctly!")
        print("✅ Selected clause evidence is being created")
        print("✅ Answers reference specific clause content")
    else:
        print("\n❌ FAILED: Clause selection routing still has issues")
    
    print("\n" + "="*50)
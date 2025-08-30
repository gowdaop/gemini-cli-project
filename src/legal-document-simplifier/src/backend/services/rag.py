from typing import Dict, Any

def answer_question(question: str, context: str = "") -> Dict[str, Any]:
    """Answer question using RAG - placeholder implementation"""
    
    return {
        "answer": f"Mock answer for: {question}",
        "sources": ["Mock source 1", "Mock source 2"],
        "confidence": 0.85
    }

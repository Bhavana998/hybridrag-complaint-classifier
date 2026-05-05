import pytest
from core.hybrid_retriever import HybridRetriever

def test_hybrid_retriever():
    retriever = HybridRetriever()
    
    documents = [
        "Credit card charged twice for subscription",
        "App keeps crashing on startup",
        "Package delivery delayed by 5 days"
    ]
    
    retriever.index_documents(documents)
    
    results = retriever.hybrid_search("billing error double charge", top_k=2)
    
    assert len(results) == 2
    assert results[0].score > 0
    assert "charged" in results[0].text.lower()
from datetime import datetime
from typing import Dict, List
import pandas as pd

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def calculate_statistics(history: List[Dict]) -> Dict:
    """Calculate statistics from classification history"""
    if not history:
        return {}
    
    categories = [h['classification']['primary_category'] for h in history]
    confidences = [h['classification']['confidence_score'] for h in history]
    processing_times = [h['processing_time'] for h in history]
    
    from collections import Counter
    category_counts = Counter(categories)
    
    return {
        'total_classifications': len(history),
        'unique_categories': len(category_counts),
        'most_common_category': category_counts.most_common(1)[0][0] if category_counts else None,
        'avg_confidence': sum(confidences) / len(confidences),
        'avg_processing_time': sum(processing_times) / len(processing_times),
        'category_distribution': dict(category_counts)
    }

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def extract_urgency_score(urgency_level: str) -> int:
    """Convert urgency level to score"""
    return {"high": 3, "medium": 2, "low": 1}.get(urgency_level.lower(), 0)
# core/groq_classifier.py - No dotenv version for Railway
import os
import json
import re
from typing import Dict, List
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration from environment variables (Railway uses these)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# Import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed. Run: pip install groq")

print("="*60)
print("🔍 Groq Classifier Initialized")
print(f"API Key: {'✅ Found' if GROQ_API_KEY else '❌ Missing'}")
print(f"Model: {GROQ_MODEL}")
print("="*60)

class ComplaintClassifier:
    """Customer complaint classifier using Groq API (Free Tier)"""
    
    def __init__(self, backend: str = "groq"):
        self.backend = backend
        self.model = GROQ_MODEL
        
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables!")
        
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package not installed. Run: pip install groq")
        
        self.client = Groq(api_key=GROQ_API_KEY)
        print(f"✅ Groq Classifier ready with model: {self.model}")
    
    def _build_prompt(self, complaint: str, retrieved_contexts: List[str]) -> str:
        """Build classification prompt"""
        context_text = ""
        if retrieved_contexts and len(retrieved_contexts) > 0:
            context_text = "\n\nReference similar complaints:\n"
            for i, ctx in enumerate(retrieved_contexts[:3], 1):
                ctx_preview = ctx[:350] + "..." if len(ctx) > 350 else ctx
                context_text += f"\n--- Example {i} ---\n{ctx_preview}\n"
        
        prompt = f"""You are an expert customer complaint classification system.

COMPLAINT: {complaint}
{context_text}

CATEGORIES: Billing, Technical, Shipping, Customer Service, Product Quality, Account, Other

Return ONLY JSON: {{"primary_category": "category", "confidence_score": 0.95, "sub_category": "issue", "urgency_level": "high/medium/low", "suggested_action": "action", "key_phrases": ["phrase1"], "sentiment": "negative/neutral/positive"}}"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse API response"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response_text)
            
            # Default fields
            defaults = {
                "primary_category": "Other",
                "confidence_score": 0.5,
                "sub_category": "Unclassified",
                "urgency_level": "medium",
                "suggested_action": "Manual review required",
                "key_phrases": [],
                "sentiment": "neutral"
            }
            
            for field, default in defaults.items():
                if field not in result:
                    result[field] = default
            
            if result["confidence_score"] > 1:
                result["confidence_score"] = result["confidence_score"] / 100
            
            return result
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return self._get_default_response()
    
    def _get_default_response(self) -> Dict:
        return {
            "primary_category": "Other",
            "confidence_score": 0.5,
            "sub_category": "Unclassified",
            "urgency_level": "medium",
            "suggested_action": "Manual review required",
            "key_phrases": [],
            "sentiment": "neutral"
        }
    
    def classify(self, complaint: str, retrieved_contexts: List[str] = None) -> Dict:
        """Classify complaint using Groq API"""
        retrieved_contexts = retrieved_contexts or []
        
        print(f"\n📤 Classifying: {complaint[:80]}...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": self._build_prompt(complaint, retrieved_contexts)}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            result = self._parse_response(response.choices[0].message.content)
            result['classified_at'] = datetime.now().isoformat()
            result['backend_used'] = 'groq'
            result['model_used'] = self.model
            
            print(f"✅ Result: {result['primary_category']}")
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return self._get_default_response()
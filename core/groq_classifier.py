# core/classifier.py - Complete Working Version for Groq API (Free)
import os
import json
import re
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import Groq (free alternative to OpenAI)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed. Run: pip install groq")

# Get configuration from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional fallback
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# Print debug info
print("="*60)
print("🔍 DEBUG: Classifier Initialization")
print(f"Groq available: {'✅ YES' if GROQ_AVAILABLE else '❌ NO'}")
print(f"Groq API Key: {'✅ Loaded' if GROQ_API_KEY else '❌ Not found'}")
if GROQ_API_KEY:
    print(f"  Key starts with: {GROQ_API_KEY[:10]}...")
print(f"Groq Model: {GROQ_MODEL}")
print(f"OpenAI API Key: {'✅ Loaded' if OPENAI_API_KEY else '❌ Not found'}")
print("="*60)

class ClassifierCache:
    """Simple cache for classification results"""
    def __init__(self, ttl: int = CACHE_TTL):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached result if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"Cache hit for key: {key[:20]}...")
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Dict):
        """Cache a result"""
        self.cache[key] = (value, time.time())
        logger.info(f"Cached result for key: {key[:20]}...")

class ComplaintClassifier:
    """
    Customer complaint classifier using Groq API (Free Tier)
    Falls back to OpenAI if Groq is unavailable
    """
    
    def __init__(self, backend: str = "groq"):
        """
        Initialize the classifier
        
        Args:
            backend: "groq" (recommended, free) or "openai" (paid)
        """
        self.backend = backend
        self.cache = ClassifierCache()
        
        # Try to initialize Groq first
        if backend == "groq" or backend == "auto":
            if GROQ_AVAILABLE and GROQ_API_KEY:
                self.backend = "groq"
                self.model = GROQ_MODEL
                self.client = Groq(api_key=GROQ_API_KEY)
                print(f"✅ Using Groq API with model: {self.model}")
                print(f"   Free tier: 30 requests/minute, no credit card needed")
            elif OPENAI_API_KEY:
                self.backend = "openai"
                from openai import OpenAI
                self.model = OPENAI_MODEL
                self.client = OpenAI(api_key=OPENAI_API_KEY)
                print(f"⚠️ Groq not available, using OpenAI with model: {self.model}")
            else:
                raise ValueError(
                    "No API key found!\n"
                    "Please either:\n"
                    "1. Get a FREE Groq API key from https://console.groq.com\n"
                    "   Add to .env: GROQ_API_KEY=your-key-here\n"
                    "2. Or use OpenAI API key in .env file"
                )
        elif backend == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found in .env file")
            from openai import OpenAI
            self.model = OPENAI_MODEL
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            print(f"✅ Using OpenAI API with model: {self.model}")
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _build_prompt(self, complaint: str, retrieved_contexts: List[str]) -> str:
        """
        Build the prompt for classification
        
        Args:
            complaint: The customer complaint text
            retrieved_contexts: List of similar complaints from hybrid search
        
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved similar complaints
        context_text = ""
        if retrieved_contexts and len(retrieved_contexts) > 0:
            context_text = "\n\n📋 **REFERENCE EXAMPLES (similar past complaints):**\n"
            context_text += "-" * 50 + "\n"
            for i, ctx in enumerate(retrieved_contexts[:3], 1):
                # Truncate long contexts
                ctx_preview = ctx[:350] + "..." if len(ctx) > 350 else ctx
                context_text += f"\nExample {i}:\n{ctx_preview}\n"
            context_text += "\n" + "-" * 50 + "\n"
        
        # Build the complete prompt
        prompt = f"""You are an expert customer complaint classification system. Analyze the complaint and provide structured output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 **CUSTOMER COMPLAINT:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{complaint}
{context_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **CLASSIFICATION CATEGORIES:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **Billing** - Charges, payments, refunds, subscriptions, invoices, fees, double charges
2. **Technical** - App crashes, bugs, errors, connectivity, performance issues, website problems
3. **Shipping** - Delivery delays, tracking issues, damaged packages, wrong address, lost packages
4. **Customer Service** - Rude agents, long wait times, unhelpful responses, no callbacks
5. **Product Quality** - Defective items, poor materials, functionality issues, broken products
6. **Account** - Login problems, password reset, profile access, account locked, 2FA issues
7. **Other** - Anything that doesn't fit above

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📤 **OUTPUT REQUIREMENTS:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY valid JSON. Do not add any other text, explanations, or markdown formatting.

Use this exact JSON structure:
{{
    "primary_category": "Billing",
    "confidence_score": 0.95,
    "sub_category": "double charge issue",
    "urgency_level": "high",
    "suggested_action": "Process refund for duplicate charge and verify subscription status",
    "key_phrases": ["charged twice", "refund", "subscription"],
    "sentiment": "negative",
    "estimated_resolution_time": "24 hours",
    "escalation_needed": false
}}

JSON OUTPUT:"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse API response and extract JSON
        
        Args:
            response_text: Raw response from API
        
        Returns:
            Parsed dictionary with classification results
        """
        try:
            # Try to extract JSON using regex
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response_text)
            
            # Ensure all required fields exist
            default_fields = {
                "primary_category": "Other",
                "confidence_score": 0.5,
                "sub_category": "Unclassified",
                "urgency_level": "medium",
                "suggested_action": "Manual review required",
                "key_phrases": [],
                "sentiment": "neutral",
                "estimated_resolution_time": "48 hours",
                "escalation_needed": False
            }
            
            # Fill missing fields with defaults
            for field, default_value in default_fields.items():
                if field not in result:
                    result[field] = default_value
            
            # Ensure confidence score is between 0 and 1
            if result["confidence_score"] > 1:
                result["confidence_score"] = result["confidence_score"] / 100
            
            # Ensure key_phrases is a list
            if not isinstance(result["key_phrases"], list):
                result["key_phrases"] = [str(result["key_phrases"])] if result["key_phrases"] else []
            
            # Limit key phrases to 5
            result["key_phrases"] = result["key_phrases"][:5]
            
            return result
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse response: {str(e)}")
            logger.error(f"Response text: {response_text[:300]}")
            return self._get_default_response()
    
    def _get_default_response(self) -> Dict:
        """
        Return default response when classification fails
        
        Returns:
            Default classification dictionary
        """
        return {
            "primary_category": "Other",
            "confidence_score": 0.5,
            "sub_category": "Unclassified",
            "urgency_level": "medium",
            "suggested_action": "Manual review required",
            "key_phrases": [],
            "sentiment": "neutral",
            "estimated_resolution_time": "48 hours",
            "escalation_needed": False,
            "classified_at": datetime.now().isoformat(),
            "backend_used": self.backend,
            "model_used": self.model
        }
    
    def classify(self, complaint: str, retrieved_contexts: List[str] = None) -> Dict:
        """
        Classify a customer complaint using the API
        
        Args:
            complaint: The customer complaint text
            retrieved_contexts: List of similar complaints from hybrid search
        
        Returns:
            Dictionary with classification results
        """
        retrieved_contexts = retrieved_contexts or []
        
        # Generate cache key
        cache_content = f"{complaint}_{str(retrieved_contexts)}"
        cache_key = hashlib.md5(cache_content.encode()).hexdigest()
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached classification result")
            return cached_result
        
        # Debug output
        print("\n" + "="*70)
        print(f"📤 CALLING {self.backend.upper()} API")
        print("="*70)
        print(f"📍 Complaint: {complaint[:150]}...")
        print(f"📚 Contexts provided: {len(retrieved_contexts)}")
        print(f"🤖 Model: {self.model}")
        print(f"💰 Backend: {self.backend.upper()} ({'Free' if self.backend == 'groq' else 'Paid'})")
        print("="*70)
        
        # Build the prompt
        prompt = self._build_prompt(complaint, retrieved_contexts)
        
        try:
            # Make API call based on backend
            if self.backend == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a complaint classification expert. Output only valid JSON. Do not add explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a complaint classification expert. Output only valid JSON. Do not add explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Debug print response
            print("\n📥 API RESPONSE:")
            print("="*70)
            print(f"Response length: {len(response_text)} characters")
            print(f"Response preview: {response_text[:300]}...")
            print("="*70)
            
            # Parse the response
            result = self._parse_response(response_text)
            
            # Add metadata
            result['classified_at'] = datetime.now().isoformat()
            result['backend_used'] = self.backend
            result['model_used'] = self.model
            result['contexts_used'] = len(retrieved_contexts)
            
            # Print success message
            print(f"\n✅ CLASSIFICATION COMPLETE:")
            print(f"   Category: {result['primary_category']}")
            print(f"   Confidence: {result['confidence_score']:.0%}")
            print(f"   Sub-category: {result['sub_category']}")
            print(f"   Urgency: {result['urgency_level'].upper()}")
            print(f"   Sentiment: {result['sentiment'].upper()}")
            print("="*70 + "\n")
            
            # Cache the result
            self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            print(f"\n❌ CLASSIFICATION ERROR:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print("="*70 + "\n")
            
            logger.error(f"API error: {str(e)}")
            
            # Return default response with error info
            default_response = self._get_default_response()
            default_response['error'] = str(e)
            default_response['classified_at'] = datetime.now().isoformat()
            
            return default_response
    
    def batch_classify(self, complaints: List[str], contexts_list: List[List[str]] = None) -> List[Dict]:
        """
        Classify multiple complaints in batch
        
        Args:
            complaints: List of complaint texts
            contexts_list: List of context lists for each complaint
        
        Returns:
            List of classification dictionaries
        """
        results = []
        contexts_list = contexts_list or [[] for _ in complaints]
        
        total = len(complaints)
        for i, (complaint, contexts) in enumerate(zip(complaints, contexts_list)):
            logger.info(f"Processing complaint {i+1}/{total}")
            print(f"\n📊 Progress: {i+1}/{total} complaints")
            
            result = self.classify(complaint, contexts)
            results.append(result)
            
            # Small delay to avoid rate limits
            if i < total - 1:
                time.sleep(0.5)
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get classifier statistics
        
        Returns:
            Dictionary with cache and usage statistics
        """
        return {
            "backend": self.backend,
            "model": self.model,
            "cache_size": len(self.cache.cache),
            "cache_ttl": self.cache.ttl
        }


# Optional: Test function
if __name__ == "__main__":
    print("\n🧪 Running classifier test...")
    
    try:
        # Initialize classifier
        classifier = ComplaintClassifier()
        
        # Test complaints
        test_complaints = [
            "I was charged twice for my subscription. Please refund me.",
            "The app crashes every time I open it.",
            "My package hasn't arrived in 2 weeks."
        ]
        
        for complaint in test_complaints:
            print("\n" + "="*50)
            print(f"Testing: {complaint}")
            result = classifier.classify(complaint)
            print(f"Result: {result['primary_category']} ({result['confidence_score']:.0%})")
        
        # Print stats
        print("\n" + "="*50)
        print("Classifier Statistics:")
        print(json.dumps(classifier.get_stats(), indent=2))
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
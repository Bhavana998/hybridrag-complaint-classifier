import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List, Optional

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
    def clean(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline"""
        # Clean text
        text = self.clean(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords and stem
        processed_tokens = []
        for token in tokens:
            if self.remove_stopwords and token in self.stop_words:
                continue
            if self.use_stemming:
                token = self.stemmer.stem(token)
            processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """Extract key phrases using simple frequency-based approach"""
        from collections import Counter
        
        processed = self.preprocess(text)
        words = processed.split()
        
        # Remove very short words
        words = [w for w in words if len(w) > 2]
        
        # Get most common words
        word_freq = Counter(words)
        top_words = word_freq.most_common(top_n)
        
        return [word for word, _ in top_words]
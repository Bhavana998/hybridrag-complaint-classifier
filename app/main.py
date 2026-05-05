# app/main.py - Complete Version with Working Voice Search
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# Import from core modules
from core.groq_classifier import ComplaintClassifier
from core.hybrid_retriever import HybridRetriever
from core.data_loader import DataLoader
from core.voice_search import VoiceSearch

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="HybridRAG - Voice Enabled Complaint Analysis",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS with Voice Search Styling
st.markdown("""
<style>
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .modern-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .modern-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .modern-header p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .result-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 5px solid;
    }
    
    .category-Billing { border-left-color: #ff6b6b; }
    .category-Technical { border-left-color: #4ecdc4; }
    .category-Shipping { border-left-color: #45b7d1; }
    .category-Customer-Service { border-left-color: #96ceb4; }
    .category-Product-Quality { border-left-color: #ffeaa7; }
    .category-Account { border-left-color: #dfe6e9; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    
    /* Key phrase tags */
    .key-phrase-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:disabled {
        background: linear-gradient(135deg, #cccccc 0%, #999999 100%);
        cursor: not-allowed;
    }
    
    /* Voice button special styling */
    .voice-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Success message */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Free badge */
    .free-badge {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    /* Voice input container */
    .voice-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'processed_count' not in st.session_state:
    st.session_state.processed_count = 0
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'complaint_text' not in st.session_state:
    st.session_state.complaint_text = ""
if 'voice_text' not in st.session_state:
    st.session_state.voice_text = ""

# Initialize voice search
voice_search = VoiceSearch()

# Modern Header
st.markdown("""
<div class="modern-header">
    <h1>🎤 HybridRAG Intelligence Platform</h1>
    <p>Voice-Enabled Customer Complaint Analysis with Groq AI + Hybrid Search</p>
    <p><span class="free-badge">⚡ FREE TIER</span> <span class="free-badge">🎤 Voice Search</span> <span class="free-badge">💰 No Credit Card</span></p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    
    # API Configuration
    st.markdown("### 🤖 Groq API Configuration")
    st.info("Groq offers a **FREE API** with 30 requests per minute. No credit card required!")
    
    # Groq API Key input
    groq_api_key = st.text_input(
        "🔑 Groq API Key", 
        type="password",
        help="Get your free API key from https://console.groq.com",
        placeholder="gsk_..."
    )
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.session_state.api_key_set = True
        st.success("✅ Groq API Key set!")
        
        # Save to .env file
        try:
            with open(".env", "w") as f:
                f.write(f"GROQ_API_KEY={groq_api_key}\n")
                f.write("GROQ_MODEL=llama-3.1-8b-instant\n")
                f.write("LLM_BACKEND=groq\n")
        except:
            pass
    else:
        if not st.session_state.api_key_set:
            st.warning("⚠️ Please enter your Groq API key to continue")
            st.markdown("""
            **Don't have a key?**
            1. Go to [console.groq.com](https://console.groq.com)
            2. Sign up (free, no credit card)
            3. Create an API key
            4. Paste it above
            """)
    
    # Model selection
    model_name = st.selectbox(
        "🧠 Model", 
        ["llama-3.1-8b-instant", "llama-3.2-3b-preview", "mixtral-8x7b-32768"],
        help="Llama 3.1 8B is fast and free. Mixtral is more accurate but slower"
    )
    
    st.markdown("---")
    
    # Search Configuration
    st.markdown("### 🔍 Search Configuration")
    alpha = st.slider(
        "🎯 Semantic vs Keyword",
        0.0, 1.0, 0.7,
        help="Higher = more meaning-based search"
    )
    top_k = st.slider("📚 Context Windows", 1, 10, 5)
    
    st.markdown("---")
    
    # Voice Settings
    st.markdown("### 🎤 Voice Settings")
    st.info("💡 Click 'Start Speaking' and say your complaint clearly")
    st.caption("🎤 Works best in Chrome or Edge browser")
    
    st.markdown("---")
    
    # Data Management
    st.markdown("### 📁 Data Source")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload Complaints (CSV)",
        type=['csv'],
        help="CSV file with 'complaint' column"
    )
    
    # Load Sample Data Button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Load Sample Data", use_container_width=True, type="primary", disabled=not st.session_state.api_key_set):
            with st.spinner("Loading sample complaints..."):
                try:
                    # Sample complaints
                    sample_complaints = [
                        "I was charged twice for my monthly subscription of $49.99, please refund the duplicate charge immediately",
                        "The mobile app crashes every time I try to upload a profile picture",
                        "My package was supposed to arrive 3 days ago but tracking shows it's still stuck in transit",
                        "Customer support put me on hold for 45 minutes and then disconnected the call",
                        "The wireless headphones stopped working after only 2 days of use",
                        "I cannot log into my account and the password reset email never arrives",
                        "Why was I charged a $50 late fee? I paid my bill on time last week",
                        "The website keeps timing out when I try to checkout my shopping cart",
                        "The product quality is terrible - the stitching came apart after first wash"
                    ]
                    
                    st.session_state.retriever = HybridRetriever()
                    st.session_state.retriever.index_documents(sample_complaints)
                    st.session_state.classifier = ComplaintClassifier(backend="groq")
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(sample_complaints)} sample complaints!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure your Groq API key is valid")
    
    with col2:
        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.classifier = None
            st.session_state.data_loaded = False
            st.session_state.history = []
            st.session_state.current_results = None
            st.session_state.complaint_text = ""
            st.success("Data cleared!")
            st.rerun()
    
    # Handle uploaded file
    if uploaded_file and st.session_state.api_key_set:
        try:
            df = pd.read_csv(uploaded_file)
            if 'complaint' in df.columns:
                complaints = df['complaint'].tolist()
                st.session_state.retriever = HybridRetriever()
                st.session_state.retriever.index_documents(complaints)
                st.session_state.classifier = ComplaintClassifier(backend="groq")
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(complaints)} complaints from file!")
                st.rerun()
            else:
                st.error("CSV must have a 'complaint' column")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Show data status
    if st.session_state.data_loaded and st.session_state.retriever:
        st.markdown("---")
        st.success("✅ **Data Ready!**")
        st.caption(f"📊 {len(st.session_state.retriever.documents)} complaints indexed")
        
        if st.session_state.history:
            avg_conf = sum(h['classification'].get('confidence_score', 0) for h in st.session_state.history) / len(st.session_state.history)
            st.metric("⭐ Avg Confidence", f"{avg_conf:.0%}")

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📝 Complaint Input")
    
    if not st.session_state.api_key_set:
        st.info("🔑 **Please enter your Groq API key in the sidebar to begin**")
        st.markdown("""
        **Get your free Groq API key:**
        1. Go to [console.groq.com](https://console.groq.com)
        2. Sign up with Google/GitHub (free)
        3. Create an API key
        4. Paste it in the sidebar
        
        💡 **Free tier includes:** 30 requests/minute, 6,000-12,000 tokens/minute
        """)
    
    # Voice Input Section
    st.markdown("#### 🎤 Voice Input")
    st.caption("Click the button below, speak your complaint clearly, then click 'Use Voice Text'")
    
    # Voice HTML component
    voice_html = voice_search.get_voice_html()
    st.components.v1.html(voice_html, height=120)
    
    # Voice result field
    voice_result = st.text_input(
        "Voice Result:", 
        key="voice_result_field", 
        placeholder="Your spoken text will appear here after speaking...",
        label_visibility="collapsed"
    )
    
    if voice_result:
        st.session_state.voice_text = voice_result
        st.success(f"🎤 Voice captured: {voice_result[:100]}...")
        
        # Button to use voice text
        if st.button("📝 Use This Voice Text", use_container_width=True):
            st.session_state.complaint_text = voice_result
            st.rerun()
    
    st.markdown("---")
    
    # Text Input Section
    st.markdown("#### ✍️ Or Type Your Complaint")
    
    complaint_text = st.text_area(
        "",
        value=st.session_state.complaint_text,
        height=150,
        placeholder="✍️ Type your complaint here or use voice input above...\n\nExample: 'I was charged twice for my subscription. Please refund me.'",
        label_visibility="collapsed",
        disabled=not st.session_state.api_key_set,
        key="complaint_text_area"
    )
    
    # Update session state when typing
    if complaint_text:
        st.session_state.complaint_text = complaint_text
    
    # Quick Examples
    st.markdown("**Quick Examples:**")
    example_cols = st.columns(3)
    examples = [
        ("💳 Billing", "I was charged twice for my monthly subscription of $49.99. Please refund the duplicate charge immediately!"),
        ("🔧 Technical", "The mobile app crashes every time I try to upload a photo. I've reinstalled 3 times."),
        ("📦 Shipping", "My package shows delivered but it's not at my door. The tracking hasn't updated in 5 days.")
    ]
    
    for idx, (label, text) in enumerate(examples):
        with example_cols[idx]:
            if st.button(label, key=f"ex_{idx}", use_container_width=True, disabled=not st.session_state.api_key_set):
                st.session_state.complaint_text = text
                st.rerun()
    
    # Analyze and Clear buttons
    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        analyze_clicked = st.button(
            "🚀 Analyze Complaint",
            type="primary",
            use_container_width=True,
            disabled=not (st.session_state.complaint_text and st.session_state.data_loaded and st.session_state.api_key_set)
        )
    
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.complaint_text = ""
            st.session_state.voice_text = ""
            st.session_state.current_results = None
            st.rerun()
    
    # Process classification
    if analyze_clicked:
        complaint_to_analyze = st.session_state.complaint_text
        
        if complaint_to_analyze:
            with st.spinner("🔄 Processing complaint through Groq AI + Hybrid Search..."):
                start_time = time.time()
                
                try:
                    # Hybrid retrieval
                    similar_complaints = st.session_state.retriever.hybrid_search(
                        complaint_to_analyze, top_k=top_k, alpha=alpha
                    )
                    
                    contexts = [result.text for result in similar_complaints]
                    
                    # Groq classification
                    classification = st.session_state.classifier.classify(complaint_to_analyze, contexts)
                    
                    processing_time = time.time() - start_time
                    
                    # Determine input method
                    input_method = "voice" if st.session_state.voice_text else "text"
                    
                    # Store in history
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'complaint': complaint_to_analyze[:100],
                        'classification': classification,
                        'processing_time': processing_time,
                        'input_method': input_method
                    })
                    st.session_state.processed_count += 1
                    
                    # Clear voice text after use
                    st.session_state.voice_text = ""
                    
                    # Store current results
                    st.session_state.current_results = {
                        'similar_complaints': similar_complaints,
                        'classification': classification,
                        'processing_time': processing_time,
                        'complaint': complaint_to_analyze
                    }
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Classification error: {str(e)}")
                    st.info("Please check your Groq API key and try again")

with col2:
    st.markdown("### 📊 Analysis Results")
    
    # Check if current_results exists and is not None
    if st.session_state.current_results is not None:
        results = st.session_state.current_results
        
        # Safely get classification with fallback
        classification = results.get('classification', None)
        
        if classification is None:
            st.error("Classification failed. Please try again.")
        else:
            # Category
            category = classification.get('primary_category', 'Other')
            category_class = category.replace(" ", "-")
            
            # Main metrics row
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                confidence = classification.get('confidence_score', 0.5)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{confidence:.0%}</h3>
                    <p>Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[1]:
                urgency = classification.get('urgency_level', 'medium').upper()
                urgency_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(urgency, "⚪")
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{urgency_icon} {urgency}</h3>
                    <p>Urgency Level</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[2]:
                sentiment = classification.get('sentiment', 'neutral').upper()
                sentiment_icon = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😞"}.get(sentiment, "😐")
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{sentiment_icon} {sentiment}</h3>
                    <p>Customer Sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Category card
            st.markdown(f"""
            <div class="result-card category-{category_class}">
                <h2 style="margin: 0; color: #2c3e50;">📌 {category}</h2>
                <p style="color: #666; margin-top: 0.5rem;">Primary Classification (Groq AI)</p>
                <hr>
                <strong>🎯 Sub-category:</strong> {classification.get('sub_category', 'N/A')}<br>
                <strong>💡 Suggested Action:</strong> {classification.get('suggested_action', 'N/A')}<br>
                <strong>⏱️ Estimated Resolution:</strong> {classification.get('estimated_resolution_time', 'N/A')}<br>
                <strong>🚨 Escalation Required:</strong> {'✅ Yes' if classification.get('escalation_needed', False) else '❌ No'}
            </div>
            """, unsafe_allow_html=True)
            
            # Key phrases
            st.markdown("### 🔑 Key Phrases Extracted")
            key_phrases = classification.get('key_phrases', [])
            if key_phrases:
                html_phrases = "".join([f'<span class="key-phrase-tag">#{phrase}</span>' for phrase in key_phrases[:6]])
                st.markdown(f'<div style="margin: 1rem 0;">{html_phrases}</div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ No key phrases extracted")
            
            # Processing info
            model_used = classification.get('model_used', 'llama-3.1-8b-instant')
            backend_used = classification.get('backend_used', 'groq')
            st.caption(f"⚡ Processing time: {results.get('processing_time', 0):.2f} seconds | 🤖 Model: {model_used} | 🔗 Backend: {backend_used}")
            
            # Retrieved contexts expander
            similar_complaints = results.get('similar_complaints', [])
            with st.expander(f"🔍 Retrieved Contexts ({len(similar_complaints)} similar complaints)"):
                for i, result in enumerate(similar_complaints[:3], 1):
                    relevance_color = "🟢" if result.score > 0.7 else "🟡" if result.score > 0.4 else "🔴"
                    st.markdown(f"""
                    **{relevance_color} Result {i}** (Relevance: {result.score:.2f})
                    > {result.text[:200]}...
                    
                    *Semantic: {result.semantic_score:.2f} | Keyword: {result.keyword_score:.2f}*
                    ---
                    """)
    
    else:
        st.info("✨ Ready for analysis! Enter your Groq API key, load data, and enter/speak a complaint to see results here.")
        
        with st.expander("📖 How Voice-Enabled HybridRAG Works"):
            st.markdown("""
            **4-Step Pipeline with Voice Input:**
            1. 🎤 **Voice Input** - Speak your complaint (or type)
            2. 🔍 **Hybrid Search** - FAISS (semantic) + TF-IDF (keyword)
            3. 📚 **Context Retrieval** - Finds similar past complaints
            4. 🚀 **Groq AI Classification** - Ultra-fast LLM analysis
            5. 📊 **Structured JSON Output** - Returns actionable insights
            
            **Voice Features (Browser-based):**
            - 🎤 Click "Start Speaking" button
            - 🌍 Works in Chrome, Edge, and modern browsers
            - 📝 Edit transcribed text before analysis
            - ✅ No installation required!
            
            **To get started:**
            1. Get a free API key from [console.groq.com](https://console.groq.com)
            2. Enter your key in the sidebar
            3. Click "Load Sample Data"
            4. Speak or type a complaint
            5. Click "Analyze Complaint"
            """)

# History section at bottom
if st.session_state.history and len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown("### 📜 Recent Analysis History")
    
    history_data = []
    for h in st.session_state.history[-5:]:
        if 'classification' in h and h['classification']:
            input_icon = "🎤" if h.get('input_method') == 'voice' else "✍️"
            history_data.append({
                'Time': h['timestamp'].strftime("%H:%M:%S"),
                'Input': input_icon,
                'Complaint': h['complaint'][:50] + "...",
                'Category': h['classification'].get('primary_category', 'N/A'),
                'Confidence': f"{h['classification'].get('confidence_score', 0):.0%}",
                'Urgency': h['classification'].get('urgency_level', 'N/A'),
                'Response Time': f"{h['processing_time']:.1f}s"
            })
    
    if history_data:
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.current_results = None
        st.session_state.complaint_text = ""
        st.rerun()
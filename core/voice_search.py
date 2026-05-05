# core/voice_search.py - Simplified Working Version
import streamlit as st

class VoiceSearch:
    """Voice search using browser's built-in speech recognition"""
    
    def __init__(self):
        pass
    
    def get_voice_html(self):
        """Returns HTML/JS for browser-based voice recognition"""
        return """
        <div id="voice-container" style="text-align: center; padding: 10px;">
            <button id="voice-btn" style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 50px;
                padding: 15px 30px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
            ">
                🎤 Start Speaking
            </button>
            <div id="voice-status" style="margin-top: 10px; font-size: 14px; color: #666;"></div>
        </div>
        
        <script>
        const voiceBtn = document.getElementById('voice-btn');
        const voiceStatus = document.getElementById('voice-status');
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            voiceStatus.innerHTML = '❌ Browser does not support speech recognition. Please use Chrome or Edge.';
            voiceStatus.style.color = 'red';
            voiceBtn.disabled = true;
        } else {
            const recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            voiceBtn.onclick = () => {
                recognition.start();
                voiceBtn.textContent = '🔴 Listening...';
                voiceBtn.style.background = 'linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)';
                voiceStatus.innerHTML = '🎤 Speak now...';
                voiceStatus.style.color = '#667eea';
            };
            
            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                voiceStatus.innerHTML = '✅ Recognized: ' + transcript;
                voiceStatus.style.color = 'green';
                voiceBtn.textContent = '🎤 Start Speaking';
                voiceBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                
                // Find the Streamlit text input and set its value
                const streamlitInput = document.querySelector('input[data-testid="stTextInput"]');
                if (streamlitInput) {
                    streamlitInput.value = transcript;
                    streamlitInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
                
                // Also try to find by placeholder
                const inputs = document.querySelectorAll('input[type="text"]');
                for (let input of inputs) {
                    if (input.placeholder && input.placeholder.includes('Voice')) {
                        input.value = transcript;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        break;
                    }
                }
            };
            
            recognition.onerror = (event) => {
                voiceStatus.innerHTML = '❌ Error: ' + event.error;
                voiceStatus.style.color = 'red';
                voiceBtn.textContent = '🎤 Start Speaking';
                voiceBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            };
            
            recognition.onend = () => {
                if (voiceBtn.textContent === '🔴 Listening...') {
                    voiceBtn.textContent = '🎤 Start Speaking';
                    voiceBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                    voiceStatus.innerHTML = '⏰ No speech detected. Try again.';
                    voiceStatus.style.color = 'orange';
                }
            };
        }
        </script>
        """
    
    def listen_and_convert(self) -> str:
        return None
    
    def check_microphone(self) -> bool:
        return True
import os
from dotenv import load_dotenv
load_dotenv()

# Test OpenAI
if os.getenv("LLM_BACKEND") == "openai":
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello, HybridRAG is working!'"}],
            max_tokens=50
        )
        print("✅ OpenAI is working:", response.choices[0].message.content)
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        print("Please check your API key in .env file")

# Test Ollama
elif os.getenv("LLM_BACKEND") == "ollama":
    import ollama
    try:
        response = ollama.chat(model=os.getenv("OLLAMA_MODEL", "llama2"), 
                              messages=[{"role": "user", "content": "Say 'Hello'"}])
        print("✅ Ollama is working:", response['message']['content'])
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("Make sure Ollama is running: 'ollama serve'")
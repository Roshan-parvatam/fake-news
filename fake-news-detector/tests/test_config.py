# test_config.py
"""Quick test for configuration system"""

def test_config_system():
    print("🧪 Testing Configuration System...")
    
    try:
        # Test config imports
        from config import get_model_config, get_prompt_template, get_settings
        print("✅ Config imports successful")
        
        # Test getting model configs
        bert_config = get_model_config('bert_classifier')
        print(f"✅ BERT config loaded: {len(bert_config)} settings")
        
        # Test getting prompt templates  
        llm_prompt = get_prompt_template('llm_explanation', 'main_explanation')
        print("✅ Prompt template loaded")
        
        # Test system settings
        settings = get_settings()
        print(f"✅ System settings loaded")
        
        print("✅ Configuration system working correctly!")
        
    except Exception as e:
        print(f"❌ Config test failed: {str(e)}")
        print("⚠️  Make sure GEMINI_API_KEY is set in environment")

if __name__ == "__main__":
    test_config_system()

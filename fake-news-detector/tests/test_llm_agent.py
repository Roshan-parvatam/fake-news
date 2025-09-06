# test_llm_agent.py
"""Quick test for LLM Explanation Agent"""

def test_llm_explanation():
    print("🧪 Testing LLM Explanation Agent...")
    
    try:
        from agents.llm_explanation import LLMExplanationAgent
        
        # Initialize agent
        agent = LLMExplanationAgent()
        print("✅ Agent initialized successfully")
        
        # Test input
        test_input = {
            "text": "A study shows 95% effectiveness with no side effects reported.",
            "prediction": "FAKE",
            "confidence": 0.85,
            "metadata": {"source": "UnknownBlog.com", "date": "2024-01-15"}
        }
        
        # Process
        result = agent.process(test_input)
        
        if result['success']:
            print("✅ Explanation generated successfully")
            print(f"   Response time: {result['result']['metadata']['response_time_seconds']}s")
            print("✅ LLM Agent working correctly!")
        else:
            print(f"❌ Processing failed: {result['error']['message']}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_llm_explanation()

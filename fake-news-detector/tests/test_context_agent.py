# test_context_agent.py
"""Quick test for Context Analyzer Agent"""

def test_context_analyzer():
    print("🧪 Testing Context Analyzer Agent...")
    
    try:
        from agents.context_analyzer import ContextAnalyzerAgent
        
        # Initialize agent
        agent = ContextAnalyzerAgent()
        print("✅ Agent initialized successfully")
        
        # Test input
        test_input = {
            "text": "This outrageous scandal exposes corrupt officials! Every patriot must wake up!",
            "previous_analysis": {
                "prediction": "FAKE", 
                "confidence": 0.65,
                "source": "test_source.com"
            }
        }
        
        # Process
        result = agent.process(test_input)
        
        if result['success']:
            context_score = result['result']['context_scores']['overall_context_score']
            risk_level = result['result']['context_scores']['risk_level']
            print(f"✅ Context score: {context_score:.1f}/10 ({risk_level})")
            print("✅ Context Analyzer Agent working correctly!")
        else:
            print(f"❌ Processing failed: {result['error']['message']}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_context_analyzer()

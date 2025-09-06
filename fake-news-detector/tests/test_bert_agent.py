# test_bert_agent.py
"""Quick test for BERT Classifier Agent"""

def test_bert_classifier():
    print("🧪 Testing BERT Classifier Agent...")
    
    try:
        from agents.bert_classifier import BERTClassifierAgent
        
        # Initialize agent
        agent = BERTClassifierAgent()
        print("✅ Agent initialized successfully")
        
        # Test input
        test_input = {
            "text": "Breaking: Scientists discover miracle cure that doctors don't want you to know about!",
            "metadata": {"source": "test_source.com"}
        }
        
        # Process
        result = agent.process(test_input)
        
        if result['success']:
            print(f"✅ Classification: {result['result']['prediction']}")
            print(f"   Confidence: {result['result']['confidence']:.2%}")
            print("✅ BERT Agent working correctly!")
        else:
            print(f"❌ Processing failed: {result['error']['message']}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_bert_classifier()

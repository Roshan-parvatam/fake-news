# test_claim_agent.py
"""Quick test for Claim Extractor Agent"""

def test_claim_extractor():
    print("🧪 Testing Claim Extractor Agent...")
    
    try:
        from agents.claim_extractor import ClaimExtractorAgent
        
        # Initialize agent
        agent = ClaimExtractorAgent()
        print("✅ Agent initialized successfully")
        
        # Test input
        test_input = {
            "text": "Study published in Nature found 85% improvement. Dr. Smith confirmed the results.",
            "bert_results": {"prediction": "REAL", "confidence": 0.78},
            "topic_domain": "health"
        }
        
        # Process
        result = agent.process(test_input)
        
        if result['success']:
            claims_count = result['result']['metadata']['total_claims_found']
            print(f"✅ Claims extracted: {claims_count}")
            print("✅ Claim Extractor Agent working correctly!")
        else:
            print(f"❌ Processing failed: {result['error']['message']}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_claim_extractor()

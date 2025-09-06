# test_source_agent.py
"""Quick test for Credible Source Agent"""

def test_credible_source():
    print("🧪 Testing Credible Source Agent...")
    
    try:
        from agents.credible_source import CredibleSourceAgent
        
        # Initialize agent
        agent = CredibleSourceAgent()
        print("✅ Agent initialized successfully")
        
        # Test input with sample claims and evidence
        test_claims = [
            {
                'text': 'Medical study shows effectiveness.',
                'verifiability_score': 8,
                'source': 'Medical researchers'
            }
        ]
        
        test_input = {
            "text": "Harvard researchers published findings in medical journal.",
            "extracted_claims": test_claims,
            "evidence_evaluation": {"overall_evidence_score": 7.2}
        }
        
        # Process
        result = agent.process(test_input)
        
        if result['success']:
            sources_count = len(result['result']['recommended_sources'])
            recommendation_score = result['result']['recommendation_scores']['overall_recommendation_score']
            print(f"✅ Sources recommended: {sources_count}")
            print(f"   Recommendation score: {recommendation_score:.1f}/10")
            print("✅ Credible Source Agent working correctly!")
        else:
            print(f"❌ Processing failed: {result['error']['message']}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_credible_source()

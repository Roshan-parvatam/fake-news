# test_evidence_agent.py
"""Quick test for Evidence Evaluator Agent"""

def test_evidence_evaluator():
    print("🧪 Testing Evidence Evaluator Agent...")
    
    try:
        from agents.evidence_evaluator import EvidenceEvaluatorAgent
        
        # Initialize agent
        agent = EvidenceEvaluatorAgent()
        print("✅ Agent initialized successfully")
        
        # Test input with sample claims
        test_claims = [
            {
                'text': 'Study published in peer-reviewed journal.',
                'claim_type': 'Research',
                'verifiability_score': 8,
                'source': 'University researchers'
            }
        ]
        
        test_input = {
            "text": "Research published in Nature Medicine shows promising results.",
            "extracted_claims": test_claims,
            "context_analysis": {"overall_context_score": 4.2}
        }
        
        # Process
        result = agent.process(test_input)
        
        if result['success']:
            evidence_score = result['result']['evidence_scores']['overall_evidence_score']
            quality_level = result['result']['evidence_scores']['quality_level']
            print(f"✅ Evidence score: {evidence_score:.1f}/10 ({quality_level})")
            print("✅ Evidence Evaluator Agent working correctly!")
        else:
            print(f"❌ Processing failed: {result['error']['message']}")
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_evidence_evaluator()

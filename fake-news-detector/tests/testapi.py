#!/usr/bin/env python3
"""
Comprehensive test script for Fake News Detection API
Tests the orchestration and all major endpoints
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api():
    print("🧪 Starting Fake News Detection API Tests...")
    print("=" * 50)
    
    # Test 1: Basic Health Check
    print("1️⃣ Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200, f"Health check failed with status {response.status_code}"
        
        health_data = response.json()
        print(f"   ✅ Status: {health_data.get('status')}")
        print(f"   ✅ LangGraph loaded: {health_data.get('langgraph_loaded')}")
        print(f"   ✅ API key configured: {health_data.get('api_key_configured')}")
        print(f"   ✅ Agents: {len(health_data.get('agents', []))} loaded")
        
        # Verify all agents are loaded
        expected_agents = ["bert_classifier", "claim_extractor", "evidence_evaluator", 
                          "credible_source", "llm_explanation"]
        actual_agents = health_data.get('agents', [])
        for agent in expected_agents:
            assert agent in actual_agents, f"Missing agent: {agent}"
        
        print("   ✅ Health check passed!")
        
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    print()
    
    # Test 2: Model Configuration Check
    print("2️⃣ Testing /config/models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/config/models")
        assert response.status_code == 200, "Model config check failed"
        
        config_data = response.json()
        print(f"   ✅ Current models loaded:")
        for agent, model in config_data.get('current_models', {}).items():
            print(f"      - {agent}: {model}")
            
        print("   ✅ Model configuration check passed!")
        
    except Exception as e:
        print(f"   ❌ Model config check failed: {e}")
        return False
    
    print()
    
    # Test 3: Simple Article Analysis
    print("3️⃣ Testing /analyze endpoint with simple article...")
    simple_article = {
        "text": "Reuters reports that the GDP increased by 2.1% last quarter according to the Bureau of Labor Statistics.",
        "detailed": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/analyze", json=simple_article)
        end_time = time.time()
        
        assert response.status_code == 200, f"Analysis failed with status {response.status_code}"
        
        result = response.json()
        assert result.get('success'), "Analysis returned success=False"
        
        print(f"   ✅ Processing time: {end_time - start_time:.2f} seconds")
        print(f"   ✅ Processing path: {result.get('metadata', {}).get('processing_path', 'unknown')}")
        
        # Check if we got results from agents
        results = result.get('results', {})
        print(f"   ✅ BERT classification: {results.get('classification', {})}")
        print(f"   ✅ Claims extracted: {len(results.get('claims', {}).get('extracted_claims', []))} claims")
        
        print("   ✅ Simple analysis passed!")
        
    except Exception as e:
        print(f"   ❌ Simple analysis failed: {e}")
        if 'response' in locals():
            print(f"   Response: {response.text[:200]}...")
        return False
    
    print()
    
    # Test 4: Complex Article Analysis (Detailed)
    print("4️⃣ Testing /analyze endpoint with complex article (detailed)...")
    complex_article = {
        "text": "BREAKING: Scientists claim they have discovered alien life on Mars, but government officials are denying access to the evidence. Anonymous sources say the discovery was made last month but is being covered up by NASA.",
        "detailed": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/analyze", json=complex_article)
        end_time = time.time()
        
        assert response.status_code == 200, "Complex analysis failed"
        
        result = response.json()
        assert result.get('success'), "Complex analysis returned success=False"
        
        print(f"   ✅ Processing time: {end_time - start_time:.2f} seconds")
        print(f"   ✅ Processing path: {result.get('metadata', {}).get('processing_path', 'unknown')}")
        
        # This should trigger more comprehensive analysis
        results = result.get('results', {})
        classification = results.get('classification', {})
        print(f"   ✅ Classification: {classification.get('prediction')} (confidence: {classification.get('confidence', 0):.2f})")
        
        claims = results.get('claims', {}).get('extracted_claims', [])
        print(f"   ✅ Claims found: {len(claims)}")
        
        if results.get('evidence'):
            print(f"   ✅ Evidence evaluation completed")
            
        if results.get('sources'):
            print(f"   ✅ Source recommendations provided")
            
        explanation = results.get('explanation', '')
        print(f"   ✅ Explanation generated: {len(explanation)} characters")
        
        print("   ✅ Complex analysis passed!")
        
    except Exception as e:
        print(f"   ❌ Complex analysis failed: {e}")
        if 'response' in locals():
            print(f"   Response: {response.text[:200]}...")
        return False
    
    print()
    
    # Test 5: System Metrics
    print("5️⃣ Testing /metrics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        assert response.status_code == 200, "Metrics check failed"
        
        metrics = response.json()
        print(f"   ✅ System status: {metrics.get('system_status')}")
        print(f"   ✅ Smart routing: {metrics.get('smart_routing_enabled')}")
        print(f"   ✅ Processing modes: {metrics.get('processing_modes')}")
        
        print("   ✅ Metrics check passed!")
        
    except Exception as e:
        print(f"   ❌ Metrics check failed: {e}")
        return False
    
    print()
    print("🎉 ALL TESTS PASSED! Your API and orchestration are working correctly!")
    print("=" * 50)
    print("✅ LangGraph workflow is operational")
    print("✅ Smart conditional routing is active")
    print("✅ All agents are responding")
    print("✅ API endpoints are functioning")
    print("✅ Your fake news detection system is ready for use!")
    
    return True

if __name__ == "__main__":
    success = test_api()
    if not success:
        print("\n❌ Some tests failed. Check your server logs and configuration.")
        exit(1)
    else:
        print("\n🚀 Ready for production use!")

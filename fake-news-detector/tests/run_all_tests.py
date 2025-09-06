# tests/run_all_tests.py
"""Run all agent tests - Fixed version"""

import sys
import os
import importlib.util

def run_all_tests():
    print("🧪 Running All Agent Tests")
    print("=" * 50)
    
    # List of test modules and their test functions
    test_modules = [
        ("BERT Classifier", "test_bert_agent", "test_bert_classifier"),
        ("LLM Explanation", "test_llm_agent", "test_llm_explanation"), 
        ("Claim Extractor", "test_claim_agent", "test_claim_extractor"),
        ("Context Analyzer", "test_context_agent", "test_context_analyzer"),
        ("Evidence Evaluator", "test_evidence_agent", "test_evidence_evaluator"),
        ("Credible Source", "test_source_agent", "test_credible_source")
    ]
    
    results = []
    
    for agent_name, module_name, function_name in test_modules:
        print(f"\n🔍 Testing {agent_name} Agent...")
        try:
            # Import the test module
            test_module = __import__(f"tests.{module_name}", fromlist=[function_name])
            test_function = getattr(test_module, function_name)
            
            # Run the test
            test_function()
            results.append((agent_name, "✅ PASSED"))
            
        except ImportError as e:
            print(f"❌ Could not import {module_name}: {str(e)}")
            results.append((agent_name, f"❌ FAILED: Import error"))
        except AttributeError as e:
            print(f"❌ Could not find function {function_name}: {str(e)}")
            results.append((agent_name, f"❌ FAILED: Function not found"))
        except Exception as e:
            print(f"❌ {agent_name} test failed: {str(e)}")
            results.append((agent_name, f"❌ FAILED: {str(e)[:50]}..."))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 50)
    
    for agent_name, status in results:
        print(f"{agent_name:<20}: {status}")
    
    passed = sum(1 for _, status in results if "PASSED" in status)
    total = len(results)
    print(f"\nOverall: {passed}/{total} agents passed tests")
    
    if passed == total:
        print("🎉 All agents are working correctly!")
        print("✅ Ready to proceed with LangGraph integration!")
    else:
        print("⚠️  Some agents need attention before LangGraph integration")

if __name__ == "__main__":
    run_all_tests()

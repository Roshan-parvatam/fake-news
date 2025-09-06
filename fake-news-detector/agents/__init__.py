# agents/__init__.py
"""
Agents Package - Fake News Detection System

Modular agents for fake news detection with LangGraph compatibility.
"""

# Import modular agents
from .bert_classifier import BERTClassifier
from .llm_explanation import LLMExplanationAgent
from .credible_source import CredibleSourceAgent
from .context_analyzer import ContextAnalyzerAgent
from .claim_extractor import ClaimExtractorAgent
from .evidence_evaluator import EvidenceEvaluatorAgent

# Import base agent
from .base import BaseAgent

__all__ = [
    "BERTClassifier",
    "LLMExplanationAgent", 
    "CredibleSourceAgent",
    "ContextAnalyzerAgent",
    "ClaimExtractorAgent",
    "EvidenceEvaluatorAgent",
    "BaseAgent"
]

__version__ = "2.0.0"

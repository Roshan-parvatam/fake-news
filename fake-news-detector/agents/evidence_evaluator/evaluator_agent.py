# agents/evidence_evaluator/evaluator_agent.py

"""
Evidence Evaluator Agent

Production-ready evidence evaluation agent that assesses evidence quality,
source credibility, and logical consistency in news articles using LLM-powered
verification with specific, actionable source recommendations.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import google.generativeai as genai

# ✅ FIXED: Correct import path for BaseAgent
from agents.base import BaseAgent
from config import get_model_config, get_settings

# ✅ FIXED: Utils import with fallback
try:
    from utils.helpers import sanitize_text
except ImportError:
    def sanitize_text(text: str) -> str:
        """Basic text sanitization fallback."""
        if not isinstance(text, str):
            return ""
        return text.strip().replace('\x00', '').replace('\r\n', '\n')

from .criteria import EvidenceQualityCriteria
from .fallacy_detection import LogicalFallacyDetector
from .prompts import get_prompt_template, PromptValidator
from .validators import validate_evidence_input
from .exceptions import (
    EvidenceEvaluatorError,
    InputValidationError,
    LLMResponseError,
    VerificationSourceError,
    ConfigurationError
)

class EvidenceEvaluatorAgent(BaseAgent):
    """
    Evidence evaluation agent with LLM-powered verification and specific source generation.
    
    Evaluates evidence quality, source credibility, and logical consistency while
    providing specific, actionable verification sources for fact-checking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evidence evaluator agent with configuration."""
        
        # ✅ GET CONFIGURATION FROM CONFIG FILES
        evidence_config = get_model_config('evidence_evaluator')
        system_settings = get_settings()
        
        if config:
            evidence_config.update(config)

        self.agent_name = "evidence_evaluator"
        super().__init__(evidence_config)

        # AI Model Configuration
        self.model_name = self.config.get('model_name', 'gemini-1.5-pro')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 3072)

        # Evaluation Settings
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        self.evidence_threshold = self.config.get('evidence_threshold', 6.0)
        self.enable_fallacy_detection = self.config.get('enable_fallacy_detection', True)
        self.max_verification_sources = self.config.get('max_verification_sources', 5)

        # Scoring Configuration
        self.scoring_weights = self.config.get('scoring_weights', {
            'source_quality': 0.35,
            'logical_consistency': 0.3,
            'evidence_completeness': 0.25,
            'verification_quality': 0.1
        })

        # ✅ FIXED: Enhanced API key loading from .env
        self.api_key = (
            os.getenv('GEMINI_API_KEY') or 
            os.getenv('GOOGLE_API_KEY') or 
            getattr(system_settings, 'gemini_api_key', None)
        )
        
        if not self.api_key:
            raise ConfigurationError(
                "Gemini API key not found. Please set GEMINI_API_KEY in your .env file"
            )

        # Rate limiting configuration
        self.rate_limit = self.config.get('rate_limit_seconds', getattr(system_settings, 'gemini_rate_limit', 1.0))
        self.max_retries = self.config.get('max_retries', getattr(system_settings, 'max_retries', 3))

        # Initialize components
        self._initialize_gemini_api()
        self.quality_criteria = EvidenceQualityCriteria(self.config)
        self.fallacy_detector = LogicalFallacyDetector(self.config)
        self.prompt_validator = PromptValidator()

        # Performance tracking
        self.evaluation_metrics = {
            'evaluations_completed': 0,
            'verification_sources_generated': 0,
            'high_quality_sources_found': 0,
            'fallacies_detected': 0,
            'evidence_quality_assessments': 0
        }

        self.last_request_time = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Evidence Evaluator Agent initialized with model {self.model_name}")

    def _initialize_gemini_api(self) -> None:
        """Initialize Gemini API with safety settings."""
        try:
            genai.configure(api_key=self.api_key)

            generation_config = {
                "temperature": self.temperature,
                "top_p": self.config.get('top_p', 0.9),
                "top_k": self.config.get('top_k', 40),
                "max_output_tokens": self.max_tokens,
            }

            safety_settings = self.config.get('safety_settings', [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ])

            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.logger.info(f"Gemini API initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Gemini API: {str(e)}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for evidence evaluation.
        
        Args:
            input_data: Dictionary containing article text, claims, and context
            
        Returns:
            Dictionary with evaluation results and verification sources
        """
        # ✅ FIXED: Enhanced input validation
        validation_result = validate_evidence_input(input_data)
        if not validation_result.is_valid:
            return self.format_error_output(
                InputValidationError(f"Input validation failed: {validation_result.errors[0]}"),
                input_data
            )

        # ✅ FIXED: Session management compatibility
        if hasattr(self, '_start_processing_session'):
            self._start_processing_session(input_data)

        start_time = time.time()
        
        try:
            article_text = input_data.get('text', '')
            extracted_claims = input_data.get('extracted_claims', [])
            context_analysis = input_data.get('context_analysis', {})
            
            # Determine analysis depth
            context_score = context_analysis.get('overall_context_score', 5.0)
            include_detailed_analysis = (
                self.enable_detailed_analysis or 
                context_score > 7.0 or 
                len(extracted_claims) < 2
            )

            # Perform evidence evaluation
            evaluation_result = self.evaluate_evidence(
                article_text=article_text,
                extracted_claims=extracted_claims,
                context_analysis=context_analysis,
                include_detailed_analysis=include_detailed_analysis
            )

            # Update metrics
            processing_time = time.time() - start_time
            self.evaluation_metrics['evaluations_completed'] += 1
            
            evidence_score = evaluation_result['evidence_scores'].get('overall_evidence_score', 5.0)
            confidence = evidence_score / 10.0

            return self.format_output(
                result=evaluation_result,
                confidence=confidence,
                metadata={
                    'processing_time': processing_time,
                    'model_used': self.model_name,
                    'detailed_analysis': include_detailed_analysis,
                    'verification_sources_count': len(evaluation_result.get('verification_sources', [])),
                    'agent_version': '3.0.0'
                }
            )

        except Exception as e:
            self.logger.error(f"Evidence evaluation failed: {str(e)}")
            return self.format_error_output(e, input_data)
        
        finally:
            # ✅ FIXED: Session cleanup compatibility
            if hasattr(self, '_end_processing_session'):
                self._end_processing_session()

    def evaluate_evidence(self,
                         article_text: str,
                         extracted_claims: List[Dict[str, Any]],
                         context_analysis: Dict[str, Any],
                         include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evidence evaluation with specific source verification.
        
        Args:
            article_text: Article content to evaluate
            extracted_claims: List of claims extracted from article
            context_analysis: Context analysis results
            include_detailed_analysis: Whether to include detailed analysis
            
        Returns:
            Dictionary containing evaluation results and verification sources
        """
        start_time = time.time()

        # Clean and prepare article text
        article_text = sanitize_text(article_text)
        max_length = self.config.get('max_article_length', 4000)
        if len(article_text) > max_length:
            article_text = article_text[:max_length] + "..."

        # Step 1: Generate specific verification sources
        verification_sources = self._generate_verification_sources(article_text, extracted_claims)

        # Step 2: Assess source quality
        source_quality_analysis = self._assess_source_quality(article_text, extracted_claims)

        # Step 3: Analyze logical consistency
        logical_consistency_analysis = self._analyze_logical_consistency(article_text, extracted_claims)

        # Step 4: Run systematic quality assessment
        quality_assessment = self.quality_criteria.assess_evidence_quality(
            article_text, extracted_claims
        )

        # Step 5: Detect logical fallacies
        fallacy_report = {}
        if self.enable_fallacy_detection:
            fallacy_report = self.fallacy_detector.detect_fallacies(article_text)

        # Step 6: Identify evidence gaps (if detailed analysis requested)
        evidence_gaps_analysis = None
        if include_detailed_analysis:
            evidence_gaps_analysis = self._identify_evidence_gaps(article_text, extracted_claims)

        # Step 7: Calculate comprehensive scores
        evidence_scores = self._calculate_evidence_scores(
            quality_assessment, verification_sources, source_quality_analysis,
            logical_consistency_analysis, fallacy_report
        )

        # Step 8: Create summary
        evidence_summary = self._create_evidence_summary(
            extracted_claims, evidence_scores, verification_sources
        )

        processing_time = time.time() - start_time

        # Update metrics
        self.evaluation_metrics['verification_sources_generated'] += len(verification_sources)
        self.evaluation_metrics['high_quality_sources_found'] += len([
            s for s in verification_sources if s.get('quality_score', 0) >= 0.8
        ])
        if fallacy_report:
            self.evaluation_metrics['fallacies_detected'] += len(fallacy_report.get('detected_fallacies', []))
        self.evaluation_metrics['evidence_quality_assessments'] += 1

        return {
            'verification_sources': verification_sources,
            'source_quality_analysis': source_quality_analysis,
            'logical_consistency_analysis': logical_consistency_analysis,
            'evidence_gaps_analysis': evidence_gaps_analysis,
            'quality_assessment': quality_assessment,
            'fallacy_report': fallacy_report,
            'evidence_scores': evidence_scores,
            'evidence_summary': evidence_summary,
            'metadata': {
                'processing_time_seconds': round(processing_time, 2),
                'model_used': self.model_name,
                'claims_evaluated': len(extracted_claims),
                'detailed_analysis_included': include_detailed_analysis,
                'verification_sources_found': len(verification_sources),
                'timestamp': datetime.now().isoformat()
            }
        }

    def _generate_verification_sources(self, article_text: str, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific verification sources using LLM with URL validation."""
        try:
            self._respect_rate_limits()
            
            prompt = get_prompt_template(
                'verification_sources',
                article_text=article_text,
                claims=claims
            )

            response = self.model.generate_content(prompt)
            
            if not self._is_valid_response(response):
                self.logger.warning("Invalid LLM response for verification sources")
                return self._create_fallback_sources(claims)

            response_text = response.candidates[0].content.parts[0].text
            verification_sources = self._parse_verification_sources(response_text)

            # Validate URL specificity
            validated_sources = []
            for source in verification_sources:
                url = source.get('url', '')
                if self.prompt_validator.validate_url_specificity(url):
                    source['quality_score'] = self._calculate_source_quality_score(source)
                    validated_sources.append(source)
                else:
                    self.logger.debug(f"Rejected generic URL: {url}")

            return validated_sources[:self.max_verification_sources]

        except Exception as e:
            self.logger.error(f"Verification source generation failed: {str(e)}")
            return self._create_fallback_sources(claims)

    def _parse_verification_sources(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured verification sources."""
        import re
        verification_sources = []

        # Split by verification source sections
        sections = re.split(r'## VERIFICATION SOURCE \d+', response_text)
        
        for section in sections[1:]:  # Skip first empty section
            try:
                source_data = {}
                
                # Extract fields using regex
                claim_match = re.search(r'\*\*CLAIM\*\*:\s*["\']?(.*?)["\']?(?=\n|\*\*)', section, re.IGNORECASE)
                institution_match = re.search(r'\*\*INSTITUTION\*\*:\s*(.*?)(?=\n|\*\*)', section, re.IGNORECASE)
                url_match = re.search(r'\*\*SPECIFIC_URL\*\*:\s*(https?://\S+)', section, re.IGNORECASE)
                type_match = re.search(r'\*\*VERIFICATION_TYPE\*\*:\s*(\w+)', section, re.IGNORECASE)
                confidence_match = re.search(r'\*\*CONFIDENCE\*\*:\s*([0-9]\.[0-9]+)', section, re.IGNORECASE)

                if claim_match and institution_match and url_match:
                    source_data = {
                        'claim': claim_match.group(1).strip(),
                        'institution': institution_match.group(1).strip(),
                        'url': url_match.group(1).strip(),
                        'verification_type': type_match.group(1).strip() if type_match else 'general',
                        'confidence': float(confidence_match.group(1)) if confidence_match else 0.5,
                        'source_type': 'llm_generated'
                    }
                    verification_sources.append(source_data)
                    
            except Exception as e:
                self.logger.debug(f"Failed to parse verification section: {str(e)}")
                continue

        return verification_sources

    def _calculate_source_quality_score(self, source: Dict[str, Any]) -> float:
        """Calculate quality score for a verification source."""
        score = 0.5  # Base score
        
        institution = source.get('institution', '').lower()
        url = source.get('url', '').lower()
        verification_type = source.get('verification_type', '').lower()

        # Institution quality bonus
        high_quality_institutions = [
            'cdc', 'who', 'nih', 'fda', 'harvard', 'stanford', 'mit',
            'pubmed', 'nature', 'science', 'reuters', 'ap news'
        ]
        
        if any(inst in institution for inst in high_quality_institutions):
            score += 0.3

        # Domain quality bonus
        high_quality_domains = ['.gov', '.edu', 'pubmed', 'nature.com', 'science.org']
        if any(domain in url for domain in high_quality_domains):
            score += 0.2

        # Verification type bonus
        if verification_type in ['primary_source', 'official_data']:
            score += 0.15
        elif verification_type in ['expert_analysis', 'research_study']:
            score += 0.1

        # Confidence factor
        confidence = source.get('confidence', 0.5)
        score = score * (0.7 + 0.3 * confidence)

        return min(1.0, max(0.1, score))

    def _assess_source_quality(self, article_text: str, claims: List[Dict[str, Any]]) -> str:
        """Assess overall source quality using LLM analysis."""
        try:
            self._respect_rate_limits()
            
            sources = [claim.get('source', 'Not specified') for claim in claims]
            sources = [s for s in sources if s != 'Not specified'][:10]

            prompt = get_prompt_template(
                'source_quality',
                article_text=article_text,
                sources=sources
            )

            response = self.model.generate_content(prompt)
            
            if self._is_valid_response(response):
                return response.candidates[0].content.parts[0].text
            else:
                return self._create_fallback_source_analysis(len(sources))

        except Exception as e:
            self.logger.error(f"Source quality assessment failed: {str(e)}")
            return self._create_fallback_source_analysis(len(claims))

    def _analyze_logical_consistency(self, article_text: str, claims: List[Dict[str, Any]]) -> str:
        """Analyze logical consistency using LLM analysis."""
        try:
            self._respect_rate_limits()
            
            claim_texts = [claim.get('text', '') for claim in claims[:6]]

            prompt = get_prompt_template(
                'logical_consistency',
                article_text=article_text,
                claims=claim_texts
            )

            response = self.model.generate_content(prompt)
            
            if self._is_valid_response(response):
                return response.candidates[0].content.parts[0].text
            else:
                return self._create_fallback_logical_analysis(len(claims))

        except Exception as e:
            self.logger.error(f"Logical consistency analysis failed: {str(e)}")
            return self._create_fallback_logical_analysis(len(claims))

    def _identify_evidence_gaps(self, article_text: str, claims: List[Dict[str, Any]]) -> str:
        """Identify evidence gaps using LLM analysis."""
        try:
            self._respect_rate_limits()

            prompt = get_prompt_template(
                'evidence_gaps',
                article_text=article_text,
                claims=claims
            )

            response = self.model.generate_content(prompt)
            
            if self._is_valid_response(response):
                return response.candidates[0].content.parts[0].text
            else:
                return "Evidence gap analysis unavailable due to content restrictions."

        except Exception as e:
            self.logger.error(f"Evidence gaps analysis failed: {str(e)}")
            return "Evidence gap analysis unavailable."

    def _calculate_evidence_scores(self,
                                  quality_assessment: Dict[str, Any],
                                  verification_sources: List[Dict[str, Any]],
                                  source_quality_analysis: str,
                                  logical_consistency_analysis: str,
                                  fallacy_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive evidence scores."""
        
        # Extract base scores
        source_quality_score = quality_assessment.get('source_quality_score', 5.0)
        completeness_score = quality_assessment.get('completeness_score', 5.0)

        # Calculate verification quality score
        verification_score = 3.0
        if verification_sources:
            avg_quality = sum(s.get('quality_score', 0.5) for s in verification_sources) / len(verification_sources)
            verification_score = 3.0 + (avg_quality * 7.0)
            
            if len(verification_sources) >= 3:
                verification_score += 1.0
                
            if len([s for s in verification_sources if s.get('quality_score', 0) >= 0.8]) >= 2:
                verification_score += 1.0

        verification_score = max(0, min(10, verification_score))

        # Calculate logical consistency score
        fallacy_count = len(fallacy_report.get('detected_fallacies', []))
        logical_score = max(1.0, min(10.0, 8.0 - fallacy_count))

        # ✅ FIXED: Safe access to scoring weights with fallbacks
        overall_score = (
            (source_quality_score * self.scoring_weights.get('source_quality', 0.35)) +
            (logical_score * self.scoring_weights.get('logical_consistency', 0.3)) +
            (completeness_score * self.scoring_weights.get('evidence_completeness', 0.25)) +
            (verification_score * self.scoring_weights.get('verification_quality', 0.1))
        )

        overall_score = max(0, min(10, overall_score))

        # Determine quality level
        if overall_score >= 8.0:
            quality_level = "HIGH QUALITY"
        elif overall_score >= 6.0:
            quality_level = "MODERATE QUALITY"
        elif overall_score >= 4.0:
            quality_level = "LOW QUALITY"
        else:
            quality_level = "POOR QUALITY"

        return {
            'source_quality_score': round(source_quality_score, 2),
            'logical_consistency_score': round(logical_score, 2),
            'evidence_completeness_score': round(completeness_score, 2),
            'verification_quality_score': round(verification_score, 2),
            'overall_evidence_score': round(overall_score, 2),
            'quality_level': quality_level,
            'verification_sources_count': len(verification_sources),
            'high_quality_sources_count': len([s for s in verification_sources if s.get('quality_score', 0) >= 0.8]),
            'fallacy_count': fallacy_count
        }

    def _create_evidence_summary(self,
                                claims: List[Dict[str, Any]],
                                evidence_scores: Dict[str, Any],
                                verification_sources: List[Dict[str, Any]]) -> str:
        """Create comprehensive evidence summary."""
        if not claims:
            return "No claims available for evidence evaluation."

        overall_score = evidence_scores.get('overall_evidence_score', 5.0)
        quality_level = evidence_scores.get('quality_level', 'UNKNOWN')
        source_count = len(verification_sources)
        high_quality_count = evidence_scores.get('high_quality_sources_count', 0)

        summary_lines = [
            "EVIDENCE EVALUATION SUMMARY",
            f"Overall Score: {overall_score:.1f}/10 ({quality_level})",
            f"Verification Sources: {source_count} total ({high_quality_count} high-quality)",
            "",
            "Component Scores:",
            f"• Source Quality: {evidence_scores.get('source_quality_score', 5.0):.1f}/10",
            f"• Logical Consistency: {evidence_scores.get('logical_consistency_score', 5.0):.1f}/10",
            f"• Evidence Completeness: {evidence_scores.get('evidence_completeness_score', 5.0):.1f}/10",
            f"• Verification Quality: {evidence_scores.get('verification_quality_score', 5.0):.1f}/10",
            "",
            "Claims Analysis:",
            f"• Total Claims Evaluated: {len(claims)}",
            f"• High Verifiability Claims: {len([c for c in claims if c.get('verifiability_score', 0) >= 7])}",
            f"• Fallacies Detected: {evidence_scores.get('fallacy_count', 0)}"
        ]

        if high_quality_count >= 3:
            summary_lines.append("✓ Excellent verification sources available")
        elif high_quality_count >= 1:
            summary_lines.append("⚠ Some high-quality verification sources available")
        elif source_count > 0:
            summary_lines.append("⚠ Basic verification sources provided")
        else:
            summary_lines.append("❌ Limited verification sources available")

        return "\n".join(summary_lines)

    def _create_fallback_sources(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create fallback verification sources when LLM generation fails."""
        fallback_sources = []
        
        for claim in claims[:3]:
            claim_text = claim.get('text', 'Unknown claim')
            fallback_sources.append({
                'claim': claim_text,
                'institution': 'Snopes Fact Checking',
                'url': 'https://www.snopes.com/fact-check/',
                'verification_type': 'fact_checker',
                'confidence': 0.6,
                'quality_score': 0.6,
                'source_type': 'fallback'
            })

        return fallback_sources

    def _create_fallback_source_analysis(self, source_count: int) -> str:
        """Create fallback source analysis when LLM analysis fails."""
        return f"""
SOURCE QUALITY ASSESSMENT

Sources Identified: {source_count}

Assessment: Automated evaluation indicates sources require human verification.

Recommendation: Cross-reference claims with established authorities and verify source credentials independently.

Quality Check: Verify sources through official channels and check for conflicts of interest.
"""

    def _create_fallback_logical_analysis(self, claim_count: int) -> str:
        """Create fallback logical analysis when LLM analysis fails."""
        return f"""
LOGICAL CONSISTENCY ASSESSMENT

Claims Analyzed: {claim_count}

Assessment: Logical structure requires human review for comprehensive evaluation.

Recommendation: Check claim-to-claim consistency, verify evidence-to-conclusion logic, and assess reasoning quality independently.

Review Focus: Look for logical fallacies, verify statistical claims, and ensure temporal consistency.
"""

    def _is_valid_response(self, response) -> bool:
        """Check if LLM response is valid and not blocked."""
        return (response and 
                response.candidates and 
                len(response.candidates) > 0 and 
                response.candidates[0].finish_reason != 2 and  # Not SAFETY
                response.candidates[0].content and 
                response.candidates[0].content.parts)

    def _respect_rate_limits(self) -> None:
        """Implement rate limiting for API calls."""
        current_time = time.time()
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        base_metrics = super().get_performance_metrics()
        evidence_metrics = {
            'evaluations_completed': self.evaluation_metrics['evaluations_completed'],
            'verification_sources_generated': self.evaluation_metrics['verification_sources_generated'],
            'high_quality_sources_found': self.evaluation_metrics['high_quality_sources_found'],
            'fallacies_detected': self.evaluation_metrics['fallacies_detected'],
            'evidence_quality_assessments': self.evaluation_metrics['evidence_quality_assessments'],
            'model_config': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        }
        
        return {**base_metrics, **evidence_metrics}

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input data for evidence evaluation."""
        if not isinstance(input_data, dict):
            return False, "Input must be a dictionary"
        
        if 'text' not in input_data:
            return False, "Missing required 'text' field"
            
        if not input_data['text'].strip():
            return False, "Article text cannot be empty"
            
        if len(input_data['text']) < 50:
            return False, "Article text too short for meaningful analysis"
        
        return True, ""

# Testing functionality
if __name__ == "__main__":
    """Test the evidence evaluator agent."""
    try:
        agent = EvidenceEvaluatorAgent()
        
        test_input = {
            "text": """
            According to a study published by researchers at Harvard University,
            the new treatment shows promising results in clinical trials with
            over 1,000 participants. The peer-reviewed research, published in
            Nature Medicine, demonstrates significant efficacy rates.
            """,
            "extracted_claims": [
                {
                    "text": "Harvard University study shows promising treatment results",
                    "verifiability_score": 8,
                    "priority": 1
                },
                {
                    "text": "Clinical trials included over 1,000 participants",
                    "verifiability_score": 7,
                    "priority": 2
                }
            ],
            "context_analysis": {
                "overall_context_score": 6.5,
                "risk_level": "MEDIUM"
            }
        }

        result = agent.process(test_input)
        
        if result['success']:
            evaluation_data = result['result']
            evidence_scores = evaluation_data.get('evidence_scores', {})
            print("✅ Evidence Evaluation Results:")
            print(f"   Overall Evidence Score: {evidence_scores.get('overall_evidence_score', 0):.1f}/10")
            print(f"   Quality Level: {evidence_scores.get('quality_level', 'UNKNOWN')}")
            print(f"   Verification Sources: {evidence_scores.get('verification_sources_count', 0)}")
            print(f"   High Quality Sources: {evidence_scores.get('high_quality_sources_count', 0)}")
            print(f"   Processing Time: {evaluation_data['metadata']['processing_time_seconds']}s")
        else:
            print(f"❌ Evaluation failed: {result['error']['message']}")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

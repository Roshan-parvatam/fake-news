// api.ts - Updated for Backend Compatibility

import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS || 360000); // default 6 minutes

const api = axios.create({
  baseURL: API_URL,
  timeout: API_TIMEOUT_MS,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface AnalysisRequest {
  text?: string;
  url?: string;
  detailed: boolean;
}

export interface Claim {
  text: string;
  priority: 'HIGH' | 'MEDIUM' | 'LOW';
  verification_status: 'VERIFIED' | 'UNVERIFIED' | 'DISPUTED';
}

export interface ContextScore {
  bias: number;
  manipulation: number;
  risk: number;
  credibility: number;
}

export interface Evidence {
  quality: 'HIGH' | 'MEDIUM' | 'LOW';
  sources: string[];
  confidence: number;
}

export interface VerificationLink {
  claim: string;
  url: string;
  explanation: string;
  source_type: string;
  quality_score: number;
  institution?: string;
  search_terms?: string;
  type?: string;
}

export interface ContextualSource {
  name: string;
  details: string;
  relevance: string;
  contact_method: string;
  type: string;
  reliability_score: number;
  url?: string;
  relevance_score?: number;
}

export interface AnalysisResponse {
  classification: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number;
  claims: Claim[];
  context_scores: ContextScore;
  evidence: Evidence;
  recommended_sources: string[];
  verification_links: VerificationLink[];
  contextual_sources: ContextualSource[];
  explanation: string;
  timestamp: string;
  analysis_metadata: {
    llm_scores_used: boolean;
    specific_links_generated: boolean;
    contextual_recommendations: boolean;
    response_time: number;
    config_version: string;
    safety_fallbacks_used?: boolean;
    quality_score?: number;
    safety_blocks_encountered?: number;
  };
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'error';
  agents: {
    name: string;
    status: 'online' | 'offline';
    lastCheck: string;
  }[];
}

export interface ModelConfig {
  name: string;
  version: string;
  capabilities: string[];
}

// ✅ UPDATED: Fixed to match backend response structure (no nested 'results')
export const analyzeContent = async (request: AnalysisRequest): Promise<AnalysisResponse> => {
  try {
    console.log('🚀 Sending analysis request:', request);
    const response = await api.post('/analyze', request);
    const backendData = response.data;
    
    // DEBUG: Log the actual response structure
    console.log('🐛 Raw backend response:', JSON.stringify(backendData, null, 2));

    // Handle error responses from backend
    if (backendData.detail) {
      throw new Error(backendData.detail);
    }

    if (!backendData.success) {
      console.error('❌ Backend returned unsuccessful response:', backendData);
      throw new Error(backendData.errors?.[0] || 'Analysis failed - no results returned');
    }

    // ✅ FIXED: Access data from backendData.results (the actual nested structure)
    const results = backendData.results;
    console.log('📊 Results structure:', results);

    // ✅ UPDATED: Transform backend response to frontend format
    const transformedResponse: AnalysisResponse = {
      // ✅ FIXED: Access classification from nested structure
      classification: (results.classification?.prediction || 'UNCERTAIN') as 'REAL' | 'FAKE' | 'UNCERTAIN',
      confidence: Math.round((results.classification?.confidence || 0) * 100),

      // ✅ FIXED: Access claims from nested structure with RAG-based verification
      claims: (results.claims?.extracted_claims || []).map((claim: any) => {
        const claimText = claim.text || claim.claim_text || claim.claim || 'Unknown claim';
        const priority = typeof claim.priority === 'number'
          ? (claim.priority <= 1 ? 'HIGH' : claim.priority <= 2 ? 'MEDIUM' : 'LOW')
          : (claim.priority || 'MEDIUM');
        
        // ✅ NEW: Use RAG verification sources to determine actual verification status
        const verificationLinks = results.evidence?.verification_links || [];
        const claimVerificationSources = verificationLinks.filter((link: any) => 
          link.claim && link.claim.toLowerCase().includes(claimText.toLowerCase().substring(0, 20))
        );
        
        // Determine verification status based on actual sources found
        let verification_status: 'VERIFIED' | 'UNVERIFIED' | 'DISPUTED';
        if (claimVerificationSources.length > 0) {
          // ✅ FIXED: Mark as VERIFIED if ANY verification source is found (not just high quality)
          verification_status = 'VERIFIED';
        } else {
          // Fallback to original scoring if no RAG sources found
          const verificationScore = claim.verifiability_score || claim.verification_score || 3;
          verification_status = (verificationScore >= 7 ? 'VERIFIED' :
                              verificationScore >= 4 ? 'UNVERIFIED' : 'DISPUTED');
        }
        
        return {
          text: claimText,
          priority: priority as 'HIGH' | 'MEDIUM' | 'LOW',
          verification_status: verification_status as 'VERIFIED' | 'UNVERIFIED' | 'DISPUTED'
        };
      }),

      // ✅ FIXED: Access context scores from nested structure
      context_scores: {
        bias: Math.round((results.context_analysis?.scores?.bias ||
                         results.context_analysis?.scores?.bias_score * 10 || 0)),
        manipulation: Math.round((results.context_analysis?.scores?.manipulation ||
                                results.context_analysis?.scores?.manipulation_score * 10 || 0)),
        risk: Math.round((results.context_analysis?.scores?.risk ||
                         results.context_analysis?.overall_score * 10 || 50)),
        credibility: Math.round((results.context_analysis?.scores?.credibility ||
                               (100 - (results.context_analysis?.overall_score || 5) * 10)))
      },

      // ✅ FIXED: Access evidence from nested structure
      evidence: {
        quality: (
          results.evidence?.quality_level === 'HIGH QUALITY' ? 'HIGH' :
          results.evidence?.quality_level === 'MEDIUM QUALITY' ? 'MEDIUM' :
          results.evidence?.quality_level === 'LOW QUALITY' ? 'LOW' : 'MEDIUM'
        ) as 'HIGH' | 'MEDIUM' | 'LOW',
        sources: results.evidence?.verification_links || [],
        confidence: Math.round(Math.max(0, Math.min(100, (results.evidence?.overall_score || 5) * 10)))
      },

      // ✅ FIXED: Parse verification links from evidence.verification_links (backend transformed from verification_sources)
      verification_links: (results.evidence?.verification_links || []).map((link: any) => ({
        claim: link.claim || link.text || link.title || 'Unknown claim',
        url: link.url || '#',
        explanation: link.explanation || link.description || link.search_terms || 'No explanation available',
        source_type: link.source_type || link.type || 'unknown',
        quality_score: typeof link.quality_score === 'number' ? link.quality_score : 0.5,
        institution: link.institution || link.source || 'Unknown Institution',
        search_terms: link.search_terms || '',
        type: link.type || 'verification'
      })),

      // ✅ FIXED: Parse contextual sources from sources.recommended_sources
      contextual_sources: (results.sources?.recommended_sources || []).map((source: any) => ({
        name: source.name || 'Unknown source',
        details: source.details || 'No details available',
        relevance: source.relevance || 'Not specified',
        contact_method: source.contact_method || 'Not specified',
        type: source.type || 'unknown',
        reliability_score: source.reliability_score || source.relevance_score || 5,
        url: source.url || undefined,
        relevance_score: source.relevance_score || 5
      })),

      // Enhanced recommended sources with better fallback logic
      recommended_sources: [
        // First try contextual sources URLs
        ...(results.sources?.recommended_sources || [])
          .map((s: any) => s.url)
          .filter(Boolean),
        // Then try top sources
        ...(results.sources?.top_sources || [])
          .map((s: any) => s.url || s.name)
          .filter(Boolean),
        // Finally fallback to defaults if nothing found
        ...(results.sources?.recommended_sources?.length > 0 || 
            results.sources?.top_sources?.length > 0 ? [] : [
          'https://snopes.com',
          'https://factcheck.org',
          'https://politifact.com'
        ])
      ].slice(0, 5), // Limit to 5 sources

      // ✅ FIXED: Use explanation from nested structure
      explanation:
        results.explanation?.text ||
        results.context_analysis?.analysis ||
        'No detailed explanation available',

      // Timestamp
      timestamp: backendData.timestamp || new Date().toISOString(),

      // Enhanced analysis metadata
      analysis_metadata: {
        llm_scores_used: !!(results.context_analysis?.scores),
        specific_links_generated: (results.evidence?.verification_links || []).length > 0,
        contextual_recommendations: (results.sources?.recommended_sources || []).length > 0,
        response_time: backendData.metadata?.processing_time_seconds || 0,
        config_version: backendData.metadata?.api_version || '3.2.0',
        safety_fallbacks_used: backendData.metadata?.enhanced_features_used?.safety_fallbacks_used || false,
        quality_score: backendData.metadata?.quality_score || 0,
        safety_blocks_encountered: backendData.metadata?.safety_blocks_encountered || 0
      }
    };

    console.log('✅ Transformed response with enhanced features:', transformedResponse);
    console.log('🔗 Verification links found:', transformedResponse.verification_links.length);
    console.log('🎯 Contextual sources found:', transformedResponse.contextual_sources.length);
    
    return transformedResponse;

  } catch (error) {
    console.error('❌ Analysis failed:', error);
    
    // Enhanced error handling
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      const errorMessage = error.response?.data?.detail ||
                          error.response?.data?.message ||
                          error.message ||
                          'Network error occurred';

      // Add specific error context based on status
      if (status === 422) {
        throw new Error(`Validation error: ${errorMessage}`);
      } else if (status === 500) {
        throw new Error(`Server error: ${errorMessage}`);
      } else if (status === 503) {
        throw new Error(`Service temporarily unavailable: ${errorMessage}`);
      } else if (status === 408 || error.code === 'ECONNABORTED') {
        throw new Error('Analysis timeout - please try again with a shorter text or disable detailed analysis');
      } else {
        throw new Error(errorMessage);
      }
    }
    
    throw error;
  }
};

export const checkHealth = async (): Promise<HealthResponse> => {
  try {
    const response = await api.get('/health');
    const backendData = response.data;
    console.log('🏥 Health check response:', backendData);

    return {
      status: backendData.status === 'healthy' ? 'healthy' : 'degraded',
      agents: (backendData.agents || []).map((agent: string) => ({
        name: agent,
        status: 'online' as const,
        lastCheck: new Date().toISOString()
      }))
    };
  } catch (error) {
    console.error('❌ Health check failed:', error);
    return {
      status: 'error',
      agents: []
    };
  }
};

export const getModelConfigs = async (): Promise<ModelConfig[]> => {
  try {
    const response = await api.get('/config/models');
    const backendData = response.data;
    
    return Object.entries(backendData.current_models || {}).map(([agent, model]) => ({
      name: model as string,
      version: '3.2.0', // ✅ Updated version
      capabilities: [`${agent}-analysis`]
    }));
  } catch (error) {
    console.error('❌ Failed to fetch model configs:', error);
    return [];
  }
};

// Helper functions remain the same
export const getAnalysisFeatures = (response: AnalysisResponse) => {
  return {
    hasLLMScoring: response.analysis_metadata.llm_scores_used,
    hasSpecificLinks: response.analysis_metadata.specific_links_generated,
    hasContextualSources: response.analysis_metadata.contextual_recommendations,
    hasSafetyFallbacks: response.analysis_metadata.safety_fallbacks_used || false,
    isEnhancedAnalysis: response.analysis_metadata.llm_scores_used &&
                       response.analysis_metadata.specific_links_generated &&
                       response.analysis_metadata.contextual_recommendations,
    qualityScore: response.analysis_metadata.quality_score || 0,
    safetyBlocksEncountered: response.analysis_metadata.safety_blocks_encountered || 0
  };
};

export const getQualityIndicators = (response: AnalysisResponse) => {
  const features = getAnalysisFeatures(response);
  return {
    scoring_quality: features.hasLLMScoring ? 'HIGH' : 'MEDIUM',
    evidence_quality: features.hasSpecificLinks ? 'HIGH' : 'MEDIUM',
    source_quality: features.hasContextualSources ? 'HIGH' : 'MEDIUM',
    overall_quality: features.isEnhancedAnalysis ? 'HIGH' : 'MEDIUM',
    safety_handling: features.safetyBlocksEncountered > 0 ? 'FALLBACK_USED' : 'NORMAL',
    quality_score: features.qualityScore
  };
};

export const getVerificationSummary = (response: AnalysisResponse) => {
  const links = response.verification_links || [];
  const highQualityLinks = links.filter(link => link.quality_score >= 0.8);
  const institutions = [...new Set(links.map(link => link.institution).filter(Boolean))];
  
  return {
    total_links: links.length,
    high_quality_links: highQualityLinks.length,
    unique_institutions: institutions.length,
    top_institutions: institutions.slice(0, 3),
    has_evidence: links.length > 0
  };
};

// Export the axios instance for potential custom requests
export { api };

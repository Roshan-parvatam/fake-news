import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const API_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS || 360000); // default 6 minutes

const api = axios.create({
  baseURL: API_URL,
  timeout: API_TIMEOUT_MS, // allow long-running detailed analyses
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

// ✅ UPDATED: Enhanced verification links interface
export interface VerificationLink {
  claim: string;
  url: string;
  explanation: string;
  source_type: string;
  quality_score: number;
  institution?: string;  // ✅ New field from backend
  search_terms?: string; // ✅ New field from backend
  type?: string;         // ✅ Additional type field
}

// ✅ UPDATED: Enhanced contextual sources interface
export interface ContextualSource {
  name: string;
  details: string;
  relevance: string;
  contact_method: string;
  type: string;
  reliability_score: number;
  url?: string;          // ✅ New optional URL field
  relevance_score?: number; // ✅ New scoring field
}

export interface AnalysisResponse {
  classification: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number;
  claims: Claim[];
  context_scores: ContextScore;
  evidence: Evidence;
  recommended_sources: string[];
  // ✅ UPDATED: Enhanced verification links and contextual sources
  verification_links: VerificationLink[];
  contextual_sources: ContextualSource[];
  explanation: string;
  timestamp: string;
  // ✅ UPDATED: Enhanced analysis metadata
  analysis_metadata: {
    llm_scores_used: boolean;
    specific_links_generated: boolean;
    contextual_recommendations: boolean;
    response_time: number;
    config_version: string;
    safety_fallbacks_used?: boolean; // ✅ New safety field
    quality_score?: number;          // ✅ New quality field
    safety_blocks_encountered?: number; // ✅ New safety field
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

    if (!backendData.success || !backendData.results) {
      console.error('❌ Backend returned unsuccessful response:', backendData);
      throw new Error(backendData.errors?.[0] || 'Analysis failed - no results returned');
    }

    const results = backendData.results;
    console.log('📊 Results structure:', results);

    // ✅ COMPLETELY UPDATED: Enhanced data extraction with proper field mapping
    const transformedResponse: AnalysisResponse = {
      // Classification with safe access
      classification: (results.classification?.prediction || 'UNCERTAIN') as 'REAL' | 'FAKE' | 'UNCERTAIN',
      confidence: Math.round((results.classification?.confidence || 0) * 100),

      // ✅ UPDATED: Claims transformation with enhanced field mapping
      claims: (results.claims?.extracted_claims || []).map((claim: any) => {
        const claimText = claim.text || claim.claim_text || claim.claim || 'Unknown claim';
        const priority = typeof claim.priority === 'number'
          ? (claim.priority <= 1 ? 'HIGH' : claim.priority <= 2 ? 'MEDIUM' : 'LOW')
          : (claim.priority || 'MEDIUM');
        const verificationScore = claim.verifiability_score || claim.verification_score || 3;

        return {
          text: claimText,
          priority: priority as 'HIGH' | 'MEDIUM' | 'LOW',
          verification_status: (verificationScore >= 7 ? 'VERIFIED' :
            verificationScore >= 4 ? 'UNVERIFIED' : 'DISPUTED') as 'VERIFIED' | 'UNVERIFIED' | 'DISPUTED'
        };
      }),

      // ✅ UPDATED: Context scores using LLM scores when available
      context_scores: {
        // Use LLM scores if available, otherwise fall back to old structure
        bias: Math.round((results.context_analysis?.llm_scores?.bias ||
          results.context_analysis?.context_scores?.bias_score * 10 || 0)),
        manipulation: Math.round((results.context_analysis?.llm_scores?.manipulation ||
          results.context_analysis?.manipulation_report?.overall_manipulation_score * 10 || 0)),
        risk: Math.round((results.context_analysis?.llm_scores?.risk ||
          results.context_analysis?.context_scores?.overall_context_score * 10 || 50)),
        credibility: Math.round((results.context_analysis?.llm_scores?.credibility ||
          (100 - (results.context_analysis?.context_scores?.overall_context_score || 5) * 10)))
      },

      // ✅ UPDATED: Evidence with enhanced quality assessment
      evidence: {
        quality: (
          results.evidence?.quality_level === 'HIGH QUALITY' ? 'HIGH' :
          results.evidence?.quality_level === 'MEDIUM QUALITY' ? 'MEDIUM' :
          results.evidence?.quality_level === 'LOW QUALITY' ? 'LOW' : 'MEDIUM'
        ) as 'HIGH' | 'MEDIUM' | 'LOW',
        sources: results.evidence?.sources || [],
        confidence: Math.round(Math.max(0, Math.min(100, (results.evidence?.overall_evidence_score || 5) * 10)))
      },

      // ✅ COMPLETELY FIXED: Parse verification links from evidence evaluator with proper field mapping
      verification_links: (results.evidence?.verification_links || []).map((link: any) => ({
        claim: link.claim || link.text || 'Unknown claim',
        url: link.url || '#',
        explanation: link.explanation || link.search_terms || 'No explanation available',
        source_type: link.source_type || link.type || 'unknown',
        quality_score: typeof link.quality_score === 'number' ? link.quality_score : 0.5,
        institution: link.institution || 'Unknown Institution',
        search_terms: link.search_terms || '',
        type: link.type || 'verification'
      })),

      // ✅ COMPLETELY FIXED: Parse contextual sources from credible source agent with proper field mapping
      contextual_sources: (results.sources?.contextual_sources || []).map((source: any) => ({
        name: source.name || 'Unknown source',
        details: source.details || 'No details available',
        relevance: source.relevance || 'Not specified',
        contact_method: source.contact_method || 'Not specified',
        type: source.type || 'unknown',
        reliability_score: source.reliability_score || source.relevance_score || 5,
        url: source.url || undefined,
        relevance_score: source.relevance_score || 5
      })),

      // ✅ UPDATED: Enhanced recommended sources with better fallback logic
      recommended_sources: [
        // First try contextual sources URLs
        ...(results.sources?.contextual_sources || [])
          .map((s: any) => s.url)
          .filter(Boolean),
        // Then try top sources
        ...(results.sources?.top_sources || [])
          .map((s: any) => s.url || s.name)
          .filter(Boolean),
        // Finally fallback to defaults if nothing found
        ...(results.sources?.contextual_sources?.length > 0 || results.sources?.top_sources?.length > 0 ? [] : [
          'https://snopes.com',
          'https://factcheck.org',
          'https://politifact.com'
        ])
      ].slice(0, 5), // Limit to 5 sources

      // ✅ UPDATED: Enhanced explanation with better fallback logic
      explanation: 
        results.explanation?.text ||
        results.context_analysis?.llm_analysis ||
        results.explanation?.explanation ||
        'No detailed explanation available',

      // Timestamp
      timestamp: backendData.metadata?.timestamp || new Date().toISOString(),

      // ✅ COMPLETELY UPDATED: Enhanced analysis metadata with safety information
      analysis_metadata: {
        llm_scores_used: !!(results.context_analysis?.llm_scores),
        specific_links_generated: (results.evidence?.verification_links || []).length > 0,
        contextual_recommendations: (results.sources?.contextual_sources || []).length > 0,
        response_time: backendData.metadata?.processing_time_seconds || 0,
        config_version: backendData.metadata?.config_version || 'unknown',
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
      version: '3.1.0', // ✅ Updated version
      capabilities: [`${agent}-analysis`]
    }));
  } catch (error) {
    console.error('❌ Failed to fetch model configs:', error);
    return [];
  }
};

// ✅ UPDATED: Helper function to check if enhanced features are available
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

// ✅ UPDATED: Helper function to get quality indicators with safety awareness
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

// ✅ NEW: Helper function to get verification link summary
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

// ✅ Export the axios instance for potential custom requests
export { api };

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

export interface AnalysisResponse {
  classification: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number;
  claims: Claim[];
  context_scores: ContextScore;
  evidence: Evidence;
  recommended_sources: string[];
  explanation: string;
  timestamp: string;
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

    // Safe data extraction with comprehensive fallbacks
    const transformedResponse: AnalysisResponse = {
      // Classification with safe access
      classification: (results.classification?.prediction || 'UNCERTAIN') as 'REAL' | 'FAKE' | 'UNCERTAIN',
      confidence: Math.round((results.classification?.confidence || 0) * 100),
      
      // Claims transformation with safe defaults
      claims: (results.claims?.extracted_claims || []).map((claim: any) => {
        // Handle different claim formats
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
      
      // Context scores with proper mapping from backend structure
      context_scores: {
        // FIX: Invert scoring - higher backend score = LOWER credibility
        bias: Math.round((results.context_analysis?.context_scores?.bias_score || 0) * 10),
        manipulation: Math.round((results.context_analysis?.manipulation_report?.overall_manipulation_score || 0) * 10),
        // CRITICAL FIX: Invert risk calculation
        risk: Math.round(Math.max(0, Math.min(100, (results.context_analysis?.context_scores?.overall_context_score || 5) * 10))),
        // CRITICAL FIX: Invert credibility - lower context score = higher credibility
        credibility: Math.round(Math.max(0, Math.min(100, (10 - (results.context_analysis?.context_scores?.overall_context_score || 5)) * 10)))
      },
    
      
      // Evidence with safe mapping
      evidence: {
        quality: (results.evidence?.quality_level || 'MEDIUM') as 'HIGH' | 'MEDIUM' | 'LOW',
        sources: results.evidence?.sources || [],
        // Convert 0-10 evidence score to 0-100 confidence percentage
        confidence: Math.round(Math.max(0, Math.min(100, (results.evidence?.overall_score || 5) * 10)))
      },
      
      // Recommended sources
      recommended_sources: results.sources?.top_sources || results.sources?.recommended_sources || [
        'https://snopes.com',
        'https://factcheck.org',
        'https://politifact.com'
      ],
      
      // Explanation
      explanation: results.explanation?.text || results.explanation?.explanation || 'No detailed explanation available',
      
      // Timestamp
      timestamp: backendData.metadata?.timestamp || new Date().toISOString()
    };

    console.log('✅ Transformed response:', transformedResponse);
    return transformedResponse;

  } catch (error) {
    console.error('❌ Analysis failed:', error);
    
    // If it's an axios error, extract more details
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.message || 
                          error.message || 
                          'Network error occurred';
      throw new Error(errorMessage);
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
      version: '2.1.0',
      capabilities: [`${agent}-analysis`]
    }));
  } catch (error) {
    console.error('❌ Failed to fetch model configs:', error);
    return [];
  }
};

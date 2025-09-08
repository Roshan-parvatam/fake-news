import React from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Eye, AlertTriangle, TrendingUp, Shield, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ContextScore } from '@/services/api';

interface ScoreChartProps {
  scores: ContextScore;
  analysisMetadata?: {
    llm_scores_used?: boolean;
    scoring_method?: string;
    config_version?: string;
  };
}

// ✅ FIX: Define proper interface for score items
interface ScoreItem {
  label: string;
  value: number;
  icon: React.ComponentType<any>;
  color: string;
  description: string;
  category: 'positive' | 'negative'; // ✅ Proper string literal type
}

export const ScoreChart: React.FC<ScoreChartProps> = ({ scores, analysisMetadata }) => {
  console.log('📊 ScoreChart received scores:', scores);
  console.log('🔧 Analysis metadata:', analysisMetadata);

  // ✅ ENHANCED: Ensure scores have valid values with better defaults
  const safeScores = {
    bias: Math.min(Math.max(scores?.bias ?? 0, 0), 100),
    manipulation: Math.min(Math.max(scores?.manipulation ?? 0, 0), 100),
    risk: Math.min(Math.max(scores?.risk ?? 0, 0), 100),
    credibility: Math.min(Math.max(scores?.credibility ?? 50, 0), 100),
  };

  // ✅ ENHANCED: Check if using LLM-driven scoring
  const isLLMScoring = analysisMetadata?.llm_scores_used || false;
  const scoringMethod = analysisMetadata?.scoring_method || 'traditional';

  // ✅ FIX: Properly typed score list with explicit category values
  const scoreList: ScoreItem[] = [
    {
      label: 'Bias Level',
      value: safeScores.bias,
      icon: Eye,
      color: 'text-purple-600',
      description: isLLMScoring 
        ? 'AI-analyzed political or ideological bias (0-100 scale)'
        : 'Detected political or ideological slant',
      category: 'negative' as const // ✅ Explicit type assertion
    },
    {
      label: 'Manipulation',
      value: safeScores.manipulation,
      icon: AlertTriangle,
      color: 'text-orange-600',
      description: isLLMScoring
        ? 'AI-detected emotional manipulation techniques (0-100 scale)'
        : 'Detects emotional manipulation tactics',
      category: 'negative' as const // ✅ Explicit type assertion
    },
    {
      label: 'Risk Score',
      value: safeScores.risk,
      icon: TrendingUp,
      color: 'text-red-600',
      description: isLLMScoring
        ? 'AI-assessed overall misinformation risk (0-100 scale)'
        : 'Overall misinformation risk assessment',
      category: 'negative' as const // ✅ Explicit type assertion
    },
    {
      label: 'Credibility',
      value: safeScores.credibility,
      icon: Shield,
      color: 'text-green-600',
      description: isLLMScoring
        ? 'AI-evaluated source and content credibility (0-100 scale)'
        : 'Source and content credibility rating',
      category: 'positive' as const // ✅ Explicit type assertion
    },
  ];

  // ✅ FIX: Properly typed function parameters
  const getScoreColor = (value: number, category: 'positive' | 'negative'): string => {
    if (category === 'positive') {
      // For credibility: higher is better
      if (value >= 80) return 'text-green-600';
      if (value >= 60) return 'text-green-500';
      if (value >= 40) return 'text-yellow-500';
      if (value >= 20) return 'text-orange-500';
      return 'text-red-600';
    } else {
      // For bias, manipulation, risk: lower is better
      if (value <= 20) return 'text-green-600';
      if (value <= 40) return 'text-green-500';
      if (value <= 60) return 'text-yellow-500';
      if (value <= 80) return 'text-orange-500';
      return 'text-red-600';
    }
  };

  // ✅ FIX: Properly typed function parameters
  const getProgressColor = (value: number, category: 'positive' | 'negative'): string => {
    if (category === 'positive') {
      // For credibility: higher is better
      if (value >= 80) return 'bg-green-600';
      if (value >= 60) return 'bg-green-500';
      if (value >= 40) return 'bg-yellow-500';
      if (value >= 20) return 'bg-orange-500';
      return 'bg-red-600';
    } else {
      // For bias, manipulation, risk: lower is better
      if (value <= 20) return 'bg-green-600';
      if (value <= 40) return 'bg-green-500';
      if (value <= 60) return 'bg-yellow-500';
      if (value <= 80) return 'bg-orange-500';
      return 'bg-red-600';
    }
  };

  // ✅ FIX: Properly typed function parameters
  const getScoreInterpretation = (value: number, category: 'positive' | 'negative'): string => {
    if (category === 'positive') {
      if (value >= 80) return 'Excellent';
      if (value >= 60) return 'Good';
      if (value >= 40) return 'Fair';
      if (value >= 20) return 'Poor';
      return 'Very Poor';
    } else {
      if (value <= 20) return 'Excellent';
      if (value <= 40) return 'Good';
      if (value <= 60) return 'Moderate';
      if (value <= 80) return 'High';
      return 'Very High';
    }
  };

  return (
    <div className="space-y-4">
      {/* ✅ NEW: Enhanced scoring indicator */}
      {isLLMScoring && (
        <div className="flex items-center gap-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <Sparkles className="w-5 h-5 text-blue-600" />
          <div className="flex-1">
            <p className="text-sm font-medium text-blue-800">
              Enhanced AI-Driven Scoring
            </p>
            <p className="text-xs text-blue-600">
              Scores generated using advanced LLM analysis for consistent, explainable results
            </p>
          </div>
        </div>
      )}

      {/* Score cards grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {scoreList.map((item, idx) => {
          const Icon = item.icon;
          const scoreColor = getScoreColor(item.value, item.category);
          const progressColor = getProgressColor(item.value, item.category);
          const interpretation = getScoreInterpretation(item.value, item.category);

          return (
            <Card key={idx} className="flex flex-col hover:shadow-md transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className={cn("w-5 h-5", item.color)} />
                    <span className="font-semibold text-gray-900">{item.label}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={cn("text-xs px-2 py-1 rounded-full bg-gray-100", scoreColor)}>
                      {interpretation}
                    </span>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="pt-0">
                <p className="text-sm text-gray-600 mb-3">{item.description}</p>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">Score</span>
                    <span className={cn("text-lg font-bold", scoreColor)}>
                      {Math.round(item.value)}%
                    </span>
                  </div>
                  
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={cn("h-3 rounded-full transition-all duration-300", progressColor)}
                      style={{ width: `${item.value}%` }}
                    />
                  </div>
                  
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>{item.category === 'positive' ? 'Poor' : 'Low'}</span>
                    <span>{item.category === 'positive' ? 'Excellent' : 'High'}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* ✅ NEW: Scoring method info */}
      <div className="text-center">
        <p className="text-xs text-gray-500">
          {isLLMScoring 
            ? `Powered by AI-driven analysis • Method: ${scoringMethod} • All scores 0-100 scale`
            : 'Traditional scoring method • Scores may vary based on available data'
          }
        </p>
      </div>
    </div>
  );
};

export default ScoreChart;

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { AlertTriangle, Shield, TrendingUp, Eye } from 'lucide-react';
import { ContextScore } from '@/services/api';
import { cn } from '@/lib/utils';

interface ScoreChartProps {
  scores: ContextScore;
}

export const ScoreChart: React.FC<ScoreChartProps> = ({ scores }) => {
  console.log('ðŸ“Š ScoreChart received scores:', scores);

  // Ensure scores have valid values with defaults
  const safeScores = {
    bias: Math.max(0, Math.min(100, scores?.bias || 0)),
    manipulation: Math.max(0, Math.min(100, scores?.manipulation || 0)),
    risk: Math.max(0, Math.min(100, scores?.risk || 0)),
    credibility: Math.max(0, Math.min(100, scores?.credibility || 50))
  };

  const scoreItems = [
    {
      label: 'Bias Level',
      value: safeScores.bias,
      icon: Eye,
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10',
      description: 'Measures political or ideological slant',
      progressColor: 'bg-purple-500'
    },
    {
      label: 'Manipulation',
      value: safeScores.manipulation,
      icon: AlertTriangle,
      color: 'text-orange-500',
      bgColor: 'bg-orange-500/10',
      description: 'Detects emotional manipulation tactics',
      progressColor: 'bg-orange-500'
    },
    {
      label: 'Risk Score',
      value: safeScores.risk,
      icon: TrendingUp,
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
      description: 'Overall misinformation risk assessment',
      progressColor: 'bg-red-500'
    },
    {
      label: 'Credibility',
      value: safeScores.credibility,
      icon: Shield,
      color: 'text-green-500',
      bgColor: 'bg-green-500/10',
      description: 'Source and content credibility rating',
      progressColor: 'bg-green-500'
    },
  ];

  const getScoreColor = (value: number, isCredibility: boolean = false) => {
    if (isCredibility) {
      if (value >= 70) return 'text-green-500';
      if (value >= 40) return 'text-yellow-500';
      return 'text-red-500';
    } else {
      if (value <= 30) return 'text-green-500';
      if (value <= 60) return 'text-yellow-500';
      return 'text-red-500';
    }
  };

  const getProgressColor = (value: number, isCredibility: boolean = false) => {
    if (isCredibility) {
      if (value >= 70) return 'bg-green-500';
      if (value >= 40) return 'bg-yellow-500';
      return 'bg-red-500';
    } else {
      if (value <= 30) return 'bg-green-500';
      if (value <= 60) return 'bg-yellow-500';
      return 'bg-red-500';
    }
  };

  return (
    <div className="grid gap-4">
      {scoreItems.map((item, index) => {
        const Icon = item.icon;
        const isCredibility = item.label === 'Credibility';
        const scoreColor = getScoreColor(item.value, isCredibility);
        const progressColor = getProgressColor(item.value, isCredibility);

        return (
          <Card key={index} className="border-l-4 border-l-gray-200">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  <div className={cn("p-2 rounded-lg", item.bgColor)}>
                    <Icon className={cn("w-4 h-4", item.color)} />
                  </div>
                  <div>
                    <h4 className="font-medium">{item.label}</h4>
                    <p className="text-xs text-muted-foreground">{item.description}</p>
                  </div>
                </div>
                <div className={cn("text-lg font-bold", scoreColor)}>
                  {Math.round(item.value)}%
                </div>
              </div>
              
              <div className="space-y-2">
                <Progress 
                  value={item.value} 
                  className="h-2"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Low</span>
                  <span>High</span>
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
};
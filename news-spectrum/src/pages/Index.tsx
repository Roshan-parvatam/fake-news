import React, { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { AnalysisForm } from '@/components/analysis/AnalysisForm';
import { ResultsDashboard } from '@/components/analysis/ResultsDashboard';
import { ProgressIndicator } from '@/components/analysis/ProgressIndicator';
import { Shield, Sparkles, Zap, Brain } from 'lucide-react';
import { AnalysisResponse } from '@/services/api';

const Index = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);

  const handleAnalysisComplete = (data: AnalysisResponse) => {
    console.log('Analysis completed, received data:', data);
    setAnalysisResult(data);
    setIsAnalyzing(false);
  };

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning models detect patterns and anomalies'
    },
    {
      icon: Shield,
      title: 'Real-time Verification',
      description: 'Cross-reference claims with trusted fact-checking databases'
    },
    {
      icon: Zap,
      title: 'Instant Results',
      description: 'Get comprehensive analysis in under 2 minutes'
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      {/* Hero Section with gradient background */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 gradient-aurora opacity-30" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background/50 to-background" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
          <div className="text-center space-y-6">
            <div className="inline-flex items-center justify-center p-2 glass rounded-full mb-4">
              <Sparkles className="h-6 w-6 text-primary mr-2" />
              <span className="text-sm font-medium px-2">AI-Powered Fact Checking</span>
            </div>
            
            <h1 className="text-5xl md:text-6xl font-bold">
              <span className="gradient-text">Verify Truth</span>
              <br />
              <span className="text-foreground">in the Age of AI</span>
            </h1>
            
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Harness the power of advanced AI to analyze news articles, detect misinformation, 
              and make informed decisions with confidence.
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-6 mt-16 mb-16">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="glass p-6 rounded-xl hover:scale-105 transition-transform duration-300"
                >
                  <div className="inline-flex p-3 glass rounded-lg mb-4">
                    <Icon className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </div>
              );
            })}
          </div>

          {/* Analysis Form Section */}
          {!analysisResult && !isAnalyzing && (
            <div className="mt-12">
              <AnalysisForm
                onAnalysisComplete={handleAnalysisComplete}
                isAnalyzing={isAnalyzing}
                setIsAnalyzing={setIsAnalyzing}
              />
            </div>
          )}

          {/* Progress Indicator */}
          {isAnalyzing && (
            <div className="mt-12 glass rounded-xl p-8">
              <ProgressIndicator isAnalyzing={isAnalyzing} />
            </div>
          )}

          {/* Results Dashboard */}
          {analysisResult && !isAnalyzing && (
            <div className="mt-12">
              <div className="text-center mb-8">
                <button
                  onClick={() => {
                    setAnalysisResult(null);
                    setIsAnalyzing(false);
                  }}
                  className="text-sm text-primary hover:underline"
                >
                  ← Analyze another article
                </button>
              </div>
              <ResultsDashboard data={analysisResult} />
            </div>
          )}
          
        </div>
      </div>
    </div>
  );
};

export default Index;
import React, { useEffect, useState } from 'react';
import { Loader2, CheckCircle, Circle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProgressIndicatorProps {
  isAnalyzing: boolean;
}

const steps = [
  { id: 1, label: 'Extracting claims', duration: 2000 },
  { id: 2, label: 'Analyzing context', duration: 3000 },
  { id: 3, label: 'Verifying evidence', duration: 4000 },
  { id: 4, label: 'Generating report', duration: 2000 },
];

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ isAnalyzing }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);

  useEffect(() => {
    if (!isAnalyzing) {
      setCurrentStep(0);
      setCompletedSteps([]);
      return;
    }

    let timeout: NodeJS.Timeout;
    let totalTime = 0;

    const runStep = (index: number) => {
      if (index >= steps.length) {
        return;
      }

      setCurrentStep(index);
      totalTime += steps[index].duration;

      timeout = setTimeout(() => {
        setCompletedSteps(prev => [...prev, steps[index].id]);
        runStep(index + 1);
      }, steps[index].duration);
    };

    runStep(0);

    return () => {
      if (timeout) clearTimeout(timeout);
    };
  }, [isAnalyzing]);

  if (!isAnalyzing && completedSteps.length === 0) {
    return null;
  }

  return (
    <div className="w-full max-w-md mx-auto">
      <div className="space-y-4">
        {steps.map((step, index) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = currentStep === index;
          const isPending = !isCompleted && !isCurrent;

          return (
            <div
              key={step.id}
              className={cn(
                "flex items-center space-x-3 transition-all duration-500",
                isCurrent && "scale-105",
                isCompleted && "opacity-60"
              )}
            >
              <div className="relative">
                {isCompleted ? (
                  <CheckCircle className="h-6 w-6 text-green-500" />
                ) : isCurrent ? (
                  <div className="relative">
                    <Circle className="h-6 w-6 text-primary" />
                    <Loader2 className="h-6 w-6 text-primary absolute inset-0 animate-spin" />
                  </div>
                ) : (
                  <Circle className="h-6 w-6 text-muted-foreground" />
                )}
              </div>
              <span
                className={cn(
                  "text-sm font-medium transition-colors",
                  isCurrent && "text-primary",
                  isCompleted && "text-muted-foreground",
                  isPending && "text-muted-foreground/50"
                )}
              >
                {step.label}
              </span>
            </div>
          );
        })}
      </div>
      
      {isAnalyzing && (
        <div className="mt-6 text-center">
          <p className="text-sm text-muted-foreground animate-pulse">
            AI agents are working on your analysis...
          </p>
        </div>
      )}
    </div>
  );
};
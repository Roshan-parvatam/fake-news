import React, { useState } from 'react';
import { Search, Link2, FileText, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { analyzeContent, AnalysisRequest } from '@/services/api';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface AnalysisFormProps {
  onAnalysisComplete: (data: any) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (value: boolean) => void;
}

const sampleUrls = [
  'https://example.com/breaking-news-ai-breakthrough',
  'https://example.com/climate-change-report-2024',
  'https://example.com/political-statement-analysis',
];

export const AnalysisForm: React.FC<AnalysisFormProps> = ({
  onAnalysisComplete,
  isAnalyzing,
  setIsAnalyzing,
}) => {
  const [inputType, setInputType] = useState<'text' | 'url'>('text');
  const [textInput, setTextInput] = useState('');
  const [urlInput, setUrlInput] = useState('');
  const [detailedAnalysis, setDetailedAnalysis] = useState(true);
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (inputType === 'text' && !textInput.trim()) {
      toast({
        title: 'Error',
        description: 'Please enter some text to analyze',
        variant: 'destructive',
      });
      return;
    }
    
    if (inputType === 'text' && textInput.trim().length < 50) {
      toast({
        title: 'Error',
        description: 'Text must be at least 50 characters long for analysis',
        variant: 'destructive',
      });
      return;
    }
    
    if (inputType === 'url' && !urlInput.trim()) {
      toast({
        title: 'Error',
        description: 'Please enter a URL to analyze',
        variant: 'destructive',
      });
      return;
    }

    setIsAnalyzing(true);
    
    const request: AnalysisRequest = {
      ...(inputType === 'text' ? { text: textInput } : { url: urlInput }),
      detailed: detailedAnalysis,
    };

    try {
      console.log('Starting analysis with request:', request);
      const response = await analyzeContent(request);
      console.log('Analysis response received:', response);
      onAnalysisComplete(response);
      toast({
        title: 'Analysis Complete',
        description: 'Your content has been analyzed successfully',
      });
    } catch (error) {
      console.error('Analysis error:', error);
      toast({
        title: 'Analysis Failed',
        description: 'An error occurred during analysis. Please try again.',
        variant: 'destructive',
      });
      setIsAnalyzing(false);
    }
  };

  const handleSampleUrl = (url: string) => {
    setUrlInput(url);
    setInputType('url');
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-6">
        <Tabs value={inputType} onValueChange={(v) => setInputType(v as 'text' | 'url')}>
          <TabsList className="grid w-full grid-cols-2 glass">
            <TabsTrigger value="text" className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Text Analysis
            </TabsTrigger>
            <TabsTrigger value="url" className="flex items-center gap-2">
              <Link2 className="h-4 w-4" />
              URL Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="text" className="mt-6">
            <div className="space-y-4">
              <Textarea
                placeholder="Paste the article or text you want to analyze..."
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                className="min-h-[200px] glass resize-none"
                disabled={isAnalyzing}
              />
              <div className="text-sm text-muted-foreground">
                Tip: Paste news articles, social media posts, or any text content for verification
              </div>
            </div>
          </TabsContent>

          <TabsContent value="url" className="mt-6">
            <div className="space-y-4">
              <Input
                type="url"
                placeholder="https://example.com/news-article"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                className="glass"
                disabled={isAnalyzing}
              />
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">Try these sample URLs:</p>
                <div className="flex flex-wrap gap-2">
                  {sampleUrls.map((url, index) => (
                    <button
                      key={index}
                      type="button"
                      onClick={() => handleSampleUrl(url)}
                      className="text-xs px-3 py-1 glass rounded-md hover:bg-primary/10 transition-colors"
                      disabled={isAnalyzing}
                    >
                      Sample {index + 1}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        <div className="flex items-center justify-between glass-subtle rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Switch
              id="detailed"
              checked={detailedAnalysis}
              onCheckedChange={setDetailedAnalysis}
              disabled={isAnalyzing}
            />
            <Label htmlFor="detailed" className="cursor-pointer">
              Detailed Analysis
            </Label>
          </div>
          <span className="text-sm text-muted-foreground">
            {detailedAnalysis ? 'Full report with all metrics' : 'Quick summary only'}
          </span>
        </div>

        <Button
          type="submit"
          size="lg"
          disabled={isAnalyzing}
          className={cn(
            "w-full gradient-primary text-primary-foreground",
            "hover:opacity-90 transition-all duration-300",
            "disabled:opacity-50 disabled:cursor-not-allowed",
            isAnalyzing && "animate-pulse"
          )}
        >
          {isAnalyzing ? (
            <>
              <Sparkles className="mr-2 h-5 w-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Search className="mr-2 h-5 w-5" />
              Analyze Content
            </>
          )}
        </Button>
      </form>
    </div>
  );
};
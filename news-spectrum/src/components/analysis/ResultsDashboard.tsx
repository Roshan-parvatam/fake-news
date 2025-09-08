import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  Copy,
  Share2,
  Download,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Info,
  FileText,
  Search,
  Shield
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AnalysisResponse } from '@/services/api';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { ScoreChart } from './ScoreChart';
import { ClaimsList } from './ClaimsList';

interface ResultsDashboardProps {
  data: AnalysisResponse;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ data }) => {
  const [expandedExplanation, setExpandedExplanation] = useState(false);
  const { toast } = useToast();

  // Debug logging
  useEffect(() => {
    console.log('🎯 ResultsDashboard received data:', data);
    console.log('📊 Context scores:', data.context_scores);
    console.log('🔍 Claims:', data.claims);
    console.log('📈 Evidence:', data.evidence);
    console.log('🔗 Verification links:', data.verification_links); // ✅ Debug verification links
    console.log('🎯 Contextual sources:', data.contextual_sources); // ✅ Debug contextual sources
  }, [data]);

  const getClassificationIcon = () => {
    switch (data.classification) {
      case 'REAL':
        return <CheckCircle className="w-6 h-6" />;
      case 'FAKE':
        return <XCircle className="w-6 h-6" />;
      case 'UNCERTAIN':
        return <AlertTriangle className="w-6 h-6" />;
      default:
        return <AlertTriangle className="w-6 h-6" />;
    }
  };

  const getCredibilityStatus = (credibility: number) => {
    if (credibility >= 70) return { status: "HIGH", color: "text-green-500", bg: "bg-green-500/10", border: "border-green-500/20" };
    if (credibility >= 50) return { status: "MEDIUM", color: "text-yellow-500", bg: "bg-yellow-500/10", border: "border-yellow-500/20" };
    if (credibility >= 30) return { status: "LOW", color: "text-orange-500", bg: "bg-orange-500/10", border: "border-orange-500/20" };
    return { status: "VERY LOW", color: "text-red-500", bg: "bg-red-500/10", border: "border-red-500/20" };
  };

  const credibilityStatus = getCredibilityStatus(data.context_scores?.credibility || 0);

  const isValidHttpUrl = (value: string): boolean => {
    try {
      const u = new URL(value);
      return u.protocol === 'http:' || u.protocol === 'https:';
    } catch (_) {
      return false;
    }
  };

  const buildSearchUrl = (query: string): string => {
    return `https://www.google.com/search?q=${encodeURIComponent(query)}`;
  };

  const getClassificationColor = () => {
    switch (data.classification) {
      case 'REAL':
        return 'text-green-500 bg-green-500/10 border-green-500/20';
      case 'FAKE':
        return 'text-red-500 bg-red-500/10 border-red-500/20';
      case 'UNCERTAIN':
        return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/20';
      default:
        return 'text-gray-500 bg-gray-500/10 border-gray-500/20';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-500';
    if (confidence >= 60) return 'text-yellow-500';
    if (confidence >= 40) return 'text-orange-500';
    return 'text-red-500';
  };

  const copyResults = () => {
    const text = `TruthLens Analysis Results
Classification: ${data.classification}
Confidence: ${data.confidence}%
Credibility: ${credibilityStatus.status} (${data.context_scores?.credibility || 0}%)
Claims Found: ${data.claims?.length || 0}
Evidence Quality: ${data.evidence?.quality || 'N/A'}
Explanation: ${data.explanation}`;
    
    navigator.clipboard.writeText(text);
    toast({
      title: 'Copied to clipboard',
      description: 'Analysis results have been copied',
    });
  };

  const shareResults = () => {
    if (navigator.share) {
      navigator.share({
        title: 'TruthLens Analysis',
        text: `Check out this news analysis: ${data.classification} (${data.confidence}% confidence, ${credibilityStatus.status} credibility)`,
        url: window.location.href,
      });
    } else {
      copyResults();
    }
  };

  const downloadResults = () => {
    const results = {
      timestamp: data.timestamp,
      classification: data.classification,
      confidence: data.confidence,
      credibility_status: credibilityStatus,
      claims: data.claims,
      context_scores: data.context_scores,
      evidence: data.evidence,
      verification_links: data.verification_links, // ✅ Include verification links
      contextual_sources: data.contextual_sources, // ✅ Include contextual sources
      recommended_sources: data.recommended_sources,
      explanation: data.explanation
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `truthlens-analysis-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const truncatedExplanation = data.explanation && data.explanation.length > 200
    ? data.explanation.substring(0, 200) + '...'
    : data.explanation;

  // ✅ UPDATED: Enhanced quality validation checks with verification links awareness
  const qualityIssues: string[] = [];
  
  if (data.context_scores?.credibility < 30) {
    qualityIssues.push("Very low credibility score - requires human review");
  }
  
  // ✅ FIXED: Check for evidence sources - prioritize verification_links and contextual_sources
  const hasVerificationLinks = data.verification_links && data.verification_links.length > 0;
  const hasContextualSources = data.contextual_sources && data.contextual_sources.length > 0;
  const hasEvidenceSources = data.evidence?.sources && data.evidence.sources.length > 0;
  
  if (!hasVerificationLinks && !hasContextualSources && !hasEvidenceSources) {
    qualityIssues.push("No specific evidence sources provided");
  }
  
  if (!data.explanation || (!data.explanation.includes('##') && !data.explanation.includes('**'))) {
    qualityIssues.push("Explanation lacks proper formatting");
  }

  // Validate data before rendering
  if (!data) {
    return (
      <Card className="border-red-200">
        <CardContent className="p-6">
          <div className="text-center text-red-600">
            <AlertTriangle className="w-12 h-12 mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">No Analysis Data</h3>
            <p className="text-sm">No analysis results to display.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Custom markdown components with proper JSX syntax
  const markdownComponents = {
    h1: ({ children, ...props }: any) => (
      <h1 className="text-xl font-bold mb-3 text-foreground" {...props}>
        {children}
      </h1>
    ),
    h2: ({ children, ...props }: any) => (
      <h2 className="text-lg font-semibold mb-2 mt-4 text-foreground border-b border-border pb-2" {...props}>
        {children}
      </h2>
    ),
    h3: ({ children, ...props }: any) => (
      <h3 className="text-base font-medium mb-2 mt-3 text-foreground" {...props}>
        {children}
      </h3>
    ),
    ul: ({ children, ...props }: any) => (
      <ul className="list-disc list-inside mb-3 space-y-1" {...props}>
        {children}
      </ul>
    ),
    ol: ({ children, ...props }: any) => (
      <ol className="list-decimal list-inside mb-3 space-y-1" {...props}>
        {children}
      </ol>
    ),
    li: ({ children, ...props }: any) => (
      <li className="text-sm leading-relaxed" {...props}>
        {children}
      </li>
    ),
    strong: ({ children, ...props }: any) => (
      <strong className="font-semibold text-foreground" {...props}>
        {children}
      </strong>
    ),
    em: ({ children, ...props }: any) => (
      <em className="italic text-muted-foreground" {...props}>
        {children}
      </em>
    ),
    blockquote: ({ children, ...props }: any) => (
      <blockquote className="border-l-4 border-primary pl-4 italic my-4" {...props}>
        {children}
      </blockquote>
    ),
    table: ({ children, ...props }: any) => (
      <table className="w-full border-collapse border border-border rounded-lg overflow-hidden my-4" {...props}>
        {children}
      </table>
    ),
    thead: ({ children, ...props }: any) => (
      <thead className="bg-muted" {...props}>
        {children}
      </thead>
    ),
    tbody: ({ children, ...props }: any) => (
      <tbody {...props}>
        {children}
      </tbody>
    ),
    tr: ({ children, ...props }: any) => (
      <tr className="border-b border-border" {...props}>
        {children}
      </tr>
    ),
    th: ({ children, ...props }: any) => (
      <th className="border border-border px-4 py-2 text-left font-semibold" {...props}>
        {children}
      </th>
    ),
    td: ({ children, ...props }: any) => (
      <td className="border border-border px-4 py-2 text-sm" {...props}>
        {children}
      </td>
    ),
    p: ({ children, ...props }: any) => (
      <p className="text-sm leading-relaxed mb-3" {...props}>
        {children}
      </p>
    ),
    code: ({ children, ...props }: any) => (
      <code className="bg-muted px-1 py-0.5 rounded text-xs font-mono" {...props}>
        {children}
      </code>
    ),
    pre: ({ children, ...props }: any) => (
      <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-xs" {...props}>
        {children}
      </pre>
    ),
  };

  return (
    <div className="space-y-6">
      {/* Action Buttons */}
      <div className="flex justify-end space-x-2">
        <Button variant="outline" size="sm" onClick={copyResults}>
          <Copy className="w-4 h-4 mr-2" />
          Copy
        </Button>
        <Button variant="outline" size="sm" onClick={shareResults}>
          <Share2 className="w-4 h-4 mr-2" />
          Share
        </Button>
        <Button variant="outline" size="sm" onClick={downloadResults}>
          <Download className="w-4 h-4 mr-2" />
          Download
        </Button>
      </div>

      {/* Quality Issues Warning */}
      {qualityIssues.length > 0 && (
        <Alert variant="default" className="border-yellow-500/50 bg-yellow-500/10">
          <AlertTriangle className="h-4 w-4 text-yellow-500" />
          <AlertTitle className="text-yellow-700 dark:text-yellow-300">Quality Issues Detected</AlertTitle>
          <AlertDescription className="text-yellow-600 dark:text-yellow-400">
            <ul className="list-disc list-inside mt-2 space-y-1">
              {qualityIssues.map((issue, idx) => (
                <li key={idx} className="text-sm">{issue}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* Credibility Warning Banner */}
      {(credibilityStatus.status === "LOW" || credibilityStatus.status === "VERY LOW") && (
        <Alert variant="destructive" className="mb-4">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Low Credibility Content Detected</AlertTitle>
          <AlertDescription>
            This content has <strong>{credibilityStatus.status}</strong> credibility ({data.context_scores?.credibility || 0}%). 
            <br />
            <strong>⚠️ Recommendation:</strong> Verify through multiple independent sources before sharing or acting on this information.
          </AlertDescription>
        </Alert>
      )}

      {/* Main Classification Result */}
      <Card className={cn("border-2", getClassificationColor())}>
        <CardHeader>
          <div className="flex items-center space-x-3">
            {getClassificationIcon()}
            <div className="flex-1">
              <CardTitle className="text-2xl">
                {data.classification}
              </CardTitle>
              <CardDescription>
                Analysis completed at {new Date(data.timestamp).toLocaleString()}
                {/* ✅ ADD: Show if enhanced features were used */}
                {data.analysis_metadata && (
                  <div className="flex items-center space-x-2 mt-1">
                    {data.analysis_metadata.llm_scores_used && (
                      <Badge variant="outline" className="text-xs">
                        <Shield className="w-3 h-3 mr-1" />
                        AI-Enhanced
                      </Badge>
                    )}
                    {data.analysis_metadata.specific_links_generated && (
                      <Badge variant="outline" className="text-xs">
                        <Search className="w-3 h-3 mr-1" />
                        Verification Links
                      </Badge>
                    )}
                  </div>
                )}
              </CardDescription>
            </div>
            <div className="text-right">
              <Badge className={cn(credibilityStatus.bg, credibilityStatus.color, credibilityStatus.border, "border")}>
                {credibilityStatus.status} CREDIBILITY
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">Classification Confidence</span>
                <span className={cn("text-sm font-medium", getConfidenceColor(data.confidence))}>
                  {data.confidence}%
                </span>
              </div>
              <Progress value={data.confidence} className="h-3" />
            </div>
            
            {/* Credibility Indicator */}
            {data.context_scores?.credibility !== undefined && (
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Content Credibility</span>
                  <div className="flex items-center space-x-2">
                    <span className={cn("text-sm font-medium", credibilityStatus.color)}>
                      {credibilityStatus.status}
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {data.context_scores.credibility}%
                    </span>
                  </div>
                </div>
                <Progress 
                  value={data.context_scores.credibility} 
                  className={cn(
                    "h-3",
                    data.context_scores.credibility < 30 ? "[&>div]:bg-red-500" :
                    data.context_scores.credibility < 50 ? "[&>div]:bg-orange-500" :
                    data.context_scores.credibility < 70 ? "[&>div]:bg-yellow-500" : "[&>div]:bg-green-500"
                  )} 
                />
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Overview Information */}
      <Card>
        <CardContent className="p-6">
          <div className="text-sm text-muted-foreground space-y-3">
            <p>
              This report classifies the article as <span className="font-medium text-foreground">{data.classification}</span>
              {' '}with <span className="font-medium">{data.confidence}% confidence</span>
              {' '}and <span className={cn("font-medium", credibilityStatus.color)}>{credibilityStatus.status} credibility</span> ({data.context_scores?.credibility || 0}%).
              {/* ✅ ADD: Show verification summary */}
              {data.verification_links && data.verification_links.length > 0 && (
                <span className="ml-2 text-green-700 dark:text-green-400">
                  <strong>{data.verification_links.length} verification sources</strong> found.
                </span>
              )}
            </p>
            <p>
              Context scores reflect potential bias and manipulation indicators (higher % means more of that signal). 
              Risk is an inverse of credibility, while Credibility estimates overall trustworthiness of the content.
            </p>
            {(credibilityStatus.status === "LOW" || credibilityStatus.status === "VERY LOW") && (
              <p className="text-orange-600 dark:text-orange-400 font-medium">
                ⚠️ Due to low credibility, we recommend additional fact-checking through independent sources.
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Analysis Tabs */}
      <Tabs defaultValue="claims" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="claims">
            Claims ({data.claims?.length || 0})
          </TabsTrigger>
          <TabsTrigger value="context">Context</TabsTrigger>
          <TabsTrigger value="evidence">
            Evidence
            {data.verification_links && data.verification_links.length > 0 && (
              <Badge variant="secondary" className="ml-1 text-xs">
                {data.verification_links.length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="sources">Sources</TabsTrigger>
        </TabsList>

        <TabsContent value="claims">
          <Card>
            <CardHeader>
              <CardTitle>Extracted Claims</CardTitle>
              <CardDescription>
                Key claims identified in the content for fact-checking
              </CardDescription>
            </CardHeader>
            <CardContent>
              {data.claims && data.claims.length > 0 ? (
                <ClaimsList claims={data.claims} />
              ) : (
                <div className="text-center py-8">
                  <Info className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">No claims were extracted from this content.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="context">
          <Card>
            <CardHeader>
              <CardTitle>Context Analysis</CardTitle>
              <CardDescription>
                Analysis of bias, manipulation tactics, and credibility indicators
              </CardDescription>
            </CardHeader>
            <CardContent>
              {data.context_scores ? (
                <ScoreChart scores={data.context_scores} />
              ) : (
                <div className="text-center py-8">
                  <Info className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">Context analysis data not available.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="evidence">
          <Card>
            <CardHeader>
              <CardTitle>Evidence Quality Assessment</CardTitle>
              <CardDescription>
                Evaluation of supporting evidence and source quality
                {data.verification_links && data.verification_links.length > 0 && (
                  <span className="ml-2 text-green-600">
                    • {data.verification_links.length} verification sources found
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Overall Quality Assessment */}
                <div className="flex justify-between items-center">
                  <span className="font-medium">Overall Quality</span>
                  <Badge variant={
                    data.evidence?.quality === 'HIGH' ? 'default' :
                    data.evidence?.quality === 'MEDIUM' ? 'secondary' : 'destructive'
                  }>
                    {data.evidence?.quality || 'UNKNOWN'}
                  </Badge>
                </div>
                
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Evidence Confidence</span>
                    <span className="text-sm text-muted-foreground">
                      {data.evidence?.confidence || 0}%
                    </span>
                  </div>
                  <Progress value={data.evidence?.confidence || 0} className="h-2" />
                </div>

                {/* ✅ COMPLETELY FIXED: Verification Links Display */}
                {data.verification_links && data.verification_links.length > 0 ? (
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <span className="font-medium">Verification Sources</span>
                      <Badge variant="outline" className="text-xs">
                        {data.verification_links.length} sources
                      </Badge>
                    </div>
                    <div className="space-y-3">
                      {data.verification_links.slice(0, 5).map((link, index) => (
                        <div key={index} className="flex items-start space-x-3 p-4 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors border">
                          <ExternalLink className="w-5 h-5 text-muted-foreground mt-0.5 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between mb-2">
                              <div className="flex-1">
                                <h4 className="font-medium text-sm text-foreground mb-1">
                                  {link.institution || link.source_type || 'Unknown Institution'}
                                </h4>
                                <p className="text-xs text-muted-foreground mb-2 line-clamp-2">
                                  <strong>Claim:</strong> {(link.claim || 'No claim specified').substring(0, 100)}
                                  {link.claim && link.claim.length > 100 && '...'}
                                </p>
                                {link.explanation && (
                                  <p className="text-xs text-muted-foreground mb-2 line-clamp-2">
                                    <strong>Details:</strong> {link.explanation.substring(0, 120)}
                                    {link.explanation.length > 120 && '...'}
                                  </p>
                                )}
                              </div>
                              <div className="text-right ml-4">
                                <div className="text-xs text-muted-foreground mb-1">
                                  Quality: {Math.round((link.quality_score || 0.5) * 100)}%
                                </div>
                                <Badge variant="outline" className="text-xs">
                                  {link.source_type || 'Unknown'}
                                </Badge>
                              </div>
                            </div>
                            <div className="flex items-center justify-between">
                              {link.url && link.url !== '#' ? (
                                <a 
                                  href={link.url} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-primary hover:text-primary/80 text-xs flex items-center"
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  Verify Source
                                </a>
                              ) : (
                                <span className="text-xs text-muted-foreground flex items-center">
                                  <Info className="w-3 h-3 mr-1" />
                                  Source reference only
                                </span>
                              )}
                              {link.search_terms && (
                                <a 
                                  href={buildSearchUrl(link.search_terms)} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-muted-foreground hover:text-foreground text-xs flex items-center"
                                >
                                  <Search className="w-3 h-3 mr-1" />
                                  Search
                                </a>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                      
                      {data.verification_links.length > 5 && (
                        <div className="text-center py-2">
                          <p className="text-xs text-muted-foreground">
                            +{data.verification_links.length - 5} more verification sources available
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                ) : data.evidence?.sources && data.evidence.sources.length > 0 ? (
                  <div>
                    <span className="font-medium mb-3 block">Supporting Evidence Sources:</span>
                    <div className="space-y-2">
                      {data.evidence.sources.map((source, index) => (
                        <div key={index} className="flex items-start space-x-3 p-3 bg-muted/50 rounded-lg">
                          <ExternalLink className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            {isValidHttpUrl(source) ? (
                              <a 
                                href={source} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="text-sm text-primary hover:underline break-all"
                              >
                                {source}
                              </a>
                            ) : (
                              <>
                                <span className="text-sm font-medium block">{source}</span>
                                <a 
                                  href={buildSearchUrl(source)} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-xs text-muted-foreground hover:underline"
                                >
                                  Search for verification →
                                </a>
                              </>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-6 bg-muted/30 rounded-lg">
                    <AlertTriangle className="w-8 h-8 mx-auto text-muted-foreground mb-2" />
                    <p className="text-sm text-muted-foreground">
                      No specific evidence sources were identified for verification.
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      This may indicate the content lacks verifiable claims or sources.
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sources">
          <Card>
            <CardHeader>
              <CardTitle>Recommended Verification Sources</CardTitle>
              <CardDescription>
                Verify this information with these trusted fact-checking resources
                {/* ✅ ADD: Show contextual sources count */}
                {data.contextual_sources && data.contextual_sources.length > 0 && (
                  <span className="ml-2 text-green-600">
                    • {data.contextual_sources.length} contextual sources found
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* ✅ FIXED: Display contextual sources first */}
                {data.contextual_sources && data.contextual_sources.length > 0 ? (
                  <div>
                    <h4 className="font-medium mb-3 text-sm">Contextual Sources:</h4>
                    <div className="space-y-3">
                      {data.contextual_sources.slice(0, 6).map((source, index) => (
                        <div key={index} className="flex items-start space-x-3 p-4 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors border">
                          <ExternalLink className="w-5 h-5 text-muted-foreground mt-0.5 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between mb-2">
                              <div className="flex-1">
                                <h5 className="font-medium text-sm text-foreground mb-1">
                                  {source.name?.replace(/\*\*/g, '') || 'Unknown Source'}
                                </h5>
                                <p className="text-xs text-muted-foreground mb-2">
                                  {source.details?.replace(/\*\*/g, '').substring(0, 150) || 'No details available'}
                                  {source.details && source.details.length > 150 && '...'}
                                </p>
                              </div>
                              <div className="text-right ml-4">
                                <Badge variant="outline" className="text-xs mb-1">
                                  {source.type?.replace('contextual_', '').toUpperCase() || 'GENERAL'}
                                </Badge>
                                <div className="text-xs text-muted-foreground">
                                  Score: {source.relevance_score || source.reliability_score || 'N/A'}
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center justify-between">
                              {source.url && source.url !== 'Contact information not specified' && !source.url.includes('not specified') ? (
                                <a 
                                  href={source.url.replace(/[\[\]]/g, '')} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-primary hover:text-primary/80 text-xs flex items-center"
                                >
                                  <ExternalLink className="w-3 h-3 mr-1" />
                                  Visit Source
                                </a>
                              ) : (
                                <span className="text-xs text-muted-foreground flex items-center">
                                  <Info className="w-3 h-3 mr-1" />
                                  Contact via institution
                                </span>
                              )}
                              <span className="text-xs text-muted-foreground">
                                Relevance: {source.relevance || 'Contextually selected'}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                      
                      {data.contextual_sources.length > 6 && (
                        <div className="text-center py-2">
                          <p className="text-xs text-muted-foreground">
                            +{data.contextual_sources.length - 6} more contextual sources available
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                ) : data.recommended_sources && data.recommended_sources.length > 0 ? (
                  <div>
                    <h4 className="font-medium mb-3 text-sm">Recommended Sources:</h4>
                    <div className="space-y-3">
                      {data.recommended_sources.map((src: any, index: number) => {
                        // Support either string URLs or objects with name/url
                        let displayText = '';
                        let href: string | null = null;
                        let description = '';
                        
                        if (typeof src === 'string') {
                          displayText = src;
                          if (src.startsWith('http')) {
                            try { 
                              href = src; 
                              displayText = new URL(src).hostname.replace('www.', ''); 
                            } catch (_) { 
                              href = null; 
                            }
                          }
                        } else if (src && typeof src === 'object') {
                          const urlCandidate = src.url || src.link || src.href || '';
                          const nameCandidate = src.name || src.title || src.source || urlCandidate || 'Source';
                          displayText = nameCandidate;
                          description = src.description || src.explanation || '';
                          if (typeof urlCandidate === 'string' && urlCandidate.startsWith('http')) {
                            try { 
                              href = urlCandidate; 
                              if (!displayText.includes('http')) {
                                displayText = new URL(urlCandidate).hostname.replace('www.', '');
                              }
                            } catch (_) { 
                              href = null; 
                            }
                          }
                        }
                        
                        return (
                          <div key={index} className="flex items-start space-x-3 p-4 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
                            <ExternalLink className="w-5 h-5 text-muted-foreground mt-0.5 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              {href ? (
                                <a
                                  href={href}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-primary hover:underline font-medium block break-all"
                                >
                                  {displayText}
                                </a>
                              ) : (
                                <>
                                  <span className="font-medium block">{displayText}</span>
                                  <a
                                    href={buildSearchUrl(displayText)}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-sm text-muted-foreground hover:underline"
                                  >
                                    Search for verification →
                                  </a>
                                </>
                              )}
                              {description && (
                                <p className="text-sm text-muted-foreground mt-1">{description}</p>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Info className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground mb-4">No specific sources recommended for this content.</p>
                    <div className="space-y-2">
                      <p className="text-sm text-muted-foreground">Try these general fact-checking resources:</p>
                      <div className="flex flex-wrap gap-2 justify-center">
                        {[
                          { name: 'Snopes', url: 'https://snopes.com' },
                          { name: 'FactCheck.org', url: 'https://factcheck.org' },
                          { name: 'PolitiFact', url: 'https://politifact.com' }
                        ].map((resource, idx) => (
                          <a
                            key={idx}
                            href={resource.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs px-3 py-1 bg-primary/10 text-primary rounded-full hover:bg-primary/20 transition-colors"
                          >
                            {resource.name}
                          </a>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* ✅ FIXED: AI Explanation with ReactMarkdown */}
      <Card className="glass">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Detailed Analysis Explanation
          </CardTitle>
          <CardDescription>
            Comprehensive breakdown of the analysis with reasoning and evidence
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {data.explanation ? (
            expandedExplanation ? (
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={markdownComponents}
                >
                  {data.explanation}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={markdownComponents}
                >
                  {truncatedExplanation}
                </ReactMarkdown>
              </div>
            )
          ) : (
            <p className="text-muted-foreground text-center py-4">
              No detailed explanation available for this analysis.
            </p>
          )}
          
          {data.explanation && data.explanation.length > 200 && (
            <div className="flex justify-center pt-4 border-t">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setExpandedExplanation(!expandedExplanation)}
                className="text-primary"
              >
                {expandedExplanation ? (
                  <>
                    <ChevronUp className="w-4 h-4 mr-2" />
                    Show Less
                  </>
                ) : (
                  <>
                    <ChevronDown className="w-4 h-4 mr-2" />
                    Read Full Analysis
                  </>
                )}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

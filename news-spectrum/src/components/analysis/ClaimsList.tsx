import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, CheckCircle2, XCircle } from 'lucide-react';
import { Claim } from '@/services/api';
import { cn } from '@/lib/utils';

interface ClaimsListProps {
  claims: Claim[];
  verificationLinks?: any[]; // Add verification links for enhanced display
}

export const ClaimsList: React.FC<ClaimsListProps> = ({ claims, verificationLinks = [] }) => {
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'HIGH':
        return 'destructive';
      case 'MEDIUM':
        return 'secondary';
      case 'LOW':
        return 'outline';
      default:
        return 'outline';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'VERIFIED':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'DISPUTED':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'UNVERIFIED':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'VERIFIED':
        return 'text-green-500 bg-green-500/10';
      case 'DISPUTED':
        return 'text-red-500 bg-red-500/10';
      case 'UNVERIFIED':
        return 'text-yellow-500 bg-yellow-500/10';
      default:
        return '';
    }
  };

  const getVerificationSourceCount = (claimText: string) => {
    if (!verificationLinks || verificationLinks.length === 0) return 0;
    
    // Count verification sources that match this claim
    return verificationLinks.filter(link => 
      link.claim && link.claim.toLowerCase().includes(claimText.toLowerCase().substring(0, 20))
    ).length;
  };

  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle>Extracted Claims Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {claims.map((claim, index) => (
            <div
              key={index}
              className="p-4 glass-subtle rounded-lg hover:bg-primary/5 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(claim.verification_status)}
                  <Badge variant={getPriorityColor(claim.priority)}>
                    {claim.priority} PRIORITY
                  </Badge>
                  {/* ✅ NEW: Show verification source count */}
                  {getVerificationSourceCount(claim.text) > 0 && (
                    <Badge variant="outline" className="text-xs">
                      {getVerificationSourceCount(claim.text)} source{getVerificationSourceCount(claim.text) !== 1 ? 's' : ''}
                    </Badge>
                  )}
                </div>
                <span className={cn(
                  "text-xs px-2 py-1 rounded-full font-medium",
                  getStatusColor(claim.verification_status)
                )}>
                  {claim.verification_status}
                </span>
              </div>
              <p className="text-sm leading-relaxed">
                {claim.text}
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
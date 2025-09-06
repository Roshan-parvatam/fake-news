import React from 'react';
import { Header } from '@/components/layout/Header';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Shield, Users, Target, Lightbulb, Globe } from 'lucide-react';

const About = () => {
  const values = [
    {
      icon: Shield,
      title: 'Truth & Accuracy',
      description: 'We prioritize factual accuracy above all else, using multiple verification methods.'
    },
    {
      icon: Users,
      title: 'Accessibility',
      description: 'Making fact-checking tools available to everyone, regardless of technical expertise.'
    },
    {
      icon: Target,
      title: 'Transparency',
      description: 'Our analysis process is clear and explainable, showing how we reach conclusions.'
    },
  ];



  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <div className="relative">
        <div className="absolute inset-0 gradient-aurora opacity-20" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center mb-16">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">
              About <span className="gradient-text">TruthLens</span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Empowering journalists, researchers, and citizens with AI-powered tools 
              to combat misinformation and promote media literacy.
            </p>
          </div>

          {/* Mission Section */}
          <Card className="glass mb-12">
            <CardHeader>
              <div className="flex items-center space-x-3 mb-2">
                <Lightbulb className="h-6 w-6 text-primary" />
                <CardTitle className="text-2xl">Our Mission</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-lg leading-relaxed">
                In an era of information overload and sophisticated disinformation campaigns, 
                TruthLens stands as a beacon of clarity. We leverage cutting-edge artificial 
                intelligence to analyze news content, identify potential misinformation, and 
                provide users with the tools they need to make informed decisions.
              </p>
            </CardContent>
          </Card>

          {/* Values Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            {values.map((value, index) => {
              const Icon = value.icon;
              return (
                <Card key={index} className="glass hover:scale-105 transition-transform">
                  <CardHeader>
                    <div className="inline-flex p-3 glass rounded-lg mb-3">
                      <Icon className="h-6 w-6 text-primary" />
                    </div>
                    <CardTitle className="text-xl">{value.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription>{value.description}</CardDescription>
                  </CardContent>
                </Card>
              );
            })}
          </div>



          {/* Technology Section */}
          <Card className="glass">
            <CardHeader>
              <div className="flex items-center space-x-3 mb-2">
                <Globe className="h-6 w-6 text-primary" />
                <CardTitle className="text-2xl">Our Technology</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="leading-relaxed">
                TruthLens employs a sophisticated multi-agent AI system that combines:
              </p>
              <ul className="space-y-3 ml-6">
                <li className="flex items-start">
                  <span className="text-primary mr-2">•</span>
                  <span>Natural Language Processing for claim extraction and context analysis</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">•</span>
                  <span>Deep learning models trained on millions of verified articles</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">•</span>
                  <span>Real-time cross-referencing with trusted fact-checking databases</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary mr-2">•</span>
                  <span>Bias detection algorithms to identify political or ideological slants</span>
                </li>
              </ul>
              <p className="leading-relaxed mt-4">
                Our system continuously learns and improves, adapting to new forms of 
                misinformation as they emerge.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default About;
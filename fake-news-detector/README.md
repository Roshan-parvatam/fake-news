# Fake News Detection System

A multi-agent hybrid AI system for detecting fake news using BERT and LLM models.

## Features
- BERT-based classification
- LLM explanations
- Credible source recommendations
- Multi-agent orchestration with LangGraph

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. API Key Configuration

The system requires a Gemini API key for LLM-based agents. You have two options:

#### Option A: Environment File (Recommended)
Create a `.env` file in the `fake-news-detector` directory:
```bash
# Create .env file
touch fake-news-detector/.env

# Add your API key
echo "GEMINI_API_KEY=your_api_key_here" >> fake-news-detector/.env
```

#### Option B: Environment Variable
```bash
export GEMINI_API_KEY=your_api_key_here
```

#### Getting Your API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and add it to your configuration

### 3. Verify Setup
Run the system to verify everything is working:
```bash
cd fake-news-detector
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage
Coming soon...

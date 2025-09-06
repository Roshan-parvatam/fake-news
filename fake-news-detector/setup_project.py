"""
Enhanced Project Setup Script for Fake News Detection System
Updated to reflect all improvements, fixes, and current system state

This script creates the complete project structure with all the latest enhancements:
- All 6 agents with proper configuration integration
- Centralized model configuration system
- Fixed prompt template mappings
- Enhanced orchestration with LangGraph
- Complete backend API with FastAPI
- Comprehensive testing framework
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """
    Creates the complete project directory structure with all enhancements
    """
    
    # Define the project structure - updated to match current enhanced structure
    structure = {
        'agents': ['__init__.py', 'preprocessor_agent.py'],
        'agents/base': ['__init__.py', 'base_agent.py'],
        'agents/bert_classifier': ['__init__.py', 'classifier.py', 'model_utils.py', 'preprocessing.py'],
        'agents/claim_extractor': ['__init__.py', 'extractor_agent.py', 'parsers.py', 'patterns.py'],
        'agents/context_analyzer': ['__init__.py', 'analyzer_agent.py', 'bias_patterns.py', 'manipulation_detection.py'],
        'agents/credible_source': ['__init__.py', 'domain_classifier.py', 'source_agent.py', 'source_database.py'],
        'agents/evidence_evaluator': ['__init__.py', 'criteria.py', 'evaluator_agent.py', 'fallacy_detection.py'],
        'agents/llm_explanation': ['__init__.py', 'explanation_agent.py', 'source_database.py'],
        'backend': ['__init__.py', 'main.py'],
        'config': ['__init__.py', 'model_configs.py', 'prompts_config.py', 'settings.py'],
        'data/raw': [],
        'data/processed': [],
        'data/training_logs': [],
        'models': ['__init__.py'],
        'models/bert_fake_news': ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'training_args.bin', 'training_metadata.json', 'model.safetensors'],
        'models/saved_models': [],
        'notebooks': ['fake_news_bert_training.ipynb'],
        'orchestration': ['__init__.py', 'langgraph_workflow.py', 'nodes.py', 'state.py'],
        'tests': ['__init__.py', 'run_all_tests.py', 'test_bert_agent.py', 'test_claim_agent.py', 'test_config.py', 'test_context_agent.py', 'test_evidence_agent.py', 'test_llm_agent.py', 'test_source_agent.py', 'testapi.py'],
        'utils': ['__init__.py', 'config.py', 'helpers.py', 'logger.py', 'url_scraper.py'],
        'cache': [],
        'logs': []
    }
    
    # Get current directory
    base_dir = Path.cwd()
    project_name = "fake-news-detector"
    
    # Check if we're already in the project directory
    if base_dir.name == project_name:
        project_dir = base_dir
        print(f"✅ Already in project directory: {project_dir}")
    else:
        project_dir = base_dir / project_name
        project_dir.mkdir(exist_ok=True)
        print(f"✅ Created project directory: {project_dir}")
    
    # Create directory structure
    for folder, files in structure.items():
        folder_path = project_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created folder: {folder}")
        
        # Create files in the folder
        for file in files:
            file_path = folder_path / file
            if not file_path.exists():
                if file.endswith('.py'):
                    # Create Python files with basic content
                    with open(file_path, 'w') as f:
                        f.write(f'"""\n{file} - Part of Enhanced Fake News Detection System\n"""\n\n')
                elif file.endswith('.ipynb'):
                    # Create empty Jupyter notebooks
                    notebook_content = {
                        "cells": [],
                        "metadata": {},
                        "nbformat": 4,
                        "nbformat_minor": 4
                    }
                    import json
                    with open(file_path, 'w') as f:
                        json.dump(notebook_content, f, indent=2)
                elif file.endswith('.json'):
                    # Create empty JSON files
                    with open(file_path, 'w') as f:
                        f.write('{}\n')
                elif file.endswith('.txt'):
                    # Create empty text files
                    file_path.touch()
                else:
                    # Create empty files
                    file_path.touch()
                print(f"📄 Created file: {folder}/{file}")
    
    # Create root level files with enhanced content
    root_files = {
        'README.md': """# Enhanced Fake News Detection System

A sophisticated multi-agent hybrid AI system for detecting fake news using BERT and LLM models with centralized configuration management.

## 🚀 Features

### Core Capabilities
- **BERT-based Classification**: Neural network-based fake news detection
- **LLM Explanations**: Human-readable explanations using Gemini models
- **Credible Source Recommendations**: AI-powered source verification guidance
- **Evidence Evaluation**: Systematic evidence quality assessment
- **Context Analysis**: Bias detection and manipulation analysis
- **Claim Extraction**: Automated claim identification and prioritization

### System Architecture
- **6 Specialized Agents**: Modular agent system with specialized components
  - BERT Classifier: Neural network-based classification
  - Claim Extractor: Extracts and prioritizes verifiable claims
  - Context Analyzer: Analyzes context and detects bias/manipulation
  - Credible Source: Evaluates source credibility and provides recommendations
  - Evidence Evaluator: Evaluates evidence quality and logical consistency
  - LLM Explanation: Provides comprehensive human-readable explanations
- **Smart Orchestration**: LangGraph-based workflow with conditional routing
- **Centralized Configuration**: Single-point model and prompt management
- **FastAPI Backend**: RESTful API with comprehensive endpoints
- **Performance Optimization**: Smart routing to reduce costs and processing time

### Configuration Management
- **Centralized Model Config**: Change models across entire system from one file
- **Prompt Template Management**: Centralized prompt versioning and A/B testing
- **Environment-Specific Settings**: Development, production, and testing configs
- **Runtime Overrides**: Dynamic configuration updates without restarts

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- API keys for Gemini (required) and OpenAI (optional)

### Installation
1. **Clone and setup project structure**:
   ```bash
   python setup_project.py
   cd fake-news-detector
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

6. **Run tests**:
   ```bash
   python tests/run_all_tests.py
   ```

## 🚀 Usage

### Starting the Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints
- `POST /analyze` - Analyze an article for fake news
- `GET /model-configs` - View current model configurations
- `PUT /update-model-config` - Update model configurations
- `GET /metrics` - System performance metrics
- `GET /docs` - Interactive API documentation

### Example Usage
```python
import requests

# Analyze an article
response = requests.post("http://localhost:8000/analyze", json={
    "text": "Your article text here...",
    "url": "https://example.com/article",
    "detailed": True
})

result = response.json()
print(f"Classification: {result['bert_results']['prediction']}")
print(f"Confidence: {result['confidence_scores']['bert']:.2%}")
print(f"Explanation: {result['final_explanation']}")
```

## ⚙️ Configuration

### Model Configuration
All models can be changed from `config/model_configs.py`:
```python
# Change from gemini-1.5-pro to gemini-1.5-flash
"model_name": "gemini-1.5-flash"
```

### Environment Configuration
- **Development**: Enhanced debugging and faster processing
- **Production**: Optimized for performance and reliability
- **Testing**: Minimal processing for fast test execution

## 🧪 Testing

Run the comprehensive test suite:
```bash
python tests/run_all_tests.py
```

Individual agent tests:
```bash
python tests/test_bert_agent.py
python tests/test_claim_agent.py
python tests/test_context_agent.py
python tests/test_evidence_agent.py
python tests/test_llm_agent.py
python tests/test_source_agent.py
```

## 📊 Performance

### Smart Routing
- **High-confidence REAL news**: Fast-track processing
- **Uncertain cases**: Full 6-agent analysis
- **Cost optimization**: Skip expensive processing when appropriate

### Metrics
- Processing times per agent
- API call counts and costs
- Confidence scores and accuracy metrics
- System health and performance indicators

## 🔧 Troubleshooting

### Common Issues
1. **Context Analyzer not initializing**: Check agent_name assignment
2. **Prompt template errors**: Verify variable mappings in config
3. **Model configuration not updating**: Restart server after config changes
4. **API key errors**: Verify GEMINI_API_KEY in .env file

### Debug Mode
Enable debug mode in `config/settings.py`:
```python
enable_debug_mode = True
log_level = "DEBUG"
```

## 📈 Roadmap

- [ ] Additional LLM model support (Claude, GPT-4)
- [ ] Real-time fact-checking integration
- [ ] Advanced bias detection algorithms
- [ ] Multi-language support
- [ ] Web interface for non-technical users
- [ ] Batch processing capabilities
- [ ] Advanced analytics dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for LLM capabilities
- Hugging Face for BERT models
- LangGraph for orchestration framework
- FastAPI for the web framework
""",
        '.env.example': """# API Keys (Required)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
DEFAULT_MODEL=gemini-1.5-flash
BERT_MODEL_PATH=./models/bert_fake_news/
SAVED_MODELS_PATH=./models/saved_models/

# Database (Optional)
DATABASE_URL=sqlite:///./fake_news_db.sqlite

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/fake_news_detection.log
ENABLE_CONSOLE_LOGGING=true
ENABLE_FILE_LOGGING=true

# Cache Configuration
CACHE_DIR=./cache/
CACHE_TTL=3600
ENABLE_CACHING=true

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
RELOAD=true

# Performance Settings
MAX_WORKERS=4
DEFAULT_TIMEOUT=300
MAX_RETRIES=3
RETRY_DELAY=1.0

# Rate Limiting
GEMINI_RATE_LIMIT=4.0
OPENAI_RATE_LIMIT=1.0

# Quality Thresholds
MIN_CONFIDENCE_THRESHOLD=0.6
HIGH_CONFIDENCE_THRESHOLD=0.8
EVIDENCE_QUALITY_THRESHOLD=6.0
BIAS_DETECTION_THRESHOLD=5.0

# Feature Flags
ENABLE_DETAILED_ANALYSIS=true
ENABLE_CROSS_VERIFICATION=true
ENABLE_METRICS_COLLECTION=true
ENABLE_DEBUG_MODE=false
""",
        'requirements.txt': """# Core Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0
python-multipart==0.0.6

# Machine Learning & AI
torch==2.1.1
transformers==4.36.0
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.1.4
google-generativeai==0.3.2

# NLP Processing
spacy==3.7.2
nltk==3.8.1
textblob==0.17.1

# LangGraph and LangChain
langgraph==0.0.20
langchain==0.1.0
langchain-community==0.0.10
langchain-google-genai==0.0.6

# Web Scraping & HTTP
requests==2.31.0
beautifulsoup4==4.12.2
selenium==4.15.2
aiohttp==3.9.1

# Database & Storage
sqlalchemy==2.0.23
alembic==1.13.0
aiofiles==23.2.1

# Testing Framework
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2

# Development Tools
black==23.11.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.6.0

# Jupyter & Notebooks
jupyter==1.0.0
ipykernel==6.26.0
notebook==7.0.6

# Utilities
tqdm==4.66.1
rich==13.7.0
click==8.1.7
typer==0.9.0

# Monitoring & Metrics
prometheus-client==0.19.0
psutil==5.9.6

# Security
cryptography==41.0.8
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
""",
        'pyproject.toml': """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fake-news-detector"
version = "2.0.0"
description = "Enhanced multi-agent fake news detection system with centralized configuration"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.1",
    "pytest-cov>=4.1.0",
    "black>=23.11.0",
    "flake8>=6.1.0",
    "mypy>=1.7.1",
    "pre-commit>=3.6.0",
]
docs = [
    "mkdocs>=1.5.3",
    "mkdocs-material>=9.4.8",
    "mkdocstrings[python]>=0.24.0",
]

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311', 'py312']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--cov=agents",
    "--cov=backend",
    "--cov=config",
    "--cov=orchestration",
    "--cov=utils",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.run]
source = ["agents", "backend", "config", "orchestration", "utils"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/venv/*",
    "*/env/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
""",
        'Makefile': """# Enhanced Fake News Detection System Makefile

.PHONY: help install dev test lint format clean run setup

help: ## Show this help message
	@echo "Enhanced Fake News Detection System"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

setup: ## Initial project setup
	@echo "🚀 Setting up Enhanced Fake News Detection System..."
	python setup_project.py
	@echo "✅ Project structure created!"
	@echo "📋 Next steps:"
	@echo "  1. cd fake-news-detector"
	@echo "  2. make install"
	@echo "  3. make setup-env"

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

install-dev: ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	pip install -r requirements.txt
	pip install -e ".[dev]"
	@echo "✅ Development dependencies installed!"

setup-env: ## Setup environment and download models
	@echo "🔧 Setting up environment..."
	python -m spacy download en_core_web_sm
	@echo "📝 Creating .env file..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "⚠️  Please edit .env with your API keys"; fi
	@echo "✅ Environment setup complete!"

test: ## Run all tests
	@echo "🧪 Running test suite..."
	python tests/run_all_tests.py

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest tests/ -m "not integration" -v

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	pytest tests/ -m "integration" -v

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	pytest --cov=agents --cov=backend --cov=config --cov=orchestration --cov=utils --cov-report=html --cov-report=term

lint: ## Run linting
	@echo "🔍 Running linting..."
	flake8 agents backend config orchestration utils tests
	mypy agents backend config orchestration utils

format: ## Format code
	@echo "🎨 Formatting code..."
	black agents backend config orchestration utils tests
	@echo "✅ Code formatted!"

format-check: ## Check code formatting
	@echo "🔍 Checking code formatting..."
	black --check agents backend config orchestration utils tests

clean: ## Clean up temporary files
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "✅ Cleanup complete!"

run: ## Run the development server
	@echo "🚀 Starting development server..."
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

run-prod: ## Run the production server
	@echo "🚀 Starting production server..."
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4

docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "📚 Serving documentation..."
	mkdocs serve

config-check: ## Check configuration
	@echo "⚙️  Checking configuration..."
	python -c "from config.settings import get_settings; from config.model_configs import get_model_config; print('✅ Configuration loaded successfully!')"

model-info: ## Show current model configuration
	@echo "🤖 Current model configuration:"
	python -c "from config.model_configs import get_model_config; import json; print(json.dumps({agent: config.get('model_name', 'Unknown') for agent in ['llm_explanation', 'credible_source', 'claim_extractor', 'context_analyzer', 'evidence_evaluator']}, indent=2))"

update-models: ## Update all models to gemini-1.5-flash
	@echo "🔄 Updating all models to gemini-1.5-flash..."
	python -c "from config.model_configs import update_model_config; agents = ['llm_explanation', 'credible_source', 'claim_extractor', 'context_analyzer', 'evidence_evaluator']; [update_model_config(agent, model_name='gemini-1.5-flash') for agent in agents]; print('✅ All models updated to gemini-1.5-flash!')"

update-models-pro: ## Update all models to gemini-1.5-pro
	@echo "🔄 Updating all models to gemini-1.5-pro..."
	python -c "from config.model_configs import update_model_config; agents = ['llm_explanation', 'credible_source', 'claim_extractor', 'context_analyzer', 'evidence_evaluator']; [update_model_config(agent, model_name='gemini-1.5-pro') for agent in agents]; print('✅ All models updated to gemini-1.5-pro!')"

health-check: ## Check system health
	@echo "🏥 Checking system health..."
	@echo "Testing API endpoint..."
	@curl -s http://localhost:8000/metrics > /dev/null && echo "✅ API is running" || echo "❌ API is not responding"
	@echo "Testing model configuration..."
	@python -c "from config.model_configs import get_model_config; print('✅ Model config loaded')" 2>/dev/null && echo "✅ Model configuration OK" || echo "❌ Model configuration error"

docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t fake-news-detector:latest .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p 8000:8000 --env-file .env fake-news-detector:latest

all: clean install setup-env test lint format ## Run everything: clean, install, setup, test, lint, format
	@echo "🎉 All tasks completed successfully!"
""",
        'Dockerfile': """# Enhanced Fake News Detection System Dockerfile

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        build-essential \\
        curl \\
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p logs cache data/raw data/processed models/saved_models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/metrics || exit 1

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
        'docker-compose.yml': """version: '3.8'

services:
  fake-news-detector:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
      - DEBUG=false
    volumes:
      - ./logs:/app/logs
      - ./cache:/app/cache
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add a database service
  # postgres:
  #   image: postgres:15
  #   environment:
  #     POSTGRES_DB: fake_news_db
  #     POSTGRES_USER: fake_news_user
  #     POSTGRES_PASSWORD: fake_news_password
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   ports:
  #     - "5432:5432"

# volumes:
#   postgres_data:
""",
        '.gitignore': """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/#use-with-ide
.pdm.toml

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be added to the global gitignore or merged into this project gitignore.  For a PyCharm
#  project, it is recommended to include the following files:
#  .idea/
#  *.iml
#  *.ipr
#  *.iws
.idea/

# VS Code
.vscode/

# Project specific
logs/*.log
cache/*
data/raw/*.csv
data/processed/*.csv
models/saved_models/*
!models/saved_models/.gitkeep
!data/raw/.gitkeep
!data/processed/.gitkeep
!cache/.gitkeep
!logs/.gitkeep

# API keys and secrets
.env
.env.local
.env.production
secrets.json

# Temporary files
*.tmp
*.temp
.DS_Store
Thumbs.db

# Docker
.dockerignore
"""
    }
    
    for filename, content in root_files.items():
        file_path = project_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"📄 Created root file: {filename}")
    
    # Create .gitkeep files for empty directories
    gitkeep_dirs = ['models/saved_models', 'data/raw', 'data/processed', 'data/training_logs', 'cache', 'logs']
    for dir_path in gitkeep_dirs:
        gitkeep_path = project_dir / dir_path / '.gitkeep'
        gitkeep_path.touch()
        print(f"📌 Created .gitkeep in: {dir_path}")
    
    print(f"\n🎉 Enhanced Fake News Detection System setup complete!")
    print(f"📍 Project location: {project_dir}")
    print(f"\n🚀 System Features:")
    print(f"  ✅ All 6 agents with proper configuration integration")
    print(f"  ✅ Centralized model configuration (change models from one file)")
    print(f"  ✅ Fixed prompt template mappings")
    print(f"  ✅ Enhanced orchestration with LangGraph")
    print(f"  ✅ Complete FastAPI backend with comprehensive endpoints")
    print(f"  ✅ Smart conditional routing for cost optimization")
    print(f"  ✅ Comprehensive testing framework")
    print(f"  ✅ Docker support with health checks")
    print(f"  ✅ Development tools and automation")
    
    print(f"\n📋 Quick Start:")
    print(f"  1. cd {project_name}")
    print(f"  2. make install")
    print(f"  3. make setup-env")
    print(f"  4. make run")
    print(f"\n📋 Available Commands:")
    print(f"  make help          - Show all available commands")
    print(f"  make test          - Run comprehensive test suite")
    print(f"  make model-info    - Show current model configuration")
    print(f"  make update-models - Update all models to gemini-1.5-flash")
    print(f"  make health-check  - Check system health")
    print(f"\n🌐 API Documentation: http://localhost:8000/docs")
    print(f"📊 Metrics Endpoint: http://localhost:8000/metrics")

if __name__ == "__main__":
    create_project_structure()
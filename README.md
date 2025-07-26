# 🚀 InnovARAG: Innovation Discovery Multi-Agent RAG Platform

An advanced AI-powered platform that combines Retrieval-Augmented Generation (RAG) with Multi-Agent Systems to discover innovation opportunities, analyze market risks, and provide strategic insights about companies, patents, and technology trends.

## 🌟 Key Features

### 🧠 **Enhanced Multi-Agent Workflow**
- **PlanningAgent**: Intelligently decomposes complex queries into focused subquestions
- **NormalizeQueryAgent**: Classifies queries and selects appropriate tools (company/patent/general)
- **GeneralizeAgent**: Synthesizes information from multiple sources with source attribution
- **Market Analysts**: Comprehensive opportunity and risk analysis with confidence scoring
- **FactCheckingAgent**: Validates responses for accuracy and flags potential issues

### 🔍 **Hybrid Retrieval System**
- **Dense + Sparse Search**: Combines vector embeddings with BM25 for optimal recall
- **Source Attribution**: Every answer includes proper citations and confidence scores
- **Intelligent Tool Selection**: Automatically chooses between exact lookup and RAG retrieval

### 🎯 **Multiple Query Modes**
- **Company Analysis**: Business focus, market opportunities, competitive positioning
- **Patent Analysis**: Technology details, patent portfolios, innovation trends
- **General Innovation**: Industry trends, market landscapes, emerging technologies
- **Freestyle Query**: Complex multi-faceted questions with automatic decomposition

### 🔧 **Professional Infrastructure**
- **Centralized Configuration**: LLM types configurable per agent in `config/agent_config.py`
- **Comprehensive Logging**: Detailed progress tracking and error handling
- **Modular Architecture**: Easy to extend with new agents and tools
- **Multiple Interfaces**: CLI, Chat mode, and Streamlit web interface

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/DarynBang/Innov-A-RAG.git
cd InnovARAG

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI, Google AI, etc.)
```

### 2. Data Ingestion

```bash
# Ingest company and patent data
python main.py --mode ingest --force_reindex

# This will:
# - Process patent data and create embeddings
# - Process company data and create embeddings  
# - Create RAG_INDEX directory with all indexed data
# - Build enhanced data mappings
```

### 3. Configure LLMs

Edit `config/agent_config.py` to configure which LLM each agent uses:

```python
agent_config = {
    "planning_agent": "openai",           # "gemini"
    "normalize_query_agent": "openai", 
    "generalize_agent": "openai",
    "market_opportunity_agent": "openai",
    "market_risk_agent": "openai",
    "market_manager_agent": "openai",
    "fact_checking_agent": "openai",
}
```

### 4. Run Queries

```bash
# Or with custom query
python main.py --mode query --query "Tell me about TechNova's business focus and market opportunities"

# Interactive chat mode
python main.py --mode chat

# Test mode with default query
python main.py --mode test

# Start interactive chat
# This will open your browser to: http://localhost:8501
streamlit run streamlit_app.py

```

## 💻 Usage Examples

### Company Analysis
```bash
python main.py --mode query --query "What are TechNova's business focus and competitive advantages?"
```

### Patent Analysis  
```bash
python main.py --mode query --query "What is the technology behind Patent 273556553 and its market potential?"
```

### Complex Multi-Company Analysis
```bash
python main.py --mode query --query "Compare TechNova and InnovateCorp's AI patent portfolios and market strategies"
```

### Industry Trend Analysis
```bash
python main.py --mode query --query "What are the emerging trends in renewable energy patents and which companies are leading?"
```

## 🌐 Streamlit Web Interface

Launch the comprehensive web interface:

```bash
streamlit run streamlit_app.py
```

### Features:
- **Interactive Query Modes**: Choose between Company, Patent, General, or Freestyle analysis
- **Real-time Workflow Visualization**: See each agent's progress and outputs
- **Result Export**: Download complete results as JSON or summary as text
- **Query History**: Track previous analyses
- **System Configuration**: View and monitor system status

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PlanningAgent  │ -> │ NormalizeAgent  │ -> │ GeneralizeAgent │
│  Query Analysis │    │  + Tool Invoke  │    │   Synthesis     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                |
                                v
                    ┌─────────────────────────┐
                    │     Information         │
                    │      Retrieval          │
                    │   (Hybrid RAG +         │
                    │    Tool Selection)      │
                    └─────────────────────────┘
                                |
                                v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MarketOpportunity│ -> │  MarketRisk     │ -> │ MarketManager   │
│     Agent       │    │    Agent        │    │    Agent        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                |
                                v
                    ┌─────────────────────────┐
                    │   FactCheckingAgent     │
                    │   (Validation &         │
                    │   Confidence Scoring)   │
                    └─────────────────────────┘
```

## 📁 Project Structure

```
InnovARAG/
├── agents/                          # Multi-agent system
│   ├── planning_agent.py           # Query decomposition
│   ├── normalize_query_agent.py    # Query classification + tools
│   ├── generalize_agent.py         # Information synthesis
│   ├── fact_checking_agent.py      # Response validation
│   ├── multi_agent_runner.py       # Workflow orchestration
│   ├── registry.py                 # Agent registry
│   └── market_analysts/
│       ├── market_opportunity_agent.py
│       ├── market_risk_agent.py
│       └── market_manager_agent.py
├── config/                          # Configuration
│   ├── agent_config.py             # LLM configurations per agent
│   ├── prompts.py                  # Centralized prompt templates
│   └── rag_config.py               # RAG system settings
├── tools/                           # LangChain tools
│   ├── company_tools.py            # Company lookup & RAG
│   ├── patent_tools.py             # Patent lookup & RAG
│   └── hybrid_rag_tools.py         # Hybrid retrieval tools
├── retrieval/                       # Enhanced retrieval system
│   └── hybrid_retriever.py         # Dense + Sparse retrieval
├── utils/                           # Utilities
│   └── logging_utils.py            # Centralized logging
├── text_generation/                 # LLM runners
│   ├── openai_runner.py
│   ├── gemini_runner.py
│   └── qwen_runner.py
├── main.py                          # CLI interface
├── streamlit_app.py                 # Web interface
├── firm_summary_rag.py             # Company RAG system
├── patent_rag.py                    # Patent RAG system
└── requirements.txt                 # Dependencies
```

## 🔧 Configuration

### LLM Configuration
Edit `config/agent_config.py`:
- **DEFAULT_LLM_TYPE**: Global default LLM
- **agent_config**: Per-agent LLM specification

### RAG Configuration  
Edit `config/rag_config.py`:
- **Embedding models**: HuggingFace embeddings configuration
- **Chunk sizes**: Text processing parameters
- **Index settings**: Vector database configuration

### Prompt Templates
All prompts are centralized in `config/prompts.py`:
- **Chain-of-Thought reasoning**
- **Source attribution requirements**
- **Confidence scoring guidelines**
- **Examples for each agent**

## 📊 Output Features

### Comprehensive Analysis
- **Source Attribution**: `[Company: TechNova]`, `[Patent: 273556553]`
- **Confidence Scores**: 1-10 reliability ratings
- **Fact-Check Validation**: Accuracy verification
- **Multi-dimensional Analysis**: Opportunities, risks, strategic recommendations

### Export Options
- **JSON**: Complete workflow results with metadata
- **Text Summary**: Executive-friendly analysis summary
- **Structured Data**: Query history and performance metrics

## 🔍 Advanced Features

### Hybrid Retrieval
- **Dense Retrieval**: Semantic vector search using HuggingFace embeddings
- **Sparse Retrieval**: BM25 keyword matching for exact terms
- **Score Fusion**: Weighted combination with normalization
- **Source Tracking**: Complete provenance for all retrieved information

### Query Intelligence
- **Automatic Decomposition**: Complex queries split into focused subquestions
- **Tool Selection**: Intelligent choice between exact lookup and RAG retrieval
- **Context Merging**: Seamless integration of multiple information sources

### Quality Assurance
- **Fact Checking**: Validates claims against source material
- **Confidence Assessment**: Multi-criteria reliability scoring
- **Error Detection**: Flags contradictions and potential hallucinations

## 🚦 Usage Modes

| Mode | Description | Command |
|------|-------------|---------|
| **ingest** | Data ingestion and indexing | `python main.py --mode ingest` |
| **query** | Single query processing | `python main.py --mode query --query "..."` |
| **test** | Test with default query | `python main.py --mode test` |
| **chat** | Interactive chat mode | `python main.py --mode chat` |
| **streamlit** | Web interface | `streamlit run streamlit_app.py` |

## 🛠️ Development

### Adding New Agents
1. Create agent class inheriting from `BaseAgent`
2. Implement the `run()` method
3. Add prompt templates to `config/prompts.py`
4. Register in `agents/registry.py`
5. Configure LLM in `config/agent_config.py`

### Adding New Tools
1. Create tool in `tools/` directory
2. Follow LangChain tool interface
3. Register in `MultiAgentRunner.register_tools()`

### Custom Prompts
All prompts are in `config/prompts.py` with:
- System prompts for each agent
- User prompt templates
- Examples and guidelines
- Source attribution requirements

## 📝 Environment Variables

Create `.env` file:
```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Google AI (Gemini)
GOOGLE_API_KEY=your_google_api_key

```


## 🚀 Example Queries for Testing Optimized Tools

### **Basic Company & Patent Lookups**
```
1. "What company information do we have for Intel?" (OK)
   - Tests: exact_company_lookup (exact match priority)
   
2. "Tell me about Patent 498964606" (OK)
   - Tests: exact_patent_lookup
   
3. "What patents does Lattice Semiconductor have?" (OK)
   - Tests: company_patents_lookup (improved display)
```

### **Optimized Hybrid Search Queries**
```
4. "Find artificial intelligence innovations and patents" (OK)
   - Tests: optimized_hybrid_rag_retrieval with advanced caching and FAISS
   
5. "What are the latest machine learning developments in tech companies?" (OK)
   - Tests: optimized hybrid search with metadata filtering
   
6. "Compare semiconductor patent strategies across different companies" (OK)
   - Tests: optimized cross-company analysis with performance monitoring
```

### **Batch Processing & Performance Queries**
```
7. "Analyze innovation trends in: AI, machine learning, robotics, biotech" (OK BUT NOT USE BATCH!)
   - Tests: optimized_retrieval for multiple related queries
   
8. "Show me performance analytics of the current system" (OK)
   - Tests: get_performance_analytics tool
   
9. "What are Lattice Semiconductor's competitive advantages and market opportunities?" (OK)
   - Tests: context preservation and optimized retrieval
```

### **Complex Analysis Queries**
```
10. "Analyze the competitive landscape for biotech companies in drug development" (OK BUT TOO LONG)
    - Tests: industry analysis with optimized search
    
11. "Find machine learning patents and their commercial applications" (OK BUT TOO LONG)
    - Tests: technology analysis with advanced caching
    
12. "What are the innovation strategies of top semiconductor companies?" (OK)
    - Tests: multi-company analysis with performance optimization
```

### **Edge Cases & Exact Matching**
```
13. "Intel" (exact company name) (OK)
    - Tests: exact match priority (should find Intel exactly, not partial matches)
    
14. "Tell me about companies with 'tech' in their name" (OK)
    - Tests: substring matching with proper fallback
    
15. "What patents are owned by Advanced Biomedical Technologies?" (OK)
    - Tests: exact company name matching and patent lookup
```

### **Performance Testing Queries**

```
1.  "Clear all caches and show performance before/after"
    - Tests: cache management and performance monitoring
    
2.  Multiple rapid queries to test caching effectiveness
    - Tests: cache hit rates and response time improvements
    
3.  Large batch query processing
    - Tests: parallel processing and efficiency gains
```

## 🔧 **Testing Instructions**

1. **Initialize System**: Use sidebar to initialize with your preferred LLM
2. **Test Basic Tools**: Start with exact lookups to verify data access
3. **Test Optimized Features**: Use Enhanced Features tab → Optimized Hybrid Search
4. **Monitor Performance**: Use Performance Analytics tab to see metrics
5. **Test Batch Processing**: Use Batch Processing tab for multiple queries
6. **Verify Fixes**: 
   - Company lookup now prioritizes exact matches
   - Company RAG retrieval shows correct Hojin IDs
   - Company patents lookup displays properly formatted results

## 📊 **Expected Results**

- **Faster responses** due to caching and FAISS optimization
- **Better accuracy** with exact match prioritization  
- **Rich metadata** including relevance scores, confidence levels
- **Performance metrics** showing cache hit rates and query statistics
- **Proper error handling** with helpful user guidance
- **Visual notifications** with clear success/error/completion messages

## 🚀 **Latest Improvements & Performance Enhancements**

### **📈 Performance Optimizations**

#### **⚡ High-Performance Search System**
- **12x Parallel Workers**: Increased from 4 to 12 for maximum parallelization
- **Redis Caching**: Distributed cache with 2-hour TTL and fallback to disk/memory
- **Semantic Caching**: Finds similar queries using 85% cosine similarity threshold
- **Query Preprocessing**: Normalizes queries and removes stop phrases for better performance
- **Memory Optimization**: Smart eviction based on system memory usage with `psutil` monitoring
- **Async Batch Processing**: Process multiple queries simultaneously with memory management

### **🎯 Tool Selection Improvements**

#### **Exact Lookup Tools** (High-confidence sources)
- `exact_company_lookup`: Complete company profiles with business focus and keywords
- `exact_patent_lookup`: Full patent specifications with abstracts and technical details  
- `company_patents_lookup`: Complete IP portfolios for specific companies

#### **Semantic Search Tools** (Discovery and themes)
- `company_rag_retrieval`: Industry landscape discovery by characteristics
- `patent_rag_retrieval`: Technology research and innovation pattern analysis

#### **Comprehensive Analysis** (Primary tool for complex queries)
- `optimized_hybrid_rag_retrieval`: Cross-domain search across companies AND patents
- `batch_optimized_retrieval`: Parallel processing for multiple related queries
- `get_performance_analytics`: System performance monitoring and metrics

### **🔍 Data Quality Fixes**

#### **Patent Data Column Mapping**
- **Fixed Column Names**: Updated from old `abstract` → new `patent_abstract`
- **Complete Patent Information**: Now returns full abstracts and metadata
- **Enhanced Patent Details**: Added filing dates, publication numbers, and dates
- **Proper Data Source**: All components now use `data/us_patent202506.csv` (1.58M patents)

### **📊 Performance Metrics**

#### **Configuration Updates**
```python
# Enhanced SearchConfig
SearchConfig(
    max_workers=12,  # ↑ from 4
    confidence_threshold=0.7,  # ↓ from 0.9 for faster early stopping
    batch_size=50,  # NEW: batch processing size
    memory_optimization=True  # NEW: smart memory management
)

# Enhanced CacheConfig  
CacheConfig(
    use_redis=True,  # ↑ with fallback to disk/memory
    memory_cache_size=2000,  # ↑ from 1000
    cache_ttl=7200,  # ↑ 2 hours for better hit rates
    semantic_cache_threshold=0.85  # NEW: semantic similarity
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with proper logging and error handling
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

## 👥 Authors

This project was developed by:
- **[Nguyen Quang Phu (pdz1804)](https://github.com/pdz1804)** and Tieu Tri Bang.

## 📞 Support

For questions, issues, or contributions:
- 🐛 **Issues**: [GitHub Issues](https://github.com/pdz1804/InnovARAG/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/pdz1804/InnovARAG/discussions)
- 📧 **Email**: [quangphunguyen1804@gmail.com](mailto:quangphunguyen1804@gmail.com)

---




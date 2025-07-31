# 🚀 InnovARAG: Innovation Discovery Multi-Agent RAG Platform

An advanced AI-powered platform that combines Retrieval-Augmented Generation (RAG) with Multi-Agent Systems to discover innovation opportunities, analyze market risks, and provide strategic insights about companies, patents, and technology trends.

## 🌟 Key Features

<details>
<summary><strong>🧠 Enhanced Multi-Agent Workflow</strong></summary>

- **PlanningAgent**: Intelligently decomposes complex queries into focused subquestions
- **NormalizeQueryAgent**: Classifies queries and selects appropriate tools (company/patent/general)
- **GeneralizeAgent**: Synthesizes information from multiple sources with source attribution
- **Market Analysts**: Comprehensive opportunity and risk analysis with confidence scoring
- **FactCheckingAgent**: Validates responses for accuracy and flags potential issues

</details>

<details>
<summary><strong>🔍 Hybrid Retrieval System</strong></summary>

- **Dense + Sparse Search**: Combines vector embeddings with BM25 for optimal recall
- **Source Attribution**: Every answer includes proper citations and confidence scores
- **Intelligent Tool Selection**: Automatically chooses between exact lookup and RAG retrieval

</details>

<details>
<summary><strong>🎯 Multiple Query Modes</strong></summary>

- **Company Analysis**: Business focus, market opportunities, competitive positioning
- **Patent Analysis**: Technology details, patent portfolios, innovation trends
- **General Innovation**: Industry trends, market landscapes, emerging technologies
- **Freestyle Query**: Complex multi-faceted questions with automatic decomposition

</details>

<details>
<summary><strong>🔧 Professional Infrastructure</strong></summary>

- **Centralized Configuration**: LLM types configurable per agent in `config/agent_config.py`
- **Comprehensive Logging**: Detailed progress tracking and error handling
- **Modular Architecture**: Easy to extend with new agents and tools
- **Multiple Interfaces**: CLI, Chat mode, and Streamlit web interface

</details>

## 🚀 Quick Start

<details>
<summary><strong>1. Installation</strong></summary>

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

</details>

<details>
<summary><strong>2. Data Ingestion</strong></summary>

```bash
# Ingest company and patent data
python main.py --mode ingest --force_reindex

# This will:
# - Process patent data and create embeddings
# - Process company data and create embeddings  
# - Create RAG_INDEX directory with all indexed data
# - Build enhanced data mappings
```

</details>

<details>
<summary><strong>3. Configure LLMs</strong></summary>

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

</details>

<details>
<summary><strong>4. Run Queries</strong></summary>

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

</details>

## 💻 Usage Examples

<details>
<summary><strong>Click to view Usage Examples</strong></summary>

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

</details>

## 🌐 Streamlit Web Interface

<details>
<summary><strong>Click to view Streamlit Interface Details</strong></summary>

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

</details>

## 🏗️ System Architecture

<details>
<summary><strong>Click to view System Architecture Diagram</strong></summary>

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

</details>

## 🎯 Product Suggestion Mode

InnovARAG features a specialized **Product Suggestion Mode** that focuses on extracting and suggesting specific products, technologies, and innovations from retrieved patent and company data. This mode provides citation-backed product recommendations without relying on external knowledge.

<details>
<summary><strong>Click to view Product Suggestion Mode Overview</strong></summary>

### Key Features

- **Citation-Based Suggestions**: All product suggestions are grounded in retrieved contexts with proper citations
- **Context-Only Analysis**: No external knowledge used - only what's found in the patent and company databases  
- **Streamlined Workflow**: Skips market analysis agents to focus specifically on product extraction
- **Source Attribution**: Every suggestion includes detailed source information and references
- **Validation System**: Built-in fact-checking ensures product suggestions are supported by evidence

### How It Works

1. **Query Processing**: User query is analyzed and split into focused subquestions
2. **Context Retrieval**: System retrieves relevant patent and company contexts using hybrid search
3. **Product Extraction**: Specialized agents extract specific products and technologies from contexts
4. **Citation Verification**: Each suggestion is validated against source material
5. **Structured Output**: Results presented with clear citations and source attribution

</details>

<details>
<summary><strong>Click to view Product Suggestion Mode Usage</strong></summary>

### Enabling Product Suggestion Mode

**Streamlit Interface:**
1. Open the InnovARAG web interface
2. In the sidebar, locate "🎯 Operation Mode"
3. Check the "🎯 Product Suggestion Mode" checkbox
4. The interface will switch to product suggestion workflow

**Command Line:**
```bash
# Enable product suggestion mode for single query
python main.py --mode query --query "artificial intelligence applications" --product_suggestion

# Interactive chat with product suggestions
python main.py --mode chat --product_suggestion
```

</details>

### Example Queries for Product Suggestion Mode

<details>
<summary><strong>Technology Discovery Queries</strong></summary>

```python
# AI and Machine Learning Products
"artificial intelligence applications and machine learning products"
"neural network implementations in consumer devices"  
"computer vision technologies for healthcare"

# Biotechnology Products
"biomedical devices and diagnostic tools"
"pharmaceutical innovations and drug delivery systems"
"genetic sequencing technologies"

# Hardware and Electronics
"semiconductor innovations and chip technologies"
"IoT devices and smart sensor applications"
"renewable energy storage solutions"
```

</details>

<details>
<summary><strong>Company-Specific Product Analysis</strong></summary>

```python
# Company Product Portfolio
"What products does Intel develop and manufacture?"
"Describe the biomedical products from Advanced Biomedical Technologies"
"List the AI-related products from Google's patent portfolio"

# Technology Focus Areas
"What are the main product categories for Toyota's innovations?"
"Describe the software products mentioned in Microsoft patents"
"What consumer electronics does Samsung focus on?"
```

</details>

<details>
<summary><strong>Click to view Product Suggestion Output Format</strong></summary>

### Expected Output Structure

Product Suggestion Mode provides structured responses with:

**1. Product Suggestions**
- Specific product names and descriptions
- Technology categories and applications
- Implementation details from patents

**2. Source Citations**
- Patent IDs and application numbers
- Company names and affiliations  
- Relevant context excerpts

**3. Validation Information**
- Confidence scores (1-10)
- Citation verification status
- Potential issues or limitations

### Sample Output Example

```
🎯 Product Suggestions

1. **Neural Processing Unit (NPU) for Edge Computing**
   - Description: Specialized AI chip for real-time neural network inference
   - Applications: Mobile devices, IoT sensors, autonomous vehicles
   - Source: Patent US123456789 from Intel Corporation
   - Context: "The NPU architecture enables 10x faster AI processing..."

2. **Biomarker Detection System**
   - Description: Lab-on-chip device for rapid disease screening
   - Applications: Point-of-care diagnostics, personalized medicine
   - Source: Patent US987654321 from Advanced Biomedical Technologies
   - Context: "The microfluidic system can detect cancer biomarkers..."

📊 Validation Results
- Overall Score: 9/10
- Confidence: High
- Citations Verified: 2/2
- Issues Found: None
```

</details>

<details>
<summary><strong>Click to view Mode Comparison</strong></summary>

### Product Suggestion vs Market Analysis Mode

| Aspect | Product Suggestion Mode | Market Analysis Mode |
|--------|------------------------|---------------------|
| **Primary Focus** | Extract specific products from data | Comprehensive market strategy analysis |
| **Data Sources** | Retrieved contexts only | Retrieved contexts + strategic analysis |
| **Output Type** | Product lists with citations | Market opportunities, risks, strategies |
| **Workflow** | Planning → Retrieval → Extraction → Validation | Full multi-agent workflow with analysis team |
| **Use Cases** | Product discovery, technology scouting | Strategic planning, competitive analysis |
| **Processing Time** | Faster (streamlined workflow) | Longer (comprehensive analysis) |
| **Citation Requirements** | Mandatory for all suggestions | Important but not exclusive focus |

### When to Use Each Mode

**Choose Product Suggestion Mode when:**
- Looking for specific products or technologies
- Need citation-backed recommendations
- Want to discover what's available in patent data
- Require fast, focused product extraction
- Building technology landscapes or competitive intelligence

**Choose Market Analysis Mode when:**
- Need strategic business insights
- Want comprehensive opportunity analysis
- Require risk assessment and recommendations
- Planning market entry or competitive positioning
- Need synthesis across multiple business dimensions

</details>

<details>
<summary><strong>Click to view Advanced Features</strong></summary>

### Advanced Product Suggestion Features

**Context Preservation**
- Sequential subquestions maintain context from previous answers
- Enhanced understanding for complex product queries
- Improved relevance across multi-part questions

**Hybrid Search Integration** 
- Dense semantic search for conceptual product matching
- Sparse keyword search for exact product name matching
- Combined results for comprehensive product discovery

**Validation System**
- Citation verification against source materials
- Confidence scoring for each product suggestion
- Flagging of potentially unsupported claims

**Performance Optimizations**
- Caching for faster repeated queries
- Parallel processing for batch product discovery
- FAISS indexing for efficient similarity search

### Limitations and Considerations

- **Data Dependency**: Limited to what's available in patent and company databases
- **Citation Requirement**: All suggestions must be supported by retrieved contexts
- **No External Knowledge**: Cannot suggest products not found in the data
- **Patent Focus**: Emphasis on patented technologies and innovations
- **Time Sensitivity**: Patent data may not reflect latest market products

</details>



## 📁 Project Structure

<details>
<summary><strong>Click to view Complete Project Structure</strong></summary>

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

</details>

## 🔧 Configuration

<details>
<summary><strong>LLM Configuration</strong></summary>

Edit `config/agent_config.py`:
- **DEFAULT_LLM_TYPE**: Global default LLM
- **agent_config**: Per-agent LLM specification

</details>

<details>
<summary><strong>RAG Configuration</strong></summary>

Edit `config/rag_config.py`:
- **Embedding models**: HuggingFace embeddings configuration
- **Chunk sizes**: Text processing parameters
- **Index settings**: Vector database configuration

</details>

<details>
<summary><strong>Prompt Templates</strong></summary>

All prompts are centralized in `config/prompts.py`:
- **Chain-of-Thought reasoning**
- **Source attribution requirements**
- **Confidence scoring guidelines**
- **Examples for each agent**

</details>

## 📊 Output Features

<details>
<summary><strong>Click to view Output Features Details</strong></summary>

### Comprehensive Analysis
- **Source Attribution**: `[Company: TechNova]`, `[Patent: 273556553]`
- **Confidence Scores**: 1-10 reliability ratings
- **Fact-Check Validation**: Accuracy verification
- **Multi-dimensional Analysis**: Opportunities, risks, strategic recommendations

### Export Options
- **JSON**: Complete workflow results with metadata
- **Text Summary**: Executive-friendly analysis summary
- **Structured Data**: Query history and performance metrics

</details>

## 🔍 Advanced Features

<details>
<summary><strong>Click to view Advanced Features</strong></summary>

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

</details>

## 🚦 Usage Modes

| Mode | Description | Command |
|------|-------------|---------|
| **ingest** | Data ingestion and indexing | `python main.py --mode ingest` |
| **query** | Single query processing | `python main.py --mode query --query "..."` |
| **test** | Test with default query | `python main.py --mode test` |
| **chat** | Interactive chat mode | `python main.py --mode chat` |
| **streamlit** | Web interface | `streamlit run streamlit_app.py` |

## 🛠️ Development

<details>
<summary><strong>Click to view Development Guidelines</strong></summary>

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

</details>

## 📝 Environment Variables

<details>
<summary><strong>Click to view Environment Configuration</strong></summary>

Create `.env` file:
```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Google AI (Gemini)
GOOGLE_API_KEY=your_google_api_key

```

</details>


## 🚀 Example Queries for Testing Optimized Tools

<details>
<summary><strong>Click to view Testing Examples</strong></summary>

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

</details>

## 🔧 **Testing Instructions**

<details>
<summary><strong>Click to view Testing Instructions</strong></summary>

1. **Initialize System**: Use sidebar to initialize with your preferred LLM
2. **Test Basic Tools**: Start with exact lookups to verify data access
3. **Test Optimized Features**: Use Enhanced Features tab → Optimized Hybrid Search
4. **Monitor Performance**: Use Performance Analytics tab to see metrics
5. **Test Batch Processing**: Use Batch Processing tab for multiple queries
6. **Verify Fixes**: 
   - Company lookup now prioritizes exact matches
   - Company RAG retrieval shows correct Hojin IDs
   - Company patents lookup displays properly formatted results

</details>

## 📊 **Expected Results**

<details>
<summary><strong>Click to view Expected Results</strong></summary>

- **Faster responses** due to caching and FAISS optimization
- **Better accuracy** with exact match prioritization  
- **Rich metadata** including relevance scores, confidence levels
- **Performance metrics** showing cache hit rates and query statistics
- **Proper error handling** with helpful user guidance
- **Visual notifications** with clear success/error/completion messages

</details>

## 🚀 **Latest Improvements & Performance Enhancements**

<details>
<summary><strong>📈 Performance Optimizations</strong></summary>

#### **⚡ High-Performance Search System**
- **12x Parallel Workers**: Increased from 4 to 12 for maximum parallelization
- **Redis Caching**: Distributed cache with 2-hour TTL and fallback to disk/memory
- **Semantic Caching**: Finds similar queries using 85% cosine similarity threshold
- **Query Preprocessing**: Normalizes queries and removes stop phrases for better performance
- **Memory Optimization**: Smart eviction based on system memory usage with `psutil` monitoring
- **Async Batch Processing**: Process multiple queries simultaneously with memory management

</details>

<details>
<summary><strong>🎯 Tool Selection Improvements</strong></summary>

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

</details>

<details>
<summary><strong>📊 Performance Metrics</strong></summary>

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

</details>

## 🤝 Contributing

<details>
<summary><strong>Click to view Contributing Guidelines</strong></summary>

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with proper logging and error handling
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

</details>

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




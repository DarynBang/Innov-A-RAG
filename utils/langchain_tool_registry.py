"""
Langchain-based Tool Registry System for InnovARAG

This module provides automatic tool registration using Langchain tools
and eliminates the need for manual tool definitions in prompts.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from typing import Dict, List, Any, Callable, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun
import json

logger = get_logger(__name__)

class CompanyLookupInput(BaseModel):
    """Input for company lookup tool."""
    company_identifier: str = Field(description="Company name or ID to look up")

class PatentLookupInput(BaseModel):
    """Input for patent lookup tool."""
    patent_id: str = Field(description="Patent ID to look up")

class RAGRetrievalInput(BaseModel):
    """Input for RAG retrieval tools."""
    query: str = Field(description="Search query for information retrieval")
    top_k: int = Field(default=3, description="Number of top results to return")

class HybridRAGInput(BaseModel):
    """Input for hybrid RAG retrieval."""
    query: str = Field(description="Search query for hybrid retrieval")
    top_k: int = Field(default=3, description="Number of top results to return")
    search_type: str = Field(default="both", description="Search type: 'company', 'patent', or 'both'")

class CompanyLookupTool(BaseTool):
    """Tool for exact company lookup with comprehensive business information."""
    name: str = "exact_company_lookup"
    description: str = """Look up detailed information for a SPECIFIC company by exact name or ID. Returns company profile, business focus, keywords, and summary.

    WHEN TO USE:
    - "Tell me about [Specific Company Name]" (e.g., "Tell me about Intel", "What does Microsoft do?")
    - "Information about [Company Name]" 
    - "Company profile for [Specific Name]"
    - Need complete company details for a known entity
    
    WHEN NOT TO USE:
    - General questions like "Which companies work in AI?" (use optimized_hybrid_rag_retrieval)
    - "Companies with X in their name" (use company_rag_retrieval)
    - Broad market research queries
    
    EXAMPLES:
    ✅ Good: "Intel" (specific company name)
    ✅ Good: "Advanced Biomedical Technologies" 
    ❌ Bad: "AI companies" (too broad)
    ❌ Bad: "tech companies" (generic search)
    
    INPUT: Single company name or ID (string)
    OUTPUT: Structured company information with business focus, keywords, and detailed summary"""
    args_schema: Type[BaseModel] = CompanyLookupInput
    tool_function: Callable = Field(description="The function to execute for this tool")
    
    def _run(
        self, 
        company_identifier: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the company lookup."""
        try:
            result = self.tool_function(company_identifier)
            return str(result)
        except Exception as e:
            return f"Error in company lookup: {str(e)}"

class PatentLookupTool(BaseTool):
    """Tool for exact patent lookup with full technical details."""
    name: str = "exact_patent_lookup"
    description: str = """Look up complete patent information by specific patent ID/number. Returns patent details, abstracts, technical specifications, and company information.

    WHEN TO USE:
    - "Patent ID 12345678" or "Tell me about patent 12345678"
    - "Information about patent application 273556553"
    - "Details for patent [specific ID]"
    - Need complete technical specifications for a known patent
    
    WHEN NOT TO USE:
    - "Patents about AI" (use patent_rag_retrieval)
    - "Machine learning patents" (use optimized_hybrid_rag_retrieval) 
    - "Company X's patents" (use company_patents_lookup)
    - Technology research without specific patent ID
    
    EXAMPLES:
    ✅ Good: "273556553" (specific patent ID)
    ✅ Good: "US20210123456" (patent application number)
    ❌ Bad: "AI patents" (too broad, use patent_rag_retrieval)
    ❌ Bad: "latest patents" (use optimized_hybrid_rag_retrieval)
    
    INPUT: Specific patent ID or application number (string)
    OUTPUT: Complete patent information including abstract, technical details, company, and full patent text"""
    args_schema: Type[BaseModel] = PatentLookupInput
    tool_function: Callable = Field(description="The function to execute for this tool")
    
    def _run(
        self, 
        patent_id: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the patent lookup."""
        try:
            result = self.tool_function(patent_id)
            return str(result)
        except Exception as e:
            return f"Error in patent lookup: {str(e)}"

class CompanyRAGTool(BaseTool):
    """Tool for semantic company search and discovery."""
    name: str = "company_rag_retrieval"
    description: str = """Semantic search through company information for discovery, patterns, and thematic queries. Best for finding companies by characteristics, industries, or business focus areas.

    WHEN TO USE:
    - "Companies working in AI/biotech/semiconductor"
    - "Companies with 'tech/bio/systems' in their name"
    - "Find companies focused on [technology/industry]"
    - "Who are the major players in [industry]?"
    - Industry landscape and competitive mapping
    - Company discovery by business characteristics
    
    WHEN NOT TO USE:
    - Specific company lookup (use exact_company_lookup)
    - Broad cross-industry queries (use optimized_hybrid_rag_retrieval)
    - Patent-related company questions (use company_patents_lookup)
    
    EXAMPLES:
    ✅ Good: "companies with tech in their name"
    ✅ Good: "biotech companies in the database" 
    ✅ Good: "semiconductor industry players"
    ✅ Good: "companies focused on machine learning"
    ❌ Bad: "Intel" (use exact_company_lookup)
    ❌ Bad: "Intel's patents" (use company_patents_lookup)
    
    INPUT: Descriptive query about company characteristics or industries
    OUTPUT: Relevant companies with business context, ranked by semantic relevance"""
    args_schema: Type[BaseModel] = RAGRetrievalInput
    tool_function: Callable = Field(description="The function to execute for this tool")
    
    def _run(
        self, 
        query: str, 
        top_k: int = 3,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the company RAG retrieval."""
        try:
            result = self.tool_function(query)
            return str(result)
        except Exception as e:
            return f"Error in company RAG retrieval: {str(e)}"

class PatentRAGTool(BaseTool):
    """Tool for semantic patent search and technology discovery."""
    name: str = "patent_rag_retrieval"
    description: str = """Semantic search through patent information for technology research, innovation patterns, and technical discovery. Best for finding patents by technology domains, technical approaches, or innovation themes.

    WHEN TO USE:
    - "Patents about machine learning/AI/biotech"
    - "Neural network patent innovations"
    - "Find patents related to [technology/method]"
    - "What are the key patents in [technical domain]?"
    - Technology landscape research
    - Innovation pattern analysis
    - Technical competitive intelligence
    
    WHEN NOT TO USE:
    - Specific patent ID lookup (use exact_patent_lookup)
    - Company-specific patent questions (use company_patents_lookup)
    - Broad industry analysis across patents + companies (use optimized_hybrid_rag_retrieval)
    
    EXAMPLES:
    ✅ Good: "machine learning patents"
    ✅ Good: "neural network innovations" 
    ✅ Good: "semiconductor manufacturing patents"
    ✅ Good: "biotech drug discovery patents"
    ❌ Bad: "Patent 273556553" (use exact_patent_lookup)
    ❌ Bad: "Intel's patents" (use company_patents_lookup)
    
    INPUT: Technology-focused query about patent innovations or technical domains
    OUTPUT: Relevant patents with abstracts, technical details, and innovation context"""
    args_schema: Type[BaseModel] = RAGRetrievalInput
    tool_function: Callable = Field(description="The function to execute for this tool")
    
    def _run(
        self, 
        query: str, 
        top_k: int = 3,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the patent RAG retrieval."""
        try:
            result = self.tool_function(query)
            return str(result)
        except Exception as e:
            return f"Error in patent RAG retrieval: {str(e)}"

class EnhancedHybridRAGTool(BaseTool):
    """Tool for enhanced hybrid RAG retrieval with true dense+sparse search."""
    name: str = "enhanced_hybrid_rag_retrieval"
    description: str = "Enhanced hybrid retrieval using both dense (semantic) and sparse (keyword) search methods. Provides comprehensive results with data mapping integration."
    args_schema: Type[BaseModel] = HybridRAGInput
    tool_function: Callable = Field(description="The function to execute for this tool")
    
    def _run(
        self, 
        query: str, 
        top_k: int = 3,
        search_type: str = "both",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the enhanced hybrid RAG retrieval."""
        try:
            result = self.tool_function(query, top_k, search_type)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error in enhanced hybrid RAG retrieval: {str(e)}"

class OptimizedHybridRAGInput(BaseModel):
    """Input for optimized hybrid RAG retrieval."""
    query: str = Field(description="Search query for optimized hybrid retrieval")
    top_k: int = Field(default=3, description="Number of top results to return")
    search_type: str = Field(default="both", description="Search type: 'company', 'patent', or 'both'")

class BatchOptimizedRetrievalInput(BaseModel):
    """Input for batch optimized retrieval."""
    queries: List[str] = Field(description="List of queries for batch processing")
    top_k: int = Field(default=3, description="Number of top results to return per query")
    search_type: str = Field(default="both", description="Search type: 'company', 'patent', or 'both'")

class OptimizedHybridRAGTool(BaseTool):
    """Tool for comprehensive cross-domain search with maximum performance."""
    name: str = "optimized_hybrid_rag_retrieval"
    description: str = """High-performance hybrid search across BOTH companies and patents using advanced caching, FAISS indexing, semantic similarity, and parallel processing. The PRIMARY tool for complex, broad, or cross-domain queries.

    WHEN TO USE (Primary tool for most queries):
    - Cross-domain analysis: "How do company strategies relate to patent innovations?"
    - Broad market research: "Who are the major players in AI industry?"
    - Comparative analysis: "Compare different approaches to semiconductor technology"
    - Industry trends: "What are the latest trends in biotech?"
    - Complex questions spanning multiple entities or domains
    - General information gathering without specific entity focus
    - Market landscape mapping across companies and technologies
    - Strategic analysis requiring both business and technical insights
    
    SEARCH TYPES:
    - search_type="both" (default): Search companies AND patents simultaneously
    - search_type="company": Focus on company information only  
    - search_type="patent": Focus on patent information only
    
    WHEN NOT TO USE:
    - Specific entity lookup (use exact_company_lookup, exact_patent_lookup)
    - Single company's patents only (use company_patents_lookup)
    
    EXAMPLES:
    ✅ Excellent: "How do AI companies approach machine learning innovation?"
    ✅ Excellent: "Compare semiconductor strategies across industry leaders"  
    ✅ Excellent: "What are the major trends in biotech patent filings?"
    ✅ Good: "Who are the key players in quantum computing?"
    ✅ Good: "Latest developments in neural network technologies"
    ❌ Bad: "Intel" (use exact_company_lookup)
    ❌ Bad: "Intel's patents" (use company_patents_lookup)
    
    INPUT: Complex or broad query requiring comprehensive search
    OUTPUT: Ranked results from both companies and patents with semantic relevance scores"""
    args_schema: Type[BaseModel] = OptimizedHybridRAGInput
    tool_function: Callable = Field(description="The function to execute for this tool")
    
    def _run(
        self, 
        query: str, 
        top_k: int = 3,
        search_type: str = "both",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the optimized hybrid RAG retrieval."""
        try:
            result = self.tool_function(query, top_k, search_type)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error in optimized hybrid RAG retrieval: {str(e)}"

class BatchOptimizedRetrievalTool(BaseTool):
    """Tool for high-performance parallel processing of multiple queries."""
    name: str = "batch_optimized_retrieval"
    description: str = """Process multiple related queries simultaneously with parallel execution, advanced caching, and memory optimization. Ideal for comprehensive research requiring multiple searches.

    WHEN TO USE:
    - Multiple related research questions that need parallel processing
    - Comprehensive competitive analysis across multiple dimensions
    - Market research requiring multiple search angles
    - Batch processing for efficiency when you have 3+ related queries
    
    WHEN NOT TO USE:
    - Single query (use appropriate individual tool)
    - Unrelated queries (process separately)
    - Simple lookup tasks
    
    EXAMPLES:
    ✅ Perfect: ["AI company strategies", "AI patent trends", "AI market opportunities"] 
    ✅ Good: ["semiconductor leaders", "chip manufacturing patents", "semiconductor industry trends"]
    ❌ Bad: ["Intel"] (single query, use exact_company_lookup)
    ❌ Bad: ["weather", "sports", "cooking"] (unrelated topics)
    
    INPUT: List of related search queries (2-10 queries recommended)
    OUTPUT: Parallel search results for all queries with performance metrics"""
    args_schema: Type[BaseModel] = BatchOptimizedRetrievalInput
    tool_function: Callable = Field(description="The function to execute for this tool")
    
    def _run(
        self, 
        queries: List[str], 
        top_k: int = 3,
        search_type: str = "both",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute batch optimized retrieval."""
        try:
            result = self.tool_function(queries, top_k, search_type)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error in batch optimized retrieval: {str(e)}"

class PerformanceAnalyticsTool(BaseTool):
    """Tool for comprehensive system performance monitoring and analytics."""
    name: str = "get_performance_analytics"
    description: str = """Retrieve detailed system performance metrics including cache hit rates, query response times, memory usage, optimization statistics, and search efficiency metrics.

    WHEN TO USE:
    - "Show me system performance metrics"
    - "How is the search system performing?"
    - "What are the cache hit rates?"
    - "Performance analytics of the current system"
    - System optimization and monitoring needs
    - Troubleshooting slow queries or system issues
    
    WHEN NOT TO USE:
    - Business or market performance questions (use optimized_hybrid_rag_retrieval)
    - Company financial performance (use company_rag_retrieval)
    - Patent filing performance (use patent_rag_retrieval)
    
    EXAMPLES:
    ✅ Perfect: "Show me performance analytics of the current system"
    ✅ Good: "How fast are queries running?"
    ✅ Good: "System optimization metrics"
    ❌ Bad: "Company performance metrics" (use optimized_hybrid_rag_retrieval)
    ❌ Bad: "Market performance" (use optimized_hybrid_rag_retrieval)
    
    INPUT: No input required (system automatically gathers current metrics)
    OUTPUT: Comprehensive performance dashboard with cache statistics, query metrics, response times, and system health indicators"""
    args_schema: Type[BaseModel] = None  # No input required
    tool_function: Callable = Field(description="The function to execute for this tool")
    
    def _run(
        self, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute performance analytics retrieval."""
        try:
            result = self.tool_function()
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error getting performance analytics: {str(e)}"

class CompanyPatentsLookupTool(BaseTool):
    """Tool for retrieving all patents owned by a specific company."""
    name: str = "company_patents_lookup"
    description: str = """Retrieve complete patent portfolio for a SPECIFIC, KNOWN company. Returns all patents owned by that company with abstracts, technical details, and patent IDs.

    WHEN TO USE:
    - "What patents does [Specific Company] own?" (e.g., "Intel patents", "Microsoft's patent portfolio")
    - "[Company Name]'s intellectual property"
    - "Patents filed by [Specific Company]"
    - "Show me [Company]'s patent applications"
    - Need complete IP portfolio analysis for a known entity
    
    WHEN NOT TO USE:
    - General patent research (use patent_rag_retrieval)
    - "Which companies have patents in AI?" (use optimized_hybrid_rag_retrieval)
    - Technology-focused searches (use patent_rag_retrieval)
    - Cross-company patent comparisons (use optimized_hybrid_rag_retrieval)
    
    EXAMPLES:
    ✅ Perfect: "Advanced Biomedical Technologies" (specific company name)
    ✅ Perfect: "Intel patents" (specific company request)
    ✅ Good: "Microsoft's patent portfolio"
    ✅ Good: "Patents owned by Samsung"
    ❌ Bad: "biotech patents" (use patent_rag_retrieval)
    ❌ Bad: "AI companies with patents" (use optimized_hybrid_rag_retrieval)
    ❌ Bad: "patent trends" (use optimized_hybrid_rag_retrieval)
    
    CRITICAL: Only use when you have an exact, specific company name. This tool searches for patents BY company, not patents ABOUT topics.
    
    INPUT: Exact company name or identifier (string)
    OUTPUT: Complete list of patents owned by that company with abstracts, patent IDs, and technical summaries"""
    args_schema: Type[BaseModel] = CompanyLookupInput
    tool_function: Callable = Field(description="The function to execute for this tool")
    
    def _run(
        self, 
        company_identifier: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the company patents lookup."""
        try:
            result = self.tool_function(company_identifier)
            return str(result)
        except Exception as e:
            return f"Error in company patents lookup: {str(e)}"

class LangchainToolRegistry:
    """Langchain-based tool registry that automatically handles tool registration."""
    
    def __init__(self):
        self.tools: List[BaseTool] = []
        self.tool_map: Dict[str, BaseTool] = {}
        logger.info("LangchainToolRegistry initialized")
    
    def register_tool_functions(self, tool_functions: Dict[str, Callable]):
        """Register tool functions as Langchain tools."""
        logger.info(f"Registering {len(tool_functions)} tool functions")
        
        # Map function names to tool classes - UPDATED FOR OPTIMIZED TOOLS
        tool_mapping = {
            # Core lookup tools (still needed)
            'exact_company_lookup': CompanyLookupTool,
            'exact_patent_lookup': PatentLookupTool,
            'company_patents_lookup': CompanyPatentsLookupTool,
            
            # Basic RAG tools (kept for fallback)
            'company_rag_retrieval': CompanyRAGTool,
            'patent_rag_retrieval': PatentRAGTool,
            
            # OPTIMIZED TOOLS (PRIMARY)
            'optimized_hybrid_rag_retrieval': OptimizedHybridRAGTool,
            'batch_optimized_retrieval': BatchOptimizedRetrievalTool,
            'get_performance_analytics': PerformanceAnalyticsTool,
            
            # OLD ENHANCED TOOLS (COMMENTED OUT - LEGACY)
            # 'enhanced_hybrid_rag_retrieval': EnhancedHybridRAGTool,  # Replaced by optimized version
        }
        
        for func_name, func in tool_functions.items():
            if func_name in tool_mapping:
                tool_class = tool_mapping[func_name]
                # Create tool instance with the function as a parameter
                tool_instance = tool_class(tool_function=func)
                self.tools.append(tool_instance)
                self.tool_map[func_name] = tool_instance
                logger.info(f"Registered {func_name} as Langchain tool")
            else:
                logger.warning(f"No Langchain tool mapping found for {func_name}")
    
    def get_tools(self) -> List[BaseTool]:
        """Get all registered Langchain tools."""
        return self.tools
    
    def get_tool_descriptions(self) -> str:
        """Generate tool descriptions for prompts."""
        descriptions = []
        for tool in self.tools:
            desc = f"- {tool.name}: {tool.description}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def get_tool_schemas(self) -> Dict[str, Any]:
        """Get tool schemas for LLM function calling."""
        schemas = {}
        for tool in self.tools:
            schemas[tool.name] = {
                "name": tool.name,
                "description": tool.description,
                # new 24/07: schema -> model_json_schema
                "parameters": tool.args_schema.model_json_schema() if tool.args_schema else {}
            }
        return schemas
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name."""
        if tool_name in self.tool_map:
            tool = self.tool_map[tool_name]
            return tool._run(**kwargs)
        else:
            return f"Tool {tool_name} not found"
    
    def generate_classification_examples(self) -> List[Dict[str, Any]]:
        """Generate comprehensive classification examples with OPTIMIZED TOOLS."""
        return [
            {
                "query": "What are TechNova's market opportunities?",
                "query_type": "company",
                "identifiers": {"companies": ["TechNova"], "patents": []},
                "recommended_tools": ["exact_company_lookup", "company_rag_retrieval", "optimized_hybrid_rag_retrieval"],
                "reasoning": "Specific company query requiring exact info, detailed analysis, and optimized cross-referencing"
            },
            {
                "query": "Tell me about Patent 273556553",
                "query_type": "patent",
                "identifiers": {"companies": [], "patents": ["273556553"]},
                "recommended_tools": ["exact_patent_lookup", "patent_rag_retrieval"],
                "reasoning": "Specific patent query requiring both exact info and technical analysis"
            },
            {
                "query": "What are the latest trends in AI patents?",
                "query_type": "general",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["optimized_hybrid_rag_retrieval", "patent_rag_retrieval"],
                "reasoning": "General industry trend query requiring optimized search across patents with advanced caching"
            },
            {
                "query": "Who are the major semiconductor companies with significant patent portfolios?",
                "query_type": "general",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["optimized_hybrid_rag_retrieval", "company_rag_retrieval"],
                "reasoning": "General identification query seeking multiple companies - requires broad search, NOT company_patents_lookup which needs specific company names"
            },
            {
                "query": "What patents does Snowflake have?",
                "query_type": "company_patents",
                "identifiers": {"companies": ["Snowflake"], "patents": []},
                "recommended_tools": ["exact_company_lookup", "company_patents_lookup"],
                "reasoning": "Company-specific patent query requiring company lookup then direct patent search by company"
            },
            {
                "query": "Tell me about patents of company Intel",
                "query_type": "company_patents",
                "identifiers": {"companies": ["Intel"], "patents": []},
                "recommended_tools": ["exact_company_lookup", "company_patents_lookup"],
                "reasoning": "Company-specific patent query requiring company lookup then direct patent search by company"
            },
            {
                "query": "Show me Advanced Biomedical Technologies patents",
                "query_type": "company_patents",
                "identifiers": {"companies": ["Advanced Biomedical Technologies"], "patents": []},
                "recommended_tools": ["exact_company_lookup", "company_patents_lookup"],
                "reasoning": "Company-specific patent query requiring company lookup then direct patent search by company"
            },
            {
                "query": "Compare TechNova and Intel's AI innovations",
                "query_type": "comparison",
                "identifiers": {"companies": ["TechNova", "Intel"], "patents": []},
                "recommended_tools": ["optimized_hybrid_rag_retrieval", "company_rag_retrieval", "patent_rag_retrieval"],
                "reasoning": "Comparison query requiring optimized search across multiple sources with advanced performance"
            },
            {
                "query": "Analyze artificial intelligence innovation strategies across companies",
                "query_type": "general",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["optimized_hybrid_rag_retrieval", "batch_optimized_retrieval"],
                "reasoning": "Complex analysis query benefiting from optimized hybrid search and batch processing capabilities"
            },
            {
                "query": "What are the competitive advantages of biotech companies in drug development?",
                "query_type": "industry_analysis",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["optimized_hybrid_rag_retrieval", "company_rag_retrieval"],
                "reasoning": "Industry analysis requiring optimized search across companies with performance monitoring"
            },
            {
                "query": "Find machine learning patents and their commercial applications",
                "query_type": "technology_analysis",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["optimized_hybrid_rag_retrieval", "patent_rag_retrieval"],
                "reasoning": "Technology analysis benefiting from optimized hybrid search with advanced caching for better performance"
            }
        ]
    
    def generate_prompt_section(self) -> str:
        """Generate complete prompt section for normalize agent."""
        tool_descriptions = self.get_tool_descriptions()
        examples = self.generate_classification_examples()
        
        examples_text = ""
        for example in examples:
            examples_text += f"""
Example:
Query: "{example['query']}"
Output: {{
    "query_type": "{example['query_type']}",
    "identifiers": {json.dumps(example['identifiers'])},
    "recommended_tools": {json.dumps(example['recommended_tools'])},
    "reasoning": "{example['reasoning']}"
}}
"""
        
        return f"""
AVAILABLE TOOLS:
{tool_descriptions}

CLASSIFICATION EXAMPLES:
{examples_text}
"""

# Global registry instance
global_langchain_registry = LangchainToolRegistry()

def get_langchain_tool_registry() -> LangchainToolRegistry:
    """Get the global Langchain tool registry instance."""
    return global_langchain_registry 

    
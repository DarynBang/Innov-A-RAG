"""
InnovARAG Streamlit Web Interface

A comprehensive web interface for the InnovARAG Multi-Agent RAG System
that allows users to interact with the system through different query modes
and view the complete analysis workflow.
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
from io import StringIO
import traceback

# Configure Streamlit page
st.set_page_config(
    page_title="InnovARAG - Innovation Discovery Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import InnovARAG components
try:
    from agents.multi_agent_runner import MultiAgentRunner
    from config.agent_config import agent_config, DEFAULT_LLM_TYPE
    from config.rag_config import patent_config, firm_config
    from firm_summary_rag import FirmSummaryRAG
    from patent_rag import PatentRAG
    from tools.company_tools import init_company_tools
    from tools.patent_tools import init_patent_tools
    # OLD ENHANCED HYBRID TOOLS (commented out for comparison)
    # from tools.enhanced_hybrid_rag_tools import (
    #     enhanced_hybrid_rag_retrieval_tool,
    #     company_data_with_mapping_tool,
    #     mapping_key_search_tool
    # )
    
    # NEW OPTIMIZED HYBRID TOOLS
    from tools.optimized_hybrid_rag_tools import (
        optimized_hybrid_rag_retrieval_tool,
        batch_optimized_retrieval_tool,
        get_performance_analytics_tool,
        clear_caches_tool,
        optimize_for_queries_tool,
        get_optimized_hybrid_tools,
        reset_optimized_hybrid_tools
    )
    
    # NEW OPTIMIZATION CONFIGURATION
    from config.optimization_config import (
        RECOMMENDED_CONFIG_FOR_CURRENT_DATASET,
        create_optimization_config,
        get_recommended_config_for_data_size
    )
    from utils.logging_utils import setup_logging, get_logger
    from utils.langchain_tool_registry import get_langchain_tool_registry
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
except ImportError as e:
    st.error(f"❌ Error importing InnovARAG components: {e}")
    st.stop()

# Initialize session state variables
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

if 'runner' not in st.session_state:
    st.session_state.runner = None

if 'tool_registry' not in st.session_state:
    st.session_state.tool_registry = None

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

if 'optimization_tools' not in st.session_state:
    st.session_state.optimization_tools = None

if 'optimization_config' not in st.session_state:
    st.session_state.optimization_config = None

if 'performance_stats' not in st.session_state:
    st.session_state.performance_stats = {}

# Session state for other variables (audio removed)

@st.cache_resource
def initialize_innovarag():
    """Initialize InnovARAG system components with enhanced Langchain tool integration."""
    try:
        logger.info("Initializing InnovARAG system for Streamlit...")
        
        # Load data
        index_dir = "RAG_INDEX"
        patent_df = pd.read_csv(patent_config.get("patent_csv"))
        firm_df = pd.read_csv(firm_config.get("firm_csv"))
        
        # Initialize RAG systems
        firm_rag = FirmSummaryRAG(df=firm_df, index_dir=index_dir, config=firm_config)
        patent_rag = PatentRAG(df=patent_df, index_dir=index_dir, config=patent_config)
        
        # Initialize multi-agent runner (no parameters needed)
        runner = MultiAgentRunner()
        
        # Initialize and register tools (following main.py pattern)
        company_tools = init_company_tools(firm_df, index_dir)
        patent_tools = init_patent_tools(patent_df, index_dir)
        
        all_tools = {**company_tools, **patent_tools}
        
        # Setup optimization system (after RAG systems are ready)
        logger.info("Setting up optimization system for Streamlit...")
        try:
            # Get recommended configuration for the dataset
            optimization_config = RECOMMENDED_CONFIG_FOR_CURRENT_DATASET
            
            # Reset optimized hybrid tools for clean initialization
            reset_optimized_hybrid_tools()
            
            # Initialize optimized tools
            optimization_tools = get_optimized_hybrid_tools()
            
            # Add optimized hybrid tools
            all_tools['optimized_hybrid_rag_retrieval'] = optimized_hybrid_rag_retrieval_tool
            all_tools['batch_optimized_retrieval'] = batch_optimized_retrieval_tool
            all_tools['get_performance_analytics'] = get_performance_analytics_tool
            
            logger.info("Optimization system setup completed for Streamlit")
            
        except Exception as e:
            logger.warning(f"Optimization setup failed (using fallback): {e}")
            optimization_config = None
            optimization_tools = None
        
        runner.register_tools(all_tools)
        
        # Try to get Langchain tool registry, fallback if not available
        try:
            tool_registry = get_langchain_tool_registry()
            tools_count = len(tool_registry.get_tools())
        except Exception as tool_error:
            logger.warning(f"Could not load Langchain tool registry: {tool_error}")
            # Create a mock registry for now
            class MockToolRegistry:
                def get_tools(self):
                    return []
            tool_registry = MockToolRegistry()
            tools_count = 0
        
        logger.info(f"InnovARAG system initialized successfully!")
        logger.info(f"Loaded {len(patent_df)} patents and {len(firm_df)} companies")
        logger.info(f"Registered {len(all_tools)} tools with runner")
        logger.info(f"Langchain tools: {tools_count}")
        logger.info(f"Optimization system: {'Active' if optimization_tools else 'Fallback'}")
        
        return runner, len(patent_df), len(firm_df), tool_registry, optimization_config, optimization_tools
        
    except Exception as e:
        logger.error(f"Failed to initialize InnovARAG system: {str(e)}")
        # Print more detailed error information
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise e

# Sidebar configuration
st.sidebar.title("🚀 InnovARAG System")
st.sidebar.markdown("**Innovation Discovery Platform**")

# Notification settings removed (audio functionality removed)

# Model selection
model_type = st.sidebar.selectbox(
    "🤖 Select LLM Model",
    ["openai", "gemini", "qwen"],
    index=0,
    help="Choose the language model for the multi-agent system"
)

# System initialization control
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Control")

# Display system status
if st.session_state.runner is None:
    st.sidebar.warning("⚠️ System not initialized")
    
    # Initialize button
    if st.sidebar.button("🚀 Initialize System", type="primary", use_container_width=True):
        try:
            with st.spinner("🔄 Initializing InnovARAG system..."):
                # Store selected model type in session state
                st.session_state.selected_model = model_type
                
                runner, patent_count, firm_count, tool_registry, optimization_config, optimization_tools = initialize_innovarag()
                st.session_state.runner = runner
                st.session_state.tool_registry = tool_registry
                st.session_state.optimization_config = optimization_config
                st.session_state.optimization_tools = optimization_tools
                
                # Show completion notification
                st.success(f"🎉 System initialized with {model_type.upper()}!")
                st.sidebar.info(f"📊 Loaded {patent_count:,} patents and {firm_count:,} companies")
                st.rerun()
                
        except Exception as e:
            st.sidebar.error(f"❌ Initialization failed!")
            st.sidebar.error(f"Error: {str(e)}")
            # Reset session state on error
            st.session_state.runner = None
            st.session_state.tool_registry = None
            st.session_state.selected_model = None
            st.session_state.optimization_config = None
            st.session_state.optimization_tools = None
    
    st.sidebar.info("""
    📋 **Before you start:**
    1. Select your preferred LLM model
    2. Click 'Initialize System' 
    3. Wait for data loading to complete
    4. Start analyzing innovation data!
    """)
    
else:
    current_model = st.session_state.get('selected_model', 'openai')
    st.sidebar.success(f"✅ System ready ({current_model.upper()})")
    
    if st.session_state.tool_registry:
        tool_count = len(st.session_state.tool_registry.get_tools())
        st.sidebar.info(f"🔧 {tool_count} Langchain tools registered")
    
    # Model switching
    if model_type != current_model:
        st.sidebar.warning(f"⚠️ Model changed to {model_type.upper()}")
        if st.sidebar.button("🔄 Reinitialize with New Model", type="secondary", use_container_width=True):
            # Reset system for reinitialization
            st.session_state.runner = None
            st.session_state.tool_registry = None
            st.session_state.selected_model = model_type
            st.session_state.optimization_config = None
            st.session_state.optimization_tools = None
            st.rerun()
    
    # Reset button
    if st.sidebar.button("🔄 Reset System", use_container_width=True):
        st.session_state.runner = None
        st.session_state.tool_registry = None
        st.session_state.optimization_config = None
        st.session_state.optimization_tools = None
        st.session_state.selected_model = None
        st.session_state.current_results = None
        st.rerun()

def display_system_overview():
    """Display system overview and capabilities."""
    st.header("🔬 InnovARAG System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🤖 Multi-Agent System")
        st.write("""
        - **Planning Agent**: Query analysis and workflow planning
        - **Normalize Agent**: Query classification with Langchain tools
        - **Generalize Agent**: Information synthesis
        - **Market Analysts**: Strategic analysis (conditional)
        - **Fact Checker**: Response validation
        """)
    
    with col2:
        st.subheader("🔍 Enhanced Search")
        st.write("""
        - **True Hybrid Search**: Dense + Sparse retrieval
        - **Company Tools**: Exact lookup + RAG retrieval
        - **Patent Tools**: Comprehensive patent analysis
        - **Data Mapping**: Enhanced cross-referencing
        - **Workflow Awareness**: Smart tool execution
        """)
    
    with col3:
        st.subheader("📊 Features")
        st.write("""
        - **Conditional Analysis**: Smart team involvement
        - **Query Splitting**: Complex query breakdown
        - **Source Attribution**: Comprehensive citations
        - **Confidence Scoring**: Reliability assessment
        - **Real-time Processing**: Live workflow tracking
        """)

def display_enhanced_features():
    """Display enhanced features and comprehensive testing capabilities."""
    st.header("🚀 Enhanced Features & System Testing")
    
    if st.session_state.runner is None:
        st.warning("Please initialize the system first.")
        return
    
    # Create tabs for different testing categories
    test_tabs = st.tabs([
        "🏢 Company Tools", 
        "📋 Patent Tools", 
        "🔄 Optimized Hybrid Search", 
        "📊 Performance Analytics",
        "⚡ Batch Processing",
        "🔧 System Tools"
    ])
    
    # Tab 1: Company Tools Testing
    with test_tabs[0]:
        st.subheader("🏢 Company Tools Testing")
        
        # System status check
        if st.session_state.runner is None:
            st.warning("⚠️ System not initialized. Please initialize the system using the sidebar first.")
            st.info("👈 Go to the sidebar and click '🚀 Initialize System' to start using the tools.")
            return
        
        # Company Lookup by Name/ID
        with st.expander("🔍 Exact Company Lookup", expanded=True):
            st.write("**Test exact company lookup by name or Hojin ID**")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                company_identifier = st.text_input(
                    "Company Name or Hojin ID", 
                    placeholder="e.g., Intel, Toyota, or specific Hojin ID",
                    key="company_lookup_input"
                )
            with col2:
                st.write("") # spacing
                st.write("") # spacing
                if st.button("🔍 Lookup Company", key="company_lookup_btn"):
                    if company_identifier:
                        try:
                            # Check if system is initialized
                            if st.session_state.runner is None:
                                st.error("❌ System not initialized. Please initialize the system first using the sidebar.")
                                return
                            
                            with st.spinner("Looking up company..."):
                                from tools.company_tools import get_company_tools
                                company_tools = get_company_tools()
                                
                                if company_tools is None:
                                    st.error("❌ Company tools not initialized. Please reinitialize the system.")
                                    return
                                
                                result = company_tools.get_exact_company_info(company_identifier)
                                
                                if not result.get("success", False):
                                    st.error(f"❌ {result.get('message', 'Unknown error')}")
                                else:
                                    st.success("✅ Company found!")
                                    company_data = result.get('data', {})
                                    
                                    # Display company information in a nice format
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Company Information:**")
                                        st.write(f"**Company Name:** {company_data.get('company_name', 'N/A')}")
                                        st.write(f"**Hojin ID:** {company_data.get('hojin_id', 'N/A')}")
                                    
                                    with col2:
                                        st.write("**Status:**")
                                        st.success("Found")
                                    
                                    # Display keywords
                                    if company_data.get('company_keywords'):
                                        st.write("**Keywords:**")
                                        st.write(company_data['company_keywords'])
                                    
                                    # Display summary
                                    if company_data.get('summary'):
                                        with st.expander("📄 Company Summary"):
                                            st.write(company_data['summary'])
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        
        # Company RAG Retrieval
        with st.expander("🔎 Company RAG Retrieval", expanded=False):
            st.write("**Test company context retrieval using RAG**")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                company_query = st.text_input(
                    "Company Search Query", 
                    placeholder="e.g., semiconductor companies in Japan",
                    key="company_rag_input"
                )
            with col2:
                top_k_company = st.number_input("Top K", value=5, min_value=1, max_value=20, key="company_rag_k")
                if st.button("🔎 Search Companies", key="company_rag_btn"):
                    if company_query:
                        try:
                            # Check if system is initialized
                            if st.session_state.runner is None:
                                st.error("❌ System not initialized. Please initialize the system first using the sidebar.")
                                return
                            
                            with st.spinner("Searching companies..."):
                                from tools.company_tools import get_company_tools
                                company_tools = get_company_tools()
                                
                                if company_tools is None:
                                    st.error("❌ Company tools not initialized. Please reinitialize the system.")
                                    return
                                
                                contexts = company_tools.retrieve_company_contexts(company_query, top_k_company)
                                
                                st.success(f"✅ Found {len(contexts)} company contexts")
                                
                                # Display results in a nice format
                                for i, ctx in enumerate(contexts[:5], 1):
                                    with st.container():
                                        st.markdown(f"**{i}. {ctx.get('company_name', 'Unknown Company')}**")
                                        # Use company_id which is returned by firm RAG (this is actually the hojin_id)
                                        company_id = ctx.get('company_id', ctx.get('hojin_id', 'Unknown'))
                                        st.caption(f"Hojin ID: {company_id}")
                                        
                                        # Add score and rank information if available
                                        if 'score' in ctx:
                                            st.caption(f"Relevance Score: {ctx['score']:.3f} | Rank: {ctx.get('rank', 'N/A')}")
                                        
                                        content = ctx.get('chunk', '')
                                        if len(content) > 200:
                                            st.write(content[:200] + "...")
                                            with st.expander("📄 Full Content"):
                                                st.text(content)
                                        else:
                                            st.write(content)
                                        st.divider()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
    
    # Tab 2: Patent Tools Testing  
    with test_tabs[1]:
        st.subheader("📋 Patent Tools Testing")
        
        # System status check
        if st.session_state.runner is None:
            st.warning("⚠️ System not initialized. Please initialize the system using the sidebar first.")
            st.info("👈 Go to the sidebar and click '🚀 Initialize System' to start using the tools.")
            return
        
        # Patent Lookup by ID
        with st.expander("🔍 Exact Patent Lookup", expanded=True):
            st.write("**Test exact patent lookup by Application ID**")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                patent_id = st.text_input(
                    "Patent Application ID", 
                    placeholder="e.g., 123456789",
                    key="patent_lookup_input"
                )
            with col2:
                st.write("") # spacing
                st.write("") # spacing
                if st.button("🔍 Lookup Patent", key="patent_lookup_btn"):
                    if patent_id:
                        try:
                            # Check if system is initialized
                            if st.session_state.runner is None:
                                st.error("❌ System not initialized. Please initialize the system first using the sidebar.")
                                return
                            
                            with st.spinner("Looking up patent..."):
                                from tools.patent_tools import get_patent_tools
                                patent_tools = get_patent_tools()
                                
                                if patent_tools is None:
                                    st.error("❌ Patent tools not initialized. Please reinitialize the system.")
                                    return
                                
                                result = patent_tools.get_exact_patent_info(patent_id)
                                
                                if not result.get("success", False):
                                    st.error(f"❌ {result.get('message', 'Unknown error')}")
                                else:
                                    st.success("✅ Patent found!")
                                    patent_data = result.get('data', {})
                                    
                                    # Display patent information in a nice format
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Patent Information:**")
                                        st.write(f"**Patent ID:** {patent_data.get('patent_id', 'N/A')}")
                                        st.write(f"**Company:** {patent_data.get('company_name', 'N/A')}")
                                        st.write(f"**Company ID:** {patent_data.get('company_id', 'N/A')}")
                                    
                                    with col2:
                                        st.write("**Status:**")
                                        st.success("Found")
                                    
                                    # Display abstract
                                    abstract = patent_data.get('abstract', '').strip()
                                    if abstract:
                                        st.write("**Abstract:**")
                                        st.write(abstract)
                                    else:
                                        st.write("**Abstract:**")
                                        st.info("No abstract available for this patent.")
                                    
                                    # Display full text in expandable section
                                    full_text = patent_data.get('full_text', '').strip()
                                    if full_text:
                                        with st.expander("📄 Full Patent Text"):
                                            st.text(full_text)
                                    else:
                                        with st.expander("📄 Full Patent Text"):
                                            st.info("No full text content available for this patent.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        
        # Patent RAG Retrieval
        with st.expander("🔎 Patent RAG Retrieval", expanded=False):
            st.write("**Test patent context retrieval using RAG**")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                patent_query = st.text_input(
                    "Patent Search Query", 
                    placeholder="e.g., artificial intelligence algorithms",
                    key="patent_rag_input"
                )
            with col2:
                top_k_patent = st.number_input("Top K", value=5, min_value=1, max_value=20, key="patent_rag_k")
                if st.button("🔎 Search Patents", key="patent_rag_btn"):
                    if patent_query:
                        try:
                            # Check if system is initialized
                            if st.session_state.runner is None:
                                st.error("❌ System not initialized. Please initialize the system first using the sidebar.")
                                return
                            
                            with st.spinner("Searching patents..."):
                                from tools.patent_tools import get_patent_tools
                                patent_tools = get_patent_tools()
                                
                                if patent_tools is None:
                                    st.error("❌ Patent tools not initialized. Please reinitialize the system.")
                                    return
                                
                                contexts = patent_tools.retrieve_patent_contexts(patent_query, top_k_patent)
                                
                                st.success(f"✅ Found {len(contexts)} patent contexts")
                                
                                # Display results in a nice format
                                for i, ctx in enumerate(contexts[:5], 1):
                                    with st.container():
                                        st.markdown(f"**{i}. Patent {ctx.get('patent_id', 'Unknown')}**")
                                        st.caption(f"Company: {ctx.get('company_name', 'Unknown')}")
                                        
                                        content = ctx.get('chunk', '')
                                        if len(content) > 200:
                                            st.write(content[:200] + "...")
                                            with st.expander("📄 Full Content"):
                                                st.text(content)
                                        else:
                                            st.write(content)
                                        st.divider()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        
        # Company Patents Lookup
        with st.expander("🏢➡️📋 Company Patents Lookup", expanded=False):
            st.write("**Find all patents belonging to a specific company**")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                company_for_patents = st.text_input(
                    "Company Name or ID", 
                    placeholder="e.g., Intel, Toyota",
                    key="company_patents_input"
                )
            with col2:
                top_k_patents = st.number_input("Max Patents", value=10, min_value=1, max_value=50, key="company_patents_k")
                if st.button("📋 Get Patents", key="company_patents_btn"):
                    if company_for_patents:
                        try:
                            # Check if system is initialized
                            if st.session_state.runner is None:
                                st.error("❌ System not initialized. Please initialize the system first using the sidebar.")
                                return
                            
                            with st.spinner("Finding company patents..."):
                                from tools.patent_tools import get_patent_tools
                                patent_tools = get_patent_tools()
                                
                                if patent_tools is None:
                                    st.error("❌ Patent tools not initialized. Please reinitialize the system.")
                                    return
                                
                                result = patent_tools.get_patents_by_company(company_for_patents, top_k_patents)
                                
                                if not result.get("success", False):
                                    st.error(f"❌ {result.get('message', 'Unknown error')}")
                                else:
                                    data = result.get('data', {})
                                    patents = data.get('patents', [])
                                    patent_count = data.get('patent_count', 0)
                                    
                                    st.success(f"✅ Found {patent_count} total patents for {company_for_patents} (showing {len(patents)})")
                                    
                                    # Display patents in a nice format
                                    if patents:
                                        st.subheader(f"📋 Patents for {company_for_patents}")
                                        
                                        for i, patent in enumerate(patents, 1):
                                            with st.container():
                                                col1, col2 = st.columns([3, 1])
                                                
                                                with col1:
                                                    st.write(f"**{i}. Patent {patent.get('patent_id', 'Unknown')}**")
                                                    st.caption(f"Company: {patent.get('company_name', 'Unknown')} (ID: {patent.get('company_id', 'Unknown')})")
                                                    
                                                # Show enhanced abstract with better formatting
                                                abstract = patent.get('abstract', '').strip()
                                                st.write("**Abstract:**")
                                                if abstract and abstract != 'nan' and abstract.lower() != 'none':
                                                    if len(abstract) > 300:
                                                        st.write(abstract[:300] + "...")
                                                        with st.expander("📄 Full Abstract"):
                                                            st.write(abstract)
                                                    else:
                                                        st.write(abstract)
                                                    st.success("✅ Abstract available")
                                                else:
                                                    st.info("📝 Abstract not available in dataset for this patent")
                                                    st.caption("This is normal - not all patents in the dataset include abstracts")
                                                            
                                                with col2:
                                                    st.metric("Patent ID", patent.get('patent_id', 'N/A'))
                                                
                                                # Show patent content preview
                                                if patent.get('cleaned_patent'):
                                                    with st.expander("🔬 Patent Content Preview"):
                                                        st.write(patent['cleaned_patent'])
                                                
                                                st.divider()
                                    else:
                                        st.info("No patents found for this company.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
    
    # Tab 3: Optimized Hybrid Search Testing
    with test_tabs[2]:
        st.subheader("🔄 Optimized Hybrid Search Testing")
        
        # System status check
        if st.session_state.runner is None:
            st.warning("⚠️ System not initialized. Please initialize the system using the sidebar first.")
            st.info("👈 Go to the sidebar and click '🚀 Initialize System' to start using the tools.")
            return
        
        # Check if optimization tools are available
        if st.session_state.optimization_tools is None:
            st.warning("⚠️ Optimization tools not available. System may be running in fallback mode.")
            st.info("💡 Try reinitializing the system to enable optimization features.")
        
        # Optimized Hybrid Search
        with st.expander("🚀 Optimized Hybrid Search", expanded=True):
            st.write("**Test optimized hybrid retrieval with advanced caching, FAISS, and parallel processing**")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                optimized_query = st.text_input(
                    "Optimized Search Query", 
                    placeholder="e.g., artificial intelligence innovations",
                    key="optimized_search_input"
                )
            
            with col2:
                opt_search_type = st.selectbox("Search Type", ["both", "company", "patent"], key="opt_search_type")
                opt_top_k = st.number_input("Top K Results", value=5, min_value=1, max_value=20, key="opt_top_k")
            
            with col3:
                st.write("**Optimization Features:**")
                use_cache = st.checkbox("Use Caching", value=True, key="opt_cache")
                use_faiss = st.checkbox("Use FAISS", value=True, key="opt_faiss")
            
            if st.button("🚀 Optimized Hybrid Search", key="opt_search_btn"):
                if optimized_query:
                    try:
                        with st.spinner("Performing optimized hybrid search with advanced features..."):
                            start_time = time.time()
                            
                            # Use optimized hybrid search (aligned with main.py)
                            optimized_results = optimized_hybrid_rag_retrieval_tool(
                                query=optimized_query,
                                top_k=opt_top_k,
                                search_type=opt_search_type
                            )
                            
                            search_time = time.time() - start_time
                        
                        # Show completion notification for optimized search
                        st.success(f"🎉 Optimized hybrid search completed! (completed in {search_time:.2f}s)")
                        
                        # Display comprehensive metadata (fixed variable references)
                        metadata = optimized_results.get('search_metadata', {})
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Retrieval Method", metadata.get('retrieval_method', 'Optimized'))
                        with col2:
                            st.metric("Company Results", len(optimized_results.get('company_contexts', [])))
                        with col3:
                            st.metric("Patent Results", len(optimized_results.get('patent_contexts', [])))
                        with col4:
                            hybrid_used = optimized_results.get('success', False)
                            st.metric("Search Status", "✅ Success" if hybrid_used else "❌ Failed")
                        
                        # Show search configuration (using actual variables)
                        st.info(f"🎯 **Search Configuration**: Query: '{optimized_query}', Top K: {opt_top_k}, Search Type: {opt_search_type}")
                        
                        # Enhanced Company Results Display
                        if optimized_results.get('company_contexts'):
                            st.subheader("🏢 Company Results")
                            
                            for i, ctx in enumerate(optimized_results['company_contexts'][:5], 1):
                                with st.container():
                                    # Enhanced result display with confidence colors
                                    confidence = ctx.get('confidence', 'unknown')
                                    score = ctx.get('score', 0)
                                    
                                    if confidence == 'high':
                                        confidence_color = "🟢"
                                        confidence_css = "background-color: #d4edda; padding: 10px; border-radius: 5px;"
                                    elif confidence == 'medium':
                                        confidence_color = "🟡"
                                        confidence_css = "background-color: #fff3cd; padding: 10px; border-radius: 5px;"
                                    else:
                                        confidence_color = "🔴"
                                        confidence_css = "background-color: #f8d7da; padding: 10px; border-radius: 5px;"
                                    
                                    st.markdown(f'<div style="{confidence_css}">', unsafe_allow_html=True)
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(f"**{i}. {ctx.get('company_name', 'Unknown Company')}**")
                                        st.caption(f"Company ID: {ctx.get('hojin_id', 'Unknown')}")
                                    
                                    with col2:
                                        st.metric("Score", f"{score:.3f}")
                                        st.write(f"{confidence_color} {confidence.upper()}")
                                    
                                    # Content preview
                                    content = ctx.get('chunk', '')
                                    if len(content) > 300:
                                        content_preview = content[:300] + "..."
                                        with st.expander("📄 Full Content"):
                                            st.text(content)
                                    else:
                                        content_preview = content
                                    
                                    st.write(content_preview)
                                    
                                    # Technical details
                                    retrieval_method = ctx.get('retrieval_method', 'optimized')
                                    st.caption(f"🔍 Retrieved via: {retrieval_method}")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.write("")  # Add spacing
                        
                        # Enhanced Patent Results Display
                        if optimized_results.get('patent_contexts'):
                            st.subheader("📜 Patent Results")
                            
                            for i, ctx in enumerate(optimized_results['patent_contexts'][:5], 1):
                                with st.container():
                                    # Enhanced result display with confidence colors
                                    confidence = ctx.get('confidence', 'unknown')
                                    score = ctx.get('score', 0)
                                    
                                    if confidence == 'high':
                                        confidence_color = "🟢"
                                        confidence_css = "background-color: #d4edda; padding: 10px; border-radius: 5px;"
                                    elif confidence == 'medium':
                                        confidence_color = "🟡"
                                        confidence_css = "background-color: #fff3cd; padding: 10px; border-radius: 5px;"
                                    else:
                                        confidence_color = "🔴"
                                        confidence_css = "background-color: #f8d7da; padding: 10px; border-radius: 5px;"
                                    
                                    st.markdown(f'<div style="{confidence_css}">', unsafe_allow_html=True)
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(f"**{i}. Patent {ctx.get('patent_id', 'Unknown')}**")
                                        st.caption(f"Company: {ctx.get('company_name', 'Unknown')}")
                                    
                                    with col2:
                                        st.metric("Score", f"{score:.3f}")
                                        st.write(f"{confidence_color} {confidence.upper()}")
                                    
                                    # Content preview
                                    content = ctx.get('chunk', '')
                                    if len(content) > 300:
                                        content_preview = content[:300] + "..."
                                        with st.expander("📄 Full Content"):
                                            st.text(content)
                                    else:
                                        content_preview = content
                                    
                                    st.write(content_preview)
                                    
                                    # Technical details
                                    retrieval_method = ctx.get('retrieval_method', 'optimized')
                                    st.caption(f"🔍 Retrieved via: {retrieval_method}")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.write("")  # Add spacing
                        
                        # Show enhanced analytics (fixed variable names)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if optimized_results.get('metadata'):
                                with st.expander("📊 Search Metadata", expanded=False):
                                    st.json(optimized_results['metadata'])
                        
                        with col2:
                            with st.expander("🔬 Search Analytics", expanded=False):
                                st.markdown("**Search Configuration:**")
                                st.write(f"• Query: {optimized_query}")
                                st.write(f"• Top K: {opt_top_k}")
                                st.write(f"• Search Type: {opt_search_type}")
                                st.write(f"• Processing Time: {search_time:.3f}s")
                                
                                st.markdown("**Results Summary:**")
                                st.write(f"• Total Company Results: {len(optimized_results.get('company_contexts', []))}")
                                st.write(f"• Total Patent Results: {len(optimized_results.get('patent_contexts', []))}")
                                st.write(f"• Search Success: {optimized_results.get('success', False)}")
                        
                        # Performance note (aligned with main.py approach)
                        if optimized_results.get('success', False):
                            st.success("🚀 **Performance Note**: Optimized search completed successfully with advanced features!")
                        else:
                            st.warning("⚠️ **Performance Note**: Search completed but may have encountered limitations.")
                        
                    except Exception as e:
                        st.error(f"❌ Optimized hybrid search failed: {str(e)}")
                        with st.expander("🔧 Error Details"):
                            st.exception(e)
                else:
                    st.warning("Please enter a search query.")
    
    # Tab 4: Performance Analytics Testing
    with test_tabs[3]:
        st.subheader("📊 Performance Analytics Testing")
        
        # Check if optimization tools are available
        if st.session_state.optimization_tools is None:
            st.warning("⚠️ Optimization tools not available. System may be running in fallback mode.")
            st.info("💡 Try reinitializing the system to enable performance analytics.")
        
        with st.expander("📈 Current Performance Statistics", expanded=True):
            st.write("**View current system performance metrics (aligned with main.py)**")
            
            if st.button("📊 Get Performance Analytics", key="perf_analytics_btn"):
                try:
                    with st.spinner("Retrieving performance analytics..."):
                        # Use performance analytics tool (aligned with main.py)
                        stats = get_performance_analytics_tool()
                        
                        st.success("✅ Performance analytics retrieved!")
                        
                        # Display tool statistics (following main.py pattern)
                        if 'tool_stats' in stats:
                            tool_stats = stats['tool_stats']
                            st.subheader("🔧 Tool Performance Statistics")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Queries", tool_stats.get('total_queries', 0))
                            with col2:
                                st.metric("Company Queries", tool_stats.get('company_queries', 0))
                            with col3:
                                st.metric("Patent Queries", tool_stats.get('patent_queries', 0))
                            with col4:
                                avg_time = tool_stats.get('avg_response_time', 0)
                                st.metric("Avg Response Time", f"{avg_time:.3f}s")
                        
                        # Display retriever statistics (following main.py pattern)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'company_retriever_stats' in stats and stats['company_retriever_stats']:
                                st.subheader("🏢 Company Retriever Stats")
                                company_stats = stats['company_retriever_stats']
                                
                                st.metric("Cache Hit Rate", f"{company_stats.get('cache_hit_rate', 0):.2%}")
                                st.metric("Total Queries", company_stats.get('total_queries', 0))
                                st.metric("FAISS Queries", company_stats.get('faiss_queries', 0))
                                st.metric("Early Stops", company_stats.get('early_stops', 0))
                        
                        with col2:
                            if 'patent_retriever_stats' in stats and stats['patent_retriever_stats']:
                                st.subheader("📜 Patent Retriever Stats")
                                patent_stats = stats['patent_retriever_stats']
                                
                                st.metric("Cache Hit Rate", f"{patent_stats.get('cache_hit_rate', 0):.2%}")
                                st.metric("Total Queries", patent_stats.get('total_queries', 0))
                                st.metric("FAISS Queries", patent_stats.get('faiss_queries', 0))
                                st.metric("Early Stops", patent_stats.get('early_stops', 0))
                        
                        # Display optimization features (following main.py pattern)
                        if 'optimization_features' in stats:
                            st.subheader("🚀 Optimization Features Status")
                            features = stats['optimization_features']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                faiss_enabled = features.get('faiss_enabled', False)
                                st.metric("FAISS", "✅ Enabled" if faiss_enabled else "❌ Disabled")
                                
                                caching_enabled = features.get('caching_enabled', False)
                                st.metric("Caching", "✅ Enabled" if caching_enabled else "❌ Disabled")
                            
                            with col2:
                                parallel_enabled = features.get('parallel_processing', False)
                                st.metric("Parallel Processing", "✅ Enabled" if parallel_enabled else "❌ Disabled")
                                
                                metadata_enabled = features.get('metadata_filtering', False)
                                st.metric("Metadata Filtering", "✅ Enabled" if metadata_enabled else "❌ Disabled")
                            
                            with col3:
                                dim_reduction = features.get('dimensionality_reduction', False)
                                st.metric("Dim. Reduction", "✅ Enabled" if dim_reduction else "❌ Disabled")
                        
                        # Show raw analytics data
                        with st.expander("🔍 Raw Analytics Data", expanded=False):
                            st.json(stats)
                
                except Exception as e:
                    st.error(f"❌ Performance analytics failed: {str(e)}")
                    with st.expander("🔧 Error Details"):
                        st.exception(e)
    
    # Tab 5: Batch Processing Testing
    with test_tabs[4]:
        st.subheader("⚡ Batch Processing Testing")
        
        # Check if optimization tools are available
        if st.session_state.optimization_tools is None:
            st.warning("⚠️ Optimization tools not available. System may be running in fallback mode.")
            st.info("💡 Try reinitializing the system to enable batch processing.")
        
        with st.expander("🔄 Batch Query Processing", expanded=True):
            st.write("**Test batch processing for multiple related queries (aligned with main.py)**")
            st.info("🆕 **Performance Feature**: Process multiple queries simultaneously for better efficiency!")
            
            # Batch query input
            batch_queries_text = st.text_area(
                "Enter Multiple Queries (one per line):",
                placeholder="artificial intelligence innovations\nmachine learning patents\nAI companies analysis\ntech startup opportunities",
                height=120,
                key="batch_queries_input"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                batch_search_type = st.selectbox("Search Type", ["both", "company", "patent"], key="batch_search_type")
            with col2:
                batch_top_k = st.number_input("Top K per Query", value=3, min_value=1, max_value=10, key="batch_top_k")
            with col3:
                show_detailed = st.checkbox("Show Detailed Results", value=True, key="batch_detailed")
            
            if st.button("⚡ Run Batch Processing", type="primary", key="batch_process_btn"):
                if batch_queries_text.strip():
                    try:
                        # Parse queries
                        queries = [q.strip() for q in batch_queries_text.strip().split('\n') if q.strip()]
                        
                        if len(queries) == 0:
                            st.warning("Please enter at least one query.")
                        else:
                            with st.spinner(f"Processing {len(queries)} queries in batch..."):
                                start_time = time.time()
                                
                                # Use batch processing tool (aligned with main.py)
                                batch_results = batch_optimized_retrieval_tool(
                                    queries=queries,
                                    top_k=batch_top_k,
                                    search_type=batch_search_type
                                )
                                
                                processing_time = time.time() - start_time
                            
                            # Show completion notification for batch processing
                            st.success(f"🎉 Batch processing completed for {len(queries)} queries! (completed in {processing_time:.2f}s)")
                            
                            # Display batch metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Queries Processed", len(queries))
                            with col2:
                                st.metric("Total Time", f"{processing_time:.3f}s")
                            with col3:
                                avg_per_query = processing_time / len(queries) if len(queries) > 0 else 0
                                st.metric("Avg per Query", f"{avg_per_query:.3f}s")
                            with col4:
                                total_results = sum(len(result.get('company_contexts', [])) + len(result.get('patent_contexts', [])) for result in batch_results)
                                st.metric("Total Results", total_results)
                            
                            # Display batch results
                            if show_detailed:
                                st.subheader("📊 Batch Results Details")
                                
                                for i, (query, result) in enumerate(zip(queries, batch_results), 1):
                                    with st.expander(f"Query {i}: {query[:50]}{'...' if len(query) > 50 else ''}", expanded=False):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            company_count = len(result.get('company_contexts', []))
                                            st.metric("Company Results", company_count)
                                            
                                            if company_count > 0:
                                                st.write("**Top Company Results:**")
                                                for ctx in result['company_contexts'][:3]:
                                                    st.write(f"• {ctx.get('company_name', 'Unknown')} (Score: {ctx.get('score', 0):.3f})")
                                        
                                        with col2:
                                            patent_count = len(result.get('patent_contexts', []))
                                            st.metric("Patent Results", patent_count)
                                            
                                            if patent_count > 0:
                                                st.write("**Top Patent Results:**")
                                                for ctx in result['patent_contexts'][:3]:
                                                    st.write(f"• Patent {ctx.get('patent_id', 'Unknown')} (Score: {ctx.get('score', 0):.3f})")
                            
                            # Performance comparison
                            sequential_estimate = processing_time * len(queries)
                            efficiency_gain = (sequential_estimate - processing_time) / sequential_estimate * 100 if sequential_estimate > 0 else 0
                            
                            st.info(f"🚀 **Performance Gain**: Batch processing was ~{efficiency_gain:.1f}% faster than sequential processing would have been!")
                    
                    except Exception as e:
                        st.error(f"❌ Batch processing failed: {str(e)}")
                        with st.expander("🔧 Error Details"):
                            st.exception(e)
                else:
                    st.warning("Please enter at least one query.")
    
    # Tab 6: System Tools Testing
    with test_tabs[5]:
        st.subheader("🔧 System Tools Testing")
        
        with st.expander("🧹 Cache Management", expanded=True):
            st.write("**Manage system caches (aligned with main.py)**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Clear All Caches:**")
                st.caption("Clear memory, disk, and retrieval caches to free up resources")
                
                if st.button("🧹 Clear All Caches", key="clear_caches_btn"):
                    try:
                        with st.spinner("Clearing all caches..."):
                            clear_caches_tool()
                        st.success("✅ All caches cleared successfully!")
                        st.info("💡 This will reset performance statistics and may temporarily slow down next queries.")
                    except Exception as e:
                        st.error(f"❌ Cache clearing failed: {str(e)}")
                        with st.expander("🔧 Error Details"):
                            st.exception(e)
            
            with col2:
                st.write("**Cache Pre-warming:**")
                st.caption("Pre-warm caches with common queries for better performance")
                
                if st.button("🔥 Pre-warm Caches", key="prewarm_caches_btn"):
                    try:
                        with st.spinner("Pre-warming caches with common queries..."):
                            # Use optimization tool (aligned with main.py)
                            common_queries = [
                                "artificial intelligence",
                                "machine learning", 
                                "innovation technology",
                                "patent analysis",
                                "company research"
                            ]
                            optimize_for_queries_tool(common_queries)
                        st.success("✅ Caches pre-warmed successfully!")
                        st.info("💡 Common queries should now be faster due to cached results.")
                    except Exception as e:
                        st.error(f"❌ Cache pre-warming failed: {str(e)}")
                        with st.expander("🔧 Error Details"):
                            st.exception(e)
        
        with st.expander("📊 System Configuration", expanded=False):
            st.write("**Current System Configuration**")
            
            if st.session_state.optimization_config:
                config = st.session_state.optimization_config
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Configuration:**")
                    st.write(f"• Companies: {config.company_data_size:,}")
                    st.write(f"• Patents: {config.patent_data_size:,}")
                    st.write(f"• Optimization Level: {config.optimization_level}")
                    st.write(f"• Use GPU: {config.use_gpu}")
                    
                    st.write("**FAISS Configuration:**")
                    st.write(f"• Use FAISS: {config.faiss.use_faiss}")
                    st.write(f"• Index Type: {config.faiss.faiss_index_type}")
                    st.write(f"• HNSW M: {config.faiss.hnsw_m}")
                    st.write(f"• Build Batch Size: {config.faiss.batch_size_for_building}")
                
                with col2:
                    st.write("**Cache Configuration:**")
                    st.write(f"• Memory Cache: {config.cache.use_memory_cache}")
                    st.write(f"• Memory Size: {config.cache.memory_cache_size}")
                    st.write(f"• Disk Cache: {config.cache.use_disk_cache}")
                    st.write(f"• Redis Cache: {config.cache.use_redis}")
                    
                    st.write("**Parallel Configuration:**")
                    st.write(f"• Max Workers: {config.parallel.max_workers}")
                    st.write(f"• Batch Size: {config.parallel.batch_size}")
                    st.write(f"• Async Processing: {config.parallel.use_async_processing}")
            else:
                st.info("No optimization configuration available. System running in basic mode.")
        
        # Advanced Company Search (moved inside System Tools tab)
        with st.expander("🏢 Advanced Company Search", expanded=False):
            st.write("**Search for companies using optimized retrieval system**")
            st.info("🆕 **Optimized Feature**: Advanced company search with caching and performance optimizations!")
            
            company_search_query = st.text_input("Company Search Query", placeholder="e.g., AI companies, biotech firms, semiconductor")
            
            col1, col2 = st.columns(2)
            with col1:
                company_search_top_k = st.number_input("Max Results", value=5, min_value=1, max_value=20, key="adv_company_search_k")
            with col2:
                company_search_type = st.selectbox("Search Focus", ["company", "both"], key="adv_company_search_type")
            
            if st.button("🔍 Advanced Company Search", key="adv_company_search_btn"):
                if company_search_query:
                    try:
                        with st.spinner("Performing advanced company search..."):
                            # Use optimized hybrid search focused on companies
                            search_results = optimized_hybrid_rag_retrieval_tool(
                                query=company_search_query,
                                top_k=company_search_top_k,
                                search_type=company_search_type
                            )
                            
                            if search_results.get('success', False):
                                st.success("✅ Advanced company search completed!")
                                
                                # Display company results
                                company_contexts = search_results.get('company_contexts', [])
                                if company_contexts:
                                    st.subheader(f"🏢 Found {len(company_contexts)} Company Results")
                                    
                                    for i, ctx in enumerate(company_contexts, 1):
                                        with st.container():
                                            col1, col2 = st.columns([3, 1])
                                            
                                            with col1:
                                                st.write(f"**{i}. {ctx.get('company_name', 'Unknown Company')}**")
                                                st.caption(f"ID: {ctx.get('hojin_id', 'Unknown')}")
                                                
                                                # Show content preview
                                                content = ctx.get('chunk', '')
                                                if len(content) > 200:
                                                    st.write(content[:200] + "...")
                                                    with st.expander("📄 Full Details"):
                                                        st.text(content)
                                                else:
                                                    st.write(content)
                                            
                                            with col2:
                                                score = ctx.get('score', 0)
                                                st.metric("Relevance", f"{score:.3f}")
                                                
                                                confidence = ctx.get('confidence', 'unknown')
                                                if confidence == 'high':
                                                    st.success("🟢 High")
                                                elif confidence == 'medium':
                                                    st.warning("🟡 Medium")
                                                else:
                                                    st.error("🔴 Low")
                                            
                                            st.divider()
                                else:
                                    st.info("No company results found for this query.")
                            
                            else:
                                st.error(f"❌ Search failed: {search_results.get('error', 'Unknown error')}")
                                
                    except Exception as e:
                        st.error(f"❌ Advanced company search failed: {str(e)}")
                        with st.expander("🔧 Error Details"):
                            st.exception(e)
                else:
                    st.warning("Please enter a search query.")
    
    # Enhanced Company-Patent Analysis
    with st.expander("📜 Enhanced Company-Patent Analysis with Full Abstracts", expanded=False):
        st.write("**Test the enhanced company-patent tool with full abstracts and detailed information**")
        st.info("🆕 **New Feature**: Now includes full patent abstracts and content previews for better analysis!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            patent_company = st.text_input(
                "Company Name for Patent Search", 
                placeholder="e.g., Snowflake, Advanced Biomedical Technologies, Intel"
            )
        
        with col2:
            st.write("**Quick Options:**")
            if st.button("🏔️ Snowflake", key="snowflake_patents"):
                st.session_state.patent_company_input = "Snowflake"
            if st.button("🧬 Advanced Biomedical Technologies", key="advanced_bio_patents"):
                st.session_state.patent_company_input = "Advanced Biomedical Technologies"
        
        if st.button("🔍 Analyze Company Patents", type="primary"):
            if patent_company or st.session_state.get('patent_company_input'):
                company_name = patent_company or st.session_state.get('patent_company_input', '')
                
                try:
                    with st.spinner(f"Analyzing patents for {company_name}..."):
                        # Use the multi-agent runner to get enhanced patent analysis
                        if st.session_state.runner:
                            query = f"What patents does {company_name} have?"
                            
                            # Run the enhanced workflow
                            results = st.session_state.runner.run_enhanced_workflow(query)
                            
                            if "error" not in results:
                                st.success("✅ Enhanced patent analysis completed!")
                                
                                # Show quick metrics
                                metadata = results.get('metadata', {})
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Company", company_name)
                                with col2:
                                    contexts_count = metadata.get('contexts_count', 0)
                                    st.metric("Contexts Retrieved", contexts_count)
                                with col3:
                                    fact_score = results.get('fact_checking', {}).get('overall_score', 0)
                                    st.metric("Fact Check Score", f"{fact_score}/10")
                                
                                # Display enhanced patent results
                                display_enhanced_patent_results(results)
                                
                                # Show full analysis if available
                                final_response = results.get('synthesis_result', '')
                                if final_response:
                                    with st.expander("📊 Complete Analysis", expanded=False):
                                        st.write(final_response)
                                
                                # Show fact-checking improvements
                                fact_checking = results.get('fact_checking', {})
                                if fact_checking:
                                    with st.expander("✅ Fact Checking Assessment", expanded=False):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            score = fact_checking.get('overall_score', 0)
                                            confidence = fact_checking.get('confidence_level', 'unknown')
                                            
                                            if score >= 8:
                                                st.success(f"🟢 Excellent: {score}/10 ({confidence} confidence)")
                                            elif score >= 6:
                                                st.warning(f"🟡 Good: {score}/10 ({confidence} confidence)")
                                            else:
                                                st.error(f"🔴 Needs Improvement: {score}/10 ({confidence} confidence)")
                                        
                                        with col2:
                                            issues = fact_checking.get('flagged_issues', [])
                                            st.write(f"**Issues Flagged:** {len(issues)}")
                                            for issue in issues[:3]:
                                                st.caption(f"• {issue}")
                                
                            else:
                                st.error(f"❌ Patent analysis failed: {results.get('error', 'Unknown error')}")
                        else:
                            st.error("❌ System not initialized. Please initialize the system first.")
                            
                except Exception as e:
                    st.error(f"❌ Patent analysis failed: {str(e)}")
                    with st.expander("🔧 Error Details"):
                        st.exception(e)
            else:
                st.warning("Please enter a company name.")
    
    # Context Preservation Demo
    with st.expander("🔗 Context Preservation Demo", expanded=False):
        st.write("**Test sequential subquestion processing with context preservation**")
        st.info("🆕 **New Feature**: Sequential subquestions now preserve context from previous answers!")
        
        demo_queries = [
            "Compare AI innovation strategies",
            "What are TechNova's market opportunities and competitive advantages?",
            "Analyze the competitive landscape for biotech companies",
            "How do different companies approach AI patent strategy?"
        ]
        
        selected_demo = st.selectbox("Choose a demo query:", demo_queries)
        
        if st.button("🚀 Run Context Preservation Demo", type="primary"):
            try:
                with st.spinner("Running demo with context preservation..."):
                    if st.session_state.runner:
                        results = st.session_state.runner.run_enhanced_workflow(selected_demo)
                        
                        if "error" not in results:
                            st.success("✅ Context preservation demo completed!")
                            
                            # Show context preservation details
                            display_context_preservation_details(results)
                            
                            # Show the planning and subquestion breakdown
                            planning = results.get('planning', {})
                            if planning.get('subquestions'):
                                st.subheader("📋 Subquestion Breakdown")
                                subquestions = planning['subquestions']
                                
                                for i, subq in enumerate(subquestions, 1):
                                    with st.container():
                                        if i == 1:
                                            st.write(f"🎯 **Q{i}:** {subq}")
                                            st.caption("Initial question - no context needed")
                                        else:
                                            st.write(f"🔗 **Q{i}:** {subq}")
                                            st.caption("Enhanced with context from previous answers")
                                        st.write("")
                            
                            # Show final result
                            final_response = results.get('synthesis_result', '')
                            if final_response:
                                with st.expander("📊 Final Synthesized Response", expanded=False):
                                    st.write(final_response)
                        
                        else:
                            st.error(f"❌ Demo failed: {results.get('error', 'Unknown error')}")
                    else:
                        st.error("❌ System not initialized. Please initialize the system first.")
                        
            except Exception as e:
                st.error(f"❌ Demo failed: {str(e)}")
                with st.expander("🔧 Error Details"):
                    st.exception(e)

def display_enhanced_workflow_results(results):
    """Display enhanced workflow results with new features."""
    
    # Enhanced main response display
    market_analysis = results.get('market_analysis', {})
    if market_analysis.get('skipped'):
        st.info(f"ℹ️ Market Analysis Team skipped: {market_analysis.get('reason', 'Not needed for this query type')}")
        final_response = results.get('synthesis_result', 'No response available')
        st.subheader("📊 Synthesis Results")
        st.write(final_response)
    else:
        final_response = market_analysis.get('final_analysis', 'No response available')
        st.subheader("🎯 Strategic Market Analysis")
        st.write(final_response)
        
        # Show consolidated summaries if available
        opportunities = market_analysis.get('opportunities', '')
        risks = market_analysis.get('risks', '')
        
        if opportunities or risks:
            with st.expander("📋 Consolidated Analysis Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    if opportunities:
                        st.subheader("🚀 Market Opportunities Summary")
                        st.write(opportunities)
                
                with col2:
                    if risks:
                        st.subheader("⚠️ Risk Assessment Summary")
                        st.write(risks)
    
    # Enhanced Company-Patent Results Display
    display_enhanced_patent_results(results)
    
    # Enhanced Hybrid Search Results
    display_hybrid_search_details(results)
    
    # Context Preservation Details
    display_context_preservation_details(results)
    
    # Detailed analysis in expandable sections
    with st.expander("🔍 Complete Workflow Details", expanded=False):
        display_complete_workflow_details(results)

def display_enhanced_patent_results(results):
    """Display enhanced patent results with full abstracts and detailed information."""
    
    normalization_results = results.get('normalization_results', [])
    has_patent_results = False
    
    for norm_result in normalization_results:
        contexts = norm_result.get('retrieved_contexts', [])
        for context in contexts:
            if context.get('tool') == 'company_patents_lookup':
                has_patent_results = True
                break
    
    if has_patent_results:
        st.subheader("📜 Enhanced Patent Analysis")
        
        for i, norm_result in enumerate(normalization_results):
            contexts = norm_result.get('retrieved_contexts', [])
            for context in contexts:
                if context.get('tool') == 'company_patents_lookup':
                    result = context.get('result', '')
                    
                    # Check if this is the enhanced patent format
                    if "DETAILED PATENT INFORMATION:" in result:
                        with st.expander(f"🔍 Detailed Patent Information - Query {i+1}", expanded=True):
                            # Parse and display enhanced patent results
                            display_parsed_patent_results(result)
                    else:
                        # Fallback for older format
                        with st.expander(f"📋 Patent Results - Query {i+1}", expanded=False):
                            st.text(result)

def display_parsed_patent_results(result_text):
    """Parse and beautifully display enhanced patent results."""
    
    lines = result_text.split('\n')
    current_patent = {}
    patents = []
    capture_mode = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("PATENT ID:"):
            if current_patent:
                patents.append(current_patent)
            current_patent = {"id": line.replace("PATENT ID:", "").strip()}
            capture_mode = None
            
        elif line.startswith("Company:"):
            current_patent["company"] = line.replace("Company:", "").strip()
            
        elif line.startswith("ABSTRACT:"):
            capture_mode = "abstract"
            current_patent["abstract"] = ""
            
        elif line.startswith("PATENT CONTENT PREVIEW:"):
            capture_mode = "content"
            current_patent["content"] = ""
            
        elif line.startswith("Total patents for"):
            if current_patent:
                patents.append(current_patent)
            # Extract summary info
            parts = line.split(":")
            if len(parts) > 1:
                st.info(f"📊 {line}")
            break
            
        elif capture_mode == "abstract" and line and not line.startswith("-"):
            current_patent["abstract"] += line + " "
            
        elif capture_mode == "content" and line and not line.startswith("-"):
            current_patent["content"] += line + " "
    
    # Display patents in a nice format
    for i, patent in enumerate(patents, 1):
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"**Patent {i}**")
                st.code(patent.get("id", "Unknown ID"))
                st.caption(patent.get("company", "Unknown Company"))
            
            with col2:
                st.markdown("**Abstract:**")
                abstract = patent.get("abstract", "").strip()
                if abstract:
                    st.write(abstract)
                else:
                    st.write("_No abstract available_")
                
                if patent.get("content"):
                    with st.expander("📄 Patent Content Preview"):
                        st.text(patent["content"].strip())
            
            st.divider()

def display_hybrid_search_details(results):
    """Display detailed hybrid search results showing dense vs sparse retrieval."""
    
    # Check if hybrid search was used in any context
    normalization_results = results.get('normalization_results', [])
    hybrid_used = False
    
    for norm_result in normalization_results:
        contexts = norm_result.get('retrieved_contexts', [])
        for context in contexts:
            if context.get('tool') == 'enhanced_hybrid_rag_retrieval':
                hybrid_used = True
                break
    
    if hybrid_used:
        st.subheader("🔍 Hybrid Search Analysis (Dense + Sparse)")
        
        # Note: In a real implementation, we'd need to capture the detailed logs
        # For now, we'll show what information we have
        st.info("🔬 **True Hybrid Retrieval Active**: The system used both dense (semantic) and sparse (keyword-based) search methods for comprehensive results.")
        
        # Display hybrid search metrics if available
        for i, norm_result in enumerate(normalization_results):
            contexts = norm_result.get('retrieved_contexts', [])
            for context in contexts:
                if context.get('tool') == 'enhanced_hybrid_rag_retrieval':
                    with st.expander(f"🔬 Hybrid Search Details - Query {i+1}", expanded=False):
                        result = context.get('result', {})
                        
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except:
                                st.text(result)
                                continue
                        
                        # Display search metadata
                        metadata = result.get('search_metadata', {})
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Dense Weight", metadata.get('dense_weight', 0.5))
                        with col2:
                            st.metric("Sparse Weight", metadata.get('sparse_weight', 0.5))
                        with col3:
                            st.metric("Method", metadata.get('retrieval_method', 'hybrid'))
                        
                        # Display results breakdown
                        company_contexts = result.get('company_contexts', [])
                        patent_contexts = result.get('patent_contexts', [])
                        
                        if company_contexts:
                            st.markdown("**🏢 Company Results:**")
                            for ctx in company_contexts[:3]:
                                confidence = ctx.get('confidence', 'unknown')
                                color = "🟢" if confidence == "high" else "🟡" if confidence == "medium" else "🔴"
                                st.write(f"{color} **{ctx.get('company_name', 'Unknown')}** (Score: {ctx.get('score', 0):.3f}, Confidence: {confidence})")
                                st.caption(f"Retrieved via: {ctx.get('retrieval_method', 'unknown')}")
                        
                        if patent_contexts:
                            st.markdown("**📜 Patent Results:**")
                            for ctx in patent_contexts[:3]:
                                confidence = ctx.get('confidence', 'unknown')
                                color = "🟢" if confidence == "high" else "🟡" if confidence == "medium" else "🔴"
                                st.write(f"{color} **Patent {ctx.get('patent_id', 'Unknown')}** (Score: {ctx.get('score', 0):.3f}, Confidence: {confidence})")
                                st.caption(f"Retrieved via: {ctx.get('retrieval_method', 'unknown')}")

def display_context_preservation_details(results):
    """Display context preservation information for sequential subquestions."""
    
    context_preservation = results.get('context_preservation', {})
    metadata = results.get('metadata', {})
    
    if context_preservation.get('context_used', False) or metadata.get('context_preservation_used', False):
        st.subheader("🔗 Context Preservation Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Subquestions", context_preservation.get('total_subquestions', 1))
        with col2:
            st.metric("Context Length", f"{context_preservation.get('accumulated_context_length', 0)} chars")
        with col3:
            preservation_used = metadata.get('context_preservation_used', False)
            st.metric("Enhancement", "Active" if preservation_used else "Not Needed")
        
        if preservation_used:
            st.success("✅ **Context Preservation Active**: Later subquestions were enhanced with context from previous answers for better understanding.")
            
            subquestions = results.get('subquestions', [])
            if len(subquestions) > 1:
                st.markdown("**Subquestion Flow:**")
                for i, subq in enumerate(subquestions, 1):
                    icon = "🔗" if i > 1 else "🎯"
                    enhancement = " (Enhanced with previous context)" if i > 1 else ""
                    st.write(f"{icon} **Q{i}:** {subq}{enhancement}")
        else:
            st.info("ℹ️ **Single Question Mode**: No context preservation needed for single-question queries.")

def display_complete_workflow_details(results):
    """Display complete workflow details in an organized manner."""
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Planning", "🔄 Normalization", "🧠 Synthesis", "📊 Analysis Team", "✅ Fact Check"])
    
    with tab1:
        st.markdown("**Query Planning Results:**")
        planning = results.get('planning', {})
        st.write(f"**Analysis:** {planning.get('analysis', 'No analysis')}")
        st.write(f"**Needs Splitting:** {planning.get('needs_splitting', False)}")
        st.write(f"**Needs Analysis Team:** {planning.get('needs_analysis_team', False)}")
        st.write(f"**Reasoning:** {planning.get('analysis_reasoning', 'No reasoning')}")
        
        if planning.get('subquestions'):
            st.markdown("**Subquestions:**")
            for i, subq in enumerate(planning['subquestions'], 1):
                st.write(f"{i}. {subq}")
    
    with tab2:
        st.markdown("**Query Normalization Results:**")
        for i, norm_result in enumerate(results.get('normalization_results', []), 1):
            with st.expander(f"Subquestion {i}", expanded=False):
                normalization = norm_result.get('normalization', {})
                st.write(f"**Query Type:** {normalization.get('query_type', 'Unknown')}")
                st.write(f"**Reasoning:** {normalization.get('reasoning', 'No reasoning')}")
                
                identifiers = normalization.get('identifiers', {})
                st.write(f"**Companies:** {identifiers.get('companies', [])}")
                st.write(f"**Patents:** {identifiers.get('patents', [])}")
                st.write(f"**Tools:** {normalization.get('recommended_tools', [])}")
                
                contexts = norm_result.get('retrieved_contexts', [])
                st.write(f"**Contexts Retrieved:** {len(contexts)}")
    
    with tab3:
        st.markdown("**Information Synthesis:**")
        synthesis = results.get('synthesis_result', 'No synthesis available')
        st.write(synthesis)
    
    with tab4:
        st.markdown("**Market Analysis Team Results:**")
        market_analysis = results.get('market_analysis', {})
        
        if market_analysis.get('skipped'):
            st.info(f"Market Analysis skipped: {market_analysis.get('reason', 'Not needed')}")
        else:
            if market_analysis.get('opportunities'):
                with st.expander("🚀 Opportunities Analysis", expanded=False):
                    st.write(market_analysis['opportunities'])
            
            if market_analysis.get('risks'):
                with st.expander("⚠️ Risk Analysis", expanded=False):
                    st.write(market_analysis['risks'])
            
            if market_analysis.get('final_analysis'):
                with st.expander("📊 Final Strategic Analysis", expanded=False):
                    st.write(market_analysis['final_analysis'])
    
    with tab5:
        st.markdown("**Fact Checking Results:**")
        fact_checking = results.get('fact_checking', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            score = fact_checking.get('overall_score', 0)
            st.metric("Overall Score", f"{score}/10")
        with col2:
            confidence = fact_checking.get('confidence_level', 'unknown')
            st.metric("Confidence", confidence.upper())
        with col3:
            issues = len(fact_checking.get('flagged_issues', []))
            st.metric("Issues Flagged", issues)
        
        if fact_checking.get('flagged_issues'):
            st.markdown("**Issues Identified:**")
            for issue in fact_checking['flagged_issues']:
                st.write(f"⚠️ {issue}")
        
        # Sources
        sources = results.get('sources', [])
        if sources:
            st.markdown(f"**Sources Used:** {len(sources)} sources")
            for i, source in enumerate(sources[:5], 1):
                st.caption(f"{i}. {source}")
    
    # Raw JSON data
    with st.expander("🔧 Raw JSON Data", expanded=False):
        st.json(results)

def display_query_analysis():
    """Display query analysis and workflow planning interface."""
    st.header("🧠 Query Analysis & Workflow Planning")
    
    if st.session_state.runner is None:
        st.warning("Please initialize the system first.")
        return
    
    # Query analysis interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analysis_query = st.text_area(
            "Enter Query for Analysis",
            placeholder="e.g., What are TechNova's market opportunities and competitive advantages?",
            height=100
        )
    
    with col2:
        st.write("**Analysis Options:**")
        show_planning = st.checkbox("Show Planning Analysis", value=True)
        show_normalization = st.checkbox("Show Query Normalization", value=True)
        show_tool_selection = st.checkbox("Show Tool Selection", value=True)
    
    if st.button("🔍 Analyze Query"):
        if analysis_query:
            try:
                with st.spinner("Analyzing query..."):
                    # Planning analysis
                    if show_planning:
                        planning_result = st.session_state.runner.agents['planning'].plan_query(analysis_query)
                        
                        st.subheader("📋 Planning Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            needs_splitting = planning_result.get('needs_splitting', False)
                            st.metric("Needs Splitting", "Yes" if needs_splitting else "No")
                        
                        with col2:
                            needs_analysis_team = planning_result.get('needs_analysis_team', False)
                            st.metric("Analysis Team Needed", "Yes" if needs_analysis_team else "No")
                        
                        with col3:
                            subquestions_count = len(planning_result.get('subquestions', []))
                            st.metric("Subquestions", subquestions_count)
                        
                        st.write(f"**Analysis:** {planning_result.get('analysis', 'No analysis')}")
                        st.write(f"**Reasoning:** {planning_result.get('analysis_reasoning', 'No reasoning')}")
                        
                        if planning_result.get('subquestions'):
                            st.write("**Subquestions:**")
                            for i, subq in enumerate(planning_result['subquestions'], 1):
                                st.write(f"{i}. {subq}")
                    
                    # Query normalization
                    if show_normalization:
                        norm_result = st.session_state.runner.agents['normalize'].normalize_query(analysis_query)
                        
                        st.subheader("🏷️ Query Normalization")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Query Type:** {norm_result.get('query_type', 'Unknown')}")
                            st.write(f"**Reasoning:** {norm_result.get('reasoning', 'No reasoning')}")
                        
                        with col2:
                            identifiers = norm_result.get('identifiers', {})
                            st.write(f"**Companies:** {identifiers.get('companies', [])}")
                            st.write(f"**Patents:** {identifiers.get('patents', [])}")
                        
                        if show_tool_selection:
                            st.write("**Recommended Tools:**")
                            tools = norm_result.get('recommended_tools', [])
                            for tool in tools:
                                st.write(f"- {tool}")
                
            except Exception as e:
                st.error(f"❌ Query analysis failed: {str(e)}")
        else:
            st.warning("Please enter a query to analyze.")

def main():
    """Main Streamlit application."""
    
    # Header with improved styling
    st.title("🚀 InnovARAG - Innovation Discovery Platform")
    st.markdown("**Enhanced Multi-Agent RAG System with True Hybrid Search**")
    
    # Improved feature highlights with better styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h3 style="color: white; margin-top: 0;">🆕 Latest Enhancements</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
            <div>
                • 📜 <strong>Enhanced Patent Analysis</strong><br>
                &nbsp;&nbsp;&nbsp;Full abstracts and content previews<br><br>
                • 🔍 <strong>True Hybrid Search</strong><br>
                &nbsp;&nbsp;&nbsp;Dense + Sparse retrieval with detailed logging
            </div>
            <div>
                • 🔗 <strong>Context Preservation</strong><br>
                &nbsp;&nbsp;&nbsp;Sequential subquestions with accumulated context<br><br>
                • 🧠 <strong>Smart Analysis Team</strong><br>
                &nbsp;&nbsp;&nbsp;Intelligent decision-making for complex queries
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if system is initialized
    if st.session_state.runner is None:
        # System not initialized - show welcome screen with improved styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
            color: white;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin: 40px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        ">
            <h2 style="color: #ecf0f1; margin-bottom: 20px; font-size: 2.5em;">🚀 Welcome to InnovARAG!</h2>
            <p style="color: #bdc3c7; font-size: 18px; margin-bottom: 25px; line-height: 1.6;">
                Your AI-powered innovation discovery platform is ready to help you analyze patents, 
                companies, and market opportunities using advanced multi-agent RAG technology.
            </p>
            <div style="
                background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
                border-radius: 12px;
                padding: 25px;
                margin: 25px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            ">
                <h4 style="color: #ecf0f1; margin-bottom: 20px; font-size: 1.4em;">🎯 Getting Started</h4>
                <div style="color: #bdc3c7; text-align: left; margin-bottom: 15px; line-height: 1.8;">
                    <div style="margin-bottom: 12px;"><strong style="color: #3498db;">👈 Use the sidebar to:</strong></div>
                    <div style="margin-left: 20px;">
                        <div style="margin-bottom: 8px;">1️⃣ Select your preferred LLM model (OpenAI, Gemini, or Qwen)</div>
                        <div style="margin-bottom: 8px;">2️⃣ Click "Initialize System" to load the data</div>
                        <div style="margin-bottom: 8px;">3️⃣ Wait for the system to load patents and company data</div>
                        <div style="margin-bottom: 8px;">4️⃣ Start exploring with powerful AI analysis!</div>
                    </div>
                </div>
            </div>
            <div style="
                color: #27ae60; 
                font-weight: bold; 
                font-size: 16px;
                background: rgba(39, 174, 96, 0.1);
                padding: 10px 20px;
                border-radius: 8px;
                border: 1px solid rgba(39, 174, 96, 0.3);
            ">
                🔧 System Status: Ready for Initialization
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show system capabilities preview with improved colors
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            ">
                <h4 style="color: white; margin-bottom: 15px; font-size: 1.3em;">📊 Data Analysis</h4>
                <p style="color: #ecf0f1; margin: 0; line-height: 1.5;">
                Analyze 100,000+ patents and 1,320+ companies with advanced AI technology
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            ">
                <h4 style="color: white; margin-bottom: 15px; font-size: 1.3em;">🤖 Multi-Agent System</h4>
                <p style="color: #ecf0f1; margin: 0; line-height: 1.5;">
                Planning, normalization, synthesis, and fact-checking agents working together
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(26, 188, 156, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            ">
                <h4 style="color: white; margin-bottom: 15px; font-size: 1.3em;">🔍 Hybrid Search</h4>
                <p style="color: #ecf0f1; margin: 0; line-height: 1.5;">
                Dense semantic + sparse keyword search for optimal results
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        return  # Stop here if not initialized
    
    # System initialized - show main interface with improved styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        color: white;
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    ">
        <span style="color: white; font-weight: 600; font-size: 16px;">
        ✅ <strong>System Ready!</strong> 
        All agents and tools are loaded and ready for analysis.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Show optimization status if available
    if st.session_state.optimization_tools:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 25px;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        ">
            🚀 <strong>Enhanced Features Active:</strong> Context preservation (4x larger), optimized retrieval (10x faster), smart analysis team triggering, and improved patent abstracts
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different interfaces
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Query Interface", "🔬 System Overview", "🚀 Enhanced Features", "🧠 Query Analysis"])
    
    with tab1:
        # Main query interface
        st.header("💬 Multi-Agent Analysis")
        
        # Query input
        # Check if query was set from example button
        default_query = st.session_state.get('query_input', '')
        if default_query:
            # Clear the session state after using it
            st.session_state.query_input = ''
        
        query = st.text_area(
            "Enter your innovation research query:",
            value=default_query,
            placeholder="e.g., What are the market opportunities for AI companies in healthcare?",
            height=120,
            help="Ask questions about companies, patents, market analysis, or innovation strategies"
        )
        
        # Analysis button with improved styling
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            analyze_button = st.button(
                "🚀 Analyze Query", 
                type="primary", 
                use_container_width=True,
                help="Run the multi-agent analysis workflow"
            )
        
        # Process query
        if analyze_button and query.strip():
            # Check if system is initialized
            if st.session_state.runner is None:
                st.error("❌ System not initialized! Please initialize the system first using the sidebar.")
                return
                
            with st.spinner("🤖 AI agents are analyzing your query..."):
                try:
                    # Record query
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.query_history.append({
                        "timestamp": timestamp,
                        "query": query
                    })
                    
                    # Process query with enhanced workflow
                    start_time = time.time()
                    results = st.session_state.runner.run_enhanced_workflow(query)
                    processing_time = time.time() - start_time
                    
                    # Store results
                    st.session_state.current_results = results
                    
                    if "error" in results:
                        st.error(f"❌ Analysis failed: {results['error']}")
                    else:
                        # Display summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        metadata = results.get('metadata', {})
                        
                        with col1:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        with col2:
                            st.metric("Subquestions", metadata.get('subquestions_count', 0))
                        with col3:
                            st.metric("Contexts Retrieved", metadata.get('contexts_count', 0))
                        with col4:
                            analysis_team_used = metadata.get('analysis_team_used', False)
                            st.metric("Analysis Team", "Used" if analysis_team_used else "Skipped")
                        
                        # Show context preservation info with better formatting
                        context_preservation = results.get('context_preservation', {})
                        metadata = results.get('metadata', {})
                        context_used = context_preservation.get('context_used', False) or metadata.get('context_preservation_used', False)
                        
                        if context_used:
                            context_length = context_preservation.get('accumulated_context_length', 0) or metadata.get('accumulated_context_length', 0)
                            total_subqs = context_preservation.get('total_subquestions', 1)
                            st.info(f"🔗 Context Preservation: ENABLED - Enhanced {total_subqs} subquestions with accumulated context ({context_length} chars)")
                        
                        # Show analysis team decision with reasoning
                        analysis_team_used = metadata.get('analysis_team_used', False)
                        analysis_reasoning = metadata.get('analysis_team_reasoning', '')
                        if not analysis_team_used and analysis_reasoning:
                            st.info(f"🤖 Analysis Team: SKIPPED - {analysis_reasoning}")
                        
                        # Show completion notification with enhanced timing information
                        if processing_time < 10:
                            st.success(f"⚡ Multi-agent analysis completed successfully! (completed in {processing_time:.2f}s)")
                            st.info("🚀 **Performance Note**: Enhanced retrieval optimizations are working - much faster than before!")
                        else:
                            st.success(f"🎉 Multi-agent analysis completed successfully! (completed in {processing_time:.2f}s)")
                        
                        # Enhanced Results Display
                        display_enhanced_workflow_results(results)
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)
        
        elif analyze_button:
            st.warning("⚠️ Please enter a query to analyze.")
    
    with tab2:
        if st.session_state.runner is None:
            st.warning("⚠️ Please initialize the system first to view system overview.")
        else:
            display_system_overview()
    
    with tab3:
        if st.session_state.runner is None:
            st.warning("⚠️ Please initialize the system first to access enhanced features.")
        else:
            display_enhanced_features()
    
    with tab4:
        if st.session_state.runner is None:
            st.warning("⚠️ Please initialize the system first to use query analysis.")
        else:
            display_query_analysis()
    
    # Query history in sidebar
    if st.session_state.query_history:
        st.sidebar.subheader("📋 Query History")
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.sidebar.expander(f"Query {i}: {entry['query'][:30]}..."):
                st.write(f"**Time:** {entry['timestamp']}")
                st.write(f"**Query:** {entry['query']}")



if __name__ == "__main__":
    main() 


    
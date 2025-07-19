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
    from tools.hybrid_rag_tools import hybrid_rag_retrieval_tool
    from utils.logging_utils import setup_logging, get_logger
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
except ImportError as e:
    st.error(f"❌ Error importing InnovARAG components: {e}")
    st.stop()

# App styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .workflow-step {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .step-complete {
        background-color: #d1edff;
        border-color: #0084ff;
    }
    .step-running {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .step-error {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.runner = None
    st.session_state.query_history = []
    st.session_state.current_results = None

@st.cache_resource
def initialize_innovarag():
    """Initialize InnovARAG system components."""
    try:
        logger.info("Initializing InnovARAG system for Streamlit...")
        
        # Load data
        index_dir = "RAG_INDEX"
        patent_df = pd.read_csv(patent_config.get("patent_csv"))
        firm_df = pd.read_csv(firm_config.get("firm_csv"))
        
        # Initialize RAG systems
        firm_rag = FirmSummaryRAG(df=firm_df, index_dir=index_dir, config=firm_config)
        patent_rag = PatentRAG(df=patent_df, index_dir=index_dir, config=patent_config)
        
        # Initialize tools
        init_company_tools(firm_df, index_dir)
        init_patent_tools(patent_df, index_dir)
        
        # Initialize Multi-Agent Runner
        runner = MultiAgentRunner()
        
        # Register tools
        company_tools = init_company_tools(firm_rag, index_dir)
        patent_tools = init_patent_tools(patent_rag, index_dir)
        
        all_tools = {**company_tools, **patent_tools}
        all_tools['hybrid_rag_retrieval'] = hybrid_rag_retrieval_tool
        
        runner.register_tools(all_tools)
        
        logger.info("InnovARAG system initialized successfully")
        return runner, len(patent_df), len(firm_df)
        
    except Exception as e:
        logger.error(f"Error initializing InnovARAG: {e}")
        raise e

def display_header():
    """Display the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 InnovARAG</h1>
        <h3>Innovation Discovery Multi-Agent RAG Platform</h3>
        <p>Discover market opportunities, analyze risks, and explore innovation landscapes through AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)

def display_workflow_progress(results):
    """Display the workflow progress and results."""
    if not results or "error" in results:
        return
    
    st.subheader("🔄 Workflow Progress")
    
    # Planning step
    planning = results.get('planning', {})
    with st.expander("📋 Step 1: Query Planning", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Analysis:**", planning.get('analysis', 'N/A'))
        with col2:
            needs_splitting = planning.get('needs_splitting', False)
            st.write("**Needs Splitting:**", "✅ Yes" if needs_splitting else "❌ No")
        
        subquestions = results.get('subquestions', [])
        if len(subquestions) > 1:
            st.write("**Subquestions:**")
            for i, subq in enumerate(subquestions, 1):
                st.write(f"{i}. {subq}")
    
    # Normalization and Retrieval
    with st.expander("🔍 Step 2: Information Retrieval", expanded=True):
        total_contexts = results.get('total_contexts', 0)
        st.metric("Contexts Retrieved", total_contexts)
        
        normalization_results = results.get('normalization_results', [])
        for i, norm_result in enumerate(normalization_results):
            with st.container():
                st.write(f"**Subquestion {i+1} Results:**")
                retrieved_contexts = norm_result.get('retrieved_contexts', [])
                for j, context in enumerate(retrieved_contexts):
                    tool = context.get('tool', 'unknown')
                    st.write(f"- Tool: {tool}")
    
    # Synthesis
    with st.expander("🧠 Step 3: Information Synthesis", expanded=False):
        synthesis = results.get('synthesis_result', '')
        if synthesis:
            st.write("**Synthesized Information:**")
            st.write(synthesis)
    
    # Market Analysis
    market_analysis = results.get('market_analysis', {})
    if market_analysis:
        with st.expander("📊 Step 4: Market Analysis", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Opportunities", "Risks", "Final Analysis"])
            
            with tab1:
                opportunities = market_analysis.get('opportunities', '')
                if opportunities:
                    st.write(opportunities)
            
            with tab2:
                risks = market_analysis.get('risks', '')
                if risks:
                    st.write(risks)
            
            with tab3:
                final_analysis = market_analysis.get('final_analysis', '')
                if final_analysis:
                    st.write(final_analysis)
    
    # Fact Checking
    fact_checking = results.get('fact_checking', {})
    if fact_checking:
        with st.expander("✅ Step 5: Fact Checking & Validation", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                overall_score = fact_checking.get('overall_score', 0)
                st.metric("Overall Score", f"{overall_score}/10")
            
            with col2:
                confidence_level = fact_checking.get('confidence_level', 'unknown')
                confidence_color = {
                    'high': '🟢', 'medium': '🟡', 'low': '🔴'
                }.get(confidence_level, '⚪')
                st.metric("Confidence", f"{confidence_color} {confidence_level.upper()}")
            
            with col3:
                flagged_issues = fact_checking.get('flagged_issues', [])
                st.metric("Issues Flagged", len(flagged_issues))
            
            if flagged_issues:
                st.write("**Issues Identified:**")
                for issue in flagged_issues[:3]:
                    st.write(f"- {issue}")

def main():
    """Main Streamlit application."""
    display_header()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display current LLM configuration
        st.info(f"**Current LLM:** {DEFAULT_LLM_TYPE}")
        st.caption("Configure LLMs in config/agent_config.py")
        
        # Query mode selection
        st.header("🎯 Query Mode")
        query_mode = st.selectbox(
            "Choose your query type:",
            options=[
                "Company Analysis",
                "Patent Analysis", 
                "General Innovation",
                "Freestyle Query"
            ],
            help="Select the type of analysis you want to perform"
        )
        
        # Display examples based on mode
        if query_mode == "Company Analysis":
            st.markdown("""
            **Examples:**
            - "Tell me about TechNova's business focus"
            - "What are InnovateCorp's market opportunities?"
            - "Compare two companies' strategies"
            """)
        elif query_mode == "Patent Analysis":
            st.markdown("""
            **Examples:**
            - "What is Patent 273556553 about?"
            - "What AI patents does TechNova hold?"
            - "Latest trends in chemical patents"
            """)
        elif query_mode == "General Innovation":
            st.markdown("""
            **Examples:**
            - "Trends in renewable energy innovation"
            - "AI technology market opportunities"
            - "Biotech investment landscape"
            """)
        else:
            st.markdown("""
            **Examples:**
            - Any complex innovation question
            - Multi-company comparisons
            - Cross-industry analysis
            """)
        
        # System status
        st.header("📊 System Status")
        if not st.session_state.initialized:
            if st.button("🔄 Initialize System"):
                with st.spinner("Initializing InnovARAG system..."):
                    try:
                        runner, n_patents, n_firms = initialize_innovarag()
                        st.session_state.runner = runner
                        st.session_state.initialized = True
                        st.success("System initialized successfully!")
                        st.metric("Patents", f"{n_patents:,}")
                        st.metric("Companies", f"{n_firms:,}")
                    except Exception as e:
                        st.error(f"Initialization failed: {e}")
        else:
            st.success("✅ System Ready")
            if st.button("🔄 Restart System"):
                st.cache_resource.clear()
                st.session_state.initialized = False
                st.rerun()
    
    # Main interface
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the system using the sidebar.")
        st.info("""
        **Welcome to InnovARAG!**
        
        This platform uses advanced AI agents to analyze:
        - 🏢 Company information and market positioning
        - 📝 Patent portfolios and technology trends  
        - 💡 Innovation opportunities and market risks
        - 🎯 Strategic recommendations with fact-checking
        
        Click "Initialize System" in the sidebar to get started.
        """)
        return
    
    # Query input
    st.header("💬 Ask Your Question")
    
    # Pre-fill based on query mode
    mode_examples = {
        "Company Analysis": "Tell me about TechNova's business focus and market opportunities",
        "Patent Analysis": "What is the technology behind Patent 273556553?",
        "General Innovation": "What are the latest trends in AI technology patents?",
        "Freestyle Query": "Compare TechNova and InnovateCorp's innovation strategies"
    }
    
    default_query = mode_examples.get(query_mode, "")
    
    query = st.text_area(
        "Enter your question:",
        value=default_query,
        height=100,
        help="Ask about companies, patents, market trends, or any innovation-related topic"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.current_results = None
        st.experimental_rerun()
    
    # Process query
    if analyze_button and query.strip():
        with st.spinner("🤖 AI agents are analyzing your query..."):
            try:
                # Record query
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.query_history.append({
                    "timestamp": timestamp,
                    "query": query,
                    "mode": query_mode
                })
                
                # Process query
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
                        confidence = metadata.get('confidence_level', 'unknown')
                        st.metric("Confidence", confidence.upper())
                    
                    st.success("✅ Analysis completed successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.current_results and "error" not in st.session_state.current_results:
        st.divider()
        display_workflow_progress(st.session_state.current_results)
        
        # Download results
        st.subheader("💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download JSON"):
                results_json = json.dumps(st.session_state.current_results, indent=2)
                st.download_button(
                    label="Download Complete Results",
                    data=results_json,
                    file_name=f"innovarag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📄 Download Summary"):
                # Create summary text
                results = st.session_state.current_results
                summary = f"""
InnovARAG Analysis Summary
=========================
Query: {results.get('query', '')}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Planning Analysis:
{results.get('planning', {}).get('analysis', '')}

Final Market Analysis:
{results.get('market_analysis', {}).get('final_analysis', '')}

Fact Check Score: {results.get('fact_checking', {}).get('overall_score', 0)}/10
Confidence Level: {results.get('fact_checking', {}).get('confidence_level', 'unknown').upper()}
                """
                
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"innovarag_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # Query history
    if st.session_state.query_history:
        st.divider()
        st.subheader("📚 Query History")
        
        with st.expander("View Previous Queries", expanded=False):
            for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):
                st.write(f"**{entry['timestamp']}** ({entry['mode']})")
                st.write(f"_{entry['query']}_")
                st.divider()

if __name__ == "__main__":
    main() 
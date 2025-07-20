"""
Multi-Agent Runner: Orchestrates the enhanced multi-agent RAG workflow.

This module coordinates the execution of different agents in sequence:
1. PlanningAgent - Analyzes queries and splits into subquestions if needed
2. NormalizeQueryAgent - Classifies queries and invokes appropriate tools for each subquestion
3. GeneralizeAgent - Synthesizes information from multiple sources with source attribution

5. Market Analysts - Analyze opportunities and risks with comprehensive scoring
6. FactCheckingAgent - Validates responses for accuracy and consistency
7. Market Manager - Provides final strategic recommendations with confidence assessment
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from typing import List, Dict, Optional, Any
from agents.base import BaseAgent
from agents.planning_agent import PlanningAgent
from agents.normalize_query_agent import NormalizeQueryAgent
from agents.generalize_agent import GeneralizeAgent
  
from agents.fact_checking_agent import FactCheckingAgent
from agents.market_analysts.market_opportunity_agent import MarketOpportunityAgent
from agents.market_analysts.market_risk_agent import MarketRiskAgent
from agents.market_analysts.market_manager_agent import MarketManagerAgent
from config.agent_config import DEFAULT_LLM_TYPE, agent_config

logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()

class MultiAgentRunner:
    """Enhanced Multi-Agent Runner with comprehensive workflow orchestration."""
    
    def __init__(self, llm_type: str = None):
        """
        Initialize the MultiAgentRunner with enhanced workflow capabilities.
        
        Args:
            llm_type: Type of LLM to use for all agents (defaults to config setting)
        """
        self.llm_type = llm_type or DEFAULT_LLM_TYPE
        self.agents: Dict[str, BaseAgent] = {}
        self.shared_memory: Dict[str, Any] = {}
        self.tools = {}
        self.workflow_results: Dict[str, Any] = {}
        
        logger.info(f"MultiAgentRunner initialized with {self.llm_type} LLM")
        
        # Initialize all agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents for the workflow."""
        try:
            logger.info("Initializing all agents for enhanced workflow")
            
            # Initialize core agents with config-specified LLMs
            planning_llm = agent_config.get("planning_agent", self.llm_type)
            self.agents['planning'] = PlanningAgent(llm_type=planning_llm)
            
            normalize_llm = agent_config.get("normalize_query_agent", self.llm_type)
            self.agents['normalize'] = NormalizeQueryAgent(name="NormalizeQueryAgent", qa_model=normalize_llm)
            
            generalize_llm = agent_config.get("generalize_agent", self.llm_type)
            self.agents['generalize'] = GeneralizeAgent(name="GeneralizeAgent", qa_model=generalize_llm)

            # Initialize market analysis agents with config-specified LLMs
            opportunity_llm = agent_config.get("market_opportunity_agent", self.llm_type)
            self.agents['market_opportunity'] = MarketOpportunityAgent(name="MarketOpportunityAgent", qa_model=opportunity_llm)
            
            risk_llm = agent_config.get("market_risk_agent", self.llm_type)
            self.agents['market_risk'] = MarketRiskAgent(name="MarketRiskAgent", qa_model=risk_llm)
            
            manager_llm = agent_config.get("market_manager_agent", self.llm_type)
            self.agents['market_manager'] = MarketManagerAgent(name="MarketManagerAgent", qa_model=manager_llm)
            
            # Initialize fact checking agent with config-specified LLM
            fact_check_llm = agent_config.get("fact_checking_agent", self.llm_type)
            self.agents['fact_checker'] = FactCheckingAgent(llm_type=fact_check_llm)
            
            logger.info(f"Successfully initialized {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise

    def register_tools(self, tools: dict):
        """Register tools for agents to use."""
        self.tools.update(tools)
        logger.info(f"Registered {len(tools)} tools: {list(tools.keys())}")
        
        # Propagate tools to all agents that support them
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'register_tools'):
                agent.register_tools(tools)
                logger.debug(f"Registered tools with {agent_name}")

    def run_enhanced_workflow(self, query: str) -> Dict[str, Any]:
        """
        Run the enhanced multi-agent workflow with comprehensive features.
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary containing all workflow results and metadata
        """
        logger.info(f"Starting enhanced multi-agent workflow for query: {query}")
        
        try:
            workflow_start_time = logger.info("Workflow started")
            
            # Step 1: Planning - Analyze query and split into subquestions if needed
            logger.info("Step 1: Query Planning and Analysis")
            planning_result = self.agents['planning'].plan_query(query)
            
            subquestions = planning_result.get('subquestions', [query])
            needs_splitting = planning_result.get('needs_splitting', False)
            
            logger.info(f"Planning complete. Needs splitting: {needs_splitting}, Subquestions: {len(subquestions)}")
            
            # Step 2: Process each subquestion through normalization and retrieval
            logger.info("Step 2: Processing subquestions through normalization and retrieval")
            all_contexts = []
            normalization_results = []
            
            for i, subq in enumerate(subquestions):
                logger.info(f"Processing subquestion {i+1}/{len(subquestions)}: {subq}")
                
                # Normalize and retrieve information for this subquestion
                norm_result = self.agents['normalize'].normalize_and_retrieve(subq)
                normalization_results.append(norm_result)
                
                # Extract contexts from this subquestion
                retrieved_contexts = norm_result.get('retrieved_contexts', [])
                all_contexts.extend(retrieved_contexts)
                
                logger.info(f"Retrieved {len(retrieved_contexts)} contexts for subquestion {i+1}")
            
            # Step 3: Synthesize information using GeneralizeAgent
            logger.info("Step 3: Information Synthesis")
            synthesis_result = self.agents['generalize'].synthesize_information(
                original_query=query,
                subquestions=subquestions,
                contexts=all_contexts
            )
            
            logger.info(f"Synthesis complete. Response length: {len(synthesis_result)} characters")
            
            # Step 4: Market Analysis - Opportunities and Risks
            logger.info("Step 4: Market Analysis")
            
            # Prepare context for market analysis
            market_context = {
                "question": query,
                "synthesis_result": synthesis_result,
                "contexts": all_contexts
            }
            
            # Market Opportunity Analysis
            opportunity_analysis = self.agents['market_opportunity'].run(market_context)
            logger.info(f"Market opportunity analysis complete")
            
            # Market Risk Analysis  
            risk_analysis = self.agents['market_risk'].run(market_context)
            logger.info(f"Market risk analysis complete")
            
            # Step 5: Market Manager Synthesis
            logger.info("Step 5: Strategic Synthesis by Market Manager")
            
            manager_input = {
                "question": query,
                "synthesis_result": synthesis_result,
                "opportunity_analysis": opportunity_analysis,
                "risk_analysis": risk_analysis,
                "contexts": all_contexts
            }
            
            final_market_analysis = self.agents['market_manager'].run(manager_input)
            logger.info(f"Market manager analysis complete")
            
            # Step 6: Fact Checking and Validation
            logger.info("Step 6: Fact Checking and Validation")
            
            # Extract sources for fact checking
            sources = self._extract_sources_from_contexts(all_contexts)
            
            # Validate the final response with enhanced source checking
            validation_result = self.agents['fact_checker'].validate_response(
                query=query,
                response=final_market_analysis,
                sources=sources,
                contexts=all_contexts  # Pass actual contexts for enhanced verification
            )
            
            logger.info(f"Fact checking complete. Overall score: {validation_result.get('overall_score', 0)}/10")
            
            # Step 7: Compile comprehensive results
            workflow_results = {
                "query": query,
                "planning": planning_result,
                "subquestions": subquestions,
                "normalization_results": normalization_results,
                "total_contexts": len(all_contexts),
                "synthesis_result": synthesis_result,
                "market_analysis": {
                    "opportunities": opportunity_analysis,
                    "risks": risk_analysis,
                    "final_analysis": final_market_analysis
                },
                "fact_checking": validation_result,
                "sources": sources,
                "metadata": {
                    "llm_type": self.llm_type,
                    "subquestions_count": len(subquestions),
                    "contexts_count": len(all_contexts),
                    "confidence_level": validation_result.get('confidence_level', 'unknown'),
                    "overall_score": validation_result.get('overall_score', 0)
                }
            }
            
            # Print summary to screen
            self._print_workflow_summary(workflow_results)
            
            logger.info("Enhanced workflow completed successfully")
            return workflow_results
            
        except Exception as e:
            logger.error(f"Error in enhanced workflow: {e}")
            return {
                "query": query,
                "error": str(e),
                "status": "failed"
            }

    def _extract_sources_from_contexts(self, contexts: List[Dict[str, Any]]) -> List[str]:
        """Extract source information from retrieved contexts."""
        sources = []
        
        for context in contexts:
            tool_name = context.get('tool', 'unknown')
            result = context.get('result', '')
            
            if isinstance(result, dict) and 'chunks' in result:
                # Handle structured results with chunks
                for chunk in result['chunks']:
                    if isinstance(chunk, dict):
                        source = chunk.get('source', f'Unknown source from {tool_name}')
                        if source not in sources:
                            sources.append(source)
            else:
                # Handle simple string results
                sources.append(f"Result from {tool_name}")
        
        return sources

    def _print_workflow_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of workflow results to the screen."""
        logger.info("\n" + "="*80)
        logger.info("ENHANCED MULTI-AGENT WORKFLOW SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\nQuery: {results['query']}")
        
        # Planning summary
        planning = results.get('planning', {})
        logger.info(f"\nPlanning: {planning.get('analysis', 'No analysis')}")
        logger.info(f"Subquestions: {len(results.get('subquestions', []))}")
        
        # Context summary
        logger.info(f"\nRetrieved Contexts: {results.get('total_contexts', 0)}")
        
        # Synthesis preview
        synthesis = results.get('synthesis_result', '')
        logger.info(f"\nSynthesis:")
        logger.info(synthesis)
        
        # Market analysis summary
        market_analysis = results.get('market_analysis', {})
        final_analysis = market_analysis.get('final_analysis', '')
        logger.info(f"\nFinal Market Analysis:")
        logger.info(final_analysis)
        
        # Fact checking summary
        fact_checking = results.get('fact_checking', {})
        overall_score = fact_checking.get('overall_score', 0)
        confidence_level = fact_checking.get('confidence_level', 'unknown')
        flagged_issues = fact_checking.get('flagged_issues', [])
        
        logger.info(f"\nFact Checking Results:")
        logger.info(f"  Overall Score: {overall_score}/10 ({confidence_level.upper()} confidence)")
        logger.info(f"  Issues Flagged: {len(flagged_issues)}")
        
        if flagged_issues:
            logger.info("  Issues:")
            for issue in flagged_issues[:3]:  # Show first 3 issues
                logger.info(f"    - {issue}")
        
        logger.info("\n" + "="*80 + "\n")

    # Alias
    run = run_enhanced_workflow



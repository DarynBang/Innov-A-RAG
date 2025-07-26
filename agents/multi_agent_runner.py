"""
Multi-Agent Runner: Orchestrates the enhanced multi-agent RAG workflow.

This module coordinates the execution of different agents in sequence:
1. PlanningAgent - Analyzes queries, splits into subquestions if needed, decides if analysis team is needed
2. NormalizeQueryAgent - Classifies queries and invokes appropriate tools for each subquestion
3. GeneralizeAgent - Synthesizes information from multiple sources with source attribution
4. Market Analysts (conditional) - Analyze opportunities and risks with comprehensive scoring
5. FactCheckingAgent - Validates responses for accuracy and consistency
6. Market Manager (conditional) - Provides final strategic recommendations with confidence assessment
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from typing import List, Dict, Any
from agents.base import BaseAgent
from agents.planning_agent import PlanningAgent
from agents.normalize_query_agent import NormalizeQueryAgent
from agents.generalize_agent import GeneralizeAgent
  
from agents.fact_checking_agent import FactCheckingAgent
from agents.market_analysts.market_opportunity_agent import MarketOpportunityAgent
from agents.market_analysts.market_risk_agent import MarketRiskAgent
from agents.market_analysts.market_manager_agent import MarketManagerAgent
from config.agent_config import DEFAULT_LLM_TYPE, agent_config
from utils.langchain_tool_registry import get_langchain_tool_registry

logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()

class MultiAgentRunner:
    """Enhanced Multi-Agent Runner with comprehensive workflow orchestration and conditional analysis team usage."""
    
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
        self.langchain_registry = get_langchain_tool_registry()
        
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
        """Register tools for agents to use with enhanced Langchain integration."""
        self.tools.update(tools)
        logger.info(f"Registered {len(tools)} tools: {list(tools.keys())}")
        
        # Register with Langchain tool registry
        self.langchain_registry.register_tool_functions(tools)
        
        # Propagate tools to all agents that support them
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'register_tools'):
                agent.register_tools(tools)
                logger.debug(f"Registered tools with {agent_name}")

    def run_enhanced_workflow(self, query: str) -> Dict[str, Any]:
        """
        Run the enhanced multi-agent workflow with comprehensive features and conditional analysis team.
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary containing all workflow results and metadata
        """
        logger.info(f"Starting enhanced multi-agent workflow for query: {query}")
        
        try:
            workflow_start_time = logger.info("Workflow started")
            
            # Step 1: Planning - Analyze query and determine workflow
            logger.info("Step 1: Query Planning and Analysis")
            planning_result = self.agents['planning'].plan_query(query)
            
            subquestions = planning_result.get('subquestions', [query])
            needs_splitting = planning_result.get('needs_splitting', False)
            needs_analysis_team = planning_result.get('needs_analysis_team', False)
            analysis_reasoning = planning_result.get('analysis_reasoning', 'No reasoning provided')
            
            logger.info(f"Planning complete. Needs splitting: {needs_splitting}, "
                       f"Needs analysis team: {needs_analysis_team}, "
                       f"Subquestions: {len(subquestions)}")
            logger.info(f"Analysis team reasoning: {analysis_reasoning}")
            
            # Step 2: Process each subquestion through normalization and retrieval
            logger.info("Step 2: Processing subquestions through normalization and retrieval")
            
            all_contexts = [] # This is from the normalized agent
            normalization_results = []
            accumulated_context = ""  # Track accumulated context from previous subquestions
            
            for i, subq in enumerate(subquestions):
                logger.info(f"Processing subquestion {i+1}/{len(subquestions)}: {subq}")
                
                # Create enhanced subquestion with accumulated context for better understanding
                enhanced_subquestion = subq
                if i > 0 and accumulated_context:
                    # Use much larger context window and implement smart truncation
                    max_context_length = 4096  # Increased from 1024
                    if len(accumulated_context) > max_context_length:
                        # Smart truncation: keep most recent information and summary
                        context_lines = accumulated_context.split('\n')
                        # Keep last 60% of lines for recent context and add summary of older context
                        recent_lines = context_lines[-int(len(context_lines) * 0.6):]
                        recent_context = '\n'.join(recent_lines)
                        
                        # If still too long, further truncate but preserve structure
                        if len(recent_context) > max_context_length:
                            recent_context = recent_context[-max_context_length:]
                        
                        # Add summary of what was truncated
                        num_truncated = len(context_lines) - len(recent_lines)
                        truncated_context = f"[Previous {num_truncated} context entries truncated for brevity]\n{recent_context}"
                    else:
                        truncated_context = accumulated_context
                    
                    enhanced_subquestion = f"Previous context: {truncated_context}\n\nCurrent question: {subq}"
                    logger.info(f"Enhanced subquestion {i+1} with accumulated context (original: {len(accumulated_context)}, used: {len(truncated_context)} chars)")
                
                # Normalize and retrieve information for this subquestion
                norm_result = self.agents['normalize'].normalize_and_retrieve(enhanced_subquestion)
                normalization_results.append(norm_result)
                
                # Extract contexts from this subquestion
                retrieved_contexts = norm_result.get('retrieved_contexts', [])
                all_contexts.extend(retrieved_contexts)
                
                # Update accumulated context with key information from this subquestion
                if retrieved_contexts:
                    # Extract key information from ALL retrieved contexts
                    subquestion_summary = f"Q{i+1}: {subq}\n"
                    for j, context in enumerate(retrieved_contexts):  # Use ALL contexts now
                        if 'result' in context and isinstance(context['result'], str):
                            # Include tool name in summary for better context
                            tool_name = context.get('tool', 'unknown_tool')
                            result_preview = context['result'][:1024] + "..." if len(context['result']) > 1024 else context['result']
                            subquestion_summary += f"[{tool_name}]: {result_preview}\n"
                        elif 'error' in context:
                            tool_name = context.get('tool', 'unknown_tool')
                            subquestion_summary += f"[{tool_name}]: Error - {context['error']}\n"
                    
                    # Add the new summary to accumulated context
                    new_context = subquestion_summary + "\n"
                    
                    # Manage total accumulated context length with more generous limits
                    max_accumulated_length = 8192  # Increased from 2048 for better context preservation
                    if len(accumulated_context + new_context) > max_accumulated_length:
                        # Trim older context but keep more recent context
                        context_lines = accumulated_context.split('\n')
                        # Keep last 30% of lines of old context + all new context
                        lines_to_keep = max(20, int(len(context_lines) * 0.3))  # Keep at least 20 lines
                        trimmed_old_context = '\n'.join(context_lines[-lines_to_keep:]) if len(context_lines) > lines_to_keep else accumulated_context
                        
                        # Add marker to show context was trimmed
                        trimmed_marker = f"[Earlier context trimmed - kept last {lines_to_keep} entries]\n" if len(context_lines) > lines_to_keep else ""
                        accumulated_context = trimmed_marker + trimmed_old_context + new_context
                        logger.info(f"Trimmed accumulated context to prevent token overflow (kept {lines_to_keep} lines)")
                    else:
                        accumulated_context += new_context
                
                logger.info(f"Retrieved {len(retrieved_contexts)} contexts for subquestion {i+1}")
                logger.info(f"Accumulated context length: {len(accumulated_context)}")
            
            # Add context summary to workflow results for debugging
            workflow_context_summary = {
                "total_subquestions": len(subquestions),
                "accumulated_context_length": len(accumulated_context),
                "context_used": bool(accumulated_context),
                "enhanced_subquestions": [subq for subq in subquestions if len(subquestions) > 1]
            }
            
            # Step 3: Synthesize information using GeneralizeAgent
            logger.info("Step 3: Information Synthesis")
            synthesis_result = self.agents['generalize'].synthesize_information(
                original_query=query,
                subquestions=subquestions,
                contexts=all_contexts,
                accumulated_context=accumulated_context if accumulated_context else None  # Pass accumulated context
            )
            
            logger.info(f"Synthesis complete. Response length: {len(synthesis_result)} characters")
            logger.info(f"Context-aware synthesis: {'Yes' if accumulated_context else 'No'}")
            
            # Conditional Steps 4-5: Market Analysis (only if needed)
            if needs_analysis_team:
                logger.info("Step 4: Market Analysis (Analysis team needed based on query)")
                
                # Prepare context for market analysis (include accumulated context)
                market_context = {
                    "question": query,
                    "synthesis_result": synthesis_result,
                    "contexts": all_contexts,
                    "accumulated_context": accumulated_context if accumulated_context else None
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
                
                final_analysis = self.agents['market_manager'].run(manager_input)
                logger.info(f"Market manager analysis complete")
                
                market_analysis = {
                    "opportunities": opportunity_analysis,
                    "risks": risk_analysis,
                    "final_analysis": final_analysis
                }
                
            else:
                logger.info("Skipping market analysis team - not needed for this query type")
                final_analysis = synthesis_result
                market_analysis = {
                    "skipped": True,
                    "reason": analysis_reasoning,
                    "final_analysis": synthesis_result
                }
            
            # Step 6: Fact Checking and Validation
            logger.info("Step 6: Fact Checking and Validation")
            
            # Extract sources for fact checking
            sources = self._extract_sources_from_contexts(all_contexts)
            
            # Validate the final response with enhanced source checking
            validation_result = self.agents['fact_checker'].validate_response(
                query=query,
                response=final_analysis,
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
                "market_analysis": market_analysis,
                "fact_checking": validation_result,
                "sources": sources,
                "context_preservation": workflow_context_summary,  # Add context summary
                "metadata": {
                    "llm_type": self.llm_type,
                    "subquestions_count": len(subquestions),
                    "contexts_count": len(all_contexts),
                    "confidence_level": validation_result.get('confidence_level', 'unknown'),
                    "overall_score": validation_result.get('overall_score', 0),
                    "analysis_team_used": needs_analysis_team,
                    "analysis_team_reasoning": analysis_reasoning,
                    "context_preservation_used": bool(accumulated_context),
                    "accumulated_context_length": len(accumulated_context)
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
        
        # Analysis team usage
        metadata = results.get('metadata', {})
        analysis_team_used = metadata.get('analysis_team_used', False)
        analysis_team_reasoning = metadata.get('analysis_team_reasoning', 'No reasoning provided')
        logger.info(f"Analysis Team Used: {analysis_team_used}")
        logger.info(f"Analysis Team Reasoning: {analysis_team_reasoning}")
        
        # Context summary
        logger.info(f"\nRetrieved Contexts: {results.get('total_contexts', 0)}")
        
        # Context preservation info
        context_preservation = results.get('context_preservation', {})
        if context_preservation.get('context_used', False):
            logger.info(f"Context Preservation: ENABLED")
            logger.info(f"  Accumulated Context Length: {context_preservation.get('accumulated_context_length', 0)}")
            logger.info(f"  Enhanced Subquestions: {context_preservation.get('total_subquestions', 1)}")
        else:
            logger.info(f"Context Preservation: Not needed (single question)")
        
        # Synthesis preview
        synthesis = results.get('synthesis_result', '')
        logger.info(f"\nSynthesis:")
        logger.info(synthesis)
        
        # Market analysis summary
        market_analysis = results.get('market_analysis', {})
        if market_analysis.get('skipped', False):
            logger.info(f"\nMarket Analysis: Skipped - {market_analysis.get('reason', 'Not needed')}")
        else:
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

    # Alias for backward compatibility
    run = run_enhanced_workflow



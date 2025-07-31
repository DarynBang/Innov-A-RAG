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

    def run_enhanced_workflow(self, query: str, product_suggestion_mode: bool = False) -> Dict[str, Any]:
        """
        Run the enhanced multi-agent workflow with comprehensive features and conditional analysis team.
        
        Args:
            query: User query to process
            product_suggestion_mode: If True, run in product suggestion mode (skip opportunity/risk agents)
            
        Returns:
            Dictionary containing all workflow results and metadata
        """
        if product_suggestion_mode:
            return self.run_production_mode_workflow(query)
        else:
            return self.run_market_analysis_workflow(query)
    
    def run_market_analysis_workflow(self, query: str) -> Dict[str, Any]:
        """
        Run the traditional market analysis workflow.
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary containing all workflow results and metadata
        """
        logger.info(f"Starting market analysis workflow for query: {query}")
        
        try:
            workflow_start_time = logger.info("Workflow started")
            
            # Step 1: Planning - Analyze query and determine workflow
            logger.info("Step 1: Query Planning and Analysis")
            planning_result = self.agents['planning'].plan_query(query, product_suggestion_mode=False)
            
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
            
            all_contexts = []
            normalization_results = []
            accumulated_context = ""
            
            for i, subq in enumerate(subquestions):
                logger.info(f"Processing subquestion {i+1}/{len(subquestions)}: {subq}")
                
                # Create enhanced subquestion with accumulated context for better understanding
                enhanced_subquestion = subq
                if i > 0 and accumulated_context:
                    max_context_length = 4096
                    if len(accumulated_context) > max_context_length:
                        context_lines = accumulated_context.split('\n')
                        recent_lines = context_lines[-int(len(context_lines) * 0.6):]
                        recent_context = '\n'.join(recent_lines)
                        
                        if len(recent_context) > max_context_length:
                            recent_context = recent_context[-max_context_length:]
                        
                        num_truncated = len(context_lines) - len(recent_lines)
                        truncated_context = f"[Previous {num_truncated} context entries truncated for brevity]\n{recent_context}"
                    else:
                        truncated_context = accumulated_context
                    
                    enhanced_subquestion = f"Previous context: {truncated_context}\n\nCurrent question: {subq}"
                    logger.info(f"Enhanced subquestion {i+1} with accumulated context (original: {len(accumulated_context)}, used: {len(truncated_context)} chars)")
                
                # Normalize and retrieve information for this subquestion
                norm_result = self.agents['normalize'].normalize_and_retrieve(enhanced_subquestion, product_suggestion_mode=False)
                
                enhanced_norm_result = {
                    **norm_result,
                    'original_query': subq,
                    'enhanced_query': enhanced_subquestion,
                    'subquestion_index': i + 1
                }
                normalization_results.append(enhanced_norm_result)
                
                # Extract contexts from this subquestion
                retrieved_contexts = norm_result.get('retrieved_contexts', [])
                all_contexts.extend(retrieved_contexts)
                
                # Update accumulated context with key information from this subquestion
                if retrieved_contexts:
                    subquestion_summary = f"Q{i+1}: {subq}\n"
                    for j, context in enumerate(retrieved_contexts):
                        if 'result' in context and isinstance(context['result'], str):
                            tool_name = context.get('tool', 'unknown_tool')
                            result_preview = context['result'][:1024] + "..." if len(context['result']) > 1024 else context['result']
                            subquestion_summary += f"[{tool_name}]: {result_preview}\n"
                        elif 'error' in context:
                            tool_name = context.get('tool', 'unknown_tool')
                            subquestion_summary += f"[{tool_name}]: Error - {context['error']}\n"
                    
                    new_context = subquestion_summary + "\n"
                    
                    max_accumulated_length = 8192
                    if len(accumulated_context + new_context) > max_accumulated_length:
                        context_lines = accumulated_context.split('\n')
                        lines_to_keep = max(20, int(len(context_lines) * 0.3))
                        trimmed_old_context = '\n'.join(context_lines[-lines_to_keep:]) if len(context_lines) > lines_to_keep else accumulated_context
                        
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
                accumulated_context=accumulated_context if accumulated_context else None,
                product_suggestion_mode=False
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
                contexts=all_contexts
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
                "context_preservation": workflow_context_summary,
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
            
            logger.info("Market analysis workflow completed successfully")
            return workflow_results
            
        except Exception as e:
            logger.error(f"Error in market analysis workflow: {e}")
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
        
        # Subquestions details
        subquestions = results.get('subquestions', [])
        if subquestions:
            logger.info(f"\nSubquestions Generated:")
            for i, sq in enumerate(subquestions, 1):
                logger.info(f"  {i}. {sq}")
        
        # Normalization results details
        norm_results = results.get('normalization_results', [])
        if norm_results:
            logger.info(f"\nNormalization Results:")
            for i, norm_result in enumerate(norm_results, 1):
                query = norm_result.get('original_query', 'Unknown')
                retrieved_contexts = norm_result.get('retrieved_contexts', [])
                company_count = 0
                patent_count = 0
                
                for context in retrieved_contexts:
                    if isinstance(context, dict):
                        tool = context.get('tool', '')
                        if 'company' in tool.lower():
                            company_count += 1
                        elif 'patent' in tool.lower():
                            patent_count += 1
                
                logger.info(f"  {i}. {query}")
                logger.info(f"     Retrieved: {len(retrieved_contexts)} contexts ({company_count} company, {patent_count} patent)")
        
        # Synthesis summary
        synthesis = results.get('synthesis_result', '')
        logger.info(f"\nSynthesis: {len(synthesis)} characters")
        
        # Market analysis summary
        market_analysis = results.get('market_analysis', {})
        if not market_analysis.get('skipped', False):
            logger.info(f"\nMarket Analysis: Complete")
            final_analysis = market_analysis.get('final_analysis', '')
            logger.info(f"Final Analysis: {len(final_analysis)} characters")
        else:
            logger.info(f"\nMarket Analysis: Skipped")
            logger.info(f"Reason: {market_analysis.get('reason', 'No reason provided')}")
        
        # Fact checking summary
        fact_checking = results.get('fact_checking', {})
        overall_score = fact_checking.get('overall_score', 0)
        confidence_level = fact_checking.get('confidence_level', 'unknown')
        flagged_issues = fact_checking.get('flagged_issues', [])
        
        logger.info(f"\nFact Checking Score: {overall_score}/10 ({confidence_level.upper()} confidence)")
        logger.info(f"Issues Flagged: {len(flagged_issues)}")
        
        if flagged_issues:
            logger.info("Issues:")
            for issue in flagged_issues[:3]:  # Show first 3 issues
                logger.info(f"  - {issue}")
        
        # Sources summary
        sources = results.get('sources', [])
        logger.info(f"\nSources: {len(sources)} references")
        
        # Final metadata
        logger.info(f"\nWorkflow Metadata:")
        logger.info(f"  LLM Type: {metadata.get('llm_type', 'unknown')}")
        logger.info(f"  Analysis Team Used: {metadata.get('analysis_team_used', 'unknown')}")
        logger.info(f"  Context Preservation: {metadata.get('context_preservation_used', 'unknown')}")
        
        logger.info("\n" + "="*80 + "\n")

    def run_production_mode_workflow(self, query: str) -> Dict[str, Any]:
        """
        Run the simplified production mode workflow: normalize -> generalize -> manager -> verifier.
        
        This workflow skips the planning agent and uses a direct, simplified flow for production use.
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary containing production workflow results and metadata
        """
        logger.info(f"Starting production mode workflow for query: {query}")
        logger.info("Production workflow: normalize -> generalize -> manager -> verifier")
        
        try:
            workflow_start_time = logger.info("Production mode workflow started")
            
            # Step 1: Normalize and retrieve (production mode - single optimized tool)
            logger.info("Step 1: Production mode normalization and retrieval")
            norm_result = self.agents['normalize'].normalize_and_retrieve(query, product_suggestion_mode=True)
            
            # Extract contexts and count actual chunks
            all_contexts = norm_result.get('retrieved_contexts', [])
            total_chunks = norm_result.get('total_contexts', 0)
            normalization_results = [{
                **norm_result,
                'original_query': query,
                'production_mode': True
            }]
            
            logger.info(f"Production mode normalization complete. Retrieved {total_chunks} total chunks from {len(all_contexts)} tools")
            
            # Step 2: Generalize information (production mode - structured dict output)
            logger.info("Step 2: Production mode information generalization")
            synthesis_result = self.agents['generalize'].synthesize_information(
                original_query=query,
                subquestions=[query],
                contexts=all_contexts,
                accumulated_context=None,  # No accumulated context in production mode
                product_suggestion_mode=True
            )
            
            logger.info(f"Production mode synthesis complete. Result length: {len(synthesis_result)} characters")
            
            # Step 3: Product suggestion generation (production mode - based only on given info)
            logger.info("Step 3: Production mode product suggestion generation")
            
            manager_input = {
                "question": query,
                "synthesis_result": synthesis_result,
                "contexts": all_contexts
            }
            
            # Run market manager in production mode
            product_suggestions = self.agents['market_manager'].run(manager_input, product_suggestion_mode=True)
            logger.info(f"Production mode product suggestion generation complete")
            
            # Step 4: Production mode validation (robust, standard, detailed, cited)
            logger.info("Step 4: Production mode fact checking and validation")
            
            # Extract sources for fact checking
            sources = self._extract_sources_from_contexts(all_contexts)
            
            # Validate with production mode criteria
            validation_result = self.agents['fact_checker'].validate_response(
                query=query,
                response=product_suggestions,
                sources=sources,
                contexts=all_contexts,
                validation_mode="product_suggestion",
                production_mode=True
            )
            
            logger.info(f"Production mode validation complete. Overall score: {validation_result.get('overall_score', 0)}/10")
            
            # Step 5: Compile production mode results
            workflow_results = {
                "query": query,
                "mode": "production",
                "workflow_type": "simplified_production",
                "normalization_results": normalization_results,
                "total_contexts": total_chunks,
                "synthesis_result": synthesis_result,
                "product_suggestions": product_suggestions,
                "fact_checking": validation_result,
                "sources": sources,
                "metadata": {
                    "llm_type": self.llm_type,
                    "workflow": "production",
                    "contexts_count": total_chunks,
                    "confidence_level": validation_result.get('confidence_level', 'unknown'),
                    "overall_score": validation_result.get('overall_score', 0),
                    "planning_skipped": True,
                    "analysis_team_skipped": True,
                    "simplified_workflow": True,
                    "workflow_steps": ["normalize", "generalize", "manager", "verifier"]
                }
            }
            
            # Print summary to screen
            self._print_production_workflow_summary(workflow_results)
            
            logger.info("Production mode workflow completed successfully")
            return workflow_results
            
        except Exception as e:
            logger.error(f"Error in production mode workflow: {e}")
            return {
                "query": query,
                "error": str(e),
                "status": "failed",
                "mode": "production"
            }

    def _print_production_workflow_summary(self, results: Dict[str, Any]):
        """
        Print a summary of the production workflow results.
        
        Args:
            results: Dictionary containing workflow results
        """
        logger.info("\n" + "="*60)
        logger.info("PRODUCTION MODE WORKFLOW SUMMARY")
        logger.info("="*60)
        
        # Query info
        query = results.get('query', '')
        logger.info(f"Query: {query}")
        logger.info(f"Mode: {results.get('mode', 'unknown')}")
        logger.info(f"Workflow: {results.get('workflow_type', 'unknown')}")
        
        # Context summary
        logger.info(f"Contexts Retrieved: {results.get('total_contexts', 0)}")
        
        # Synthesis result summary
        synthesis = results.get('synthesis_result', '')
        logger.info(f"Synthesis Result Length: {len(synthesis)} characters")
        
        # Product suggestions summary
        product_suggestions = results.get('product_suggestions', '')
        if product_suggestions:
            logger.info(f"Product Suggestions Generated: {len(product_suggestions)} characters")
        else:
            logger.warning("No product suggestions generated")
        
        # Validation summary
        fact_checking = results.get('fact_checking', {})
        overall_score = fact_checking.get('overall_score', 0)
        confidence_level = fact_checking.get('confidence_level', 'unknown')
        flagged_issues = fact_checking.get('flagged_issues', [])
        
        logger.info(f"Validation Score: {overall_score}/10 ({confidence_level.upper()} confidence)")
        logger.info(f"Issues Flagged: {len(flagged_issues)}")
        
        if flagged_issues:
            logger.info("Issues:")
            for issue in flagged_issues[:3]:  # Show first 3 issues
                logger.info(f"  - {issue}")
        
        # Production criteria if available
        production_criteria = fact_checking.get('production_criteria', {})
        if production_criteria:
            logger.info("Production Criteria:")
            for criterion, result in production_criteria.items():
                score = result.get('score', 0) if isinstance(result, dict) else 0
                logger.info(f"  {criterion.capitalize()}: {score}/10")
        
        logger.info("\n" + "="*60 + "\n")

    # Alias for backward compatibility
    run = run_enhanced_workflow


    
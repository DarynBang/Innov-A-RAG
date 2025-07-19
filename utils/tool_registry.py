"""
Tool Registry System for InnovARAG

This module provides automatic tool registration and description generation
to improve the Normalized Query Agent's tool handling capabilities.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

import inspect
import json
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass

logger = get_logger(__name__)

@dataclass
class ToolInfo:
    """Information about a registered tool."""
    name: str
    function: Callable
    description: str
    parameters: List[Dict[str, Any]]
    return_type: str
    category: str
    examples: List[str]
    requires_followup: bool = False
    followup_tools: List[str] = None

class ToolRegistry:
    """Registry for managing and auto-generating tool information."""
    
    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}
        self.categories = {
            "exact_lookup": "Direct information retrieval by ID/name",
            "rag_retrieval": "Semantic search and analysis",
            "hybrid": "Combined retrieval methods"
        }
        logger.info("ToolRegistry initialized")
    
    def register_tool(self, 
                     name: str, 
                     function: Callable, 
                     category: str = "general",
                     examples: List[str] = None,
                     requires_followup: bool = False,
                     followup_tools: List[str] = None) -> None:
        """
        Register a tool with automatic description generation.
        
        Args:
            name: Tool name
            function: Tool function
            category: Tool category for grouping
            examples: Usage examples
            requires_followup: Whether this tool typically needs followup
            followup_tools: List of tools commonly used after this one
        """
        try:
            # Auto-generate description from docstring
            description = self._extract_description(function)
            
            # Extract parameter information
            parameters = self._extract_parameters(function)
            
            # Extract return type
            return_type = self._extract_return_type(function)
            
            tool_info = ToolInfo(
                name=name,
                function=function,
                description=description,
                parameters=parameters,
                return_type=return_type,
                category=category,
                examples=examples or [],
                requires_followup=requires_followup,
                followup_tools=followup_tools or []
            )
            
            self.tools[name] = tool_info
            logger.info(f"Registered tool: {name} (category: {category})")
            
        except Exception as e:
            logger.error(f"Error registering tool {name}: {e}")
    
    def _extract_description(self, function: Callable) -> str:
        """Extract description from function docstring."""
        doc = inspect.getdoc(function)
        if doc:
            # Take first line or paragraph of docstring
            lines = doc.strip().split('\n')
            description = lines[0].strip()
            
            # If first line is short, try to get more context
            if len(description) < 30 and len(lines) > 1:
                for line in lines[1:]:
                    if line.strip() and not line.strip().startswith('Args'):
                        description += " " + line.strip()
                        break
            
            return description
        else:
            return f"Function {function.__name__} - no description available"
    
    def _extract_parameters(self, function: Callable) -> List[Dict[str, Any]]:
        """Extract parameter information from function signature."""
        try:
            sig = inspect.signature(function)
            parameters = []
            
            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "type": str(param.annotation) if param.annotation != param.empty else "Any",
                    "required": param.default == param.empty,
                    "default": str(param.default) if param.default != param.empty else None
                }
                parameters.append(param_info)
            
            return parameters
            
        except Exception as e:
            logger.warning(f"Could not extract parameters for {function.__name__}: {e}")
            return []
    
    def _extract_return_type(self, function: Callable) -> str:
        """Extract return type from function signature."""
        try:
            sig = inspect.signature(function)
            if sig.return_annotation != sig.empty:
                return str(sig.return_annotation)
            else:
                return "Any"
        except Exception:
            return "Any"
    
    def get_tool_descriptions(self) -> str:
        """Generate formatted tool descriptions for prompts."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        
        # Group tools by category
        by_category = {}
        for tool in self.tools.values():
            category = tool.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(tool)
        
        # Format descriptions
        for category, tools in by_category.items():
            descriptions.append(f"\n{category.upper()} TOOLS:")
            
            for tool in tools:
                desc = f"- {tool.name}: {tool.description}"
                
                # Add parameter info
                if tool.parameters:
                    required_params = [p for p in tool.parameters if p["required"]]
                    if required_params:
                        param_names = [p["name"] for p in required_params]
                        desc += f" (requires: {', '.join(param_names)})"
                
                # Add followup info
                if tool.requires_followup and tool.followup_tools:
                    desc += f" [Note: Often followed by {', '.join(tool.followup_tools)}]"
                
                descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def get_tool_workflow_suggestions(self, tool_name: str) -> List[str]:
        """Get suggested followup tools for a given tool."""
        if tool_name in self.tools:
            return self.tools[tool_name].followup_tools
        return []
    
    def get_tool_by_name(self, name: str) -> Optional[ToolInfo]:
        """Get tool information by name."""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[ToolInfo]:
        """Get all tools in a specific category."""
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def generate_examples(self) -> List[Dict[str, Any]]:
        """Generate comprehensive examples including mixed queries."""
        examples = [
            {
                "query": "What are TechNova's market opportunities?",
                "type": "company",
                "identifiers": {"companies": ["TechNova"], "patents": []},
                "tools": ["company_rag_retrieval"],
                "reasoning": "Specific company query requiring detailed market analysis"
            },
            {
                "query": "What is the technology behind Patent 273556553?",
                "type": "patent",
                "identifiers": {"companies": [], "patents": ["273556553"]},
                "tools": ["exact_patent_lookup", "patent_rag_retrieval"],
                "reasoning": "Specific patent query requiring both basic info and technical analysis"
            },
            {
                "query": "What are the latest trends in AI patents?",
                "type": "general",
                "identifiers": {"companies": [], "patents": []},
                "tools": ["patent_rag_retrieval", "hybrid_rag_retrieval"],
                "reasoning": "General industry trend query requiring broad patent analysis"
            },
            {
                "query": "Compare TechNova's AI patents with those of InnovateCorp",
                "type": "general",
                "identifiers": {"companies": ["TechNova", "InnovateCorp"], "patents": []},
                "tools": ["company_rag_retrieval", "patent_rag_retrieval", "hybrid_rag_retrieval"],
                "reasoning": "Comparative analysis requiring both company and patent information"
            },
            {
                "query": "How does Patent 123456 relate to TechNova's business strategy?",
                "type": "general",
                "identifiers": {"companies": ["TechNova"], "patents": ["123456"]},
                "tools": ["exact_patent_lookup", "company_rag_retrieval", "hybrid_rag_retrieval"],
                "reasoning": "Mixed query requiring both patent details and company analysis"
            },
            {
                "query": "What companies are working on similar technology to Patent 789012?",
                "type": "general",
                "identifiers": {"companies": [], "patents": ["789012"]},
                "tools": ["exact_patent_lookup", "patent_rag_retrieval", "hybrid_rag_retrieval"],
                "reasoning": "Patent-centered query requiring technology analysis and company discovery"
            }
        ]
        
        return examples
    
    def export_for_prompt(self) -> str:
        """Export tool registry information for use in prompts."""
        tool_descriptions = self.get_tool_descriptions()
        examples = self.generate_examples()
        
        prompt_section = f"""
AVAILABLE TOOLS:
{tool_descriptions}

CLASSIFICATION EXAMPLES:
"""
        for example in examples:
            prompt_section += f"""
Query: "{example['query']}"
Output: {{
    "query_type": "{example['type']}",
    "identifiers": {json.dumps(example['identifiers'])},
    "recommended_tools": {json.dumps(example['tools'])},
    "reasoning": "{example['reasoning']}"
}}"""
        
        return prompt_section


class AdvancedNormalizeQueryAgent:
    """Enhanced version of NormalizeQueryAgent with improved tool handling."""
    
    def __init__(self, base_agent, tool_registry: ToolRegistry):
        self.base_agent = base_agent
        self.tool_registry = tool_registry
        logger.info("AdvancedNormalizeQueryAgent initialized")
    
    def normalize_and_retrieve_advanced(self, query: str) -> Dict[str, Any]:
        """
        Enhanced normalize and retrieve with sequential tool execution.
        
        Args:
            query: User query
            
        Returns:
            Enhanced results with sequential tool execution
        """
        logger.info(f"Advanced normalization and retrieval for: {query}")
        
        try:
            # First, get basic normalization
            basic_result = self.base_agent.normalize_query(query)
            
            # Get recommended tools
            recommended_tools = basic_result.get('recommended_tools', [])
            identifiers = basic_result.get('identifiers', {})
            
            # Execute tools with workflow awareness
            retrieved_contexts = []
            executed_tools = set()
            
            for tool_name in recommended_tools:
                contexts = self._execute_tool_workflow(
                    tool_name, query, identifiers, executed_tools
                )
                retrieved_contexts.extend(contexts)
            
            # Combine results
            result = {
                "normalization": basic_result,
                "retrieved_contexts": retrieved_contexts,
                "total_contexts": len(retrieved_contexts),
                "workflow_executed": list(executed_tools)
            }
            
            logger.info(f"Advanced retrieval complete: {len(retrieved_contexts)} contexts from {len(executed_tools)} tools")
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced normalize and retrieve: {e}")
            return {
                "normalization": {"error": str(e)},
                "retrieved_contexts": [],
                "total_contexts": 0,
                "error": str(e)
            }
    
    def _execute_tool_workflow(self, 
                              tool_name: str, 
                              query: str, 
                              identifiers: Dict[str, List[str]], 
                              executed_tools: set) -> List[Dict[str, Any]]:
        """
        Execute a tool and its recommended followup tools.
        
        Args:
            tool_name: Primary tool to execute
            query: Original query
            identifiers: Extracted identifiers
            executed_tools: Set of already executed tools
            
        Returns:
            List of contexts from tool execution
        """
        contexts = []
        
        # Skip if already executed
        if tool_name in executed_tools:
            return contexts
        
        # Execute primary tool
        if tool_name in self.base_agent.tool_executor:
            try:
                logger.debug(f"Executing primary tool: {tool_name}")
                
                tool_input = self.base_agent._prepare_tool_input(tool_name, query, identifiers)
                tool_result = self.base_agent.tool_executor[tool_name](tool_input)
                
                contexts.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result": tool_result,
                    "execution_type": "primary"
                })
                
                executed_tools.add(tool_name)
                
                # Check for followup tools
                tool_info = self.tool_registry.get_tool_by_name(tool_name)
                if tool_info and tool_info.requires_followup:
                    followup_contexts = self._execute_followup_tools(
                        tool_info.followup_tools, query, identifiers, 
                        executed_tools, tool_result
                    )
                    contexts.extend(followup_contexts)
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                contexts.append({
                    "tool": tool_name,
                    "error": str(e),
                    "execution_type": "primary"
                })
        
        return contexts
    
    def _execute_followup_tools(self, 
                               followup_tools: List[str], 
                               query: str, 
                               identifiers: Dict[str, List[str]], 
                               executed_tools: set,
                               primary_result: Any) -> List[Dict[str, Any]]:
        """Execute followup tools based on primary tool results."""
        contexts = []
        
        for followup_tool in followup_tools:
            if followup_tool in executed_tools or followup_tool not in self.base_agent.tool_executor:
                continue
            
            try:
                logger.debug(f"Executing followup tool: {followup_tool}")
                
                # Prepare input for followup tool (could be enhanced based on primary result)
                tool_input = self._prepare_followup_input(
                    followup_tool, query, identifiers, primary_result
                )
                
                tool_result = self.base_agent.tool_executor[followup_tool](tool_input)
                
                contexts.append({
                    "tool": followup_tool,
                    "input": tool_input,
                    "result": tool_result,
                    "execution_type": "followup"
                })
                
                executed_tools.add(followup_tool)
                
            except Exception as e:
                logger.error(f"Error executing followup tool {followup_tool}: {e}")
                contexts.append({
                    "tool": followup_tool,
                    "error": str(e),
                    "execution_type": "followup"
                })
        
        return contexts
    
    def _prepare_followup_input(self, 
                               tool_name: str, 
                               query: str, 
                               identifiers: Dict[str, List[str]], 
                               primary_result: Any) -> str:
        """Prepare input for followup tools, potentially enhanced by primary results."""
        # For now, use the same logic as base agent
        # This could be enhanced to use information from primary_result
        return self.base_agent._prepare_tool_input(tool_name, query, identifiers)


# Global registry instance
global_tool_registry = ToolRegistry()

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return global_tool_registry

def register_innovarag_tools(tools_dict: Dict[str, Callable]) -> None:
    """Register standard InnovARAG tools with enhanced metadata."""
    registry = get_tool_registry()
    
    # Register company tools
    if "exact_company_lookup" in tools_dict:
        registry.register_tool(
            "exact_company_lookup",
            tools_dict["exact_company_lookup"],
            category="exact_lookup",
            examples=["TechNova", "InnovateCorp"],
            requires_followup=True,
            followup_tools=["company_rag_retrieval"]
        )
    
    if "company_rag_retrieval" in tools_dict:
        registry.register_tool(
            "company_rag_retrieval",
            tools_dict["company_rag_retrieval"],
            category="rag_retrieval",
            examples=["TechNova market opportunities", "InnovateCorp business strategy"]
        )
    
    # Register patent tools
    if "exact_patent_lookup" in tools_dict:
        registry.register_tool(
            "exact_patent_lookup",
            tools_dict["exact_patent_lookup"],
            category="exact_lookup",
            examples=["273556553", "US123456789"],
            requires_followup=True,
            followup_tools=["patent_rag_retrieval"]
        )
    
    if "patent_rag_retrieval" in tools_dict:
        registry.register_tool(
            "patent_rag_retrieval",
            tools_dict["patent_rag_retrieval"],
            category="rag_retrieval",
            examples=["AI machine learning patents", "renewable energy technology"]
        )
    
    # Register hybrid tools
    if "hybrid_rag_retrieval" in tools_dict:
        registry.register_tool(
            "hybrid_rag_retrieval",
            tools_dict["hybrid_rag_retrieval"],
            category="hybrid",
            examples=["AI industry analysis", "TechNova competitive landscape"]
        )
    
    # Register enhanced hybrid tools
    if "enhanced_hybrid_rag_retrieval" in tools_dict:
        registry.register_tool(
            "enhanced_hybrid_rag_retrieval",
            tools_dict["enhanced_hybrid_rag_retrieval"],
            category="enhanced_hybrid",
            examples=["AI industry comprehensive analysis", "TechNova full ecosystem mapping"]
        )
    
    if "company_data_with_mapping" in tools_dict:
        registry.register_tool(
            "company_data_with_mapping",
            tools_dict["company_data_with_mapping"],
            category="enhanced_lookup",
            examples=["TechNova", "InnovateCorp"]
        )
    
    if "mapping_key_search" in tools_dict:
        registry.register_tool(
            "mapping_key_search",
            tools_dict["mapping_key_search"],
            category="mapping_search",
            examples=["Search by specific company+hojin+chunk combinations"]
        )
    
    logger.info("Standard and enhanced InnovARAG tools registered with enhanced metadata") 
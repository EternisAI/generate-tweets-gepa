#!/usr/bin/env python3
"""
Exa Search Tool Call Integration
Makes Exa search available as a function call for LLMs during inference
"""

import json
import os
from typing import Dict, List, Optional, Any
from exa_search_tool import ExaSearchTool
from rich.console import Console

console = Console()

class ExaToolCall:
    """
    Exa Search as a tool call for LLM function calling
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the Exa tool call handler"""
        self.search_tool = ExaSearchTool(api_key)
        self.tool_calls_made = []
    
    def get_tool_definition(self) -> Dict:
        """
        Get the tool definition for LLM function calling
        
        Returns:
            Tool definition in OpenAI function calling format
        """
        return {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for current information, news, facts, or context using Exa's neural search. Use this when you need recent information, want to verify facts, or need additional context for tweet generation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. Be specific and include relevant keywords. Examples: 'OpenAI GPT-5 latest news 2024', 'remote work productivity studies', 'AI safety research recent developments'"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of search results to return (1-10). Default is 3 for tweet context.",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5
                        },
                        "category": {
                            "type": "string",
                            "description": "Content type filter - use 'news' for current events, 'research paper' for studies, 'company' for business info, 'tweet' for social media, etc.",
                            "enum": ["news", "research paper", "company", "github", "tweet", "general"],
                            "default": "general"
                        },
                        "recent_only": {
                            "type": "boolean",
                            "description": "Whether to focus on recent content (last 30 days). Useful for trending topics.",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute_tool_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool call
        
        Args:
            function_name: Name of the function called
            arguments: Arguments passed to the function
            
        Returns:
            Tool call result
        """
        
        if function_name != "search_web":
            return {
                "error": f"Unknown function: {function_name}",
                "available_functions": ["search_web"]
            }
        
        try:
            # Extract arguments with defaults
            query = arguments.get("query", "")
            num_results = arguments.get("num_results", 3)
            category = arguments.get("category", "general")
            recent_only = arguments.get("recent_only", True)
            type_filter = "keyword"  # Always use keyword search by default
            
            if not query:
                return {"error": "Query parameter is required"}
            
            console.print(f"[cyan]🔍 Tool Call: Searching for '{query}'[/cyan]")
            
            # Perform search
            if category == "general":
                category = None  # Let Exa decide
            
            search_results = self.search_tool.search_for_tweet_context(
                query=query,
                num_results=min(num_results, 10),  # Cap at 10
                recent_only=recent_only,
                type_filter=type_filter
            )
            
            # Format results for LLM consumption
            formatted_results = self._format_for_llm(search_results, query)
            
            # Track tool call
            self.tool_calls_made.append({
                "function": function_name,
                "arguments": arguments,
                "timestamp": search_results.get('search_metadata', {}).get('timestamp'),
                "results_count": len(search_results.get('results', []))
            })
            
            console.print(f"[green]✓ Found {len(search_results.get('results', []))} results[/green]")
            
            return formatted_results
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            console.print(f"[red]❌ {error_msg}[/red]")
            return {"error": error_msg}
    
    def _format_for_llm(self, search_results: Dict, query: str) -> Dict[str, Any]:
        """
        Format search results for LLM consumption
        
        Args:
            search_results: Raw search results from Exa
            query: Original search query
            
        Returns:
            Formatted results for LLM
        """
        
        if 'error' in search_results:
            return {"error": search_results['error']}
        
        results = search_results.get('results', [])
        if not results:
            return {
                "query": query,
                "message": "No relevant results found for this query.",
                "results": []
            }
        
        # Format each result
        formatted_results = []
        for result in results[:5]:  # Limit to top 5 for LLM context
            formatted_result = {
                "title": result.get('title', 'Untitled'),
                "url": result.get('url', ''),
                "published_date": result.get('publishedDate', 'Unknown'),
                "summary": result.get('summary', '')[:300] + "..." if result.get('summary', '') else "",
                "highlights": result.get('highlights', [])[:3],  # Top 3 highlights
            }
            
            # Add key facts if available from tweet-optimized search
            if 'formatted_for_tweets' in search_results:
                tweet_data = next(
                    (item for item in search_results['formatted_for_tweets'] 
                     if item.get('url') == result.get('url')), 
                    {}
                )
                if tweet_data.get('key_facts'):
                    formatted_result['key_facts'] = tweet_data['key_facts'][:3]
            
            formatted_results.append(formatted_result)
        
        return {
            "query": query,
            "search_timestamp": search_results.get('search_metadata', {}).get('timestamp'),
            "autoprompt_used": search_results.get('autopromptString', ''),
            "total_results": len(results),
            "results": formatted_results,
            "message": f"Found {len(results)} relevant results for '{query}'. Use this information to enhance your tweet generation with current context."
        }
    
    def get_tool_calls_summary(self) -> Dict:
        """Get summary of tool calls made during session"""
        return {
            "total_calls": len(self.tool_calls_made),
            "calls": self.tool_calls_made
        }

def create_tool_call_handler() -> Optional[ExaToolCall]:
    """
    Create and return an Exa tool call handler if API key is available
    
    Returns:
        ExaToolCall instance or None if not available
    """
    try:
        return ExaToolCall()
    except Exception as e:
        console.print(f"[yellow]Warning: Exa tool call not available: {e}[/yellow]")
        return None

# Tool call registry for easy integration
AVAILABLE_TOOLS = {
    "search_web": {
        "handler_class": ExaToolCall,
        "description": "Neural web search using Exa API",
        "requires_api_key": "EXA_API_KEY"
    }
}

def get_all_tool_definitions() -> List[Dict]:
    """Get all available tool definitions"""
    tools = []
    
    # Try to create Exa tool
    try:
        exa_tool = ExaToolCall()
        tools.append(exa_tool.get_tool_definition())
    except:
        pass  # Skip if not available
    
    return tools

def execute_tool_call(tool_handler: ExaToolCall, function_name: str, arguments: Dict) -> Dict:
    """
    Execute a tool call with the given handler
    
    Args:
        tool_handler: Tool handler instance
        function_name: Function to call
        arguments: Function arguments
        
    Returns:
        Tool call result
    """
    return tool_handler.execute_tool_call(function_name, arguments)

def main():
    """Test the tool call functionality"""
    
    console.print("[bold blue]🧪 Testing Exa Tool Call Integration[/bold blue]\n")
    
    try:
        # Create tool handler
        tool_handler = ExaToolCall()
        
        # Get tool definition
        tool_def = tool_handler.get_tool_definition()
        console.print("[green]✓ Tool definition created[/green]")
        console.print(f"[dim]Function: {tool_def['function']['name']}[/dim]")
        
        # Test tool call
        test_args = {
            "query": "latest AI developments 2024",
            "num_results": 3,
            "recent_only": True
        }
        
        console.print(f"\n[cyan]Testing tool call with args: {test_args}[/cyan]")
        
        result = tool_handler.execute_tool_call("search_web", test_args)
        
        if "error" in result:
            console.print(f"[red]❌ Tool call failed: {result['error']}[/red]")
        else:
            console.print(f"[green]✅ Tool call successful![/green]")
            console.print(f"[dim]Query: {result['query']}[/dim]")
            console.print(f"[dim]Results: {result['total_results']}[/dim]")
            
            # Show first result
            if result['results']:
                first_result = result['results'][0]
                console.print(f"\n[bold]First Result:[/bold] {first_result['title']}")
                console.print(f"[dim]URL: {first_result['url']}[/dim]")
                if first_result['summary']:
                    console.print(f"[blue]Summary:[/blue] {first_result['summary'][:100]}...")
        
        # Show tool calls summary
        summary = tool_handler.get_tool_calls_summary()
        console.print(f"\n[yellow]Tool Calls Made: {summary['total_calls']}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")

if __name__ == "__main__":
    main()

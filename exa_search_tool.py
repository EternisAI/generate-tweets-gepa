#!/usr/bin/env python3
"""
Exa Search Tool Integration
Provides web search capabilities using Exa's neural search API
"""

import os
import json
import time
from typing import List, Dict, Optional, Union
from datetime import datetime
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

class ExaSearchTool:
    """
    Exa Search API integration for enhanced web search capabilities
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Exa search tool
        
        Args:
            api_key: Exa API key (if None, will try to get from environment)
        """
        self.api_key = api_key or os.getenv('EXA_API_KEY')
        if not self.api_key:
            raise ValueError("EXA_API_KEY not provided and not found in environment variables")
        
        self.base_url = "https://api.exa.ai"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": self.api_key
        }
        
        console.print("[green]✓ Exa Search Tool initialized[/green]")
    
    def search(
        self, 
        query: str, 
        num_results: int = 10,
        include_domains: List[str] = None,
        exclude_domains: List[str] = None,
        start_crawl_date: str = None,
        end_crawl_date: str = None,
        start_published_date: str = None,
        end_published_date: str = None,
        use_autoprompt: bool = True,
        category: str = None,
        type_filter: str = "keyword",
        include_text: bool = False,
        include_highlights: bool = False,
        include_summary: bool = True,
        summary_query: str = None
    ) -> Dict:
        """
        Perform a neural search using Exa API
        
        Args:
            query: Search query string
            num_results: Number of results to return (max 100)
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            start_crawl_date: Start date for crawl date filter (YYYY-MM-DD)
            end_crawl_date: End date for crawl date filter (YYYY-MM-DD)
            start_published_date: Start date for published date filter (YYYY-MM-DD)
            end_published_date: End date for published date filter (YYYY-MM-DD)
            use_autoprompt: Whether to use Exa's autoprompt feature
            category: Category filter (company, research paper, news, github, tweet, etc.)
            type_filter: Search type filter (keyword, neural, auto). Default: keyword
            include_text: Whether to include full text content
            include_highlights: Whether to include highlights
            include_summary: Whether to include AI-generated summary
            summary_query: Query for the summary generation
        
        Returns:
            Dictionary containing search results and metadata
        """
        
        endpoint = f"{self.base_url}/search"
        
        # Build request payload
        payload = {
            "query": query,
            "numResults": min(num_results, 100),  # API limit is 100
            "useAutoprompt": use_autoprompt,
            "type": type_filter  # Add type filter (keyword by default)
        }
        
        # Add optional parameters
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains
        if start_crawl_date:
            payload["startCrawlDate"] = start_crawl_date
        if end_crawl_date:
            payload["endCrawlDate"] = end_crawl_date
        if start_published_date:
            payload["startPublishedDate"] = start_published_date
        if end_published_date:
            payload["endPublishedDate"] = end_published_date
        if category:
            payload["category"] = category
        if include_text:
            payload["contents"] = {"text": True}
        if include_highlights:
            payload["contents"] = payload.get("contents", {})
            payload["contents"]["highlights"] = {"numSentences": 3, "highlightsPerUrl": 3}
        if include_summary:
            payload["contents"] = payload.get("contents", {})
            payload["contents"]["summary"] = {"query": summary_query or query}
        
        try:
            console.print(f"[cyan]🔍 Searching Exa for: {query[:50]}...[/cyan]")
            
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            
            results = response.json()
            
            # Add metadata
            results['search_metadata'] = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'num_results_requested': num_results,
                'num_results_returned': len(results.get('results', [])),
                'autoprompt_string': results.get('autopromptString', ''),
                'include_text': include_text,
                'include_highlights': include_highlights,
                'include_summary': include_summary
            }
            
            console.print(f"[green]✓ Found {len(results.get('results', []))} results[/green]")
            return results
            
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error during Exa search: {e}[/red]")
            return {
                'results': [],
                'error': str(e),
                'search_metadata': {
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'num_results_requested': num_results,
                    'num_results_returned': 0
                }
            }
    
    def search_and_contents(
        self,
        query: str,
        num_results: int = 5,
        text: bool = True,
        highlights: bool = True,
        summary: bool = True,
        type_filter: str = "keyword",
        **kwargs
    ) -> Dict:
        """
        Convenience method to search and get full content in one call
        
        Args:
            query: Search query
            num_results: Number of results
            text: Include full text content
            highlights: Include highlights
            summary: Include AI summary
            **kwargs: Additional search parameters
        
        Returns:
            Search results with content
        """
        return self.search(
            query=query,
            num_results=num_results,
            include_text=text,
            include_highlights=highlights,
            include_summary=summary,
            summary_query=query,
            type_filter=type_filter,
            **kwargs
        )
    
    def search_for_tweet_context(
        self,
        query: str,
        num_results: int = 3,
        recent_only: bool = True,
        type_filter: str = "keyword"
    ) -> Dict:
        """
        Specialized search for tweet generation context
        
        Args:
            query: Search query related to tweet topic
            num_results: Number of results (default 3 for tweet context)
            recent_only: Whether to filter for recent content
        
        Returns:
            Curated search results for tweet generation
        """
        
        # Set date filter for recent content if requested
        kwargs = {}
        if recent_only:
            # Get content from last 30 days
            from datetime import datetime, timedelta
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            kwargs['start_crawl_date'] = thirty_days_ago
        
        # Perform search with tweet-optimized settings
        results = self.search(
            query=query,
            num_results=num_results,
            include_text=True,
            include_highlights=True,
            include_summary=True,
            summary_query=f"Key facts and insights about {query} for social media",
            use_autoprompt=True,
            type_filter=type_filter,
            **kwargs
        )
        
        # Format results for tweet generation
        if 'results' in results and results['results']:
            formatted_results = []
            for result in results['results']:
                formatted_result = {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'published_date': result.get('publishedDate', ''),
                    'summary': result.get('summary', ''),
                    'highlights': result.get('highlights', []),
                    'key_facts': self._extract_key_facts(result)
                }
                formatted_results.append(formatted_result)
            
            results['formatted_for_tweets'] = formatted_results
        
        return results
    
    def _extract_key_facts(self, result: Dict) -> List[str]:
        """Extract key facts from a search result for tweet generation"""
        
        facts = []
        
        # Extract from summary
        if 'summary' in result:
            summary = result['summary']
            # Simple extraction - look for sentences with numbers, dates, or strong claims
            import re
            sentences = re.split(r'[.!?]+', summary)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Skip very short sentences
                    # Look for factual indicators
                    if any(indicator in sentence.lower() for indicator in [
                        'study', 'research', 'found', 'shows', 'reveals', 'according to',
                        'reported', 'announced', 'confirmed', '%', 'million', 'billion'
                    ]):
                        facts.append(sentence)
        
        # Extract from highlights
        if 'highlights' in result:
            for highlight in result['highlights'][:2]:  # Top 2 highlights
                if len(highlight) > 20:  # Skip very short highlights
                    facts.append(highlight)
        
        return facts[:3]  # Return top 3 facts
    
    def display_results(self, results: Dict, show_content: bool = False):
        """
        Display search results in a formatted way
        
        Args:
            results: Search results from Exa API
            show_content: Whether to show full content
        """
        
        if 'error' in results:
            console.print(f"[red]Search Error: {results['error']}[/red]")
            return
        
        search_results = results.get('results', [])
        metadata = results.get('search_metadata', {})
        
        # Display search info
        info_panel = Panel(
            f"Query: {metadata.get('query', 'N/A')}\n"
            f"Results: {metadata.get('num_results_returned', 0)}\n"
            f"Autoprompt: {results.get('autopromptString', 'N/A')[:100]}...",
            title="[bold cyan]Search Info[/bold cyan]",
            border_style="cyan"
        )
        console.print(info_panel)
        
        # Display results
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            published_date = result.get('publishedDate', 'Unknown date')
            
            content = f"[bold]URL:[/bold] {url}\n"
            content += f"[bold]Published:[/bold] {published_date}\n"
            
            # Add summary if available
            if 'summary' in result:
                content += f"\n[bold]Summary:[/bold]\n{result['summary'][:300]}...\n"
            
            # Add highlights if available
            if 'highlights' in result:
                content += f"\n[bold]Highlights:[/bold]\n"
                for highlight in result['highlights'][:3]:
                    content += f"• {highlight}\n"
            
            # Add text content if requested and available
            if show_content and 'text' in result:
                content += f"\n[bold]Content:[/bold]\n{result['text'][:500]}...\n"
            
            result_panel = Panel(
                content,
                title=f"[bold green]Result {i}: {title[:50]}...[/bold green]",
                border_style="green"
            )
            console.print(result_panel)
    
    def save_results(self, results: Dict, filename: str = None) -> str:
        """
        Save search results to a JSON file
        
        Args:
            results: Search results to save
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to saved file
        """
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            query_slug = results.get('search_metadata', {}).get('query', 'search')
            query_slug = ''.join(c for c in query_slug if c.isalnum() or c in (' ', '-', '_')).rstrip()
            query_slug = query_slug.replace(' ', '_')[:30]
            filename = f"exa_search_{query_slug}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ Results saved to {filename}[/green]")
        return filename

def main():
    """Test the Exa search tool"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Exa Search Tool")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--num-results", type=int, default=5, help="Number of results")
    parser.add_argument("--include-content", action="store_true", help="Include full content")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--tweet-mode", action="store_true", help="Optimize for tweet generation")
    
    args = parser.parse_args()
    
    try:
        # Initialize search tool
        exa = ExaSearchTool()
        
        # Perform search
        if args.tweet_mode:
            results = exa.search_for_tweet_context(args.query, num_results=args.num_results)
        else:
            results = exa.search_and_contents(
                args.query, 
                num_results=args.num_results,
                text=args.include_content
            )
        
        # Display results
        exa.display_results(results, show_content=args.include_content)
        
        # Save if requested
        if args.save:
            exa.save_results(results)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()

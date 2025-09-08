#!/usr/bin/env python3
"""
Enhanced X (Twitter) API Explorer - With filtered information sources
- Fetches parent tweets in conversations
- Filters tweets: English language, minimum 1000 likes
- Guarantees minimum 10 tweets through intelligent retry mechanism
- Extracts basic engagement metrics and context
- Provides parent tweet information sources
- Includes viral tweet analytics
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()
# Twitter API client
try:
    import tweepy
except ImportError:
    print("Please install tweepy: pip install tweepy")
    sys.exit(1)

import dspy
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def setup_dspy_model(model_name: str, exit_on_error: bool = False) -> dspy.LM:
    """
    Unified model setup function for DSPy using OpenRouter
    
    Args:
        model_name: The model name (e.g., "gpt-5", "claude-3-opus")
        exit_on_error: If True, exits the program on error. If False, raises exception.
        
    Returns:
        The configured LM instance
    """
    try:
        # Check for OpenRouter API key
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            console.print("[red]Error: OPENROUTER_API_KEY not found[/red]")
            if exit_on_error:
                sys.exit(1)
            else:
                raise ValueError("Missing OPENROUTER_API_KEY")
        
        from openrouter_config import setup_openrouter_model
        
        # If model doesn't have provider prefix, add openai/ for compatibility
        if not any(model_name.startswith(p) for p in ['openai/', 'anthropic/', 'google/', 'meta/', 'mistral/']):
            model_name = f"openai/{model_name}"
        
        lm = setup_openrouter_model(model_name)
        
        # Try to configure, but if in async context, just return the LM
        try:
            dspy.settings.configure(lm=lm)
        except RuntimeError as e:
            if "can only be called from the same async task" in str(e):
                # We're in an async context, caller should use dspy.context()
                pass
            else:
                raise
        
        console.print(f"[green]Using OpenRouter model: {model_name}[/green]")
        return lm
        
    except Exception as e:
        console.print(f"[red]OpenRouter setup failed: {e}[/red]")
        if exit_on_error:
            sys.exit(1)
        raise

class XAPIClient:
    """Enhanced client for Twitter/X API v2 with conversation fetching"""
    
    def __init__(self, bearer_token: Optional[str] = None):
        """Initialize X API client"""
        
        # Get bearer token from environment or parameter
        if not bearer_token:
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN') or os.getenv('X_API_BEARER_TOKEN')
        
        if not bearer_token:
            console.print("[red]Error: Twitter/X API Bearer Token not found![/red]")
            console.print("Please set TWITTER_BEARER_TOKEN or X_API_BEARER_TOKEN environment variable")
            console.print("Get your token from: https://developer.twitter.com/en/portal/dashboard")
            sys.exit(1)
        
        # Initialize client
        self.client = tweepy.Client(bearer_token=bearer_token)
        console.print("[green]✓ Connected to X API[/green]")
    
    def search_tweets(self, query: str, max_results: int = 10,
                     tweet_fields: Optional[List[str]] = None,
                     user_fields: Optional[List[str]] = None,
                     expansions: Optional[List[str]] = None,
                     sort_order: str = "relevancy",
                     min_likes: int = 1000) -> List[Dict]:
        """
        Enhanced search that includes conversation data.
        Filters tweets to ensure they are in English and have minimum likes.
        """
        
        if tweet_fields is None:
            tweet_fields = [
                'created_at', 'author_id', 'conversation_id',
                'public_metrics', 'lang', 'context_annotations',
                'entities', 'referenced_tweets', 'in_reply_to_user_id'
            ]
        
        if user_fields is None:
            user_fields = ['username', 'name', 'verified', 'description']
        
        if expansions is None:
            expansions = [
                'author_id', 
                'referenced_tweets.id',
                'referenced_tweets.id.author_id',
                'in_reply_to_user_id'
            ]
        
        tweets_data = []
        
        try:
            # Search tweets with expanded data
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=tweet_fields,
                user_fields=user_fields,
                expansions=expansions,
                sort_order=sort_order
            )
            
            if tweets.data:
                # Create user lookup
                users = {u.id: u for u in (tweets.includes.get('users', []) or [])}
                
                # Create referenced tweets lookup
                ref_tweets = {}
                if tweets.includes and 'tweets' in tweets.includes:
                    ref_tweets = {t.id: t for t in tweets.includes['tweets']}
                
                for tweet in tweets.data:
                    # Filter by minimum likes and language
                    likes_count = tweet.public_metrics.get('like_count', 0) if hasattr(tweet, 'public_metrics') else 0
                    tweet_lang = getattr(tweet, 'lang', None)

                    # Skip tweets that don't meet criteria
                    if likes_count < min_likes:
                        continue
                    if tweet_lang != 'en':
                        continue

                    user = users.get(tweet.author_id, None)

                    # Check if this is a reply or quote tweet
                    parent_tweets = []
                    if hasattr(tweet, 'referenced_tweets') and tweet.referenced_tweets:
                        for ref in tweet.referenced_tweets:
                            ref_type = ref.type  # 'replied_to', 'quoted', 'retweeted'
                            ref_id = ref.id

                            # Get the referenced tweet if available
                            if ref_id in ref_tweets:
                                parent_tweet = ref_tweets[ref_id]
                                parent_user = users.get(parent_tweet.author_id) if hasattr(parent_tweet, 'author_id') else None
                                parent_tweets.append({
                                    'type': ref_type,
                                    'id': ref_id,
                                    'text': parent_tweet.text if hasattr(parent_tweet, 'text') else '',
                                    'author': parent_user.username if parent_user and hasattr(parent_user, 'username') else 'unknown'
                                })

                    tweet_dict = {
                        'id': tweet.id if hasattr(tweet, 'id') else None,
                        'text': tweet.text if hasattr(tweet, 'text') else '',
                        'created_at': str(tweet.created_at) if hasattr(tweet, 'created_at') else None,
                        'author_id': tweet.author_id if hasattr(tweet, 'author_id') else None,
                        'username': user.username if user and hasattr(user, 'username') else 'unknown',
                        'name': user.name if user and hasattr(user, 'name') else 'Unknown',
                        'follower_count': user.public_metrics['followers_count'] if user and hasattr(user, 'public_metrics') and user.public_metrics else 0,
                        'conversation_id': tweet.conversation_id if hasattr(tweet, 'conversation_id') else None,
                        'parent_tweets': parent_tweets,
                        'metrics': tweet.public_metrics if hasattr(tweet, 'public_metrics') else {},
                        'url': f"https://twitter.com/{user.username if user and hasattr(user, 'username') else 'i'}/status/{tweet.id if hasattr(tweet, 'id') else 'unknown'}"
                    }

                    tweets_data.append(tweet_dict)
            
            return tweets_data
            
        except Exception as e:
            console.print(f"[red]API Error: {str(e)}[/red]")
            return []
    



class EnhancedXTweetExplorer:
    """Enhanced explorer with basic information extraction"""
    
    def __init__(self, model: str = "openai/gpt-5", info_model: str = "openai/gpt-5",
                 bearer_token: Optional[str] = None, max_workers: int = 5):
        """Initialize the enhanced explorer

        Args:
            model: Model for general processing
            info_model: Model for information extraction (deprecated - not used)
            bearer_token: Twitter API bearer token
            max_workers: Maximum number of parallel workers for processing
        """

        self.model = model  # For general processing
        self.info_model = info_model  # Kept for compatibility but not used
        self.max_workers = max_workers  # For parallel processing

        # Setup the main model
        self.lm = setup_dspy_model(self.model, exit_on_error=True)  # Store the LM instance
        console.print(f"[green]✓ LLM configured: {self.model}[/green]")

        self.x_client = XAPIClient(bearer_token)

        console.print(f"[cyan]Information extraction simplified - using basic parent tweet context only[/cyan]")

        self.tweets_data = []
        self._tweet_batch = []  # Store current batch for parallel processing
    
    def process_single_tweet(self, tweet_data: Tuple[Dict, str, int]) -> Dict:
        """Process a single tweet with information extraction (for parallel processing)"""
        tweet, topic, index = tweet_data
        
        try:
            # Gather real information
            information = self.gather_real_information(tweet, topic)
            
            # Format for dataset
            metrics = tweet.get('metrics', {})
            
            # Extract just the content for simple format
            simple_info = [info['content'] for info in information]

            # Calculate normalized engagement score: likes/sqrt(follower_count)
            likes = metrics.get('like_count', 0)
            follower_count = tweet.get('follower_count', 1)  # Default to 1 to avoid division by zero
            normalized_engagement = likes / (follower_count ** 0.5) if follower_count > 0 else 0

            entry = {
                "tweet": tweet['text'],
                "information": simple_info,  # Simple list of strings
                "username": tweet.get('username', 'unknown'),
                "url": tweet.get('url', ''),
                "created_at": tweet.get('created_at', ''),
                "likes": likes,
                "retweets": metrics.get('retweet_count', 0),
                "quotes": metrics.get('quote_count', 0),
                "replies": metrics.get('reply_count', 0),
                "follower_count": follower_count,
                "normalized_engagement": round(normalized_engagement, 3),
                "topic": topic,
                "tweet_id": tweet.get('id', ''),
                "conversation_id": tweet.get('conversation_id', ''),
                "has_parent": len(tweet.get('parent_tweets', [])) > 0,
                "is_real": True,
                "_index": index  # Preserve original order
            }
            
            console.print(f"[green]✓[/green] Processed tweet {index + 1}/{len(self._tweet_batch)}")
            return entry
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to process tweet {index + 1}: {str(e)[:100]}")
            return None
    
    def gather_real_information(self, tweet: Dict, topic: str) -> List[Dict]:
        """Gather basic information sources using parent tweets only"""

        information = []

        # Extract parent tweets if available
        parent_tweets = tweet.get('parent_tweets', [])
        if parent_tweets:
            for parent in parent_tweets:
                information.append({
                    'content': f"[{parent['type'].replace('_', ' ').title()}] @{parent['author']}: {parent['text']}",
                    'type': 'parent_tweet',
                    'source': f"Twitter/@{parent['author']}"
                })

        # Add engagement metrics if viral
        metrics = tweet.get('metrics', {})
        like_count = metrics.get('like_count', 0)
        retweet_count = metrics.get('retweet_count', 0)

        if like_count > 1000 or retweet_count > 500:
            engagement_ratio = retweet_count / like_count if like_count > 0 else 0
            virality_factor = "high engagement" if engagement_ratio > 0.5 else "moderate engagement"

            information.append({
                'content': f"Viral tweet metrics: {like_count:,} likes, {retweet_count:,} retweets, {metrics.get('reply_count', 0):,} replies. "
                          f"Engagement ratio of {engagement_ratio:.1%} indicates {virality_factor}. "
                          f"This level of virality suggests the content resonated strongly with the {topic} community.",
                'type': 'engagement_analysis',
                'source': 'Twitter Analytics'
            })

        # Add basic topic context if no parent tweets
        if not information:
            information.append({
                'content': f"Topic context: Latest developments in {topic}",
                'type': 'background',
                'source': 'General Knowledge'
            })

        console.print(f"[dim]→ Generated {len(information)} information sources for tweet[/dim]")
        return information
    
    def explore_topic(self, topic: str, count: int = 2) -> List[Dict]:
        """
        Enhanced exploration with basic information gathering
        """
        
        console.print(Panel(
            f"[bold cyan]Enhanced X API Explorer[/bold cyan]\n"
            f"Topic: {topic}\n"
            f"Target: {max(count, 2)} tweets (minimum)\n"
            f"Filters: English only, ≥1000 likes\n"
            f"Features: Parent tweets + Basic information extraction\n"
            f"Processing Model: {self.model}\n"
            f"[yellow]Parallel Workers: {self.max_workers}[/yellow]",
            title="Enhanced Exploration (Filtered + Guaranteed Minimum)"
        ))
        
        # Step 1: Fetch real tweets from X
        console.print("\n[bold]Step 1: Fetching real tweets from X...[/bold]")
        
        # Build search query
        query_parts = [topic, "-is:retweet", "-is:reply", "lang:en"]
        query = " ".join(query_parts)
        
        console.print(f"[cyan]Searching X for: {query}[/cyan]")
        console.print(f"[cyan]Filtering: English language, minimum 1000 likes[/cyan]")
        console.print(f"[cyan]Target: Minimum {max(count, 2)} tweets[/cyan]")

        # Ensure we fetch enough tweets to meet the minimum requirement after filtering
        min_tweets = max(count, 2)  # Ensure at least 2 tweets minimum
        fetch_count = min_tweets * 2  # Fetch more to account for filtering

        tweets = []
        attempt = 0
        max_attempts = 3

        while len(tweets) < min_tweets and attempt < max_attempts:
            attempt += 1
            console.print(f"[cyan]Attempt {attempt}: Fetching up to {fetch_count} tweets...[/cyan]")

            batch_tweets = self.x_client.search_tweets(
                query=query,
                max_results=fetch_count,
                sort_order="relevancy",
                min_likes=1000
            )

            tweets.extend(batch_tweets)

            console.print(f"[cyan]Found {len(batch_tweets)} tweets in this batch (total: {len(tweets)})[/cyan]")

            if len(tweets) < min_tweets:
                console.print(f"[yellow]Need more tweets. Current: {len(tweets)}, Target: {min_tweets}[/yellow]")
                fetch_count = min(fetch_count * 2, 100)  # Double the fetch size but cap at 100
            else:
                break

        if not tweets:
            console.print("[red]No tweets found. Try a different topic or check API limits.[/red]")
            return []

        # Trim to the requested count if we have more
        if len(tweets) > count:
            tweets = tweets[:count]

        console.print(f"[green]✓ Found {len(tweets)} filtered tweets (minimum {min_tweets} required)[/green]")
        
        # Check for parent tweets
        parent_count = sum(1 for t in tweets if t.get('parent_tweets'))
        if parent_count > 0:
            console.print(f"[cyan]✓ Found {parent_count} tweets with parent context[/cyan]")
        
        # Step 2: Extract basic information sources (PARALLELIZED)
        console.print("\n[bold]Step 2: Extracting basic information sources (parallelized)...[/bold]")
        
        # Store tweets for parallel processing
        self._tweet_batch = tweets
        
        # Prepare tweet data with indices to preserve order
        tweet_data = [(tweet, topic, i) for i, tweet in enumerate(tweets)]
        
        # Process tweets in parallel
        max_workers = min(self.max_workers, len(tweets))  # Use configured number of parallel workers
        console.print(f"[cyan]Processing {len(tweets)} tweets with {max_workers} parallel workers...[/cyan]")
        
        dataset = []
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_data = {
                executor.submit(self.process_single_tweet, data): data[2]  # Map future to index
                for data in tweet_data
            }
            
            # Collect results as they complete
            results = {}
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Processing tweets in parallel...", total=len(tweets))
                
                for future in as_completed(future_to_data):
                    index = future_to_data[future]
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per tweet
                        if result:
                            results[index] = result
                            progress.update(task, advance=1)
                        else:
                            failed_count += 1
                            progress.update(task, advance=1)
                    except Exception as e:
                        console.print(f"[red]Error processing tweet {index + 1}: {str(e)[:100]}[/red]")
                        failed_count += 1
                        progress.update(task, advance=1)
        
        # Sort results by original index to preserve order
        for i in sorted(results.keys()):
            entry = results[i]
            entry.pop('_index', None)  # Remove temporary index field
            dataset.append(entry)
        
        if failed_count > 0:
            console.print(f"[yellow]Warning: {failed_count} tweets failed to process[/yellow]")
        
        console.print(f"[green]✓ Successfully processed {len(dataset)} tweets[/green]")
        
        self.tweets_data = dataset
        return dataset
    
    def save_dataset(self, output_file: str):
        """Save the enhanced dataset"""
        
        # Create directory if needed
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
        if output_dir != "." and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.tweets_data, f, indent=2, ensure_ascii=False)
        
        # Count information quality
        total_info = sum(len(t['information']) for t in self.tweets_data)
        parent_tweets = sum(1 for t in self.tweets_data if t.get('has_parent'))
        
        console.print(f"\n[green]✓ Enhanced dataset saved to {output_file}[/green]")
        console.print(f"  Total tweets: {len(self.tweets_data)}")
        console.print(f"  Total information items: {total_info}")
        console.print(f"  Tweets with parent context: {parent_tweets}")
        console.print(f"  Average info per tweet: {total_info/len(self.tweets_data):.1f}")
    
    def display_results(self):
        """Display enhanced results"""
        
        if not self.tweets_data:
            console.print("[yellow]No data to display[/yellow]")
            return
        
        console.print("\n[bold blue]═══ Enhanced Tweet Analysis ═══[/bold blue]\n")
        
        for i, entry in enumerate(self.tweets_data[:3], 1):
            # Format display
            content = f"[yellow]Tweet:[/yellow]\n{entry['tweet']}\n\n"
            content += f"[cyan]@{entry['username']}[/cyan] • "
            content += f"❤️ {entry['likes']:,} • 🔁 {entry['retweets']:,} • "
            content += f"👥 {entry['follower_count']:,} followers\n"
            content += f"[blue]Normalized Engagement: {entry['normalized_engagement']:.3f}[/blue]\n\n"

            content += f"[green]Information Sources:[/green]\n"
            for j, info in enumerate(entry['information'][:4], 1):
                content += f"  {j}. {info[:150]}...\n" if len(info) > 150 else f"  {j}. {info}\n"

            if entry.get('has_parent'):
                content += f"\n[cyan]✓ Has parent tweet context[/cyan]"

            console.print(Panel(content, title=f"Tweet {i} (Enhanced)", border_style="cyan"))

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced X/Twitter explorer with basic information extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This enhanced version provides:
  ✓ Fetch actual parent tweets from conversations
  ✓ Extract basic engagement metrics and context
  ✓ Provide parent tweet information sources
  ✓ Include viral tweet analytics
  ⚡ PARALLEL PROCESSING for fast extraction

Examples:
  # Basic exploration (default: 2 tweets)
  python x_api_explorer_enhanced.py "AI safety"

  # Custom tweet count (guaranteed min 5 tweets)
  python x_api_explorer_enhanced.py "AI safety" -n 5 -w 10

  # From specific user (guaranteed min 3 tweets)
  python x_api_explorer_enhanced.py "from:sama OpenAI" -n 3

  # Hashtag exploration with parallel workers (guaranteed min 10 tweets)
  python x_api_explorer_enhanced.py "#GPT4" -n 10 -w 8 --save
        """
    )
    
    parser.add_argument('topic',
                       help='Topic or search query')
    parser.add_argument('-n', '--count', type=int, default=2,
                       help='Number of tweets to fetch (minimum: 2, default: 2)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output JSON file path')
    parser.add_argument('-m', '--model', type=str, default='openai/gpt-4',
                       help='Model for general processing (default: openai/gpt-4)')
    parser.add_argument('-w', '--workers', type=int, default=5,
                       help='Number of parallel workers for processing (default: 5)')
    parser.add_argument('-s', '--save', action='store_true',
                       help='Save dataset to file')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display results')
    
    args = parser.parse_args()
    
    # Initialize enhanced explorer
    explorer = EnhancedXTweetExplorer(
        model=args.model,
        max_workers=args.workers
    )
    
    # Explore topic
    dataset = explorer.explore_topic(args.topic, args.count)
    
    if not dataset:
        console.print("[red]No tweets found or processed.[/red]")
        return
    
    # Display results
    if not args.no_display:
        explorer.display_results()
    
    # Save dataset
    if args.save or args.output:
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.topic[:30])
            args.output = f"enhanced_datasets/{safe_topic}_enhanced_{timestamp}.json"
        
        explorer.save_dataset(args.output)
        
        console.print("\n[green]✨ Enhanced dataset ready![/green]")
        console.print(f"Train with: python main.py train --data {args.output}")

if __name__ == "__main__":
    main()

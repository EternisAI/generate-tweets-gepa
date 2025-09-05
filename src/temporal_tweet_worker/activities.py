import asyncio
import json
import multiprocessing
import os
import random
import sys
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

from temporalio import activity
from temporalio.common import RetryPolicy
from temporalio.exceptions import CancelledError

# Add parent directory to path to import x_api_explorer_enhanced
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from x_api_explorer_enhanced import EnhancedXTweetExplorer
from add_humor_analysis import analyze_tweet_humor
from gepa_official_optimizer import GEPATweetOptimizer, TweetGeneratorModule
from openrouter_config import setup_openrouter_model

def analyze_media_for_selected_tweets(tweets_data):
    """Analyze media only for tweets we actually keep"""
    print(f"[Media Analysis] Processing {len(tweets_data)} tweets...")

    for i, tweet_data in enumerate(tweets_data):
        try:
            print(f"[Media Analysis] Tweet {i+1} keys: {list(tweet_data.keys())}")

            # Check if the tweet already has media data from the original fetch
            quote_tweet = tweet_data.get('quote_tweet', {})
            quoted_tweet = tweet_data.get('quoted_tweet', {})

            print(f"[Media Analysis] Quote tweet keys: {list(quote_tweet.keys()) if quote_tweet else 'None'}")
            print(f"[Media Analysis] Quoted tweet keys: {list(quoted_tweet.keys()) if quoted_tweet else 'None'}")

            # Try to get media from quoted tweet first (more likely to have media)
            media_data = quoted_tweet.get('media') if quoted_tweet else None
            print(f"[Media Analysis] Quoted tweet media raw: {media_data}")

            if not media_data:
                # Fall back to quote tweet media
                media_data = quote_tweet.get('media') if quote_tweet else None
                print(f"[Media Analysis] Quote tweet media raw: {media_data}")

            # Ensure media_data is a list
            if media_data is None:
                media_data = []
                print(f"[Media Analysis] Media data was None, set to empty list")
            elif not isinstance(media_data, list):
                media_data = [media_data]
                print(f"[Media Analysis] Media data converted to list: {media_data}")

            print(f"[Media Analysis] Final media data: {media_data}")
            print(f"[Media Analysis] Media data found: {len(media_data)} items")

            if media_data:
                print(f"[Media Analysis] Found {len(media_data)} media items for tweet {i+1}")
                # Add media analysis to the tweet data
                tweet_data['media_analysis_data'] = media_data
            else:
                print(f"[Media Analysis] No media found for tweet {i+1}")

            print(f"[Media Analysis] Processed tweet {i+1}/{len(tweets_data)}")

        except Exception as e:
            print(f"[Media Analysis] Error processing tweet {i+1}: {str(e)[:100]}")
            continue

    return tweets_data

# Import viral_tweets and extract_tweets functionality
from viral_tweets import get_viral_tweets


from .db import (
    create_workflow_record,
    update_exploration_results,
    update_prompt,
    update_tweet_drafts,
    get_workflow_status
)

@activity.defn
async def explore(topic: str, workflow_id: int = None) -> Dict[str, Any]:
    """Use x_api_explorer_enhanced to gather real Twitter insights about a topic"""
    
    try:
        # Generate workflow UUID
        import uuid
        workflow_uuid = str(uuid.uuid4())
        print(f"[Explore Activity] Generated workflow UUID: {workflow_uuid}")

        # Create workflow record if not provided
        if workflow_id is None:
            workflow_id = create_workflow_record(topic)
        
        # Log what we're learning
        print(f"[Explore Activity] Starting exploration for topic: {topic}")
        # Initialize the enhanced explorer with model names
        explorer = EnhancedXTweetExplorer(
            model="openai/gpt-5",  # Pass model name directly
            info_model="openai/gpt-5",  # Use same model for info extraction
            max_workers=5  # Parallel processing for speed
        )
        
        # Log configuration
        print(f"[Explore Activity] Initialized EnhancedXTweetExplorer with GPT-5 model")
        
        # Run exploration in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        dataset = await loop.run_in_executor(
            None,
            explorer.explore_topic,
            topic,
            10  # Get 5 tweets for analysis
        )
        
        # Log results
        print(f"[Explore Activity] Found {len(dataset)} tweets with insights")
        
                # Extract insights from the dataset
        if not dataset:
            print("[Explore Activity] No tweets found, using fallback data")
            fallback_result = {
                "topic": topic,
                "insights": ["No recent tweets found for analysis"],
                "keywords": [topic],
                "engagement_score": 0.5,
                "tweets_analyzed": 0,
                "information_sources": [],
                "tweets": [],
                "workflow_id": workflow_id
            }
            if workflow_id:
                update_exploration_results(
                    workflow_id=workflow_id,
                    topic=topic,
                    articles=[],
                    exp_result=fallback_result
                )
            return fallback_result
        usernames = ["growing_daniel", "allgarled"]

        # Loop over usernames to scrape viral quote tweets from each profile (max 2 total)
        print("[Explore Activity] Scraping viral quote tweets from multiple profiles (max 2 total)...")
        viral_tweets_data = []
        max_viral_tweets = 2  # Limit to maximum 2 viral tweets

        for username in usernames:
            # Check if we already have enough viral tweets
            if len(viral_tweets_data) >= max_viral_tweets:
                print(f"[Explore Activity] Already have {len(viral_tweets_data)} viral tweets, skipping @{username}")
                continue

            print(f"[Explore Activity] Processing profile: @{username}")
            try:
                user_tweets = get_viral_tweets(
                    username=username,
                    min_likes=100,        # Minimum engagement threshold
                    language="en",        # Focus on English tweets
                    output_file=None,     # Don't save to file, process in memory
                    max_tweets=None,      # Don't limit here, we'll limit after filtering
                    include_media_analysis=True  # Skip media analysis for now
                )
                # Filter out any None or invalid tweets
                if user_tweets:
                    valid_tweets = [tweet for tweet in user_tweets if tweet and isinstance(tweet, dict)]
                                    # Only take what we need to reach the limit
                remaining_slots = max_viral_tweets - len(viral_tweets_data)
                tweets_to_add = valid_tweets[:remaining_slots]

                viral_tweets_data.extend(tweets_to_add)
                print(f"[Explore Activity] Added {len(tweets_to_add)} viral tweets from @{username} (total: {len(viral_tweets_data)})")

                # Stop if we've reached the limit
                if len(viral_tweets_data) >= max_viral_tweets:
                    print(f"[Explore Activity] Reached viral tweet limit of {max_viral_tweets}")
                    break
                else:
                    print(f"[Explore Activity] No viral tweets found from @{username}")
            except Exception as e:
                print(f"[Explore Activity] Error processing @{username}: {str(e)[:100]}")
                continue

        print(f"[Explore Activity] Total viral tweets collected: {len(viral_tweets_data)} (limited to {max_viral_tweets})")

        # Do media analysis only on tweets we actually keep
        if viral_tweets_data:
            print(f"[Explore Activity] Performing media analysis on {len(viral_tweets_data)} selected tweets...")
            viral_tweets_data = analyze_media_for_selected_tweets(viral_tweets_data)

        # Aggregate insights from real tweets and viral tweets
        all_keywords = set()

        # Process tweets from EnhancedXTweetExplorer
        total_engagement = 0
        for tweet_data in dataset:
            # Extract keywords from tweet text
            words = tweet_data['tweet'].lower().split()
            keywords = [w.strip('#.,!?') for w in words if len(w) > 4 and not w.startswith('http')]
            all_keywords.update(keywords[:5])

            # Calculate engagement
            likes = tweet_data.get('likes', 0)
            retweets = tweet_data.get('retweets', 0)

            total_engagement += (likes + retweets * 2)  # Weight retweets more       
        # Log final results
        result = {
            "topic": topic,
            "keywords": list(all_keywords)[:10],
            "tweets_analyzed": len(dataset)
        }

        print(f"[Explore Activity] Exploration complete: {len(result['keywords'])} keywords, {len(dataset)} tweets analyzed")

        # Create topic-based articles content
        topic_articles = []
        if topic:
            topic_articles.append({
                'type': 'topic_content',
                'content': f"Topic analysis for: {topic}",
                'source': 'topic_input'
            })
        usernames = ["growing_daniel", "allgarled"]

        # Loop over usernames to scrape viral quote tweets from each profile (max 2 total)
        print("[Explore Activity] Scraping viral quote tweets from multiple profiles (max 2 total)...")
        viral_tweets_data = []
        max_viral_tweets = 2  # Limit to maximum 2 viral tweets

        for username in usernames:
            # Check if we already have enough viral tweets
            if len(viral_tweets_data) >= max_viral_tweets:
                print(f"[Explore Activity] Already have {len(viral_tweets_data)} viral tweets, skipping @{username}")
                continue

            print(f"[Explore Activity] Processing profile: @{username}")
            try:
                user_tweets = get_viral_tweets(
                    username=username,
                    min_likes=100,        # Minimum engagement threshold
                    language="en",        # Focus on English tweets
                    output_file=None,     # Don't save to file, process in memory
                    max_tweets=None,      # Don't limit here, we'll limit after filtering
                    include_media_analysis=True  # Skip media analysis for now
                )
                # Filter out any None or invalid tweets
                if user_tweets:
                    valid_tweets = [tweet for tweet in user_tweets if tweet and isinstance(tweet, dict)]
                                    # Only take what we need to reach the limit
                remaining_slots = max_viral_tweets - len(viral_tweets_data)
                tweets_to_add = valid_tweets[:remaining_slots]

                viral_tweets_data.extend(tweets_to_add)
                print(f"[Explore Activity] Added {len(tweets_to_add)} viral tweets from @{username} (total: {len(viral_tweets_data)})")

                # Stop if we've reached the limit
                if len(viral_tweets_data) >= max_viral_tweets:
                    print(f"[Explore Activity] Reached viral tweet limit of {max_viral_tweets}")
                    break
                else:
                    print(f"[Explore Activity] No viral tweets found from @{username}")
            except Exception as e:
                print(f"[Explore Activity] Error processing @{username}: {str(e)[:100]}")
                continue

        print(f"[Explore Activity] Total viral tweets collected: {len(viral_tweets_data)} (limited to {max_viral_tweets})")

        # Aggregate insights from real tweets and viral tweets

        def remove_links(text):
            """Remove URLs from text"""
            import re
            # Remove http/https URLs
            text = re.sub(r'https?://[^\s]+', '', text)
            # Remove twitter.com links (common in tweets)
            text = re.sub(r'twitter\.com/[^\s]+', '', text)
            # Remove x.com links (Twitter's new domain)
            text = re.sub(r'x\.com/[^\s]+', '', text)
            # Clean up extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Do media analysis only on tweets we actually keep
        if viral_tweets_data:
            print(f"[Explore Activity] Performing media analysis on {len(viral_tweets_data)} selected tweets...")
            viral_tweets_data = analyze_media_for_selected_tweets(viral_tweets_data)

        # Create enhanced viral tweets data in transformed format
        viral_tweets_enhanced = []
        for viral_tweet in viral_tweets_data:
            quote_tweet = viral_tweet.get('quote_tweet', {})
            quoted_tweet = viral_tweet.get('quoted_tweet', {})

            # Extract basic tweet information
            tweet_text = remove_links(quote_tweet.get('text', ''))
            metrics = quote_tweet.get('metrics', {})
            author_info = quote_tweet.get('author', {})

            # Create enhanced information analysis
            quoted_text = remove_links(quoted_tweet.get('text', ''))
            media_info = quoted_tweet.get('media', [])

            # Build media analysis
            media_analysis = []
            # First try to use the media analysis data we collected
            media_analysis_data = viral_tweet.get('media_analysis_data', [])

            if media_analysis_data:
                for media in media_analysis_data:
                    if isinstance(media, dict) and 'analysis' in media:
                        # Remove links from media analysis as well
                        clean_analysis = remove_links(media['analysis'])
                        media_analysis.append(clean_analysis)
            elif media_info:
                # Fall back to original media_info if no analysis data
                for media in media_info:
                    if isinstance(media, dict) and 'analysis' in media:
                        # Remove links from media analysis as well
                        clean_analysis = remove_links(media['analysis'])
                        media_analysis.append(clean_analysis)

            # Analyze humor in the quote tweet pair
            try:
                import os
                if os.getenv('OPENROUTER_API_KEY'):
                    humor_analysis = analyze_tweet_humor(tweet_text, quoted_text)
                    # Escape quotes and newlines for JSON
                    humor_analysis = humor_analysis.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                else:
                    print("[Explore Activity] OPENROUTER_API_KEY not set, skipping humor analysis")
                    humor_analysis = "Humor analysis unavailable - API key not configured"
            except Exception as e:
                print(f"[Explore Activity] Error analyzing humor: {str(e)[:100]}")
                humor_analysis = f"Humor analysis failed: {str(e)[:50]}"

            # Create comprehensive information entry using proper JSON
            import json

            info_dict = {
                "text": quoted_text,
                "media_analysis": media_analysis if media_analysis else ["No media analysis available"],
                "humor_analysis": humor_analysis
            }

            info_content = json.dumps(info_dict, indent=2, ensure_ascii=False)

            # Create enhanced entry in transformed format
            enhanced_entry = {
                "tweet": tweet_text,
                "username": author_info.get('username', 'unknown'),
                "created_at": quote_tweet.get('created_at', ''),
                "retweets": metrics.get('retweets', 0),
                "replies": metrics.get('replies', 0),
                "likes": metrics.get('likes', 0),
                "quotes": metrics.get('quotes', 0),
                "information": [info_content],
                "follower_count": author_info.get('follower_count', 0),
                "normalized_engagement": metrics.get('normalized_likes', 0),
                "is_viral": True
            }

            viral_tweets_enhanced.append(enhanced_entry)

        # Create articles from viral tweets
        viral_articles = []
        for i, enhanced_tweet in enumerate(viral_tweets_enhanced):
            # Get the corresponding viral tweet data to extract quoted tweet directly
            quoted_tweet_text = ""
            if i < len(viral_tweets_data):
                viral_tweet = viral_tweets_data[i]
                quoted_tweet = viral_tweet.get('quoted_tweet', {})
                quoted_tweet_text = remove_links(quoted_tweet.get('text', ''))

            # Combine quote tweet and quoted tweet content
            combined_content = f"Quote: {enhanced_tweet['tweet']}\n\nQuoted: {quoted_tweet_text}"

            viral_articles.append({
                'type': 'viral_quote_tweet_enhanced',
                'content': quoted_tweet_text,
                'likes': enhanced_tweet['likes'],
                'normalized_score': enhanced_tweet['normalized_engagement'],
                'follower_count': enhanced_tweet['follower_count'],
                'quote_tweet': enhanced_tweet['tweet'],
                'quoted_tweet': quoted_tweet_text
            })

        # Combine topic articles with viral articles
        all_articles = topic_articles + viral_articles
        print(f"[Explore Activity] Topic articles: {len(topic_articles)}")
        print(f"[Explore Activity] Viral articles: {len(viral_articles)}")
        print(f"[Explore Activity] Total articles: {len(all_articles)}")

        # Add viral tweets to exp_result in enhanced format
        # result['viral_tweets'] = viral_tweets_data
        result['tweets'] = viral_tweets_enhanced
        result['total_viral_tweets'] = len(viral_tweets_data)

        # Update database with exploration results
        print(f"[Explore Activity] Preparing to update database with {len(all_articles)} articles")
        print(f"[Explore Activity] Articles content preview: {all_articles[:1] if all_articles else 'No articles'}")

        # Check if database URL is set
        import os
        if not os.getenv('DATABASE_URL'):
            print(f"[Explore Activity] DATABASE_URL not set - skipping database update")
            print(f"[Explore Activity] Set DATABASE_URL environment variable to enable database storage")
        else:
            try:
                print(f"[Explore Activity] Calling update_exploration_results...")
                update_exploration_results(
                    workflow_id=workflow_id,
                    topic=topic,
                    articles=all_articles,
                    exp_result=result
                )
                print(f"[Explore Activity] Database updated successfully with {len(all_articles)} articles")
            except Exception as db_error:
                print(f"[Explore Activity] Database update error: {str(db_error)}")
                print(f"[Explore Activity] Error type: {type(db_error).__name__}")
                import traceback
                print(f"[Explore Activity] Full traceback:")
                traceback.print_exc()

            # Save enhanced viral tweets to file in transformed format
            if viral_tweets_enhanced:
                import json
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"viral_tweets_enhanced_{timestamp}.json"

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(viral_tweets_enhanced, f, indent=2, ensure_ascii=False)

                print(f"[Explore Activity] Enhanced viral tweets saved to {output_file}")
                result['viral_tweets_output_file'] = output_file
        
        # Add workflow identifiers to result for downstream activities
        result["workflow_id"] = workflow_id  # Database integer ID
        result["workflow_uuid"] = workflow_uuid  # UUID identifier
        return result
        
    except Exception as e:
        print(f"[Explore Activity] Error during exploration: {str(e)}")
        error_msg = str(e)[:50]  # Capture error message within exception scope
        # Generate UUID for error case if not already generated
        try:
            error_uuid = workflow_uuid
        except NameError:
            import uuid
            error_uuid = str(uuid.uuid4())

        # Fallback to mock data on error
        return {
            "topic": topic,
            "insights": [f"Error during exploration: {error_msg}"],
            "keywords": [topic],
            "engagement_score": 0.5,
            "tweets_analyzed": 0,
            "information_sources": [],
            "tweets": [],
            "workflow_id": workflow_id or 0,
            "workflow_uuid": error_uuid
        }




@activity.defn
async def gen_gepa_prompt(exp_result: Dict[str, Any]) -> str:
    """Generate a GEPA-optimized prompt for tweet generation"""
    
    topic = exp_result.get("topic", "general")
    keywords = exp_result.get("keywords", [])
    engagement_score = exp_result.get("engagement_score", 0.5)
    information_sources = exp_result.get("information_sources", [])
    tweets = exp_result.get("tweets", [])
    tweets_analyzed = exp_result.get("tweets_analyzed", 0)
    
    # Log GEPA prompt generation
    print(f"[GEPA Prompt] Generating optimized prompt for topic: {topic}")
    print(f"[GEPA Prompt] Using {tweets_analyzed} analyzed tweets, engagement score: {engagement_score}")
    
    # Get workflow_id at the start
    workflow_id = exp_result.get("workflow_id")
    
    try:
        # Import required modules
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        import dspy
        from gepa_official_optimizer import TweetGeneratorModule
        from data_processor import TweetData
        
        # Get API key for GEPA training
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        # Convert tweets to training data
        training_data = []
        for tweet in tweets:
            tweet_data = TweetData(
                tweet=tweet.get("tweet", ""),
                username=tweet.get("username", ""),
                created_at=tweet.get("created_at", ""),
                retweets=tweet.get("metrics", {}).get("retweet_count", 0),
                replies=tweet.get("metrics", {}).get("reply_count", 0),
                likes=tweet.get("metrics", {}).get("like_count", 0),
                quotes=tweet.get("metrics", {}).get("quote_count", 0),
                information=tweet.get("information_detailed", [])
            )
            training_data.append(tweet_data)
        
        print(f"[GEPA Prompt] Training on {len(training_data)} tweets")
        
        # Create and train tweet generator in a separate process
        print("[GEPA Prompt] Training on examples...")
        
        # Run training in the activity process
        print("[GEPA Prompt] Starting GEPA training...")
        try:
            # Setup OpenRouter model for GEPA training with GPT-5
            lm = setup_openrouter_model()  # Uses default GPT-5
            
            # Create generator and optimizer inside context
            with dspy.context(lm=lm):
                generator = TweetGeneratorModule()
                optimizer = GEPATweetOptimizer(judge_lm=lm, reflection_lm=lm)
            
            # Convert data to DSPy examples
            examples = []
            for i, tweet_data in enumerate(training_data):
                example = dspy.Example(
                    information_context=tweet_data.information,
                    original_tweet=tweet_data.tweet
                ).with_inputs('information_context')
                examples.append(example)
                # Heartbeat is synchronous - do NOT await
                activity.heartbeat({
                    "stage": "prep",
                    "count": len(examples),
                    "total": len(training_data),
                    "current": i + 1
                })
            
            print(f"[GEPA Prompt] Prepared {len(examples)} training examples")
            
            # ---------- SAFE DEFAULTS ----------
            output_dir = "trained_models"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            trained_module = None
            prompt = None
            prompt_file = None
            model_file = None
            improvement = None
            percent_gain = None
            
            # Run optimization with explicit settings inside dspy context
            try:
                with dspy.context(lm=lm):
                    # Configure GEPA optimization parameters
                    max_generations =None  # Reduced from 3 to 1 for faster training
                    max_metric_calls = 1  # Let GEPA auto-configure based on dataset size
                    
                    print(f"[GEPA Prompt] Starting optimization with:")
                    print(f"[GEPA Prompt] - Max generations: {max_generations}")
                    print(f"[GEPA Prompt] - Max metric calls: {max_metric_calls}")
                    print(f"[GEPA Prompt] - Training examples: {len(examples)}")
                    
                    trained_module = optimizer.optimize(
                        student_module=generator,
                        trainset=examples,
                        auto=None,  # Disable auto mode
                        max_generations=max_generations,
                        max_metric_calls=max_metric_calls,
                        track_stats=True,
                        log_dir=output_dir  # Provide log directory to GEPA
                    )
                    
                    if trained_module is None:
                        raise RuntimeError("GEPA optimization returned None")
                    
                    print(f"[GEPA Prompt] Training completed successfully: {type(trained_module)}")
                    
                    # Extract and save the evolved prompt
                    prompt = None
                    if hasattr(trained_module.generate, 'predict') and hasattr(trained_module.generate.predict, 'signature'):
                        signature = trained_module.generate.predict.signature
                        if hasattr(signature, 'instructions'):
                            prompt = signature.instructions
                        elif hasattr(signature, '_instructions'):
                            prompt = signature._instructions
                    
                    if not prompt and hasattr(trained_module.generate, 'extended_signature'):
                        if hasattr(trained_module.generate.extended_signature, 'instructions'):
                            prompt = trained_module.generate.extended_signature.instructions
                    
                    if not prompt and hasattr(trained_module.generate, 'predictor') and hasattr(trained_module.generate.predictor, 'signature'):
                        if hasattr(trained_module.generate.predictor.signature, 'instructions'):
                            prompt = trained_module.generate.predictor.signature.instructions
                    
                    # Check if optimization improved the score
                    improvement = optimizer.last_best_score - optimizer.baseline_score
                    if improvement > 0:
                        # Save optimized prompt and model if there was improvement
                        if prompt:
                            # Save prompt to text file
                            prompt_file = os.path.join(output_dir, f"gepa_prompt_{timestamp}.txt")
                            with open(prompt_file, 'w') as f:
                                f.write("=== GEPA-Optimized Tweet Generation Prompt ===\n\n")
                                f.write(prompt)
                                f.write("\n\n=== Training Stats ===\n")
                                f.write(f"Baseline Score: {optimizer.baseline_score:.3f}\n")
                                f.write(f"Best Score: {optimizer.last_best_score:.3f}\n")
                                if optimizer.baseline_score > 0:
                                    percent_gain = (improvement / optimizer.baseline_score * 100)
                                    f.write(f"Improvement: +{improvement:.3f} ({percent_gain:.1f}% gain)\n")
                                else:
                                    f.write(f"Improvement: +{improvement:.3f} (baseline was 0)\n")
                                f.write(f"Average Score: {optimizer.last_avg_score:.3f}\n")
                            print(f"[GEPA Prompt] Saved evolved prompt to {prompt_file}")
                            
                            # Read the full prompt from file to get the complete instructions
                            with open(prompt_file, 'r') as f:
                                full_prompt = f.read()
                            
                            # Update database with the complete prompt
                            if workflow_id:
                                update_prompt(workflow_id, full_prompt)
                        else:
                            print("[GEPA Prompt] Warning: Could not extract evolved prompt")
                    else:
                        print("[GEPA Prompt] No improvement in score, using fallback prompt")
                        # Save fallback prompt to optimized_prompt.txt
                        keywords_str = ", ".join(keywords[:10])
                        fallback_prompt = f"""Create an engaging tweet about {topic} that:
                - Uses keywords: {keywords_str}
                - Targets high engagement (current score: {engagement_score:.2f})
                - Based on {tweets_analyzed} analyzed tweets
                - Includes relevant hashtags
                - Stays under 280 characters
                - Encourages engagement through questions or calls to action
                - Follows viral tweet patterns for maximum reach"""
                        
                        # Update database with fallback prompt
                        if workflow_id:
                            update_prompt(workflow_id, fallback_prompt)
                        # Prepare information context
                        info_context = []
                        for source in information_sources[:5]:  # Use top 5 sources
                            info_context.append(f"[{source.get('type', 'unknown')}] {source.get('content', '')[:200]}")
                        
                        # Generate optimized prompt using trained module
                        print(f"[GEPA Prompt] Generating prompt with trained module")
                        
                        prompt = f"""You are an expert viral tweet writer trained on real high-engagement tweets.
                Generate a highly engaging tweet about {topic} using GEPA optimization.

                CONTEXT INFORMATION:
                {chr(10).join(info_context) if info_context else 'General knowledge about ' + topic}

                OPTIMIZATION TARGETS:
                - Keywords to naturally include: {', '.join(keywords[:7])}
                - Style: Viral tweet patterns with high engagement potential
                - Based on {tweets_analyzed} analyzed tweets

                CONSTRAINTS:
                - Maximum 280 characters
                - Include 1-2 relevant hashtags
                - Natural, conversational tone
                - No promotional language

                Generate a tweet that would achieve maximum engagement based on the trained patterns."""

                        print(f"[GEPA Prompt] Generated optimized prompt with {len(info_context)} information sources")
                        
                        
                    
                    # Save trained module (now we're sure it exists and is inside DSPy context)
                    model_file = os.path.join(output_dir, f"gepa_model_{timestamp}.pkl")
                    trained_module.save(model_file)
                    print(f"[GEPA Prompt] Saved trained model to {model_file}")
                    
                    # Final heartbeat with success status
                    activity.heartbeat({
                        "stage": "complete",
                        "examples": len(examples),
                        "generations": max_generations,
                        "metric_calls": max_metric_calls,
                        "prompt_file": prompt_file if prompt else None,
                        "improvement": f"+{improvement:.3f} ({percent_gain:.1f}% gain)" if optimizer.baseline_score > 0 else f"+{improvement:.3f}"
                    })
                    
            except CancelledError:
                print("[GEPA Prompt] Activity cancelled during optimization")
                activity.heartbeat({
                    "stage": "cancelled",
                    "examples": len(examples)
                })
                raise  # Let Temporal record the cancellation
            except Exception as e:
                print(f"[GEPA Prompt] Optimization failed: {str(e)}")
                activity.heartbeat({
                    "stage": "failed",
                    "error": str(e),
                    "examples": len(examples)
                })
                raise
            
        except Exception as e:
            print(f"[GEPA Prompt] Training failed with error: {str(e)}")
            print(f"[GEPA Prompt] Error type: {type(e)}")
            raise
        
        
        
       
        
    except Exception as e:
        print(f"[GEPA Prompt] Failed to use GEPA optimization: {str(e)}")
        print(f"[GEPA Prompt] Falling back to standard prompt generation")
        
        # Only create fallback prompt if GEPA failed
        keywords_str = ", ".join(keywords[:10])
        fallback_prompt = f"""Create an engaging tweet about {topic} that:
- Uses keywords: {keywords_str}
- Targets high engagement (current score: {engagement_score:.2f})
- Based on {tweets_analyzed} analyzed tweets
- Includes relevant hashtags
- Stays under 280 characters
- Encourages engagement through questions or calls to action
- Follows viral tweet patterns for maximum reach"""
        
        # Update database with fallback prompt
        if workflow_id:
            update_prompt(workflow_id, fallback_prompt)
        return fallback_prompt

    return prompt


@activity.defn
async def gen_tweet(prompt: str, workflow_id: int) -> List[str]:
    """Generate tweets for each article using GEPA-optimized prompt"""
    
    print("[Gen Tweet] Starting tweet generation")
    
    # Get articles from database
    status = get_workflow_status(workflow_id)
    if not status:
        raise ValueError(f"No workflow found with ID {workflow_id}")
    
    articles = status.get("articles", [])
    if not articles:
        print("[Gen Tweet] No articles found in database")
        return []
    
    print(f"[Gen Tweet] Found {len(articles)} articles to process")
    
    # Check if this is a GEPA-optimized prompt
    is_gepa_prompt = "GEPA OPTIMIZATION" in prompt or "OPTIMIZATION TARGETS" in prompt
    generated_tweets = []
    
    # Setup OpenRouter model for generation
    lm = setup_openrouter_model()  # Uses default GPT-5
    
    # Try to use GEPA if available
    gepa_module = None
    if is_gepa_prompt:
        print(f"[Gen Tweet] Using GEPA-optimized generation strategy")
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from gepa_official_optimizer import TweetGeneratorModule
            
            # Initialize GEPA module and load the latest trained model
            output_dir = "trained_models"
            model_files = [f for f in os.listdir(output_dir) if f.startswith("gepa_model_") and f.endswith(".pkl")]
            if model_files:
                latest_model = max(model_files)
                model_path = os.path.join(output_dir, latest_model)
                print(f"[Gen Tweet] Loading trained model from {model_path}")
                gepa_module = TweetGeneratorModule()
                gepa_module.load(model_path)
        except Exception as e:
            print(f"[Gen Tweet] Could not load GEPA module: {str(e)}")
            print("[Gen Tweet] Falling back to template generation")
    
    # Process articles in parallel
    max_workers = 20  # Configure number of parallel workers
    print(f"[Gen Tweet] Processing {len(articles)} articles with {max_workers} parallel workers")
    
    def process_article(article_data):
        article, index = article_data
        article_content = article.get('content', '')
        if not article_content:
            print(f"[Gen Tweet] Skipping article {index + 1} with no content")
            return None
        
        # Combine prompt with article for context
        full_context = f"{prompt}\n\nARTICLE CONTENT:\n{article_content[:500]}"
        print(f"[Gen Tweet] Generating tweet for article {index + 1}/{len(articles)} (length: {len(article_content)} chars)")
        
        try:
            # Generate tweet using appropriate method
            with dspy.context(lm=lm):
                if gepa_module:
                    # Use GEPA module
                    generated = gepa_module.generate(
                        information_context=full_context,
                        # max_length=1024 # Twitter character limit
                    )
                    print(f"[Gen Tweet] Generated tweet: {generated}")
                    # Extract the actual tweet from the prediction
                    prediction_str = str(generated)
                    print(f"[Gen Tweet] Raw prediction: {prediction_str}")
                    
                    # Try to extract tweet from prediction string
                    tweet = None
                    
                    # Try to find the tweet between single quotes
                    import re
                    tweet_matches = re.findall(r"'([^']*)'", prediction_str)
                    if tweet_matches:
                        # Take the first quoted string that's not 'reasoning'
                        for match in tweet_matches:
                            if not match.startswith('reasoning='):
                                tweet = match
                                break
                    
                    # If no quoted tweet found, try other methods
                    if not tweet:
                        if hasattr(generated, 'generated_tweet'):
                            tweet = generated.generated_tweet
                        elif hasattr(generated, 'tweet'):
                            tweet = generated.tweet
                        else:
                            # Try to extract from prediction format
                            if 'generated_tweet=' in prediction_str:
                                parts = prediction_str.split('generated_tweet=')
                                if len(parts) > 1:
                                    tweet_part = parts[1]
                                    if ',' in tweet_part:
                                        tweet = tweet_part.split(',')[0]
                                    else:
                                        tweet = tweet_part
                                    tweet = tweet.strip("' \n)")
                    
                    # If still no tweet, use error message
                    if not tweet:
                        print(f"[Gen Tweet] Warning: Could not extract tweet from prediction for article {index + 1}")
                        tweet = "Error: Could not extract tweet from prediction"
                    
                    # Validate and clean the tweet
                    tweet = tweet.strip()
                    if not tweet:
                        print(f"[Gen Tweet] Warning: Empty tweet generated for article {index + 1}")
                        tweet = "Error: Failed to generate tweet"
                    elif len(tweet) > 280:
                        print(f"[Gen Tweet] Warning: Tweet {index + 1} too long ({len(tweet)} chars), truncating...")
                        tweet = tweet[:277] + "..."
                    
                    print(f"[Gen Tweet] Generated tweet {index + 1} using GEPA model ({len(tweet)} chars):")
                    print(f"[Gen Tweet] {tweet}")
                else:
                    # Use template generation with DSPy
                    completion = lm(full_context, max_tokens=100)
                    print(f"[Gen Tweet] Generated tweet: {completion}")
                    tweet = str(completion).strip()
                    if not tweet:
                        print(f"[Gen Tweet] Warning: Empty tweet generated for article {index + 1}")
                        tweet = "Error: Failed to generate tweet"
                    elif len(tweet) > 280:
                        print(f"[Gen Tweet] Warning: Tweet {index + 1} too long ({len(tweet)} chars), truncating...")
                        tweet = tweet[:277] + "..."
                    
                    print(f"[Gen Tweet] Generated tweet {index + 1} using template ({len(tweet)} chars):")
                    print(f"[Gen Tweet] {tweet}")
                
                return {"index": index, "tweet": tweet}
        except Exception as e:
            print(f"[Gen Tweet] Error generating tweet {index + 1}: {str(e)}")
            return None
    
    # Process articles in parallel
    from concurrent.futures import ThreadPoolExecutor
    
    article_data = [(article, i) for i, article in enumerate(articles)]
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_data = {
            executor.submit(process_article, data): data[1]  # Map future to index
            for data in article_data
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_data):
            result = future.result()
            if result:
                results[result["index"]] = result["tweet"]
    
    # Sort results by original index to preserve order
    generated_tweets = [results[i] for i in sorted(results.keys())]
    print(f"[Gen Tweet] Successfully generated {len(generated_tweets)} tweets in parallel")
    
    # Update database with generated tweets
    if workflow_id:
        # Get existing tweet drafts
        status = get_workflow_status(workflow_id)
        tweet_drafts = status.get("tweet_drafts", []) if status else []
        
        # Add new tweets with metadata
        for i, (tweet, article) in enumerate(zip(generated_tweets, articles)):
            tweet_drafts.append({
                "content": tweet,
                "is_gepa": bool(gepa_module),  # More accurate than checking prompt
                "article_length": len(article.get('content', '')),
                "article_index": i,
                "article_type": article.get('type', 'unknown'),
                "generated_at": datetime.now().isoformat()
            })
        
        # Update database
        update_tweet_drafts(workflow_id, tweet_drafts)
    
    return generated_tweets

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
from gepa_official_optimizer import GEPATweetOptimizer, TweetGeneratorModule
from openrouter_config import setup_openrouter_model


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
        
        # Aggregate insights from real tweets
        all_keywords = set()
        all_insights = []
        total_engagement = 0
        sentiments = []
        information_sources = []
        
        for tweet_data in dataset:
            # Extract keywords from tweet text
            words = tweet_data['tweet'].lower().split()
            keywords = [w.strip('#.,!?') for w in words if len(w) > 4 and not w.startswith('http')]
            all_keywords.update(keywords[:5])
            
            # Collect insights from information sources
            for info in tweet_data.get('information_detailed', []):
                if info['type'] not in ['parent_tweet', 'search_query']:
                    all_insights.append(f"{info['type']}: {info['content'][:100]}")
                    information_sources.append(info)
            
            # Calculate engagement
            likes = tweet_data.get('likes', 0)
            retweets = tweet_data.get('retweets', 0)
            total_engagement += (likes + retweets * 2)  # Weight retweets more
            
            
        
        # Calculate engagement score with better normalization
        avg_engagement = total_engagement / max(1, len(dataset))
        # Normalize by a soft cap so a single viral tweet doesn't pin to 1.0
        engagement_score = min(1.0, avg_engagement / 500.0)  # 500 is a reasonable baseline for good engagement
        
        # Log final results
        result = {
            "topic": topic,
            "insights": all_insights[:5] if all_insights else ["Analysis of recent Twitter discussions"],
            "keywords": list(all_keywords)[:10],
            "engagement_score": round(engagement_score, 2),
            "tweets_analyzed": len(dataset),
            "information_sources": information_sources[:10],  # Include top information sources
            "tweets": dataset  # Include the actual tweets
        }
        
        print(f"[Explore Activity] Exploration complete: {len(result['insights'])} insights, {len(result['keywords'])} keywords")
        print(f"[Explore Activity] Engagement score: {result['engagement_score']}")
        tweets = result.get("tweets", [])
        # Extract information with consistent field names
        flat_infos = [
            item for t in tweets 
            for item in t.get("information_detailed", [])
            if item.get("type") not in ["parent_tweet", "search_query"]  # Filter out non-content sources
        ]
        # Update database with exploration results
        update_exploration_results(
            workflow_id=workflow_id,
            topic=topic,
            articles=flat_infos,
            exp_result=result
        )
        
        # Add workflow_id to result for downstream activities
        result["workflow_id"] = workflow_id
        return result
        
    except Exception as e:
        print(f"[Explore Activity] Error during exploration: {str(e)}")
        error_msg = str(e)[:50]  # Capture error message within exception scope
        # Fallback to mock data on error
        return {
            "topic": topic,
            "insights": [f"Error during exploration: {error_msg}"],
            "keywords": [topic],
            "engagement_score": 0.5,
            "tweets_analyzed": 0,
            "information_sources": [],
            "tweets": [],
            "workflow_id": workflow_id
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

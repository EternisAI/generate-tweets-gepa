import tweepy
import json
import requests
import base64
from typing import Optional, Dict, List, Tuple
from math import sqrt
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tweepy.errors import TooManyRequests
import random

def process_single_media(media_info: Tuple[str, str, str, Optional[str]]) -> Optional[Dict]:
    """
    Process a single media item
    Args:
        media_info: Tuple of (media_url, media_type, media_key, alt_text)
    Returns:
        Dict with media information and analysis
    """
    media_url, media_type, media_key, alt_text = media_info
    try:
        analysis = analyze_media_content(media_url)
        if analysis:
            return {
                'type': media_type,
                'url': media_url,
                'media_key': media_key,
                'alt_text': alt_text,
                'analysis': analysis
            }
    except Exception as e:
        print(f"Error processing media {media_key}: {str(e)}")
    return None

def analyze_media_batch(media_urls: List[str], batch_size: int = 5) -> List[Optional[str]]:
    """
    Analyze a batch of media URLs in parallel using OpenRouter's GPT-5
    """
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return [None] * len(media_urls)

    headers = {
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'https://github.com/OpenRouterTeam/openrouter-python',
        'X-Title': 'Viral Tweet Analysis',
    }

    def download_and_encode_image(url: str) -> Optional[str]:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            print(f"Error downloading image {url}: {str(e)}")
        return None

    def analyze_single_image(image_base64: str) -> Optional[str]:
        if not image_base64:
            return None
            
        payload = {
            'model': 'openai/gpt-5',
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Analyze this image and describe: 1) What is shown? 2) Any text visible? 3) What makes it engaging/viral? Be concise.'
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{image_base64}'
                            }
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
            
            print(f"Error from OpenRouter: {response.text}")
        except Exception as e:
            print(f"Error making OpenRouter API call: {str(e)}")
        return None

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # First, download and encode all images in parallel
        print(f"\nDownloading and encoding {len(media_urls)} images...")
        encoded_images = list(executor.map(download_and_encode_image, media_urls))
        
        # Then, make API calls in parallel batches
        print(f"Making API calls in batches of {batch_size}...")
        for i in range(0, len(encoded_images), batch_size):
            batch = encoded_images[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(encoded_images) + batch_size - 1)//batch_size}")
            
            futures = [executor.submit(analyze_single_image, img) for img in batch if img]
            batch_results = []
            for future in as_completed(futures):
                result = future.result()
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Add a small delay between batches to avoid rate limits
            if i + batch_size < len(encoded_images):
                time.sleep(1)
    
    return results

def analyze_media_content(media_url: str) -> Optional[str]:
    """
    Analyze a single media URL using the batch processor
    """
    results = analyze_media_batch([media_url])
    return results[0] if results else None

def fetch_with_retry(func, max_retries=5, initial_wait=5):
    """Helper function to handle rate limits with exponential backoff"""
    wait_time = initial_wait
    for attempt in range(max_retries):
        try:
            return func()
        except TooManyRequests as e:
            if attempt == max_retries - 1:
                raise e
            
            # Add some randomness to prevent all retries happening at exactly the same time
            jitter = random.uniform(0.1, 1.0)
            sleep_time = wait_time + jitter
            
            print(f"\nRate limit hit. Waiting {sleep_time:.1f} seconds before retry {attempt + 1}/{max_retries}...")
            time.sleep(sleep_time)
            
            # Exponential backoff
            wait_time *= 2
            
def get_viral_tweets(username: str, min_likes: int = 1000, language: Optional[str] = "en", output_file: Optional[str] = None, max_tweets: Optional[int] = None, include_media_analysis: bool = True) -> List[Dict]:
    # Get bearer token from environment variable (more secure)
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        raise ValueError("Please set TWITTER_BEARER_TOKEN environment variable")

    client = tweepy.Client(bearer_token=bearer_token)

    # 1. Get user ID with retry
    user = fetch_with_retry(lambda: client.get_user(username=username))
    user_id = user.data.id

    # 2. Fetch tweets with pagination (4 pages)
    all_tweets_data = []
    all_includes = {'tweets': [], 'users': [], 'media': []}
    pagination_token = None
    pages_fetched = 0
    max_pages = 1

    while pages_fetched < max_pages:
        print(f"\nFetching page {pages_fetched + 1}/{max_pages}...")
        # Use retry mechanism for tweet fetching
        tweets = fetch_with_retry(
            lambda: client.get_users_tweets(
    id=user_id,
                tweet_fields=["referenced_tweets", "created_at", "public_metrics", "lang", "author_id", "attachments", "entities"],
                expansions=[
                    "referenced_tweets.id",
                    "referenced_tweets.id.author_id",
                    "author_id",
                    "attachments.media_keys",
                    "referenced_tweets.id.attachments.media_keys"
                ],
                user_fields=["public_metrics"],
                media_fields=["url", "preview_image_url", "type", "alt_text", "variants"],
                max_results=100,
                pagination_token=pagination_token
            )
        )

        if not tweets.data:
            break

        all_tweets_data.extend(tweets.data)
        
        # Merge includes
        if tweets.includes:
            if 'tweets' in tweets.includes:
                all_includes['tweets'].extend(tweets.includes['tweets'])
            if 'users' in tweets.includes:
                all_includes['users'].extend(tweets.includes['users'])
            if 'media' in tweets.includes:
                all_includes['media'].extend(tweets.includes['media'])

        pages_fetched += 1
        
        # Get next pagination token
        if 'next_token' not in tweets.meta or pages_fetched >= max_pages:
            break
        pagination_token = tweets.meta['next_token']
        
        print(f"Found {len(tweets.data)} tweets on this page")
        print(f"Total tweets so far: {len(all_tweets_data)}")

    print(f"\nFetched {pages_fetched} pages, total {len(all_tweets_data)} tweets")
    
    if not all_tweets_data:
        return []
        
    # Create a new response object with all the data
    class CombinedResponse:
        def __init__(self, data, includes):
            self.data = data
            self.includes = includes
    
    tweets = CombinedResponse(all_tweets_data, all_includes)

    # Debug: Print tweet data
    print("\nDebug: Tweet Data Structure")
    print("Includes:", tweets.includes.keys() if tweets.includes else "No includes")
    if tweets.includes and "media" in tweets.includes:
        print("Media count:", len(tweets.includes["media"]))
    
    # Build lookup for quoted tweets, users, and media
    quoted_lookup = {}
    user_lookup = {}
    media_lookup = {}
    
    if tweets.includes:
        if "tweets" in tweets.includes:
            for qt in tweets.includes["tweets"]:
                quoted_lookup[qt.id] = qt
                # Debug: Print quoted tweet data
                print(f"\nQuoted tweet {qt.id}:")
                print("Has attachments:", hasattr(qt, 'attachments'))
                print("Has entities:", hasattr(qt, 'entities'))
                if hasattr(qt, 'entities'):
                    print("Entities keys:", qt.entities.keys() if qt.entities else "No entities")
                    if qt.entities and 'urls' in qt.entities:
                        print("URLs:")
                        for url in qt.entities['urls']:
                            print(f"  - {url.get('expanded_url', '')}")

        if "users" in tweets.includes:
            for user in tweets.includes["users"]:
                user_lookup[user.id] = user

        if "media" in tweets.includes:
            for media in tweets.includes["media"]:
                media_lookup[media.media_key] = media
                # Debug: Print media data
                print(f"\nMedia {media.media_key}:")
                print("Type:", getattr(media, 'type', None))
                print("URL:", getattr(media, 'url', None))
                print("Preview URL:", getattr(media, 'preview_image_url', None))

    # 3. Filter tweets based on criteria and process metrics
    processed_tweets = []
    for t in tweets.data:
        # Skip if doesn't meet language criteria
        if language and getattr(t, 'lang', None) != language:
            continue

        # Skip if doesn't meet likes criteria
        if t.public_metrics['like_count'] < min_likes:
            continue

        if t.referenced_tweets:
            for ref in t.referenced_tweets:
                if ref.type == "quoted":
                    quoted = quoted_lookup.get(ref.id)
                    if quoted:
                        # Get author info
                        author = user_lookup.get(t.author_id)
                        follower_count = author.public_metrics['followers_count'] if author else 0
                        
                        # Calculate normalized likes
                        likes = t.public_metrics['like_count']
                        normalized_likes = likes / sqrt(follower_count) if follower_count > 0 else 0
                        
                        # Process media for both tweets (only if requested)
                        quote_media = []
                        quoted_media = []

                        def process_tweet_media(tweet):
                            """Helper function to process media from a tweet"""
                            # First, collect all media to process
                            media_to_process = []
                            
                            if hasattr(tweet, 'attachments') and tweet.attachments is not None:
                                media_keys = tweet.attachments.get('media_keys', [])
                                for media_key in media_keys:
                                    media = media_lookup.get(media_key)
                                    if media:
                                        # Get the direct media URL from the media object
                                        media_url = getattr(media, 'url', None)
                                        if not media_url and media.type == 'video':
                                            media_url = getattr(media, 'preview_image_url', None)
                                        
                                        if media_url:
                                            media_to_process.append((
                                                media_url,
                                                media.type,
                                                media_key,
                                                getattr(media, 'alt_text', None)
                                            ))
                            
                            if not media_to_process:
                                return None
                                
                            # Process media in parallel
                            media_items = []
                            total_media = len(media_to_process)
                            processed = 0
                            start_time = time.time()
                            
                            print(f"\nProcessing {total_media} media items...")
                            with ThreadPoolExecutor(max_workers=min(10, total_media)) as executor:
                                future_to_media = {
                                    executor.submit(process_single_media, media_info): media_info 
                                    for media_info in media_to_process
                                }
                                
                                for future in as_completed(future_to_media):
                                    media_info = future_to_media[future]
                                    processed += 1
                                    current_time = time.time()
                                    elapsed = current_time - start_time
                                    avg_time_per_item = elapsed / processed
                                    remaining_items = total_media - processed
                                    estimated_remaining = avg_time_per_item * remaining_items
                                    
                                    print(f"Progress: {processed}/{total_media} ({processed/total_media*100:.1f}%) - "
                                          f"Elapsed: {elapsed:.1f}s, "
                                          f"Avg per item: {avg_time_per_item:.1f}s, "
                                          f"Est. remaining: {estimated_remaining:.1f}s")
                                    
                                    result = future.result()
                                    if result:
                                        media_items.append(result)
                            
                            total_time = time.time() - start_time
                            print(f"Media processing completed in {total_time:.1f} seconds "
                                  f"({total_time/total_media:.1f}s per item average)")
                            
                            return media_items if media_items else None
                        
                        # Process media for both tweets (only if requested)
                        if include_media_analysis:
                            quote_media = process_tweet_media(t)
                            quoted_media = process_tweet_media(quoted)
                        else:
                            quote_media = []
                            quoted_media = []
                        
                        tweet_data = {
                            'quote_tweet': {
                                'id': str(t.id) if hasattr(t, 'id') else 'unknown',
                                'text': t.text if hasattr(t, 'text') else '',
                                'created_at': t.created_at.isoformat() if hasattr(t, 'created_at') else None,
                                'metrics': {
                                    'likes': t.public_metrics.get('like_count', 0) if t.public_metrics else 0,
                                    'retweets': t.public_metrics.get('retweet_count', 0) if t.public_metrics else 0,
                                    'replies': t.public_metrics.get('reply_count', 0) if t.public_metrics else 0,
                                    'normalized_likes': round(normalized_likes, 3)
                                },
                                'author': {
                                    'id': str(t.author_id),
                                    'follower_count': follower_count
                                },
                                'language': getattr(t, 'lang', None),
                                'media': quote_media if quote_media else None
                            },
                            'quoted_tweet': {
                                'id': str(quoted.id) if hasattr(quoted, 'id') else 'unknown',
                                'text': quoted.text if hasattr(quoted, 'text') else '',
                                'created_at': quoted.created_at.isoformat() if hasattr(quoted, 'created_at') else None,
                                'media': quoted_media if quoted_media else None
                            }
                        }

                        # Only add tweet if it has valid data
                        if tweet_data['quote_tweet']['text'] and tweet_data['quoted_tweet']['text']:
                            processed_tweets.append(tweet_data)
    
    # Sort by normalized_likes in descending order
    processed_tweets.sort(key=lambda x: x['quote_tweet']['metrics']['normalized_likes'], reverse=True)
    
    # Apply max_tweets limit if specified
    if max_tweets is not None and len(processed_tweets) > max_tweets:
        print(f"Limiting results to {max_tweets} tweets (found {len(processed_tweets)})")
        processed_tweets = processed_tweets[:max_tweets]

    # Save to JSON file if output_file is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'tweets': processed_tweets,
                'metadata': {
                    'username': username,
                    'min_likes': min_likes,
                    'language': language,
                    'max_tweets': max_tweets,
                    'total_tweets': len(processed_tweets),
                    'generated_at': datetime.now().isoformat()
                }
            }, f, indent=2, ensure_ascii=False)

    return processed_tweets

if __name__ == "__main__":
    # Example usage
    output_file = f"viral_tweets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    tweets = get_viral_tweets(
        username="shauseth",
        min_likes=100,  # Only tweets with at least 100 likes
        language="en",  # Only English tweets
        output_file=output_file,
        max_tweets=2   # Limit to 2 tweets for testing
    )
    
    print(f"Found {len(tweets)} viral quote tweets")
    print(f"Results saved to {output_file}")
    
    # Print top 5 tweets by normalized likes
    for i, tweet in enumerate(tweets[:5], 1):
        qt = tweet['quote_tweet']
        quoted = tweet['quoted_tweet']
        print(f"\n=== #{i} Quote Tweet (Normalized Likes: {qt['metrics']['normalized_likes']}) ===")
        print(f"Likes: {qt['metrics']['likes']} | Retweets: {qt['metrics']['retweets']} | Replies: {qt['metrics']['replies']}")
        print(f"Author Followers: {qt['author']['follower_count']}")
        print(f"Created: {qt['created_at']}")
        print(f"Text: {qt['text']}")
        print("\n--- Quoted Tweet ---")
        print(f"Created: {quoted['created_at']}")
        print(f"Text: {quoted['text']}")
        print("=" * 80)
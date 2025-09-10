"""
Twitter API client using RapidAPI's Twitter API endpoint.
This module provides functionality to search and extract tweets using the RapidAPI Twitter endpoint.
"""

import os
import json
import time
import html
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote
import requests
from http import HTTPStatus

@dataclass
class MediaItem:
    """Represents a media item in a tweet."""
    type: str
    url: str
    preview_url: Optional[str] = None
    display_url: Optional[str] = None
    expanded_url: Optional[str] = None
    aspect_ratio: Optional[List[int]] = None
    duration_millis: Optional[int] = None
    video_variants: Optional[List[Dict]] = None

    def to_dict(self) -> Dict:
        """Convert MediaItem to dictionary for JSON serialization."""
        data = {
            'type': self.type,
            'url': self.url,
            'preview_url': self.preview_url,
            'display_url': self.display_url,
            'expanded_url': self.expanded_url
        }
        if self.aspect_ratio:
            data['aspect_ratio'] = self.aspect_ratio
        if self.duration_millis:
            data['duration_millis'] = self.duration_millis
        if self.video_variants:
            data['video_variants'] = self.video_variants
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'MediaItem':
        """Create a MediaItem from a dictionary."""
        return cls(
            type=data.get("type", "unknown"),
            url=data.get("media_url_https", ""),
            preview_url=data.get("preview_url"),
            display_url=data.get("display_url"),
            expanded_url=data.get("expanded_url"),
            aspect_ratio=data.get("aspect_ratio"),
            duration_millis=data.get("duration_millis"),
            video_variants=data.get("video_variants")
        )

class TwitterAPIError(Exception):
    """Custom exception for Twitter API errors."""
    pass

class Tweet:
    """Represents a tweet with its metadata."""
    
    def __init__(self, id: str, text: str, author: str, created_at: str,
                 likes: int = 0, retweets: int = 0, media: List[Dict] = None,
                 quoted_tweet_id: Optional[str] = None, quoted_tweet: Optional['Tweet'] = None,
                 client = None):
        self.id = id
        self.text = text if not isinstance(text, str) else html.unescape(text)
        self.author = author
        self.created_at = created_at
        self.likes = likes
        self.retweets = retweets
        self.media = []
        if media:
            for m in media:
                if "media_url_https" in m:
                    self.media.append(MediaItem(
                        type=m.get("type", "unknown"),
                        url=m.get("media_url_https", ""),
                        preview_url=m.get("expanded_url") or m.get("display_url")
                    ))
        self.quoted_tweet_id = quoted_tweet_id
        self.quoted_tweet = quoted_tweet
        self._client = client

    def __str__(self) -> str:
        """Return a string representation of the tweet."""
        output = [
            f"Tweet by {self.author}:",
            f"Text: {self.text}",
            f"Created at: {self.created_at}",
            f"Likes: {self.likes}, Retweets: {self.retweets}"
        ]
        
        if self.media:
            output.append("Media:")
            for m in self.media:
                output.append(f"- {m.type}: {m.url}")
                
        return "\n".join(output)

    def to_dict(self) -> Dict:
        """Convert Tweet to dictionary for JSON serialization."""
        media_list = [m.to_dict() if isinstance(m, MediaItem) else m for m in self.media] if self.media else []
        
        data = {
            'id': self.id,
            'text': self.text,
            'author': self.author,
            'created_at': self.created_at,
            'likes': self.likes,
            'retweets': self.retweets,
            'media': media_list
        }
        
        if self.quoted_tweet_id:
            data['quoted_tweet_id'] = self.quoted_tweet_id
            if self.quoted_tweet:
                data['quoted_tweet'] = self.quoted_tweet.to_dict()
        
        return data

    @classmethod
    def from_api_response(cls, data: Dict, client = None) -> Optional['Tweet']:
        """Create a Tweet object from API response data."""
        try:
            # Extract core tweet data
            core_data = data.get("core", {}).get("user_results", {}).get("result", {})
            if not core_data:
                return None
            
            # Get tweet ID and text
            tweet_id = data.get("rest_id")
            tweet_text = data.get("legacy", {}).get("full_text", "")
            
            # Get author info
            author = core_data.get("legacy", {}).get("screen_name", "unknown")
            
            # Get engagement metrics
            legacy_data = data.get("legacy", {})
            likes = legacy_data.get("favorite_count", 0)
            retweets = legacy_data.get("retweet_count", 0)
            created_at = legacy_data.get("created_at", "")
            
            # Extract media
            media = []
            for m in legacy_data.get("extended_entities", {}).get("media", []) or []:
                if not m.get("media_url_https"):
                    continue
                    
                item = {
                    "type": m.get("type", "unknown"),
                    "media_url_https": m.get("media_url_https"),
                    "expanded_url": m.get("expanded_url"),
                    "display_url": m.get("display_url")
                }
                
                # Add video info if present
                vinfo = m.get("video_info", {})
                if vinfo:
                    item["aspect_ratio"] = vinfo.get("aspect_ratio")
                    item["duration_millis"] = vinfo.get("duration_millis")
                    item["video_variants"] = [
                        {k: v.get(k) for k in ("bitrate", "content_type", "url") if k in v}
                        for v in vinfo.get("variants", [])
                    ]
                media.append(item)
            
            # Extract quoted tweet information
            quoted_tweet_id = None
            quoted_tweet = None
            
            # Check for quoted tweet ID
            if "quoted_status_id_str" in legacy_data:
                quoted_tweet_id = legacy_data["quoted_status_id_str"]
                # print(f"Quoted tweet ID: {quoted_tweet_id}")
                # Try to get quoted tweet data from response first
                quoted_status = data.get("quoted_status_result", {}).get("result")
                if quoted_status:
                    quoted_tweet = cls.from_quoted_api_response(quoted_status, client)
                
                # If no quoted tweet data in response and we have a client, fetch it
                elif client:
                    # time.sleep(1)  # Add delay to avoid rate limits
                    quoted_tweet = client.get_tweet_by_id(quoted_tweet_id)
                    # print(f"Quoted tweet: {quoted_tweet}")
            return cls(
                id=tweet_id,
                text=tweet_text,
                author=author,
                created_at=created_at,
                likes=likes,
                retweets=retweets,
                media=media,
                quoted_tweet_id=quoted_tweet_id,
                quoted_tweet=quoted_tweet,
                client=client
            )
            
        except Exception:
            return None
            
    @classmethod
    def from_quoted_api_response(cls, data: Dict, client = None) -> Optional['Tweet']:
        """Create a Tweet object from quoted tweet API response data."""
        try:
            tr = (data.get("data") or data).get("tweetResult", {}).get("result", {})
            if not tr:
                return None

            legacy = tr.get("legacy", {})
            if not legacy:
                return None

            # Get author info from either legacy or core data
            author_legacy = (tr.get("core", {})
                          .get("user_results", {})
                          .get("result", {})
                          .get("legacy", {}))
            author_core = (tr.get("core", {})
                          .get("user_results", {})
                          .get("result", {})
                          .get("core", {}))
            author_sn = author_legacy.get("screen_name") or author_core.get("screen_name")
            if not author_sn:
                return None

            # Extract media items
            media_out = []
            for m in legacy.get("extended_entities", {}).get("media", []) or []:
                if not m.get("media_url_https"):
                    continue
                    
                item = {
                    "type": m.get("type", "unknown"),
                    "media_url_https": m.get("media_url_https"),
                    "expanded_url": m.get("expanded_url"),
                    "display_url": m.get("display_url")
                }
                media_out.append(item)

            # Check for quoted tweet ID
            quoted_tweet_id = legacy.get("quoted_status_id_str")
            quoted_tweet = None
            
            # If we have a quoted tweet ID and client, try to fetch it
            if quoted_tweet_id and client:
                try:
                    # time.sleep(1)  # Add delay to avoid rate limits
                    quoted_tweet = client.get_tweet_by_id(quoted_tweet_id)
                except Exception:
                    pass  # Ignore errors fetching quoted tweet
            
            return cls(
                id=tr.get("rest_id"),
                text=legacy.get("full_text"),
                author=author_sn,
                created_at=legacy.get("created_at"),
                likes=legacy.get("favorite_count"),
                retweets=legacy.get("retweet_count"),
                media=media_out,
                quoted_tweet_id=quoted_tweet_id,
                quoted_tweet=quoted_tweet,
                client=client
            )
            
        except Exception as e:
            print(f"Error parsing quoted tweet: {str(e)}")
            return None
            
        

class TwitterRapidAPI:
    """Client for interacting with Twitter API through RapidAPI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Twitter API client."""
        self.api_key = api_key or os.getenv('RAPIDAPI_KEY')
        if not self.api_key:
            raise TwitterAPIError("No API key provided. Set RAPIDAPI_KEY environment variable or pass key to constructor.")
        
        self.base_url = "https://twitter283.p.rapidapi.com"
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': "twitter283.p.rapidapi.com"
        }
        self._retries = 3
        self._retry_delay = 1  # seconds

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make an HTTP request with retries."""
        url = f"{self.base_url}{endpoint}"
        retries = 0
        
        while retries <= self._retries:
            try:
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == HTTPStatus.TOO_MANY_REQUESTS and retries < self._retries:
                    retry_after = int(response.headers.get('Retry-After', self._retry_delay))
                    time.sleep(retry_after)
                    retries += 1
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if retries < self._retries:
                    time.sleep(self._retry_delay)
                    retries += 1
                    continue
                raise TwitterAPIError(f"Request failed: {str(e)}")
            
        raise TwitterAPIError("Max retries exceeded")

    def get_tweet_by_id(self, tweet_id: str) -> Optional[Tweet]:
        """Fetch a single tweet by its ID."""
        try:
            data = self._make_request("/TweetArticle", params={"tweet_id": tweet_id})
            
            if "data" in data:
                quoted_tweet = Tweet.from_quoted_api_response(data["data"], self)
                # if quoted_tweet:
                #     print("\nFetched quoted tweet details:")
                #     print(f"Author: {quoted_tweet.author}")
                #     print(f"Text: {quoted_tweet.text}")
                #     print(f"Created at: {quoted_tweet.created_at}")
                #     print(f"Likes: {quoted_tweet.likes}, Retweets: {quoted_tweet.retweets}")
                #     if quoted_tweet.media:
                #         print("\nMedia:")
                #         for media in quoted_tweet.media:
                #             print(f"- {media.type}: {media.url}")
                return quoted_tweet
            return None
            
        except Exception:
            return None

    def search_tweets(self, query: str, target_count: int = 50, safe_search: bool = True) -> List[Tweet]:
        """Search for tweets matching the given query.
        
        Args:
            query: The search query
            target_count: Target number of tweets to fetch
            safe_search: Whether to enable safe search
            
        Returns:
            List of Tweet objects
        """
        try:
            all_tweets = []
            seen = set()  # For deduplication
            batch_size = 20  # RapidAPI /Search usually tops out around 20 per page
            cursor = None
            
            while len(all_tweets) < target_count:
                params = {
                    "q": query,
                    "type": "Latest",  # Use Latest for better coverage
                    "count": batch_size,
                    "safe_search": str(safe_search).lower()
                }
                
                if cursor:
                    params["cursor"] = cursor
                
                print(f"\rFetched {len(all_tweets)} tweets...", end="")
                
                data = self._make_request("/Search", params=params)
                new_tweets = self.extract_tweets(data)
                
                # Deduplicate and add new tweets
                for tweet in new_tweets:
                    if tweet and tweet.id and tweet.id not in seen:
                        seen.add(tweet.id)
                        all_tweets.append(tweet)
                
                if len(all_tweets) >= target_count:
                    break
                
                if not new_tweets:
                    print("\nNo more tweets available")
                    break
                
                # Get cursor for next page (BOTTOM)
                timeline = data.get("data", {}).get("search_by_raw_query", {}).get("search_timeline", {}).get("timeline", {})
                cursor = None
                for instr in timeline.get("instructions", []):
                    for entry in instr.get("entries", []):
                        c = entry.get("content", {})
                        if c.get("__typename") == "TimelineTimelineCursor" and c.get("cursor_type") == "Bottom":
                            cursor = c.get("value")
                            break
                    if cursor:
                        break
                
                if not cursor:
                    print("\nNo more pages available")
                    break
                
                # Add delay between pages
                time.sleep(0.6)  # Reduced delay between pages
            
            print(f"\nTotal tweets fetched: {len(all_tweets)}")
            return all_tweets[:target_count]  # Ensure we don't return more than requested
            
        except Exception as e:
            raise TwitterAPIError(f"Request failed: {str(e)}")
        
    def search_tweets_sharded(self, base_query: str, start: str, end: str, days_per_shard: int = 7, target_count: int = 2000) -> List[Tweet]:
        """Search for tweets using time-based sharding for better coverage.
        
        Args:
            base_query: The base search query without time constraints
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            days_per_shard: Number of days per time shard
            target_count: Target number of tweets to fetch
            
        Returns:
            List of Tweet objects
        """
        from datetime import datetime, timedelta
        def d(s): return datetime.strptime(s, "%Y-%m-%d")
        
        cur, stop = d(start), d(end)
        all_tweets, seen = [], set()
        
        while cur < stop and len(all_tweets) < target_count:
            shard_end = min(cur + timedelta(days=days_per_shard), stop)
            q = f"{base_query} since:{cur.date()} until:{shard_end.date()}"
            page_cursor = None
            pages = 0
            
            while pages < 10 and len(all_tweets) < target_count:
                params = {
                    "q": q,
                    "type": "Latest",
                    "count": 20,
                    "safe_search": "true"
                }
                if page_cursor:
                    params["cursor"] = page_cursor
                    
                data = self._make_request("/Search", params=params)
                new = self.extract_tweets(data)
                got = 0
                
                for t in new:
                    if t and t.id and t.id not in seen:
                        seen.add(t.id)
                        all_tweets.append(t)
                        got += 1
                
                # Get cursor for next page
                timeline = data.get("data", {}).get("search_by_raw_query", {}).get("search_timeline", {}).get("timeline", {})
                page_cursor = None
                for instr in timeline.get("instructions", []):
                    for entry in instr.get("entries", []):
                        c = entry.get("content", {})
                        if c.get("__typename") == "TimelineTimelineCursor" and c.get("cursor_type") == "Bottom":
                            page_cursor = c.get("value")
                            break
                    if page_cursor:
                        break
                
                if not page_cursor or got == 0:
                    break
                    
                pages += 1
                time.sleep(0.6)
                
            cur = shard_end
            
        return all_tweets[:target_count]

    def extract_tweets(self, payload: Dict) -> List[Tweet]:
        """Extract tweet data from API response payload."""
        tweets = []
        try:
            timeline = payload.get("data", {}).get("search_by_raw_query", {}).get("search_timeline", {}).get("timeline", {})
            
            if not timeline:
                return []

            for instruction in timeline.get("instructions", []):
                if instruction.get("__typename") != "TimelineAddEntries":
                    continue

                for entry in instruction.get("entries", []):
                    content = entry.get("content", {})
                    if content.get("__typename") != "TimelineTimelineItem":
                        continue
                        
                    tweet_content = content.get("content", {})
                    if tweet_content.get("__typename") != "TimelineTweet":
                        continue
                        
                    tweet_result = tweet_content.get("tweet_results", {}).get("result")
                    if tweet_result:
                        tweet = Tweet.from_api_response(tweet_result, self)
                        if tweet:
                            tweets.append(tweet)
                    
            return tweets
            
        except Exception:
            return []

def save_tweets_to_json(tweets: List[Tweet], filename: str) -> str:
    """Save all tweets to a JSON file."""
    try:
        all_tweets = []
        for tweet in tweets:
            tweet_data = tweet.to_dict()
            all_tweets.append(tweet_data)
        
        if not all_tweets:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{filename}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_tweets, f, indent=2, ensure_ascii=False)
            
        return output_file
        
    except Exception as e:
        raise TwitterAPIError(f"Failed to save tweets: {str(e)}")

def save_quoted_tweets_to_json(tweets: List[Tweet], filename: str) -> str:
    """Save only quoted tweets to a JSON file."""
    try:
        quoted_tweets = []
        for tweet in tweets:
            if tweet.quoted_tweet:
                quoted_tweets.append({
                    'original_tweet': tweet.to_dict(),
                    'quoted_tweet': tweet.quoted_tweet.to_dict()
                })
        
        if not quoted_tweets:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{filename}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(quoted_tweets, f, indent=2, ensure_ascii=False)
            
        return output_file
        
    except Exception as e:
        raise TwitterAPIError(f"Failed to save quoted tweets: {str(e)}")

def main():
    """Example usage of the TwitterRapidAPI client."""
    try:
        client = TwitterRapidAPI()
        
        # Search for tweets from the last 3 months
        print("Starting tweet search for the last 3 months...")
        # Search for tweets using time sharding for better coverage
        print("Starting tweet search with time sharding...")
        base_query = "from:FT"
        tweets = client.search_tweets_sharded(
            base_query=base_query,
            start="2025-01-10",
            end="2025-09-10",
            days_per_shard=7,
            target_count=2000
        )
        print(f"\nSearch complete. Found {len(tweets)} tweets.")
        
        # Save all tweets
        print("\nSaving all tweets...")
        output_file = save_tweets_to_json(tweets, "all_tweets")
        if output_file:
            print(f"Saved all tweets to: {output_file}")
        else:
            print("No tweets found")
            
        # Save quoted tweets separately
        print("\nSaving quoted tweets...")
        quoted_output_file = save_quoted_tweets_to_json(tweets, "quoted_tweets")
        if quoted_output_file:
            print(f"Saved quoted tweets to: {quoted_output_file}")
        else:
            print("No quoted tweets found")
            
        print("\nDone!")
            
    except TwitterAPIError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
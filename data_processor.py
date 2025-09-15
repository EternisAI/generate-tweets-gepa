import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

@dataclass
class TweetData:
    """Dataclass to represent a tweet with its information context and media analysis"""
    tweet: str
    username: str
    created_at: str
    retweets: int
    replies: int
    likes: int
    quotes: int
    information: List[Dict]
    media_analysis: Optional[str] = None  # Media analysis from GPT-5
    engagement_score: float = field(init=False)
    parsed_date: Optional[datetime] = field(init=False)
    
    def __post_init__(self):
        # Calculate engagement score as a weighted sum
        self.engagement_score = (
            self.likes * 1.0 + 
            self.retweets * 2.0 + 
            self.quotes * 3.0 + 
            self.replies * 0.5
        )
        
        # Parse and fix date
        self.parsed_date = self._parse_and_fix_date(self.created_at)
    
    def _parse_and_fix_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string and fix future dates"""
        if not date_str:
            return None
        
        try:
            # Parse the date
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

            
            return dt
        except Exception as e:
            print(f"[DataProcessor] Could not parse date '{date_str}': {e}")
            return None

class DataProcessor:
    """Process and manage tweet dataset for training"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tweets = []
        self.load_data()
        print(f"[DataProcessor] Loaded {len(self.tweets)} tweets from dataset")
    
    def load_data(self):
        """Load and parse the JSON dataset with robust field handling"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        skipped_count = 0
        for i, item in enumerate(raw_data):
            try:
                # Handle converted dataset format with <think> tags
                if 'tweet' in item and '<think>' in item.get('tweet', ''):
                    # This is the converted format - extract the generated tweet
                    tweet_text, information_context = self._extract_from_think_format(item)
                    if not tweet_text:
                        skipped_count += 1
                        continue
                    
                    # Create tweet with extracted data
                    tweet = TweetData(
                        tweet=tweet_text,
                        username=item.get('username', 'unknown'),
                        created_at=item.get('created_at', ''),
                        retweets=item.get('retweets', 0),
                        replies=0,  # Not available in converted format
                        likes=item.get('likes', 0),
                        quotes=0,   # Not available in converted format
                        information=[information_context] if information_context else [],
                        media_analysis=None
                    )
                    self.tweets.append(tweet)
                    
                else:
                    # Original format
                    if 'tweet' not in item:
                        print(f"[DataProcessor] Skipping row {i}: missing 'tweet' field")
                        skipped_count += 1
                        continue
                    
                    # Create tweet with safe field access
                    tweet = TweetData(
                        tweet=item.get('tweet', ''),
                        username=item.get('username', 'unknown'),
                        created_at=item.get('created_at', ''),
                        retweets=item.get('retweets', 0),
                        replies=item.get('replies', 0),
                        likes=item.get('likes', 0),
                        quotes=item.get('quotes', 0),
                        information=item.get('information', []),
                        media_analysis=item.get('media_analysis')  # Include media analysis if present
                    )
                    self.tweets.append(tweet)
                    
            except Exception as e:
                print(f"[DataProcessor] Error loading row {i}: {e}")
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"[DataProcessor] Skipped {skipped_count} rows with missing/invalid data")
    
    def _extract_from_think_format(self, item: dict) -> tuple:
        """Extract generated tweet and information from converted format with <think> tags"""
        import re
        import json
        
        tweet_text = item.get('tweet', '')
        information_context = item.get('information', '')
        
        # Extract the JSON part after </think>
        think_match = re.search(r'<think>.*?</think>', tweet_text, re.DOTALL)
        if think_match:
            remaining_text = tweet_text[think_match.end():].strip()
            
            try:
                # Try to parse the JSON
                tweet_json = json.loads(remaining_text)
                generated_tweet = tweet_json.get('generated_tweet', '')
                return generated_tweet, information_context
            except json.JSONDecodeError:
                # Fallback: look for "generated_tweet" in the text
                tweet_match = re.search(r'"generated_tweet":\s*"([^"]*)"', remaining_text)
                if tweet_match:
                    return tweet_match.group(1), information_context
        
        return "", information_context
    
    def get_high_engagement_tweets(self, top_k: int = None, min_engagement: float = 100):
        """Get tweets with high engagement scores"""
        filtered = [t for t in self.tweets if t.engagement_score >= min_engagement]
        sorted_tweets = sorted(filtered, key=lambda x: x.engagement_score, reverse=True)
        return sorted_tweets[:top_k] if top_k else sorted_tweets
    
    def split_dataset(self, train_ratio: float = 0.8, seed: int = 42):
        """Split dataset into train and test sets"""
        random.seed(seed)
        shuffled = self.tweets.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train_data = shuffled[:split_idx]
        test_data = shuffled[split_idx:]
        
        print(f"[DataProcessor] Train set: {len(train_data)} tweets, Test set: {len(test_data)} tweets")
        return train_data, test_data
    
    def format_information(self, information: List, media_analysis: Optional[str] = None) -> str:
        """Format information sources and media analysis into a single string context
        
        Handles both formats:
        - Simple: List of strings (from GPT-5 enhanced extraction)
        - Complex: List of dicts with tweet/username/url fields
        
        Also includes media analysis if available
        """
        formatted = []
        
        # Add media analysis first if available
        if media_analysis:
            formatted.append("[Media Analysis]")
            formatted.append(media_analysis)
        
        # Add information sources
        if information:
            formatted.append("\n[Information Sources]")
            for i, info in enumerate(information, 1):
                # Handle string format (GPT-5 enhanced datasets)
                if isinstance(info, str):
                    formatted.append(f"[Source {i}]: {info}")
                # Handle dict format (original datasets)
                elif isinstance(info, dict):
                    if 'tweet' in info:
                        username = info.get('username', 'Unknown')
                        tweet_text = info.get('tweet', '')
                        formatted.append(f"[Source {i}] @{username}: {tweet_text}")
                    elif 'content' in info:
                        # For information_detailed format
                        formatted.append(f"[Source {i}]: {info.get('content', '')}")
                    elif 'url' in info:
                        # Could be a tweet URL or article URL
                        formatted.append(f"[Source {i}] URL: {info.get('url', '')}")
                else:
                    # Fallback for any other type
                    formatted.append(f"[Source {i}]: {str(info)}")
        elif not media_analysis:
            return "No additional context available."
        
        return "\n\n".join(formatted)
    
    def get_statistics(self):
        """Get dataset statistics with safe handling"""
        if not self.tweets:
            print("[DataProcessor] No tweets loaded")
            return {}
        
        engagements = [t.engagement_score for t in self.tweets]
        info_counts = [len(t.information) for t in self.tweets]
        
        # Count tweets with missing fields
        missing_stats = {
            'missing_username': sum(1 for t in self.tweets if t.username == 'unknown'),
            'missing_created_at': sum(1 for t in self.tweets if not t.created_at),
            'invalid_dates': sum(1 for t in self.tweets if t.created_at and not t.parsed_date),
            'future_dates_fixed': sum(1 for t in self.tweets if t.parsed_date and '2025' in t.created_at),
            'zero_engagement': sum(1 for t in self.tweets if t.engagement_score == 0)
        }
        
        stats = {
            'total_tweets': len(self.tweets),
            'avg_engagement': np.mean(engagements) if engagements else 0,
            'median_engagement': np.median(engagements) if engagements else 0,
            'max_engagement': max(engagements) if engagements else 0,
            'min_engagement': min(engagements) if engagements else 0,
            'avg_info_sources': np.mean(info_counts) if info_counts else 0,
            'max_info_sources': max(info_counts) if info_counts else 0,
            'tweets_with_info': sum(1 for c in info_counts if c > 0),
            **missing_stats
        }
        
        print("\n=== Dataset Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Warn about data quality issues
        if stats.get('missing_username', 0) > 0:
            print(f"[Warning] {stats['missing_username']} tweets have missing username")
        if stats.get('future_dates_fixed', 0) > 0:
            print(f"[Info] Fixed {stats['future_dates_fixed']} future dates (2025 -> 2024)")
        if stats.get('invalid_dates', 0) > 0:
            print(f"[Warning] {stats['invalid_dates']} tweets have invalid dates")
        if stats.get('zero_engagement', 0) > len(self.tweets) * 0.2:
            print(f"[Warning] {stats['zero_engagement']} tweets have zero engagement")
        
        print("========================\n")
        
        return stats

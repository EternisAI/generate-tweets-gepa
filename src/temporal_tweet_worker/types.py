from dataclasses import dataclass
from typing import List


@dataclass
class TweetGenerationRequest:
    """Request to generate tweets about a topic"""
    topic: str
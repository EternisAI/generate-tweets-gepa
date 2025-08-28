from dataclasses import dataclass
from typing import List


@dataclass
class TweetGenerationRequest:
    topic: str
    articles: List[str]
import json
import random
from typing import Dict, Any

from temporalio import activity


@activity.defn
async def explore(topic: str) -> Dict[str, Any]:
    mock_insights = [
        "trending hashtags analysis",
        "competitor content review",
        "audience sentiment data",
        "engagement patterns",
        "optimal posting times",
    ]

    mock_keywords = [
        "innovation",
        "technology",
        "growth",
        "success",
        "strategy",
        "leadership",
        "digital",
        "transformation",
        "future",
        "trends",
    ]

    return {
        "topic": topic,
        "insights": random.sample(mock_insights, k=random.randint(2, 4)),
        "keywords": random.sample(mock_keywords, k=random.randint(3, 6)),
        "engagement_score": round(random.uniform(0.1, 1.0), 2),
        "sentiment": random.choice(["positive", "neutral", "mixed"]),
        "target_audience": random.choice(
            ["professionals", "entrepreneurs", "tech_enthusiasts", "general"]
        ),
    }


@activity.defn
async def gen_gepa_prompt(exp_result: Dict[str, Any]) -> str:
    topic = exp_result.get("topic", "general")
    keywords = ", ".join(exp_result.get("keywords", []))
    sentiment = exp_result.get("sentiment", "neutral")
    audience = exp_result.get("target_audience", "general")

    prompt = f"""Create an engaging tweet about {topic} that:
- Uses keywords: {keywords}
- Maintains a {sentiment} tone
- Targets {audience} audience
- Includes relevant hashtags
- Stays under 280 characters
- Encourages engagement through questions or calls to action
"""

    return prompt


@activity.defn
async def gen_tweet(prompt: str, article: str) -> str:
    mock_tweet_templates = [
        "🚀 {topic}: {insight} What's your take? #{hashtag}",
        "💡 Just read about {topic} - {insight}. Thoughts? #{hashtag}",
        "🔥 Hot take on {topic}: {insight} Do you agree? #{hashtag}",
        "📈 {topic} update: {insight} Share your experience! #{hashtag}",
        "⚡ Breaking: {topic} insights reveal {insight} What do you think? #{hashtag}",
    ]

    mock_topics = ["AI", "Crypto", "Tech", "Business", "Innovation", "Leadership"]
    mock_insights = [
        "the future is here",
        "game-changing opportunities ahead",
        "disruption is accelerating",
        "new paradigms emerging",
        "transformation in progress",
    ]
    mock_hashtags = [
        "TechTrends",
        "Innovation",
        "Future",
        "Growth",
        "Leadership",
        "Digital",
    ]

    template = random.choice(mock_tweet_templates)
    tweet = template.format(
        topic=random.choice(mock_topics),
        insight=random.choice(mock_insights),
        hashtag=random.choice(mock_hashtags),
    )

    return tweet

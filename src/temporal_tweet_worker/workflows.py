from datetime import timedelta
from typing import List

from temporalio import workflow

from .activities import explore, gen_gepa_prompt, gen_tweet
from .types import TweetGenerationRequest


@workflow.defn
class GenerateTweetWorkflow:
    @workflow.run
    async def run(self, request: TweetGenerationRequest) -> List[str]:
        exp_result = await workflow.execute_activity(
            explore,
            request.topic,
            start_to_close_timeout=timedelta(seconds=30),
        )
        
        prompt = await workflow.execute_activity(
            gen_gepa_prompt,
            exp_result,
            start_to_close_timeout=timedelta(seconds=30),
        )
        
        tweet_drafts = []
        for article in request.articles:
            tweet_draft = await workflow.execute_activity(
                gen_tweet,
                args=[prompt, article],
                start_to_close_timeout=timedelta(seconds=30),
            )
            tweet_drafts.append(tweet_draft)
        
        return tweet_drafts
from datetime import timedelta
from typing import List

from temporalio import workflow
from temporalio.common import RetryPolicy

from .types import TweetGenerationRequest


@workflow.defn
class GenerateTweetWorkflow:
    @workflow.run
    async def run(self, request: TweetGenerationRequest) -> List[str]:
        # Use string-based activity names to avoid importing HTTP libraries in workflow
        # Configure retry policy for API calls
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(minutes=1),
            maximum_attempts=3,
            non_retryable_error_types=["ValueError"]
        )
        
        # Start exploration
        exp_result = await workflow.execute_activity(
            "explore",
            args=[request.topic],
            start_to_close_timeout=timedelta(minutes=5),  # Increased timeout for API calls
            retry_policy=retry_policy
        )
        
        # Get workflow_id for subsequent activities
        workflow_id = exp_result.get("workflow_id")
        
        # Configure retry policy for GEPA training
        gepa_retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=5),
            maximum_interval=timedelta(minutes=5),
            maximum_attempts=5,  # More retries for training
            non_retryable_error_types=["ValueError"]
        )
        
        prompt = await workflow.execute_activity(
            "gen_gepa_prompt",
            args=[exp_result],
            start_to_close_timeout=timedelta(minutes=30),  # Longer timeout for GEPA training
            retry_policy=gepa_retry_policy
        )
        
        # Configure retry policy for tweet generation
        gen_retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(minutes=1),
            maximum_attempts=3,
            non_retryable_error_types=["ValueError"]
        )
        
        # Generate tweets using articles from exploration
        tweet_drafts = await workflow.execute_activity(
            "gen_tweet",
            args=[prompt, workflow_id],
            start_to_close_timeout=timedelta(minutes=5),  # Standard timeout for generation
            retry_policy=gen_retry_policy
        )
        
        return tweet_drafts
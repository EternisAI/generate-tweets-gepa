import asyncio
from datetime import datetime
from typing import List

from temporalio.client import Client

from .types import TweetGenerationRequest
from .workflows import GenerateTweetWorkflow


async def run_client():
    client = await Client.connect("localhost:7233")

    request = TweetGenerationRequest(
        topic="Artificial Intelligence"
    )

    print(f"Starting workflow for topic: {request.topic}")
    print("Processing articles from Twitter exploration...")

    result = await client.execute_workflow(
        GenerateTweetWorkflow.run,
        request,
        id=f"generate-tweet-workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        task_queue="tweet-generation",  # Use dedicated queue
    )

    print("\n🐦 Generated Tweet Drafts:")
    print("=" * 50)
    for i, tweet in enumerate(result, 1):
        print(f"{i}. {tweet}")
    print("=" * 50)


def main():
    asyncio.run(run_client())


if __name__ == "__main__":
    main()

import asyncio
from typing import List

from temporalio.client import Client

from .types import TweetGenerationRequest
from .workflows import GenerateTweetWorkflow


async def run_client():
    client = await Client.connect("localhost:7233")

    request = TweetGenerationRequest(
        topic="Artificial Intelligence",
        articles=[
            "AI breakthrough in natural language processing",
            "Machine learning transforms healthcare diagnostics",
            "Ethical considerations in AI development",
        ]
    )

    print(f"Starting workflow for topic: {request.topic}")
    print(f"Processing {len(request.articles)} articles...")

    result = await client.execute_workflow(
        GenerateTweetWorkflow.run,
        request,
        id="generate-tweet-workflow-001",
        task_queue="default",
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

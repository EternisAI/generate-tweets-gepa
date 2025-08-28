import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from .activities import explore, gen_gepa_prompt, gen_tweet
from .workflows import GenerateTweetWorkflow


async def run_worker():
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="default",
        workflows=[GenerateTweetWorkflow],
        activities=[explore, gen_gepa_prompt, gen_tweet],
    )

    print("Starting worker on task queue: default")
    await worker.run()


def main():
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()

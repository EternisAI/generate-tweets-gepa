import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from .activities import explore, gen_gepa_prompt, gen_tweet
from .workflows import GenerateTweetWorkflow


async def run_worker():
    # Get Temporal settings from environment variables
    temporal_address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    temporal_namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    temporal_api_key = os.getenv("TEMPORAL_CLOUDAPIKEY")

    print(f"Connecting to Temporal server at {temporal_address} (namespace: {temporal_namespace})")
    
    # Configure client connection
    client_options = {
        "namespace": temporal_namespace,
    }
    
    # Add API key if provided
    if temporal_api_key:
        from temporalio.service import TLSConfig
        client_options["tls"] = TLSConfig(
            server_root_ca_cert=None,  # Use system CA certificates
            client_cert=None,
            client_private_key=None
        )
        client_options["api_key"] = temporal_api_key
    
    # Connect to Temporal
    client = await Client.connect(temporal_address, **client_options)

    # Get task queue from environment
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "tweet-generation")
    print(f"Using task queue: {task_queue}")

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[GenerateTweetWorkflow],
        activities=[explore, gen_gepa_prompt, gen_tweet],
    )

    print("Starting worker on task queue: tweet-generation")
    await worker.run()


def main():
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()

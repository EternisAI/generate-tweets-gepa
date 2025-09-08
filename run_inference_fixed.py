#!/usr/bin/env python3
"""
Run parallel inference using the optimized GEPA model to generate tweets
"""

import dspy
import json
import os
import asyncio
import aiohttp
import re
from datetime import datetime
from typing import List, Dict
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from gepa_official_optimizer import TweetGeneratorModule

console = Console()

# Number of concurrent API calls
MAX_CONCURRENT_CALLS = 5

def load_optimized_module(model_dir):
    """Load the optimized module and its configuration"""
    
    # Create module with optimized configuration
    module = TweetGeneratorModule()
    prompt = '''Generate an original, punchy tweet using the provided "Information Sources" and/or "Media Analysis." Mirror the intended comedic pattern and tone, but keep it novel, concrete, and tight.

What you'll receive
- information_context: A structured bundle that may include:
  - [Media Analysis]: What's shown in an image/screenshot, why it's engaging/viral, notable props/era cues, visible UI text.
  - [Information Sources]: Humor analysis scaffolding with fields such as:
    - setup_context: about, referenced_context, quote_interplay
    - humor_elements: primary_mechanisms (e.g., misdirection, irony, register_clash, understatement, exaggeration), why_funny, notable_lines
    - tone_style: register (e.g., serious-on-silly), delivery (e.g., deadpan), voice_notes
    - cultural_meme_alignment: patterns_referenced, why_pattern_helps, relatability_lever, replication_guide (template, slots, dos, donts)
    - funniness_strength: label/score/rationale
    - humor_archetype and/or structural_blueprint
  - [Cultural Context]: domain (tech, finance, lifestyle, viral), locale shibboleths, platform idioms, props, entities.

Output format
- reasoning: 1–3 concise sentences naming the blueprint you used, your twist, and the one concrete detail you integrated.
- generated_tweet: The tweet text only (no explanations). Keep it brief and platform-native.

Core goals
- Match the specified structure and cadence of the source pattern.
- Be novel: do not copy notable_lines or obvious phrasing verbatim—paraphrase, invert, or escalate.
- Add exactly one left-field twist or one hyper-specific detail that sharpens the joke and invites replies.
- Stay deadpan, concise, and confident. No emojis, no hashtags, no moral disclaimers, no stat dumps.'''
    
    # Set the optimized prompt
    module.generate.predict.signature.instructions = prompt
    
    return module

def setup_model():
    """Configure the language model"""
    
    # Import OpenRouter configuration
    from openrouter_config import setup_openrouter_model
    
    # Configure model through OpenRouter
    try:
        lm = setup_openrouter_model("openai/gpt-5")  # Using GPT-5 for best results
        dspy.settings.configure(lm=lm)
    except Exception as e:
        raise ValueError(f"Failed to setup OpenRouter model: {e}")

async def generate_tweet_async(session: aiohttp.ClientSession, 
                             api_key: str, 
                             prompt: str, 
                             information_context: str) -> str:
    """Generate a tweet using OpenRouter API directly"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/OpenRouterAI/openrouter",
        "Content-Type": "application/json",
        "X-Title": "Tweet Generator",  # Identify your application
        "Accept": "application/json"
    }
    
    # Format the prompt to focus on tweet generation - very explicit
    formatted_prompt = f"""{prompt}

CRITICAL: Output ONLY the tweet text. No explanations, no quotes, no prefixes, no thinking. Just the raw tweet that would be posted on Twitter."""

    data = {
        "model": "openai/gpt-5",
        "messages": [
            {"role": "system", "content": formatted_prompt},
            {
                "role": "user",
                "content": (
                    f"Information: {information_context}\n\n"
                    "Generate the tweet. Output ONLY the tweet text, no explanations."
                )
            }
        ],

        # --- generation ---
        "temperature": 1.0,
        "top_p": 1,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,

        # IMPORTANT: give enough tokens for final text; must exceed reasoning budget
        "max_tokens": 1500,

        # Force plain text back on chat/completions
        "response_format": { "type": "text" },

        # Disable tool paths
        "tool_choice": "none",

        # Keep reasoning minimal so budget isn't eaten
        "reasoning": { "effort": "low" }
    }
    
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                console.print(f"[red]API Error ({response.status}): {error_text}[/red]")
                return f"Error: API returned status {response.status}"
            
            result = await response.json()

            tweet = ""

            # Parse Chat Completions API response
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                tweet = message.get("content", "").strip()

            # Fallback: check for direct content field (some APIs return this)
            if not tweet:
                if "content" in result:
                    tweet = result["content"].strip()
                elif "text" in result:
                    tweet = result["text"].strip()

            if not tweet:
                console.print("[red]Error: No tweet content found in response[/red]")
                return "Error: No tweet content found"

            # Clean up the response
            tweet = tweet.replace('"', '').replace('Tweet:', '').strip()
            if len(tweet) > 280:
                tweet = tweet[:277] + "..."

            return tweet
    except Exception as e:
        console.print(f"[red]Error calling API: {str(e)}[/red]")
        return f"Error: {str(e)}"

def load_training_data(file_path):
    """Load the training dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)
        # Convert the list format to the expected format with information_context and original_tweet
        if isinstance(data, list):
            return [
                {
                    'information_context': item.get('information', ''),
                    'original_tweet': item.get('tweet', '')
                }
                for item in data
            ]
        return data.get('training_data', [])

async def process_batch(session: aiohttp.ClientSession,
                      api_key: str,
                      prompt: str,
                      batch: List[Dict],
                      progress,
                      progress_task_id) -> List[Dict]:
    """Process a batch of examples"""
    
    # Process tweet generation
    tasks = []
    for item in batch:
        # Create task for this example
        task = asyncio.create_task(
            generate_tweet_async(session, api_key, prompt, item['information_context'])
        )
        tasks.append((task, item['information_context'], item['original_tweet']))

    # Wait for all tweet generation tasks to complete
    results = []
    for task, information_context, original_tweet in tasks:
        generated_tweet = await task
        result = {
            'information_context': information_context,
            'original_tweet': original_tweet,
            'generated_tweet': generated_tweet
        }
        results.append(result)
        progress.update(progress_task_id, advance=1)

    return results

async def async_main():
    # Setup
    console.print("[bold blue]Setting up model...[/bold blue]")
    setup_model()
    
    # Load optimized module
    console.print("[bold blue]Loading optimized module...[/bold blue]")
    module = load_optimized_module("results_20250906_183215_fresh")
    
    # Get the optimized prompt
    prompt = module.generate.predict.signature.instructions
    
    # Load training data
    console.print("[bold blue]Loading training data...[/bold blue]")
    training_data = load_training_data("topic_extraction.json")
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    # Process examples in parallel batches
    results = []
    
    # Create progress bar
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    )
    
    with progress:
        task = progress.add_task("[cyan]Processing tweets...", total=len(training_data))
        
        # Process in batches
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(training_data), MAX_CONCURRENT_CALLS):
                batch = training_data[i:i + MAX_CONCURRENT_CALLS]
                batch_results = await process_batch(session, api_key, prompt, batch, progress, task)
                results.extend(batch_results)
                
                # Small delay between batches to avoid rate limits
                await asyncio.sleep(1.0)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generated_tweets_{timestamp}.json"
    
    console.print(f"[bold blue]Saving results to {output_file}...[/bold blue]")
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'metadata': {
                'model_dir': "results_20250906_183215_fresh",
                'timestamp': timestamp,
                'total_examples': len(results),
                'batch_size': MAX_CONCURRENT_CALLS
            }
        }, f, indent=2)
    
    console.print(f"[bold green]Done! Processed {len(results)} tweets[/bold green]")
    console.print(f"Results saved to: {output_file}")

def main():
    """Entry point that runs the async main function"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()

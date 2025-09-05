import json
import os
import requests
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def analyze_tweet_humor(quote_tweet: str, original_tweet: str) -> str:
    """Analyze the humor in a quote tweet pair using GPT-5"""

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    headers = {
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'https://github.com/OpenRouterTeam/openrouter-python',
        'X-Title': 'Tweet Humor Analysis',
    }

    prompt = f"""Analyze the humor in this quote tweet and its original tweet. Follow this structured analysis:

Quote Tweet: "{quote_tweet}"
Original Tweet: "{original_tweet}"

Answer these questions in detail:

1. SETUP & EXPECTATION
   • What is the original quote or context?
   • What expectation does it create in the reader?

2. THE TWIST
   • Where does the reply diverge from the expected response?
   • Is the twist exaggeration, understatement, absurdity, misdirection, or irony?

3. TONE/STYLE
   • Is the humor delivered in a serious register applied to a silly situation?
   • Or in a silly register applied to a serious situation?
   • Deadpan, sarcastic, hyperbolic, etc.?

4. CULTURAL/MEME ALIGNMENT
   • Does it riff on a familiar internet pattern?
   • How does recognizing that structure make it funnier?

5. THE INCONGRUITY
   • Why is it funny that this serious-sounding phrasing applies to a trivial/absurd situation?
   • State the mismatch clearly: "The joke works because it treats [trivial event] as though it were [serious phenomenon]."

6. FUNNINESS STRENGTH
   • Dry wit (amusing but subtle)?
   • Punchy dunk (likely to go viral)?
   • Niche joke (funny only to insiders)?

Provide a comprehensive analysis that explains why this specific quote tweet is funny."""

    payload = {
        'model': 'openai/gpt-5',
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }

    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return "Analysis failed due to API error"

    except Exception as e:
        print(f"Error analyzing humor: {str(e)}")
        return f"Analysis failed: {str(e)}"

def process_single_humor_analysis(task: dict) -> dict:
    """Process a single humor analysis task"""
    quote_tweet = task['quote_tweet']
    original_tweet = task['original_tweet']
    info_context = task['info_context']
    example = task['example']

    # Analyze humor
    humor_analysis = analyze_tweet_humor(quote_tweet, original_tweet)

    # Add humor analysis to the information context
    try:
        context_dict = json.loads(info_context)
        context_dict['humor_analysis'] = humor_analysis
        enhanced_info_context = json.dumps(context_dict, indent=2)
    except:
        # If info_context isn't JSON, create new structure
        context_dict = {
            'text': info_context,
            'humor_analysis': humor_analysis
        }
        enhanced_info_context = json.dumps(context_dict, indent=2)

    # Create enhanced example
    enhanced_example = example.copy()
    enhanced_example['information_context'] = enhanced_info_context

    return enhanced_example

def add_humor_to_training_data(input_file: str, output_file: str, batch_size: int = 5):
    """Add humor analysis to GEPA training data with parallel processing"""

    print("Loading training data...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    training_data = data['training_data']
    total_examples = len(training_data)

    print(f"Processing {total_examples} examples with batch size {batch_size}...")

    enhanced_data = []

    # Process in batches to avoid overwhelming the API
    for batch_start in range(0, total_examples, batch_size):
        batch_end = min(batch_start + batch_size, total_examples)
        batch = training_data[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_examples + batch_size - 1)//batch_size}")
        print(f"Examples {batch_start + 1}-{batch_end}")

        # Prepare batch data for parallel processing
        batch_tasks = []
        for i, example in enumerate(batch):
            quote_tweet = example['original_tweet']
            info_context = example['information_context']

            # Extract the original tweet from the information context
            try:
                context_data = json.loads(info_context)
                original_tweet = context_data.get('text', 'No original tweet found')
            except:
                original_tweet = info_context  # Fallback

            batch_tasks.append({
                'index': batch_start + i,
                'quote_tweet': quote_tweet,
                'original_tweet': original_tweet,
                'info_context': info_context,
                'example': example
            })

        # Process batch in parallel
        print(f"Making {len(batch_tasks)} parallel API calls...")

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(batch_size, len(batch_tasks))) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_humor_analysis, task): task
                for task in batch_tasks
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1

                try:
                    result = future.result()
                    enhanced_data.append(result)
                    print(f"  ✓ Completed analysis for example {task['index'] + 1} ({completed}/{len(batch_tasks)})")
                except Exception as e:
                    print(f"  ✗ Failed analysis for example {task['index'] + 1}: {str(e)}")
                    # Add original example without humor analysis
                    enhanced_data.append(task['example'])

        # Brief pause between batches to be respectful to the API
        if batch_end < total_examples:
            print("Taking a brief pause between batches...")
            time.sleep(3)

    # Keep original order for consistency

    # Save enhanced data
    enhanced_json = {
        'training_data': enhanced_data,
        'metadata': {
            'source_file': input_file,
            'examples': len(enhanced_data),
            'generated_at': datetime.now().isoformat(),
            'enhancement': f'Added parallel humor analysis using GPT-5 (batch_size={batch_size})'
        }
    }

    print(f"\nSaving enhanced training data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(enhanced_json, f, indent=2)

    print(f"✅ Successfully added humor analysis to {len(enhanced_data)} training examples")
    print(f"📁 Output saved to: {output_file}")
    print(f"⚡ Processed in parallel batches of {batch_size} for maximum speed")

if __name__ == "__main__":
    import sys

    # Default values
    input_file = "gepa_training_data_20250903_175705.json"
    batch_size = 5

    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            print("Error: batch_size must be an integer")
            sys.exit(1)

    output_file = f"enhanced_humor_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print("🚀 Starting Parallel Humor Analysis")
    print(f"📁 Input: {input_file}")
    print(f"📤 Output: {output_file}")
    print(f"⚡ Batch size: {batch_size}")
    print("=" * 50)

    add_humor_to_training_data(input_file, output_file, batch_size)

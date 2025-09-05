#!/usr/bin/env python3
"""
Self-Evolving Tweet Prompt Optimizer using DSPy GEPA

This system learns to generate viral tweets by:
1. Analyzing information sources (tweets, articles)
2. Generating tweet candidates
3. Evaluating them with an LLM judge
4. Evolving prompts using GEPA (Gradient-Enriched Prompt Adaptation)
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import warnings
warnings.filterwarnings('ignore')

from training_pipeline import TrainingPipeline
from data_processor import DataProcessor
from gepa_official_optimizer import TweetGeneratorModule, GEPATweetOptimizer
import dspy

console = Console()

def load_config(config_path: str = None) -> Dict:
    """Load configuration from file or use defaults"""
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        console.print(f"[green]Loaded configuration from {config_path}[/green]")
        
        # Always create a new timestamped output directory to avoid reusing old logs
        if 'output_dir' in config:
            base_dir = config['output_dir']
            # Append timestamp to make it unique
            config['output_dir'] = f"{base_dir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            console.print(f"[cyan]Output directory: {config['output_dir']}[/cyan]")
        
        return config
    
    # Default configuration
    config = {
        'data_path': 'ubi_trending_with_information.json',
        'model': 'gpt-5',  # Per user preference for GPT-5 model
        'train_ratio': 0.8,
        'seed': 42,
        'output_dir': f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'num_generations': 10,
        'population_size': 8,
        'mutation_rate': 0.3,
        'elite_ratio': 0.25,
        'samples_per_evaluation': 5,
        'initial_prompt': """You are a viral tweet expert. Analyze the provided information and create a tweet that:
1. Synthesizes the most compelling insights
2. Uses emotional hooks and surprising facts
3. Creates urgency or controversy
4. Speaks directly to the audience
5. Stays under 280 characters

Focus on being provocative, insightful, and shareable."""
    }
    
    return config

def setup_environment():
    """Set up environment variables"""
    
    # Try to load from .env file
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
        console.print("[green]Loaded environment variables from .env[/green]")
    
    # Check for required API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not openai_key and not anthropic_key:
        console.print("[yellow]Warning: No API keys found in environment[/yellow]")
        console.print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        
        # Prompt for API key
        choice = Prompt.ask(
            "Which API would you like to use?",
            choices=["openai", "anthropic", "skip"],
            default="skip"
        )
        
        if choice == "openai":
            key = Prompt.ask("Enter your OpenAI API key", password=True)
            os.environ['OPENAI_API_KEY'] = key
        elif choice == "anthropic":
            key = Prompt.ask("Enter your Anthropic API key", password=True)
            os.environ['ANTHROPIC_API_KEY'] = key

def train_model(config: Dict):
    """Train the tweet generation model using official DSPy GEPA"""
    
    console.print(Panel.fit(
        "[bold blue]Starting Official DSPy GEPA Training[/bold blue]\n"
        "Using reflective prompt evolution with Pareto selection\n"
        "Learning to generate viral tweets from information sources",
        title="DSPy GEPA Tweet Optimizer",
        border_style="blue"
    ))
    
    # Log everything as per user preference
    console.print("[cyan]System is logging all learning and changes...[/cyan]")
    console.print("[cyan]Using official DSPy GEPA with rich textual feedback[/cyan]\n")
    
    # Initialize and run training pipeline
    pipeline = TrainingPipeline(config)
    
    # Start training with official GEPA
    pipeline.train(num_generations=config['num_generations'])
    
    console.print("\n[bold green]GEPA optimization completed successfully![/bold green]")
    console.print(f"Results saved to: {config['output_dir']}")
    
    return pipeline

def generate_tweet(config: Dict, module_path: str = None):
    """Generate a single tweet using GEPA-optimized module"""
    
    # Set up DSPy
    setup_dspy_model(config['model'])
    
    # Load optimized module or create new one
    if module_path and os.path.exists(module_path):
        try:
            import pickle
            with open(module_path, 'rb') as f:
                generator = pickle.load(f)
            console.print(f"[green]Loaded optimized module from {module_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not load module: {e}[/yellow]")
            console.print("[cyan]Using new base module[/cyan]")
            generator = TweetGeneratorModule()
    else:
        # Look for optimized module in results
        results_dirs = [d for d in os.listdir('.') if d.startswith('results_')]
        if results_dirs:
            latest_dir = sorted(results_dirs)[-1]
            module_path = os.path.join(latest_dir, 'optimized_module.pkl')
            if os.path.exists(module_path):
                try:
                    import pickle
                    with open(module_path, 'rb') as f:
                        generator = pickle.load(f)
                    console.print(f"[green]Loaded optimized module from {module_path}[/green]")
                except:
                    generator = TweetGeneratorModule()
            else:
                generator = TweetGeneratorModule()
        else:
            generator = TweetGeneratorModule()
    
    # Get information context
    console.print("\n[cyan]Enter information sources (tweets/articles) to generate from:[/cyan]")
    console.print("(Enter each source on a new line. Type 'done' when finished)\n")
    
    sources = []
    while True:
        source = Prompt.ask(f"Source {len(sources) + 1}")
        if source.lower() == 'done':
            break
        sources.append(source)
    
    if not sources:
        console.print("[red]No sources provided[/red]")
        return
    
    # Format sources
    information_context = "\n\n".join([f"[Source {i+1}] {s}" for i, s in enumerate(sources)])
    
    # Generate tweet
    console.print("\n[cyan]Generating tweet with GEPA-optimized module...[/cyan]")
    
    result = generator(information_context)
    
    console.print("\n[bold green]Generated Tweet:[/bold green]")
    console.print(Panel(
        result.generated_tweet,
        title="Tweet",
        border_style="green"
    ))
    
    # Optionally evaluate
    if Confirm.ask("\nWould you like to evaluate this tweet?"):
        original = Prompt.ask("Enter the original tweet for comparison (or 'skip')")
        
        if original != 'skip':
            optimizer = GEPATweetOptimizer()
            gold = {'information_context': information_context, 'original_tweet': original}
            # Call with 5 arguments as required by GEPA
            evaluation = optimizer.metric(gold, result, trace=None, pred_name="manual", pred_trace=None)
            
            console.print("\n[bold blue]GEPA Evaluation Results:[/bold blue]")
            console.print(f"Score: {evaluation.score:.3f}/1.0")
            console.print(f"\nFeedback:\n{evaluation.feedback}")

def setup_dspy_model(model_name: str):
    """Configure DSPy with specified model
    
    Supports:
    - OpenRouter models (e.g., 'claude-3-opus', 'llama-3.1-405b')
    - Direct OpenAI models (e.g., 'gpt-4', 'gpt-5')
    - Direct Anthropic models (e.g., 'claude-3-opus-20240229')
    """
    
    # Try OpenRouter first if not explicitly OpenAI/Anthropic
    if not model_name.startswith('openai/') and not model_name.startswith('anthropic/'):
        # Check if OpenRouter is available
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if openrouter_key:
            try:
                from openrouter_config import setup_openrouter_model
                lm = setup_openrouter_model(model_name, openrouter_key)
                dspy.settings.configure(lm=lm)
                console.print("[dim]Logging all learning and changes...[/dim]")
                return
            except Exception as e:
                console.print(f"[yellow]OpenRouter setup failed: {e}[/yellow]")
                console.print("[cyan]Falling back to direct provider...[/cyan]")
    
    # DSPy 3.0+ syntax for direct providers
    if 'gpt' in model_name.lower():
        api_key = os.getenv('OPENAI_API_KEY')
        # Handle GPT models - GPT-5 needs special parameters [[memory:7215547]]
        if '5' in model_name or model_name == 'gpt-5':
            # GPT-5 is a reasoning model requiring temperature=1.0 and max_tokens>=20000
            lm = dspy.LM('openai/gpt-5', api_key=api_key, temperature=1.0, max_tokens=20000)
            console.print("[cyan]Using GPT-5 reasoning model as configured[/cyan]")
        elif '4' in model_name or model_name == 'gpt-4':
            lm = dspy.LM('openai/gpt-4', api_key=api_key, max_tokens=500)
            console.print("[cyan]Using GPT-4 model[/cyan]")
        else:
            # Map model names to DSPy format
            if model_name == 'gpt-3.5-turbo':
                lm = dspy.LM('openai/gpt-3.5-turbo', api_key=api_key, max_tokens=500)
            else:
                lm = dspy.LM(f'openai/{model_name}', api_key=api_key, max_tokens=500)
    elif 'claude' in model_name.lower():
        api_key = os.getenv('ANTHROPIC_API_KEY')
        # Map Claude models to DSPy format
        lm = dspy.LM(f'anthropic/{model_name}', api_key=api_key)
    else:
        # Default to gpt-5 per user preference
        api_key = os.getenv('OPENAI_API_KEY')
        lm = dspy.LM('openai/gpt-5', api_key=api_key, temperature=1.0, max_tokens=20000)
        console.print("[cyan]Defaulting to GPT-5[/cyan]")
    
    dspy.settings.configure(lm=lm)
    console.print("[dim]Logging all learning and changes...[/dim]")

def analyze_dataset(config: Dict):
    """Analyze the tweet dataset"""
    
    console.print("[bold blue]Dataset Analysis[/bold blue]\n")
    
    processor = DataProcessor(config['data_path'])
    processor.get_statistics()
    
    # Show top tweets
    top_tweets = processor.get_high_engagement_tweets(top_k=5)
    
    console.print("\n[bold cyan]Top 5 High-Engagement Tweets:[/bold cyan]\n")
    
    for i, tweet in enumerate(top_tweets, 1):
        console.print(Panel(
            f"[bold]Tweet:[/bold] {tweet.tweet[:280]}\n\n"
            f"[cyan]@{tweet.username}[/cyan] | "
            f"❤️ {tweet.likes} | 🔁 {tweet.retweets} | 💬 {tweet.replies}\n"
            f"[dim]Engagement Score: {tweet.engagement_score:.0f}[/dim]\n"
            f"[dim]Sources: {len(tweet.information)} information items[/dim]",
            title=f"#{i} Top Tweet",
            border_style="cyan"
        ))

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Self-Evolving Tweet Prompt Optimizer using DSPy GEPA"
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'generate', 'analyze'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=10,
        help='Number of generations for training'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5',
        help='Model to use (gpt-5, gpt-3.5-turbo, claude-3-opus, etc)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        help='Path to prompt file for generation'
    )
    
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.generations:
        config['num_generations'] = args.generations
    if args.model:
        config['model'] = args.model
    
    # Log configuration (per user preference to log everything)
    console.print("\n[dim]Configuration:[/dim]")
    for key, value in config.items():
        if key != 'initial_prompt':  # Don't print the long prompt
            console.print(f"[dim]{key}: {value}[/dim]")
    console.print()
    
    # Execute command
    if args.command == 'train':
        train_model(config)
    
    elif args.command == 'generate':
        generate_tweet(config, args.prompt)
    
    elif args.command == 'analyze':
        analyze_dataset(config)
    
    console.print("\n[green]Done![/green]")

if __name__ == "__main__":
    main()

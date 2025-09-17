from training_pipeline import TrainingPipeline
import os
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Run GEPA training for tweet generation',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Judge Types:
  cognitive  - Original LLM cognitive analysis (strategic thinking, audience psychology)
             - Use --curriculum to gradually introduce penalties during training
  penalty    - Pure rule-based penalty enforcement (length, format, patterns)

Examples:
  python run_gepa_training.py --judge cognitive --model deepseek/deepseek-r1
  python run_gepa_training.py --judge penalty --generations 20
  python run_gepa_training.py --prompt custom_prompt.txt --judge cognitive
  python run_gepa_training.py --prompt my_prompt.txt --model nousresearch/hermes-4-405b
  python run_gepa_training.py --judge cognitive --curriculum --model deepseek/deepseek-r1
    """
)
parser.add_argument('--model', type=str, default='deepseek/deepseek-r1',
                    help='Model to use (e.g., deepseek/deepseek-r1, nousresearch/hermes-4-405b, moonshotai/kimi-k2-0905)')
parser.add_argument('--data', type=str, default='converted_final_output.json',
                    help='Path to training data')
parser.add_argument('--generations', type=int, default=15,
                    help='Number of GEPA optimization iterations')
parser.add_argument('--judge', type=str, choices=['cognitive', 'penalty'], 
                    default='cognitive',
                    help='Type of judge to use for evaluation (default: cognitive)')
parser.add_argument('--curriculum', action='store_true',
                    help='Enable curriculum penalty learning for cognitive judge (gradually introduces penalties)')
parser.add_argument('--prompt', type=str, 
                    help='Path to text file containing the initial prompt (if not provided, uses default prompt)')
args = parser.parse_args()

# Determine output directory based on model and judge
model_prefix = args.model.split('/')[-1].replace('-', '_').replace('.', '_')[:20]
output_dir = f'{model_prefix}_{args.judge}_judge_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

def load_initial_prompt(prompt_file_path: str = None) -> str:
    """Load initial prompt from file or return default prompt
    
    Args:
        prompt_file_path: Path to the prompt text file
        
    Returns:
        The prompt string to use for training
    """
    
    if prompt_file_path and os.path.exists(prompt_file_path):
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            print(f"✓ Loaded custom prompt from: {prompt_file_path}")
            return prompt
        except Exception as e:
            print(f"⚠️  Error reading prompt file {prompt_file_path}: {e}")
            print("Using default prompt instead...")
    elif prompt_file_path:
        print(f"⚠️  Prompt file not found: {prompt_file_path}")
        print("Using default prompt instead...")
    
    # Default prompt if no file provided or file reading failed
    return """You are an expert tweet strategist with deep understanding of viral content psychology.

Analyze the given information context and generate a viral tweet that demonstrates sophisticated strategic thinking:

REASONING PROCESS:
1. Identify the core insight or angle from the information, use web search tools if needed to get current information
2. Analyze target audience psychology and motivations  
3. Select optimal engagement mechanics (hooks, triggers, social proof)
4. Consider cultural context and trending dynamics, use web search tools if needed to get current information
5. Assess risks and optimize for maximum viral potential

TWEET REQUIREMENTS:
- Strategic positioning that goes beyond surface-level content
- Audience-aware language and psychological triggers
- Cultural fluency with current trends and memes, use web search tools if needed to get current information     
- Engaging hook that stops scroll and drives interaction
- Clear, concise execution under 280 characters
- Synthesized insight, not just information repetition

Think step-by-step about the strategic decisions, then generate the optimized viral tweet."""

# Configure training
config = {
    'data_path': args.data,  # Using converted cognitive analysis dataset
    'model': args.model,  # Model specified via command line
    'judge_type': args.judge,  # Judge type specified via command line
    'enable_curriculum': args.curriculum,  # Enable curriculum penalty learning
    'train_ratio': 0.8,  # 80% training, 20% validation
    'seed': 42,
    'output_dir': output_dir,  # Dynamic output directory based on model and judge
    'num_generations': 6,  # Number of GEPA optimization iterations
    'population_size': 32,
    'mutation_rate': 0.3,
    'elite_ratio': 0.2,
    'samples_per_evaluation': 25,
    'enable_tool_calling': True,  # Enable web search tools in GEPA optimization
    'initial_prompt': load_initial_prompt(args.prompt)  # Load from file or use default
}

def main():
    # Ensure OpenRouter API key is set
    if not os.getenv('OPENROUTER_API_KEY'):
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    # Print configuration
    prompt_source = args.prompt if args.prompt else "default (built-in)"
    print("=" * 70)
    print(f"🚀 GEPA Training Configuration")
    print("=" * 70)
    print(f"Model: {args.model}")
    judge_desc = 'Cognitive Analysis' if args.judge == 'cognitive' else 'Penalty Enforcement'
    if args.curriculum and args.judge == 'cognitive':
        judge_desc += ' (with Curriculum Learning)'
    print(f"Judge: {args.judge} ({judge_desc})")
    print(f"Prompt: {prompt_source}")
    print(f"Data: {args.data}")
    print(f"Generations: {args.generations}")
    print(f"Output: {config['output_dir']}")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config)
    
    # Run training
    pipeline.train(num_generations=args.generations)
    
    print(f"\nTraining complete! Check {config['output_dir']} for results")
    print("Key files:")
    print(f"- {config['output_dir']}/optimized_prompt.txt (The evolved prompt)")
    print(f"- {config['output_dir']}/optimized_module.json (Model configuration)")
    print(f"- {config['output_dir']}/training_metrics.json (Training statistics)")

if __name__ == "__main__":
    main()

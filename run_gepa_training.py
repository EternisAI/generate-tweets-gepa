from training_pipeline import TrainingPipeline
import os
from datetime import datetime

# Configure training
config = {
    'data_path': 'extracted_tweets_gepa_final.json',  # Using extracted tweets dataset
    'model': 'gpt-5',  # Using GPT-5 for best results
    'train_ratio': 0.8,  # 80% training, 20% validation
    'seed': 42,
    'output_dir': f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_fresh',  # Force fresh output directory
    'num_generations': 20,  # Number of GEPA optimization iterations
    'population_size': 32,
    'mutation_rate': 0.3,
    'elite_ratio': 0.125,
    'samples_per_evaluation': 25,
    'initial_prompt': """Generate a viral tweet that synthesizes the key information from the provided tweet and its media information if it's provided.
            Focus on creating engagement through surprising insights and emotional hooks.
            Be conversational, funny, provocative, and authentic.
            Maximum 280 characters. Just generate the tweet, don't include any other text."""
}

def main():
    # Ensure OpenRouter API key is set
    if not os.getenv('OPENROUTER_API_KEY'):
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config)
    
    # Run training
    pipeline.train(num_generations=config['num_generations'])
    
    print(f"\nTraining complete! Check {config['output_dir']} for results")
    print("Key files:")
    print(f"- {config['output_dir']}/optimized_prompt.txt (The evolved prompt)")
    print(f"- {config['output_dir']}/optimized_module.json (Model configuration)")
    print(f"- {config['output_dir']}/training_metrics.json (Training statistics)")

if __name__ == "__main__":
    main()

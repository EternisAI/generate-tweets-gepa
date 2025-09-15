from training_pipeline import TrainingPipeline
import os
from datetime import datetime

# Configure training
config = {
    'data_path': 'converted_final_output.json',  # Using converted cognitive analysis dataset
    'model': 'deepseek/deepseek-r1',  # Using DeepSeek R1 via OpenRouter for reasoning capabilities
    'train_ratio': 0.8,  # 80% training, 20% validation
    'seed': 42,
    'output_dir': f'deepseek_cognitive_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',  # DeepSeek cognitive analysis results
    'num_generations': 15,  # Number of GEPA optimization iterations
    'population_size': 32,
    'mutation_rate': 0.3,
    'elite_ratio': 0.125,
    'samples_per_evaluation': 25,
    'initial_prompt': """You are an expert tweet strategist with deep understanding of viral content psychology.

Analyze the given information context and generate a viral tweet that demonstrates sophisticated strategic thinking:

REASONING PROCESS:
1. Identify the core insight or angle from the information
2. Analyze target audience psychology and motivations  
3. Select optimal engagement mechanics (hooks, triggers, social proof)
4. Consider cultural context and trending dynamics
5. Assess risks and optimize for maximum viral potential

TWEET REQUIREMENTS:
- Strategic positioning that goes beyond surface-level content
- Audience-aware language and psychological triggers
- Cultural fluency with current trends and memes
- Engaging hook that stops scroll and drives interaction
- Clear, concise execution under 280 characters
- Synthesized insight, not just information repetition

Think step-by-step about the strategic decisions, then generate the optimized viral tweet."""
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

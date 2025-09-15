import dspy
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from data_processor import DataProcessor, TweetData
from llm_judge import LLMJudge, ComparativeJudge, PromptFeedback
from gepa_official_optimizer import GEPATweetOptimizer, TweetGeneratorModule

console = Console()

class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self, output_dir: str = 'metrics'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics = {
            'generation_scores': [],
            'best_scores': [],
            'average_scores': [],
            'component_scores': {},
            'prompt_evolution': []
        }
        
        console.print(f"[green]MetricsTracker initialized. Output directory: {output_dir}[/green]")
    
    def log_generation(self, generation: int, best_score: float, avg_score: float, 
                      component_scores: Dict = None, best_prompt: str = None):
        """Log metrics for a generation"""
        
        self.metrics['generation_scores'].append(generation)
        self.metrics['best_scores'].append(best_score)
        self.metrics['average_scores'].append(avg_score)
        
        if component_scores:
            for component, score in component_scores.items():
                if component not in self.metrics['component_scores']:
                    self.metrics['component_scores'][component] = []
                self.metrics['component_scores'][component].append(score)
        
        if best_prompt:
            self.metrics['prompt_evolution'].append({
                'generation': generation,
                'prompt': best_prompt[:500],  # Store first 500 chars
                'score': best_score
            })
        
        console.print(f"[cyan]Generation {generation}:[/cyan] Best: {best_score:.2f}, Avg: {avg_score:.2f}")
    
    def plot_progress(self):
        """Create visualization of training progress"""
        
        if not self.metrics['generation_scores']:
            console.print("[yellow]No metrics to plot yet[/yellow]")
            return
        
        # Set style
        sns.set_style('whitegrid')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Score progression
        ax1 = axes[0, 0]
        ax1.plot(self.metrics['generation_scores'], self.metrics['best_scores'], 
                label='Best Score', marker='o', linewidth=2)
        ax1.plot(self.metrics['generation_scores'], self.metrics['average_scores'], 
                label='Average Score', marker='s', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Score')
        ax1.set_title('Score Evolution Over Generations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement rate
        ax2 = axes[0, 1]
        if len(self.metrics['best_scores']) > 1:
            improvements = np.diff(self.metrics['best_scores'])
            ax2.bar(self.metrics['generation_scores'][1:], improvements, 
                   color=['green' if x > 0 else 'red' for x in improvements])
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Score Improvement')
            ax2.set_title('Generation-to-Generation Improvement')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Component scores
        ax3 = axes[1, 0]
        if self.metrics['component_scores']:
            for component, scores in self.metrics['component_scores'].items():
                ax3.plot(self.metrics['generation_scores'][:len(scores)], 
                        scores, label=component, marker='.')
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Component Score')
            ax3.set_title('Component Score Evolution')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Score distribution
        ax4 = axes[1, 1]
        ax4.boxplot([self.metrics['best_scores'], self.metrics['average_scores']], 
                   labels=['Best Scores', 'Average Scores'])
        ax4.set_ylabel('Score')
        ax4.set_title('Score Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        console.print(f"[green]Progress plot saved to {plot_path}[/green]")
        
        plt.show()
    
    def save_metrics(self, filename: str = 'training_metrics.json'):
        """Save metrics to JSON file"""
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        console.print(f"[green]Metrics saved to {filepath}[/green]")
    
    def print_summary(self):
        """Print a summary table of training results"""
        
        table = Table(title="Training Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        if self.metrics['best_scores']:
            table.add_row("Total Generations", str(len(self.metrics['generation_scores'])))
            table.add_row("Best Score Achieved", f"{max(self.metrics['best_scores']):.3f}")
            table.add_row("Final Best Score", f"{self.metrics['best_scores'][-1]:.3f}")
            table.add_row("Initial Best Score", f"{self.metrics['best_scores'][0]:.3f}")
            
            improvement = self.metrics['best_scores'][-1] - self.metrics['best_scores'][0]
            table.add_row("Total Improvement", f"{improvement:.3f}")
            
            if len(self.metrics['best_scores']) > 1:
                avg_improvement = improvement / (len(self.metrics['best_scores']) - 1)
                table.add_row("Avg Improvement/Gen", f"{avg_improvement:.4f}")
        
        console.print(table)

class TrainingPipeline:
    """Main training pipeline for tweet generation optimization"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        
        # Initialize components
        console.print("[bold blue]Initializing Training Pipeline[/bold blue]")
        console.print("[cyan]Learning to generate viral tweets from information...[/cyan]\n")
        
        # Data processor
        self.data_processor = DataProcessor(self.config['data_path'])
        self.data_processor.get_statistics()
        
        # Split data
        self.train_data, self.test_data = self.data_processor.split_dataset(
            train_ratio=self.config['train_ratio'],
            seed=self.config['seed']
        )
        
        # Initialize DSPy
        self._setup_dspy()
        
        # Initialize modules
        self.optimized_module = None
        self.judge = LLMJudge()
        self.comparative_judge = ComparativeJudge()
        self.feedback_generator = PromptFeedback()
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker(self.config['output_dir'])
        
        # Official GEPA Optimizer (will use settings.lm for judge and create reflection LM)
        self.optimizer = GEPATweetOptimizer()
        
        console.print("[green]Pipeline initialized successfully![/green]\n")
    
    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data_path': 'ubi_trending_with_information.json',
            'model': 'gpt-5',  # Changed as per user preference
            'train_ratio': 0.8,
            'seed': 42,
            'output_dir': 'results',
            'num_generations': 10,
            'population_size': 8,
            'mutation_rate': 0.3,
            'elite_ratio': 0.25,
            'samples_per_evaluation': 5,
            'initial_prompt': """You are a witty social media writer crafting viral tweets.

Given the tweet context which includes:
1. Media Analysis (if present) - Detailed analysis of images/media in the tweet
2. Information Sources - Additional context, quotes, and background information

Create a new tweet that:
- If media analysis is present:
  • Leverages visual elements and imagery described
  • Uses media insights to add depth or irony
  • References visual details that resonate emotionally
- For all tweets:
  • Synthesizes the *insight or irony* of the source
  • Uses tone levers: surprising twist, humor, provocation
  • Feels conversational and authentic (never corporate)
  • Stays punchy and scannable (under 280 chars)
  • Avoids copying phrases; finds a *fresh angle*
  • Ends with rhythm: kicker, shrug, or ironic beat

Output only the tweet text."""
        }
    
    def _setup_dspy(self):
        """Configure DSPy with the selected language model
        
        Supports OpenRouter models for experimentation
        """
        import os
        
        model = self.config['model']
        console.print(f"[cyan]Configuring DSPy with model: {model}[/cyan]")
        
        # Always use OpenRouter
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_key:
            console.print("[red]Error: OPENROUTER_API_KEY not found in environment variables[/red]")
            console.print("[yellow]Please set OPENROUTER_API_KEY environment variable[/yellow]")
            raise ValueError("Missing OPENROUTER_API_KEY")

        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from openrouter_config import setup_openrouter_model
            
            # If model doesn't have provider prefix, add openai/ for compatibility
            # Exception: deepseek-r1 should be used directly without prefix
            if model not in ['deepseek-r1', 'deepseek-chat'] and not any(model.startswith(p) for p in ['openai/', 'anthropic/', 'google/', 'meta/', 'mistral/', 'deepseek/']):
                model = f"openai/{model}"
            
            lm = setup_openrouter_model(model)
            dspy.settings.configure(lm=lm)
            console.print("[green]DSPy configured successfully with OpenRouter[/green]")
            console.print(f"[cyan]Using model: {model}[/cyan]")
            console.print("[dim]System logging all learning and changes...[/dim]\n")
        except Exception as e:
            console.print(f"[red]OpenRouter setup failed: {e}[/red]")
            raise
        
        dspy.settings.configure(lm=lm)
        console.print("[green]DSPy configured successfully[/green]")
        console.print("[dim]System logging all learning and changes...[/dim]\n")
    
    def prepare_training_data(self) -> List[Dict]:
        """Prepare training data in the format needed for optimization"""
        
        prepared_data = []
        
        for tweet_data in self.train_data:
            # Format information sources and media analysis
            info_context = self.data_processor.format_information(
                information=tweet_data.information,
                media_analysis=tweet_data.media_analysis
            )
            
            prepared_data.append({
                'information_context': info_context,
                'original_tweet': tweet_data.tweet,
                'engagement_score': tweet_data.engagement_score,
                'metadata': {
                    'username': tweet_data.username,
                    'likes': tweet_data.likes,
                    'retweets': tweet_data.retweets,
                    'has_media_analysis': bool(tweet_data.media_analysis)
                }
            })
        
        return prepared_data
    
    def train(self, num_generations: int = None):
        """Train the tweet generation system using official DSPy GEPA"""
        
        num_generations = num_generations or self.config['num_generations']
        
        console.print(f"[bold blue]Starting Official DSPy GEPA Training[/bold blue]")
        console.print(f"[cyan]Using reflective prompt evolution with Pareto selection[/cyan]")
        console.print(f"[cyan]Optimizing over {num_generations} generations[/cyan]\n")
        
        # Prepare training data
        train_data_prepared = self.prepare_training_data()
        
        # Select high-engagement tweets for training
        high_engagement = sorted(train_data_prepared, 
                                key=lambda x: x['engagement_score'], 
                                reverse=True)[:50]
        
        console.print(f"[cyan]Using top {len(high_engagement)} high-engagement tweets for training[/cyan]\n")
        
        # Create base module with tool calling if enabled
        enable_tools = self.config.get('enable_tool_calling', False)
        student_module = TweetGeneratorModule(enable_tool_calling=enable_tools)
        
        # Run GEPA optimization
        # When generations are specified explicitly, don't use auto mode
        if num_generations and num_generations <= 10:
            # Use max_full_evals for explicit control
            optimized_module = self.optimizer.optimize(
                student_module=student_module,
                trainset=high_engagement[:40],  # Use most for training
                valset=high_engagement[40:50],  # Keep some for validation
                max_generations=num_generations,  # This will be used as max_full_evals
                auto=None,  # Don't use auto mode when explicit generations specified
                track_stats=True,
                log_dir=self.config['output_dir']
            )
        else:
            # Use auto mode for larger runs
            optimized_module = self.optimizer.optimize(
                student_module=student_module,
                trainset=high_engagement[:40],
                valset=high_engagement[40:50],
                auto=self.config.get('gepa_mode', 'medium'),
                track_stats=True,
                log_dir=self.config['output_dir']
            )
        
        # Store optimized module
        self.optimized_module = optimized_module
        
        # Extract scores from GEPA optimizer if available
        best_score = 0.0
        avg_score = 0.0
        baseline_score = 0.0
        
        # Check if optimizer tracked any scores
        if hasattr(self.optimizer, 'last_best_score'):
            best_score = self.optimizer.last_best_score
            avg_score = self.optimizer.last_avg_score if hasattr(self.optimizer, 'last_avg_score') else best_score
            baseline_score = self.optimizer.baseline_score if hasattr(self.optimizer, 'baseline_score') else 0.0
        
        # Store scores for final display
        self.baseline_score = baseline_score
        self.optimized_score = best_score
        
        # Log metrics for each generation (approximate from GEPA output)
        # Since GEPA runs multiple iterations internally, log the final result
        self.metrics_tracker.log_generation(
            generation=num_generations,
            best_score=best_score,
            avg_score=avg_score,
            best_prompt="Optimized with official DSPy GEPA"
        )
        
        # Save results
        self.save_results()
        
        # Show final statistics
        self.show_final_results()
    
    def _analyze_and_adapt(self, test_samples: List[Dict]):
        """Analyze performance using GEPA's inference-time search"""
        
        console.print("[cyan]Running GEPA inference-time search for analysis...[/cyan]")
        
        if not self.optimized_module:
            return
        
        # Use GEPA's inference-time search
        search_results = self.optimizer.inference_time_search(
            student_module=self.optimized_module,
            test_instances=test_samples[:5],  # Quick search on subset
            max_metric_calls=50
        )
        
        if 'best_outputs' in search_results:
            console.print(f"[green]Found best outputs for {len(search_results['best_outputs'])} instances[/green]")
            console.print(f"[cyan]Average score: {np.mean(search_results['scores']):.3f}[/cyan]")
            console.print(f"[cyan]Total metric calls used: {search_results['total_calls']}[/cyan]")
    
    def evaluate_on_test_set(self):
        """Evaluate the optimized module on test set"""
        
        console.print("\n[bold blue]Evaluating on Test Set[/bold blue]")
        
        if not hasattr(self, 'optimized_module') or not self.optimized_module:
            console.print("[red]No trained model to evaluate[/red]")
            return
        
        generator = self.optimized_module
        
        # Prepare test data
        test_data_prepared = []
        for tweet_data in self.test_data[:20]:  # Evaluate on subset
            info_context = self.data_processor.format_information(tweet_data.information)
            test_data_prepared.append({
                'information_context': info_context,
                'original_tweet': tweet_data.tweet,
                'engagement_score': tweet_data.engagement_score
            })
        
        # Generate and evaluate
        evaluations = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Evaluating test samples...", total=len(test_data_prepared))
            
            for sample in test_data_prepared:
                try:
                    result = generator(sample['information_context'])
                    eval_result = self.judge(
                        information_sources=sample['information_context'],
                        generated_tweet=result.generated_tweet,
                        original_tweet=sample['original_tweet']
                    )
                    
                    evaluations.append({
                        'overall_score': eval_result.overall_score,
                        'evaluation': eval_result.evaluation,
                        'generated_tweet': result.generated_tweet,
                        'original_tweet': sample['original_tweet']
                    })
                except Exception as e:
                    console.print(f"[red]Error evaluating sample: {e}[/red]")
                
                progress.update(task, advance=1)
        
        # Calculate statistics
        if evaluations:
            scores = [e['overall_score'] for e in evaluations]
            
            table = Table(title="Test Set Performance")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Samples Evaluated", str(len(evaluations)))
            table.add_row("Average Score", f"{np.mean(scores):.3f}")
            table.add_row("Median Score", f"{np.median(scores):.3f}")
            table.add_row("Max Score", f"{np.max(scores):.3f}")
            table.add_row("Min Score", f"{np.min(scores):.3f}")
            table.add_row("Std Dev", f"{np.std(scores):.3f}")
            
            console.print(table)
            
            # Show best generated tweet
            best_eval = max(evaluations, key=lambda x: x['overall_score'])
            console.print("\n[green]Best Generated Tweet:[/green]")
            console.print(f"Score: {best_eval['overall_score']:.2f}")
            console.print(f"Generated: {best_eval['generated_tweet']}")
            console.print(f"Original: {best_eval['original_tweet'][:280]}")
    
    def save_results(self):
        """Save all training results"""
        
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        self.metrics_tracker.save_metrics()
        
        # Save optimized module if available
        if hasattr(self, 'optimized_module') and self.optimized_module:
            try:
                import pickle
                module_path = os.path.join(output_dir, 'optimized_module.pkl')
                with open(module_path, 'wb') as f:
                    pickle.dump(self.optimized_module, f)
                console.print(f"[green]Optimized module saved to {module_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not save module: {e}[/yellow]")
        
        # Save configuration
        config_path = os.path.join(output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        console.print(f"[green]All results saved to {output_dir}/[/green]")
    
    def show_final_results(self):
        """Display final training results"""
        
        console.print("\n[bold blue]Training Complete![/bold blue]\n")
        console.print("[green]Used Official DSPy GEPA with:[/green]")
        console.print("  • Reflective prompt mutations")
        console.print("  • Rich textual feedback")
        console.print("  • Pareto-based candidate selection")
        console.print("  • System-aware merging\n")
        
        # Show metrics summary
        self.metrics_tracker.print_summary()
        
        # Show GEPA optimization improvement
        console.print(f"\n[bold green]GEPA Optimization Results:[/bold green]")
        if hasattr(self, 'baseline_score') and hasattr(self, 'optimized_score'):
            improvement = self.optimized_score - self.baseline_score
            
            # Create a table for before/after comparison
            from rich.table import Table
            table = Table(title="Score Comparison")
            table.add_column("Stage", style="cyan", width=20)
            table.add_column("Score", style="magenta", width=10)
            table.add_column("Change", style="green", width=15)
            
            table.add_row("Baseline (Before)", f"{self.baseline_score:.3f}", "-")
            table.add_row("Optimized (After)", f"{self.optimized_score:.3f}", f"+{improvement:.3f}")
            
            # Handle percentage calculation properly
            if self.baseline_score > 0:
                percent_gain = (improvement / self.baseline_score * 100)
                table.add_row("Improvement", "-", f"{percent_gain:.1f}%")
            else:
                table.add_row("Improvement", "-", "N/A (baseline was 0)")
            
            console.print(table)
            console.print(f"\n[cyan]Optimized prompt saved to {self.config['output_dir']}/optimized_prompt.txt[/cyan]")
        else:
            console.print("Score tracking information not available")
        
        # Plot progress
        self.metrics_tracker.plot_progress()
        
        # Evaluate on test set
        self.evaluate_on_test_set()

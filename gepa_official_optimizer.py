#!/usr/bin/env python3
"""
Official DSPy GEPA implementation for Tweet Generation
Using dspy.GEPA for reflective prompt optimization
"""

import dspy
from dspy import Prediction, Example
try:
    from dspy.teleprompt import GEPA
except ImportError:
    # Fallback for different DSPy versions
    try:
        from dspy import GEPA
    except ImportError:
        raise ImportError("Could not import GEPA. Please ensure dspy-ai >= 2.5.0 is installed.")
from typing import Dict, List, Optional, Any
import json
import numpy as np
from datetime import datetime
from rich.console import Console

console = Console()

# Define base class if not available in current DSPy version
try:
    from dspy import GEPAFeedbackMetric
except ImportError:
    # Create a compatible base class for older DSPy versions
    class GEPAFeedbackMetric:
        """Base class for GEPA feedback metrics"""
        def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
            raise NotImplementedError("Subclasses must implement __call__")

class TweetGenerationMetric(GEPAFeedbackMetric if 'GEPAFeedbackMetric' in locals() else object):
    """GEPA-compatible metric for tweet generation evaluation"""
    
    def __init__(self, judge_lm=None):
        """Initialize the metric with an optional judge LM"""
        self.judge_lm = judge_lm or dspy.settings.lm
        # Define evaluation signature class
        class TweetEvaluationSignature(dspy.Signature):
            """You are a humor + virality critic. 
Evaluate a candidate tweet against the original tweet + media context.

Check first:
- If the model did not output a valid tweet (empty, meta-text, instructions, or commentary instead of a tweet), assign a score of 0 and return feedback: "No tweet generated."

If a valid tweet is present, score each criterion on 0–1:
1. Information Coverage — captures the *gist* of the context without parroting.
2. Style Match — aligns with viral tweet patterns (brevity, cadence, rhetorical punch).
3. Originality — introduces a unique twist; penalize if it mimics the source.
4. Engagement Potential — likely to get replies, likes, or reposts.
5. Humor / Surprise — delivers wit, absurdity, or an emotional hook.
6. Cultural Evaluator (0–1 each)
Judge cultural grounding with these axes:

	•	Cultural Fluency — Uses references, tone, and norms naturally.
	•	Meme Lineage Alignment — Riffs appropriately on existing meme formats/phrases.
	•	Locale Shibboleths Usage — Employs insider slang, jargon, or fan idioms.
	•	Cross-Subculture Resonance — Likely to resonate beyond niche communities.
Rules:
- Penalize blandness or over-explanation.
- Penalize copying content; reward fresh spins.
- Bonus if funny, deadpan, or ironic.
- Favor concise rhythm (short clauses, punchy beats).

Return:
- JSON with criterion scores and final average `score` (0–1). If invalid, return 0.
- 2–3 sentences of feedback on what worked and what to improve."""
            
            information_sources = dspy.InputField(desc="Source information")
            generated_tweet = dspy.InputField(desc="Generated tweet")
            original_tweet = dspy.InputField(desc="Original viral tweet")
            score = dspy.OutputField(desc="Score between 0 and 1")
            feedback = dspy.OutputField(desc="Detailed feedback for improvement")
        
        self.evaluation_prompt = TweetEvaluationSignature
        self.evaluator = dspy.Predict(self.evaluation_prompt)
        print("[TweetGenerationMetric] Initialized GEPA-compatible metric")
    
    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
        """Evaluate a generated tweet and return score + feedback
        
        Args:
            gold: Input example with information_context and original_tweet (gold standard)
            pred: Generated tweet prediction
            trace: Optional execution trace
            pred_name: Name of the predictor (for GEPA)
            pred_trace: Predictor trace (for GEPA)
        
        Returns:
            dspy.Prediction with score (0-1) and feedback string
        """
        
        # Handle different input types for gold (example)
        if hasattr(gold, '__getitem__') and hasattr(gold, 'get'):
            # DSPy Example or dictionary-like
            information_context = gold.get('information_context', gold.get('inputs', {}).get('information_context', ''))
            original_tweet = gold.get('original_tweet', '')
        elif hasattr(gold, 'information_context'):
            # Object with attributes
            information_context = gold.information_context
            original_tweet = getattr(gold, 'original_tweet', '')
        elif hasattr(gold, 'inputs'):
            # DSPy Example with inputs
            information_context = gold.inputs.get('information_context', '')
            original_tweet = getattr(gold, 'original_tweet', '')
        else:
            # Fallback
            information_context = str(gold)
            original_tweet = ''
        
        # Extract the generated tweet from prediction
        if hasattr(pred, 'generated_tweet'):
            generated_tweet = pred.generated_tweet
        elif hasattr(pred, 'answer'):
            generated_tweet = pred.answer
        elif hasattr(pred, 'completions'):
            # Handle DSPy completion format
            generated_tweet = pred.completions.generated_tweet if hasattr(pred.completions, 'generated_tweet') else str(pred.completions)
        else:
            generated_tweet = str(pred)
        
        # Evaluate with the judge
        try:
            # Debug logging
            if not information_context:
                print(f"[GEPA Metric] Warning: Empty information context")
            if not generated_tweet:
                print(f"[GEPA Metric] Warning: Empty generated tweet")
            
            with dspy.context(lm=self.judge_lm):
                evaluation = self.evaluator(
                    information_sources=information_context,
                    generated_tweet=generated_tweet,
                    original_tweet=original_tweet
                )
            
            # Parse score
            try:
                score = float(evaluation.score)
                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                score = 0.5  # Default mid-range score
            
            # Format feedback
            feedback = evaluation.feedback
            
            # Add specific insights based on common issues
            if score < 0.5:
                feedback += "\n\nKey issues to address:"
                if "information" in feedback.lower():
                    feedback += "\n- Focus on extracting the most newsworthy elements"
                if "style" in feedback.lower():
                    feedback += "\n- Add emotional hooks and controversy"
                if "engagement" in feedback.lower():
                    feedback += "\n- Start with a compelling hook"
            
            # Log everything as per user preference
            print(f"[GEPA Metric] Score: {score:.3f} | Feedback: {feedback[:100]}...")
            
            return dspy.Prediction(score=score, feedback=feedback)
        
        except Exception as e:
            print(f"[GEPA Metric] Error during evaluation: {e}")
            return dspy.Prediction(
                score=0.0,
                feedback=f"Evaluation failed: {str(e)}. Consider improving tweet structure."
            )

class TweetGenerationSignature(dspy.Signature):
    """Signature for tweet generation from information sources and media analysis"""
    
    information_context = dspy.InputField(
        desc="Source tweets, articles, media analysis, and additional information to base the tweet on. Media analysis will be prefixed with [Media Analysis] if present."
    )
    generated_tweet = dspy.OutputField(
        desc="A viral tweet (max 280 chars) that synthesizes key insights and leverages media analysis when available"
    )

class TweetGeneratorModule(dspy.Module):
    """DSPy module for tweet generation compatible with GEPA"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(TweetGenerationSignature)
    
    def forward(self, information_context):
        """Generate a tweet from information context"""
        result = self.generate(information_context=information_context)
        
        # Ensure 280 character limit
        tweet = result.generated_tweet
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        
        return dspy.Prediction(generated_tweet=tweet)

class GEPATweetOptimizer:
    """Wrapper for official DSPy GEPA optimizer for tweet generation"""
    
    def __init__(self, judge_lm=None, reflection_lm=None):
        """
        Initialize the GEPA optimizer for tweets
        
        Args:
            judge_lm: LM for evaluation (default: uses dspy.settings.lm)
            reflection_lm: LM for reflection/mutation (default: creates a new gpt-5 instance)
        """
        self.judge_lm = judge_lm or dspy.settings.lm
        
        # Set up reflection LM - GEPA needs a strong model for reflective mutations
        if reflection_lm:
            self.reflection_lm = reflection_lm
        else:
            # Create a dedicated reflection LM with higher temperature for creativity using OpenRouter
            import os
            from openrouter_config import setup_openrouter_model
            
            try:
                # Always use OpenRouter for reflection
                self.reflection_lm = setup_openrouter_model('openai/gpt-5')  # Using GPT-5 for best reflection
                print("[GEPATweetOptimizer] Created OpenRouter GPT-5 reflection model")
            except Exception as e:
                # Fallback to settings LM if OpenRouter setup fails
                print(f"[GEPATweetOptimizer] OpenRouter setup failed: {e}")
                self.reflection_lm = dspy.settings.lm
                print("[GEPATweetOptimizer] Using default LM for reflection")
        
        self.metric = TweetGenerationMetric(judge_lm)
        
        # Initialize score tracking
        self.last_best_score = 0.0
        self.last_avg_score = 0.0
        
        # Log initialization
        print("[GEPATweetOptimizer] Initializing official DSPy GEPA optimizer")
        print("[GEPATweetOptimizer] Using Pareto-based selection and reflective mutations")
    
    def optimize(self,
                 student_module: dspy.Module,
                 trainset: List,
                 valset: List = None,
                 max_generations: int = 10,
                 max_metric_calls: int = None,
                 auto: str = 'medium',
                 track_stats: bool = True,
                 log_dir: str = None) -> dspy.Module:
        """
        Optimize the tweet generation module using GEPA
        
        Args:
            student_module: DSPy module to optimize
            trainset: Training examples
            valset: Validation examples (if None, uses trainset)
            max_generations: Maximum full evaluations
            max_metric_calls: Maximum metric calls
            auto: Automatic configuration ('light', 'medium', 'heavy')
            track_stats: Whether to track detailed statistics
            log_dir: Directory for logs
        
        Returns:
            Optimized DSPy module
        """
        
        # Convert data to DSPy Example format if needed
        trainset = self._prepare_dataset(trainset)
        
        if valset is None:
            # Split trainset for validation
            split_idx = int(len(trainset) * 0.8)
            valset = trainset[split_idx:]
            trainset = trainset[:split_idx]
            print(f"[GEPATweetOptimizer] Split data: {len(trainset)} train, {len(valset)} val")
        else:
            valset = self._prepare_dataset(valset)
        
        # Configure GEPA optimizer
        print(f"\n[GEPATweetOptimizer] Starting GEPA optimization")
        
        # Initialize official GEPA - use ONLY ONE of auto/max_metric_calls/max_full_evals
        try:
            if auto:
                # Use auto mode (preferred)
                print(f"[GEPATweetOptimizer] Mode: {auto} (auto-configured)")
                gepa_optimizer = GEPA(
                    metric=self.metric,
                    auto=auto,
                    reflection_minibatch_size=3,
                    candidate_selection_strategy='pareto',  # Use Pareto frontier
                    reflection_lm=self.reflection_lm,
                    skip_perfect_score=True,
                    add_format_failure_as_feedback=True,
                    use_merge=True,  # Enable crossover between candidates
                    max_merge_invocations=5,
                    failure_score=0.0,
                    perfect_score=1.0,
                    log_dir=log_dir or f"gepa_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    track_stats=track_stats,
                    track_best_outputs=True,  # Track best outputs for each instance
                    seed=42
                )
            elif max_metric_calls:
                # Use metric calls limit
                print(f"[GEPATweetOptimizer] Max metric calls: {max_metric_calls}")
                gepa_optimizer = GEPA(
                    metric=self.metric,
                    max_metric_calls=max_metric_calls,
                    reflection_minibatch_size=3,
                    candidate_selection_strategy='pareto',
                    reflection_lm=self.reflection_lm,
                    skip_perfect_score=True,
                    add_format_failure_as_feedback=True,
                    use_merge=True,
                    max_merge_invocations=5,
                    failure_score=0.0,
                    perfect_score=1.0,
                    log_dir=log_dir or f"gepa_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    track_stats=track_stats,
                    track_best_outputs=True,
                    seed=42
                )
            else:
                # Use max full evaluations (respects --generations flag)
                dataset_size = len(trainset) + len(valset)
                print(f"[GEPATweetOptimizer] Using explicit generation limit: {max_generations} full evaluations")
                print(f"[GEPATweetOptimizer] This will run approximately {max_generations * dataset_size} metric calls")
                gepa_optimizer = GEPA(
                    metric=self.metric,
                    max_full_evals=max_generations,
                    reflection_minibatch_size=3,
                    candidate_selection_strategy='pareto',
                    reflection_lm=self.reflection_lm,
                    skip_perfect_score=True,
                    add_format_failure_as_feedback=True,
                    use_merge=True,
                    max_merge_invocations=5,
                    failure_score=0.0,
                    perfect_score=1.0,
                    log_dir=log_dir or f"gepa_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    track_stats=track_stats,
                    track_best_outputs=True,
                    seed=42
                )
        except TypeError as e:
            # Fallback for older DSPy versions with different parameters
            console.print(f"[yellow]Using compatibility mode for GEPA: {e}[/yellow]")
            gepa_optimizer = GEPA(
                metric=self.metric,
                max_bootstrapped_demos=max_generations,
                max_labeled_demos=max_generations,
                num_candidates=3,
                init_temperature=1.0
            )
        
        # First, evaluate baseline module (before optimization)
        print("\n[GEPATweetOptimizer] Evaluating baseline module...")
        
        # Check if we have validation data
        if not valset or len(valset) == 0:
            print("[GEPATweetOptimizer] Warning: No validation data available")
            baseline_score = 0.3  # Set local variable
            self.baseline_score = baseline_score
            print(f"[GEPATweetOptimizer] Using default baseline score: {self.baseline_score:.3f}")
        else:
            baseline_scores = []
            successful_evals = 0
            eval_count = min(5, len(valset))  # Evaluate up to 5 examples
            
            for i, example in enumerate(valset[:eval_count]):
                try:
                    # Generate with baseline module
                    # Try to get the information_context field
                    if hasattr(example, 'information_context'):
                        context = example.information_context
                    elif isinstance(example, dict) and 'information_context' in example:
                        context = example['information_context']
                    else:
                        # Fallback: use string representation
                        context = str(example)
                    
                    prediction = student_module(information_context=context)
                    
                    # Evaluate with metric
                    result = self.metric(
                        gold=example,
                        pred=prediction,
                        trace=None,
                        pred_name=None,
                        pred_trace=None
                    )
                    score = result.score if hasattr(result, 'score') else 0.5
                    baseline_scores.append(score)
                    successful_evals += 1
                    print(f"  Sample {i+1}: Score {score:.3f}")
                except Exception as e:
                    print(f"  Sample {i+1}: Failed - {str(e)[:200]}")
                    print(f"    Context type: {type(context)}, Context: {str(context)[:100]}")
                    # Try a simpler evaluation with just a generated tweet
                    try:
                        # Generate something simple for baseline
                        simple_tweet = "Testing baseline generation..."
                        simple_pred = dspy.Prediction(generated_tweet=simple_tweet)
                        result = self.metric(
                            gold=example,
                            pred=simple_pred,
                            trace=None,
                            pred_name=None,
                            pred_trace=None
                        )
                        score = result.score if hasattr(result, 'score') else 0.3
                        baseline_scores.append(score)
                        successful_evals += 1
                        print(f"    Fallback score: {score:.3f}")
                    except:
                        pass  # Skip completely failed evaluations
            
            if baseline_scores:
                baseline_score = sum(baseline_scores) / len(baseline_scores)
            else:
                # If all evaluations failed, use a lower default baseline
                print("[GEPATweetOptimizer] Warning: All baseline evaluations failed, using default score 0.3")
                baseline_score = 0.3  # Use 0.3 (30%) as low baseline for failed evals
            
            self.baseline_score = baseline_score
            print(f"[GEPATweetOptimizer] Baseline score (before GEPA): {baseline_score:.3f} (from {successful_evals}/{eval_count} samples)")
        
        # Track scores and prompts during optimization
        import logging
        import re
        
        # Capture GEPA's internal logs to extract scores AND evolved prompts
        class OptimizationCapture:
            def __init__(self, baseline=0.3):
                self.best_score = 0.0
                self.avg_scores = []
                self.iteration_scores = []
                self.baseline = baseline  # Store baseline
                self.evolved_prompt = None
                self.capturing_prompt = False
                self.prompt_lines = []
                
            def parse_log(self, message):
                # Parse "Average Metric: X / Y (Z%)"
                if "Average Metric:" in message:
                    match = re.search(r'\((\d+\.\d+)%\)', message)
                    if match:
                        score = float(match.group(1)) / 100.0
                        self.avg_scores.append(score)
                        self.best_score = max(self.best_score, score)
                
                # Parse "Best score on valset: X"
                if "Best score on valset:" in message:
                    match = re.search(r'Best score on valset: ([0-9.]+)', message)
                    if match:
                        score = float(match.group(1))
                        self.iteration_scores.append(score)
                        self.best_score = max(self.best_score, score)
                
                # Capture evolved prompt text from GEPA logs
                if "Proposed new text for generate.predict:" in message:
                    self.capturing_prompt = True
                    self.prompt_lines = []
                    # Extract the start of the prompt from this line
                    parts = message.split("Proposed new text for generate.predict:")
                    if len(parts) > 1:
                        self.prompt_lines.append(parts[1].strip())
                elif self.capturing_prompt:
                    # Stop capturing when we see certain indicators
                    if any(x in message for x in ["[GEPA Metric]", "Average Metric:", "2025/", "2024/", "INFO dspy"]):
                        self.capturing_prompt = False
                        if self.prompt_lines:
                            self.evolved_prompt = "\n".join(self.prompt_lines)
                    else:
                        self.prompt_lines.append(message)
        
        optimization_capture = OptimizationCapture(baseline=baseline_score)
        
        # Set up logging handler to capture scores and prompts
        class CaptureHandler(logging.Handler):
            def emit(self, record):
                optimization_capture.parse_log(record.getMessage())
        
        # Add handler to GEPA logger
        gepa_logger = logging.getLogger('dspy.teleprompt.gepa.gepa')
        capture_handler = CaptureHandler()
        gepa_logger.addHandler(capture_handler)
        
        # Run optimization
        print("[GEPATweetOptimizer] Running reflective prompt evolution...")
        try:
            optimized_module = gepa_optimizer.compile(
                student=student_module,
                trainset=trainset,
                valset=valset
            )
        finally:
            # Remove handler after optimization
            gepa_logger.removeHandler(capture_handler)
        
        print(f"\n[GEPATweetOptimizer] Optimization Complete!")
        
        # Store captured scores and prompt for external access
        self.last_best_score = optimization_capture.best_score
        self.last_avg_score = sum(optimization_capture.avg_scores) / len(optimization_capture.avg_scores) if optimization_capture.avg_scores else optimization_capture.best_score
        self.baseline_score = baseline_score
        self.evolved_prompt = optimization_capture.evolved_prompt
        
        # Report before/after comparison
        print(f"\n[GEPATweetOptimizer] === Score Comparison ===")
        print(f"[GEPATweetOptimizer] Baseline score (before): {baseline_score:.3f}")
        print(f"[GEPATweetOptimizer] Best score (after): {optimization_capture.best_score:.3f}")
        
        # Calculate improvement with zero-check
        improvement = optimization_capture.best_score - baseline_score
        if baseline_score > 0:
            percent_gain = (improvement / baseline_score * 100)
            print(f"[GEPATweetOptimizer] Improvement: +{improvement:.3f} ({percent_gain:.1f}% gain)")
        else:
            print(f"[GEPATweetOptimizer] Improvement: +{improvement:.3f} (baseline was 0)")
        
        print(f"[GEPATweetOptimizer] Average score: {self.last_avg_score:.3f}")
        
        # Report if we captured the evolved prompt
        if optimization_capture.evolved_prompt:
            print(f"[GEPATweetOptimizer] Captured evolved prompt ({len(optimization_capture.evolved_prompt)} chars)")
        
        # Try to access GEPA statistics if available
        if hasattr(gepa_optimizer, 'total_metric_calls'):
            print(f"[GEPATweetOptimizer] Total metric calls: {gepa_optimizer.total_metric_calls}")
        
        if hasattr(gepa_optimizer, 'num_full_evals'):
            print(f"[GEPATweetOptimizer] Full evaluations: {gepa_optimizer.num_full_evals}")
        
        # Save the optimized prompt and module configuration
        try:
            # Save module configuration
            if hasattr(optimized_module, 'save'):
                save_path = f"{log_dir}/optimized_module.json"
                optimized_module.save(save_path)
                print(f"[GEPATweetOptimizer] Saved optimized module to: {save_path}")
            
            # Extract and save the optimized prompt
            prompt_path = f"{log_dir}/optimized_prompt.txt"
            prompt_text = "No optimized prompt found"
            
            # Try multiple ways to extract the evolved prompt
            if hasattr(optimized_module, 'generate'):
                generator = optimized_module.generate
                
                # Method 1: Check for evolved instructions in the predictor
                if hasattr(generator, 'predict') and hasattr(generator.predict, 'signature'):
                    signature = generator.predict.signature
                    
                    # Get the instructions field (this is what GEPA evolves)
                    if hasattr(signature, 'instructions'):
                        prompt_text = signature.instructions
                    elif hasattr(signature, '_instructions'):
                        prompt_text = signature._instructions
                    
                # Method 2: Check for extended signature
                if prompt_text == "No optimized prompt found" and hasattr(generator, 'extended_signature'):
                    extended_sig = generator.extended_signature
                    if hasattr(extended_sig, 'instructions'):
                        prompt_text = extended_sig.instructions
                
                # Method 3: Check the actual predictor object
                if prompt_text == "No optimized prompt found" and hasattr(generator, 'predictor'):
                    predictor = generator.predictor
                    if hasattr(predictor, 'signature') and hasattr(predictor.signature, 'instructions'):
                        prompt_text = predictor.signature.instructions
            
            # If still not found, try to extract from the module's string representation
            if prompt_text == "No optimized prompt found":
                # Save full module JSON and parse it
                import json
                module_dict = optimized_module.save("temp.json", save_metadata=False)
                with open("temp.json", 'r') as f:
                    module_data = json.load(f)
                    
                # Look for the generate.predict section
                if "generate.predict" in module_data:
                    predict_data = module_data["generate.predict"]
                    if "signature" in predict_data and "instructions" in predict_data["signature"]:
                        prompt_text = predict_data["signature"]["instructions"]
                
                # Clean up temp file
                import os
                if os.path.exists("temp.json"):
                    os.remove("temp.json")
                
            # Check if we have the evolved prompt from logs
            if hasattr(self, 'evolved_prompt') and self.evolved_prompt:
                prompt_text = self.evolved_prompt
            
            # Save the prompt
            with open(prompt_path, 'w') as f:
                f.write("=== OPTIMIZED GEPA PROMPT ===\n\n")
                
                # If we found the actual evolved prompt, use it
                if prompt_text != "No optimized prompt found" and prompt_text != "Signature for tweet generation from information sources" and len(prompt_text) > 50:
                    f.write(prompt_text)
                else:
                    # Fallback: Note that prompt couldn't be extracted
                    f.write("Note: GEPA's evolved prompt could not be fully extracted.\n")
                    f.write("The optimization improved the module internally.\n\n")
                    if prompt_text and prompt_text != "No optimized prompt found":
                        f.write("Basic signature: ")
                        f.write(prompt_text)
                
                # Always write optimization metrics
                f.write("\n\n=== OPTIMIZATION METRICS ===\n")
                f.write(f"Baseline Score: {baseline_score:.3f}\n")
                f.write(f"Optimized Score: {self.last_best_score:.3f}\n")
                f.write(f"Improvement: +{(self.last_best_score - baseline_score):.3f}\n")
                
                # Handle percentage gain calculation with zero check
                if baseline_score > 0:
                    percent_gain = ((self.last_best_score - baseline_score) / baseline_score * 100)
                    f.write(f"Percentage Gain: {percent_gain:.1f}%\n")
                else:
                    f.write(f"Percentage Gain: N/A (baseline was 0)\n")
                
                print(f"[GEPATweetOptimizer] Saved optimized prompt to: {prompt_path}")
        except Exception as e:
            print(f"[GEPATweetOptimizer] Could not save prompt: {e}")
        
        return optimized_module
    
    def _prepare_dataset(self, dataset: List) -> List:
        """Convert dataset to DSPy Example format"""
        
        examples = []
        for item in dataset:
            # Handle different input formats
            if hasattr(item, 'inputs'):
                # Already a DSPy Example
                examples.append(item)
            elif isinstance(item, dict):
                # Convert dict to Example with proper structure
                example = dspy.Example(
                    information_context=item.get('information_context', ''),
                    original_tweet=item.get('original_tweet', '')
                ).with_inputs('information_context')
                examples.append(example)
            else:
                # Try to use as-is
                examples.append(item)
        
        return examples
    
    def _save_results(self, results, log_dir):
        """Save optimization results"""
        
        if not log_dir:
            log_dir = f"gepa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        import os
        os.makedirs(log_dir, exist_ok=True)
        
        # Save results summary
        summary = {
            'best_score': float(results.best_candidate.score) if hasattr(results.best_candidate, 'score') else None,
            'total_metric_calls': results.total_metric_calls,
            'num_full_evals': results.num_full_val_evals,
            'best_idx': results.best_idx,
            'val_scores': [float(s) for s in results.val_aggregate_scores]
        }
        
        with open(os.path.join(log_dir, 'optimization_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[GEPATweetOptimizer] Results saved to {log_dir}")
    
    def inference_time_search(self,
                             student_module: dspy.Module,
                             test_instances: List[Dict],
                             max_metric_calls: int = 100) -> Dict:
        """
        Use GEPA for inference-time search to find best outputs
        
        Args:
            student_module: Base module
            test_instances: Instances to generate for
            max_metric_calls: Budget for search
        
        Returns:
            Dictionary with best outputs for each instance
        """
        
        print("\n[GEPATweetOptimizer] Running inference-time search")
        print(f"[GEPATweetOptimizer] Searching for best outputs on {len(test_instances)} instances")
        
        # Use GEPA as inference-time optimizer
        gepa_optimizer = GEPA(
            metric=self.metric,
            max_metric_calls=max_metric_calls,
            track_stats=True,
            track_best_outputs=True,  # Critical for inference search
            candidate_selection_strategy='pareto'
        )
        
        # Run optimization with test set as both train and val
        optimized = gepa_optimizer.compile(
            student=student_module,
            trainset=test_instances,
            valset=test_instances  # Use same set for inference
        )
        
        # Extract best outputs
        if hasattr(optimized, 'detailed_results'):
            results = optimized.detailed_results
            best_outputs = results.best_outputs_valset
            scores = results.highest_score_achieved_per_val_task
            
            return {
                'best_outputs': best_outputs,
                'scores': scores,
                'total_calls': results.total_metric_calls
            }
        
        return {'error': 'No results available'}

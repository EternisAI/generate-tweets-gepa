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
import concurrent.futures
from datetime import datetime
from rich.console import Console

# Import tool calling functionality
try:
    from exa_tool_call import ExaToolCall, get_all_tool_definitions
    TOOL_CALLING_AVAILABLE = True
except ImportError:
    TOOL_CALLING_AVAILABLE = False
    print("[GEPA] Warning: Tool calling not available - exa_tool_call not found")

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
    """GEPA-compatible metric for tweet generation evaluation with variant support"""
    
    def __init__(self, judge_lm=None):
        """Initialize the metric with an optional judge LM"""
        self.judge_lm = judge_lm or dspy.settings.lm
        
        # Track best variants
        self.best_variants = {}  # {task_id: [(score, variant_id, tweet)]}
        
        # Configure thread pool for parallel evaluation
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Define evaluation signature class
        class TweetEvaluationSignature(dspy.Signature):
            """You are a humor + virality critic with expertise in engagement analytics.
Evaluate a candidate tweet against the original tweet + media context, accounting for temporal and audience factors.

Check first:
- If the model did not output a valid tweet (empty, meta-text, instructions, or commentary instead of a tweet), assign a score of 0 and return feedback: "No tweet generated."

If a valid tweet is present, score each criterion on 0–1:

1. Content Quality (0-1):
   - Information Coverage — captures key insights without verbatim copying
   - Originality — introduces unique perspective or angle
   - Clarity — message is clear and well-structured

2. Engagement Optimization (0-1):
   - Time-of-Day Alignment — matches optimal posting windows
   - Audience Targeting — appeals to core demographic segments
   - Call-to-Action — encourages interaction naturally

3. Viral Mechanics (0-1):
   - Hook Strength — compelling opening that stops scrolling
   - Share-worthiness — readers likely to repost/quote
   - Discussion Potential — sparks conversation threads

4. Cultural Resonance (0-1):
   - Trend Alignment — fits current conversation climate
   - Reference Fluency — uses cultural touchstones effectively
   - Cross-Community Appeal — bridges multiple audience segments

5. Technical Optimization (0-1):
   - Length Optimization — uses space efficiently
   - Format Optimization — leverages platform features
   - Rich Media Hooks — references visuals/links effectively

Normalization Factors:
- Time Decay: Weight recent engagement patterns more heavily
- Impression Normalization: Adjust metrics by audience reach
- Audience Overlap: Account for community cross-pollination
- Platform Velocity: Factor in current platform activity levels

Rules:
- Normalize engagement expectations by time-of-day
- Account for audience size and overlap
- Consider platform-specific engagement patterns
- Evaluate share/reply ratio potential
- Factor in current trending topics and memes

Return:
- JSON with normalized criterion scores and final score (0–1)
- Feedback on optimization opportunities
- Specific suggestions for timing and audience targeting"""
            
            # Input fields
            information_sources = dspy.InputField(desc="Source information")
            generated_tweet = dspy.InputField(desc="Generated tweet")
            original_tweet = dspy.InputField(desc="Original viral tweet")
            
            # Engagement context fields
            post_time = dspy.InputField(desc="Target posting time (HH:MM format)")
            audience_size = dspy.InputField(desc="Estimated audience size")
            platform_activity = dspy.InputField(desc="Current platform activity level (0-1)")
            trending_topics = dspy.InputField(desc="Current trending topics and hashtags")
            
            # Output fields
            score = dspy.OutputField(desc="Normalized score between 0 and 1")
            feedback = dspy.OutputField(desc="Detailed feedback for improvement")
            timing_guidance = dspy.OutputField(desc="Specific timing and audience targeting suggestions")
        
        self.evaluation_prompt = TweetEvaluationSignature
        self.evaluator = dspy.Predict(self.evaluation_prompt)
        print("[TweetGenerationMetric] Initialized GEPA-compatible metric")
    
    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
        """Evaluate a generated tweet and return score + feedback
        
        Args:
            gold: Input example with information_context and original_tweet (gold standard)
            pred: Generated tweet prediction
            trace: Optional execution trace
            pred_name: Name of the predictor (for GEPA)
            pred_trace: Predictor trace (for GEPA)
        
        Returns:
            float score between 0 and 1
        """
        
        # Extract fields using getattr with consistent fallbacks
        information_context = ''
        original_tweet = ''
        
        # First try direct attribute access
        if hasattr(gold, 'information_context'):
            information_context = getattr(gold, 'information_context')
        elif hasattr(gold, 'inputs'):
            # Try inputs.information_context
            inputs = getattr(gold, 'inputs')
            if hasattr(inputs, 'information_context'):
                information_context = getattr(inputs, 'information_context')
            elif isinstance(inputs, dict):
                information_context = inputs.get('information_context', '')
        elif isinstance(gold, dict):
            # Handle dictionary input
            information_context = gold.get('information_context', gold.get('inputs', {}).get('information_context', ''))
        else:
            # Last resort fallback
            information_context = str(gold)
            
        # Get original tweet with getattr
        original_tweet = getattr(gold, 'original_tweet', '')
        
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
            
            # Get current time and platform context
            from datetime import datetime
            current_time = datetime.now().strftime("%H:%M")
            
            # Get engagement context from gold example if available
            audience_size = getattr(gold, 'audience_size', '100000')  # Default audience size
            platform_activity = getattr(gold, 'platform_activity', '0.5')  # Default activity
            trending_topics = getattr(gold, 'trending_topics', '')  # Default empty
            
            # Evaluate with engagement context
            with dspy.context(lm=self.judge_lm):
                evaluation = self.evaluator(
                    information_sources=information_context,
                    generated_tweet=generated_tweet,
                    original_tweet=original_tweet,
                    post_time=current_time,
                    audience_size=str(audience_size),
                    platform_activity=str(platform_activity),
                    trending_topics=trending_topics
                )
            
            # Parse normalized score
            try:
                raw_score = float(evaluation.score)
                
                # Apply time-of-day normalization
                hour = int(current_time.split(":")[0])
                # Higher weights during peak hours (8-22)
                time_factor = 1.0 if 8 <= hour <= 22 else 0.7
                
                # Apply audience size normalization
                try:
                    audience = float(audience_size)
                    # Log scale normalization
                    import math
                    audience_factor = min(1.0, math.log10(audience) / math.log10(1000000))
                except:
                    audience_factor = 0.5
                
                # Apply platform activity normalization
                try:
                    activity = float(platform_activity)
                    activity_factor = min(1.0, max(0.1, activity))
                except:
                    activity_factor = 0.5
                
                # Combine normalizations
                score = raw_score * time_factor * audience_factor * activity_factor
                
                # Ensure final score is between 0 and 1
                score = max(0.0, min(1.0, score))
                
            except (ValueError, TypeError):
                score = 0.2  # Default low score for failed evaluations
            
            # Format feedback with timing guidance
            feedback = evaluation.feedback
            if hasattr(evaluation, 'timing_guidance'):
                feedback += f"\n\nTiming Guidance: {evaluation.timing_guidance}"
            
            # Add specific insights based on common issues
            if score < 0.2:
                feedback += "\n\nKey issues to address:"
                if "information" in feedback.lower():
                    feedback += "\n- Focus on extracting the most newsworthy elements"
                if "style" in feedback.lower():
                    feedback += "\n- Add emotional hooks and controversy"
                if "engagement" in feedback.lower():
                    feedback += "\n- Start with a compelling hook"
            
            # Get module score if available
            module_score = None
            if pred_trace and isinstance(pred_trace, dict):
                module_score = pred_trace.get('score')
            
            # Use module score if available, otherwise use our computed score
            final_score = module_score if module_score is not None else score
            
            # Create feedback object compatible with GEPA
            feedback_obj = dspy.Prediction(
                score=final_score,
                feedback=feedback,
                normalized_score=score,  # Keep original score for reference
                computed_score=score  # Store our computed score
            )
            
            # Store feedback in trace if available and it's a dict
            if trace is not None and isinstance(trace, dict):
                trace.update({
                    'score': final_score,
                    'feedback': feedback,
                    'normalized_score': score,
                    'computed_score': score
                })
            elif trace is not None:
                print(f"[GEPA Metric] Warning: trace must be a dict to store feedback")
            
            # Log everything as per user preference
            print(f"[GEPA Metric] Score: {final_score:.3f} (computed: {score:.3f}) | Feedback: {feedback[:100]}...")
            
            return feedback_obj
        
        except Exception as e:
            print(f"[GEPA Metric] Error during evaluation: {e}")
            error_feedback = f"Evaluation failed: {str(e)}. Consider improving tweet structure."
            
            # Get module score if available
            module_score = None
            if pred_trace and isinstance(pred_trace, dict):
                module_score = pred_trace.get('score', 0.0)
            
            # Use module score if available, otherwise use 0.0
            final_score = module_score if module_score is not None else 0.0
            
            # Create error feedback object
            error_obj = dspy.Prediction(
                score=final_score,
                feedback=error_feedback,
                normalized_score=0.0,
                computed_score=0.0
            )
            
            # Store error feedback in trace
            if trace is not None and isinstance(trace, dict):
                trace.update({
                    'score': final_score,
                    'feedback': error_feedback,
                    'normalized_score': 0.0,
                    'computed_score': 0.0
                })
            
            return error_obj
        
    def __del__(self):
        """Cleanup thread pool on deletion"""
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            print(f"[TweetGenerationMetric] Warning: Error during thread pool shutdown: {e}")
            pass
    
    def score_variants(self, gold, variants, task_id=None):
        """Score multiple tweet variants and track the best ones
        
        Args:
            gold: Input example with context
            variants: List of variant dicts with 'tweet' and 'variant_id'
            task_id: Optional identifier for tracking best variants
            
        Returns:
            List of (score, variant_id, tweet) tuples, sorted by score
        """
        scores = []
        futures = []
        
        def score_single_variant(variant):
            """Helper function to score a single variant"""
            tweet = variant['tweet']
            variant_id = variant.get('variant_id', 'v0')
            
            # Create prediction object
            pred = dspy.Prediction(generated_tweet=tweet)
            
            try:
                # Score the variant
                result = self(gold, pred)
                # Extract numeric score from the feedback object
                score = float(result.score) if hasattr(result, 'score') else 0.0
                return score, variant_id, tweet
            except Exception as e:
                print(f"[TweetGenerationMetric] Failed to score variant {variant_id}: {e}")
                return 0.0, variant_id, tweet
        
        try:
            # Submit scoring tasks to thread pool
            for variant in variants:
                future = self._executor.submit(score_single_variant, variant)
                futures.append(future)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    score, variant_id, tweet = future.result()
                    scores.append((score, variant_id, tweet))
                except Exception as e:
                    print(f"[TweetGenerationMetric] Error collecting variant score: {e}")
                    continue
            
        except Exception as e:
            print(f"[TweetGenerationMetric] Error during parallel scoring: {e}")
            # Fallback to sequential scoring
            for variant in variants:
                result = score_single_variant(variant)
                scores.append(result)
        
        # Sort by score descending
        scores.sort(reverse=True)
        
        # Track best variants if task_id provided
        if task_id is not None:
            self.best_variants[task_id] = scores
            
            # Log best variant
            if scores:
                best_score, best_id, best_tweet = scores[0]
                print(f"[TweetGenerationMetric] Best variant for task {task_id}:")
                print(f"  Score: {best_score:.3f}")
                print(f"  ID: {best_id}")
                print(f"  Tweet: {best_tweet[:100]}...")
        
        return scores

class TweetGenerationSignature(dspy.Signature):
    """Signature for tweet generation from information sources and media analysis"""
    
    information_context = dspy.InputField(
        desc="Source tweets, articles, media analysis, and additional information to base the tweet on. Media analysis will be prefixed with [Media Analysis] if present."
    )
    generated_tweet = dspy.OutputField(
        desc="A viral tweet (max 280 chars) that synthesizes key insights and leverages media analysis when available"
    )

class TweetGeneratorModule(dspy.Module):
    """DSPy module for tweet generation compatible with GEPA with structured generation and tool calling"""
    
    # Define available hook styles
    HOOK_STYLES = {
        'question': 'Start with an intriguing question',
        'statistic': 'Lead with a surprising statistic or number',
        'contrast': 'Present a striking contrast or contradiction',
        'challenge': 'Challenge a common assumption',
        'quote': 'Begin with a relevant quote or statement',
        'scenario': 'Paint a brief scenario or situation'
    }
    
    # Define tweet structure templates
    STRUCTURES = {
        'hook_evidence_punchline': 'Start with a hook, present key evidence, end with impact',
        'claim_support_call': 'Make a claim, support it, call to action',
        'problem_solution_twist': 'Present problem, offer solution, add unexpected twist',
        'setup_context_reveal': 'Set up context, build tension, reveal insight',
        'compare_contrast_conclude': 'Compare perspectives, highlight contrast, conclude'
    }
    
    def __init__(self, 
                 hook_style: str = None,
                 structure: str = None,
                 evidence_density: float = 0.5,
                 num_candidates: int = 3,
                 temperature: float = 0.7,
                 seed: int = 42,
                 enable_tool_calling: bool = True):
        """Initialize tweet generator with structured parameters
        
        Args:
            hook_style: Style of opening hook (from HOOK_STYLES)
            structure: Tweet structure template (from STRUCTURES)
            evidence_density: How much evidence to include (0.0-1.0)
            num_candidates: Number of candidates to generate
            temperature: Generation temperature for variety
            seed: Random seed for deterministic generation
            enable_tool_calling: Whether to enable web search tool calling
        """
        super().__init__()
        
        # Store generation parameters
        self.hook_style = hook_style
        self.structure = structure
        self.evidence_density = max(0.0, min(1.0, evidence_density))
        self.num_candidates = max(1, num_candidates)
        self.temperature = temperature
        self.seed = seed
        self.enable_tool_calling = enable_tool_calling and TOOL_CALLING_AVAILABLE
        
        # Initialize tool handler if enabled
        self.tool_handler = None
        if self.enable_tool_calling:
            try:
                self.tool_handler = ExaToolCall()
                print(f"[TweetGenerator] Tool calling enabled")
            except Exception as e:
                print(f"[TweetGenerator] Tool calling setup failed: {e}")
                self.enable_tool_calling = False
        
        # Set random seed for determinism
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize generation stats
        self.generation_stats = {
            'total_generations': 0,
            'total_variants': 0,
            'parameter_history': []
        }
        
        # Create signature with structured guidance and optional tool calling
        if self.enable_tool_calling:
            class ToolCallingTweetSignature(TweetGenerationSignature):
                """Enhanced signature with structural guidance and tool calling"""
                
                # Add fields for generation guidance
                hook_guidance = dspy.InputField(desc="Style guidance for the opening hook")
                structure_template = dspy.InputField(desc="Overall tweet structure template")
                evidence_guidance = dspy.InputField(desc="Guidance on evidence density")
                tool_instructions = dspy.InputField(desc="Instructions for when to use web search tools")
                
                # Add tool calling outputs
                tool_calls = dspy.OutputField(desc="JSON array of tool calls made (empty array if none)")
                reasoning = dspy.OutputField(desc="Strategic reasoning including tool usage decisions")
            
            self.generate = dspy.ChainOfThought(ToolCallingTweetSignature)
        else:
            class StructuredTweetSignature(TweetGenerationSignature):
                """Enhanced signature with structural guidance"""
                
                # Add fields for generation guidance
                hook_guidance = dspy.InputField(desc="Style guidance for the opening hook")
                structure_template = dspy.InputField(desc="Overall tweet structure template")
                evidence_guidance = dspy.InputField(desc="Guidance on evidence density")
            
            self.generate = dspy.ChainOfThought(StructuredTweetSignature)
        
        # Log configuration
        print(f"[TweetGenerator] Initialized with:")
        print(f"  Hook style: {hook_style or 'default'}")
        print(f"  Structure: {structure or 'default'}")
        print(f"  Evidence density: {evidence_density:.2f}")
        print(f"  Candidates: {num_candidates}")
        print(f"  Tool calling: {'enabled' if self.enable_tool_calling else 'disabled'}")
    
    def _prepare_guidance(self):
        """Prepare structured guidance for generation"""
        
        # Format hook guidance
        if self.hook_style and self.hook_style in self.HOOK_STYLES:
            hook_guidance = f"Hook style: {self.HOOK_STYLES[self.hook_style]}"
        else:
            hook_guidance = "Use an engaging opening hook"
        
        # Format structure guidance
        if self.structure and self.structure in self.STRUCTURES:
            structure_guidance = f"Structure: {self.STRUCTURES[self.structure]}"
        else:
            structure_guidance = "Use a clear and engaging structure"
        
        # Format evidence guidance based on density
        if self.evidence_density <= 0.3:
            evidence_guidance = "Focus on high-level insights, minimal raw evidence"
        elif self.evidence_density <= 0.7:
            evidence_guidance = "Balance key evidence with engaging presentation"
        else:
            evidence_guidance = "Include detailed evidence and supporting facts"
        
        return hook_guidance, structure_guidance, evidence_guidance
    
    def _format_search_results(self, tool_result: dict) -> str:
        """Format search results for context enhancement"""
        
        if not tool_result.get('results'):
            return "No search results found."
        
        formatted_results = []
        for i, result in enumerate(tool_result['results'][:3], 1):  # Top 3 results
            title = result.get('title', 'Untitled')
            summary = result.get('summary', '')[:150] + "..." if result.get('summary') else ""
            
            formatted_result = f"{i}. {title}"
            if summary:
                formatted_result += f" - {summary}"
            
            formatted_results.append(formatted_result)
        
        return "\n".join(formatted_results)
    
    def forward(self, information_context):
        """Generate multiple tweet candidates with structured guidance and optional tool calling"""
        
        # Prepare structural guidance
        hook_guidance, structure_guidance, evidence_guidance = self._prepare_guidance()
        
        # Set deterministic seed for this generation
        import random
        import numpy as np
        generation_seed = self.seed + self.generation_stats['total_generations']
        random.seed(generation_seed)
        np.random.seed(generation_seed)
        
        candidates = []
        for i in range(self.num_candidates):
            # Set variant-specific seed
            variant_seed = generation_seed + i
            random.seed(variant_seed)
            np.random.seed(variant_seed)
            
            # Generate with structural guidance and optional tool calling
            with dspy.settings.context(temperature=self.temperature):
                if self.enable_tool_calling:
                    # Add tool calling instructions
                    tool_instructions = """You have access to web search tools. If you need current information, recent news, or additional context to create a better tweet, you can call the search_web function.

Available Tools:
- search_web(query, num_results=5, category="general", recent_only=true): Search the web for current information
  * Use category="news" for current events and breaking news
  * Use category="research paper" for studies and academic content
  * Use category="company" for business and corporate information

Use tools when:
- The information context seems outdated or lacks recent details
- You need to verify facts or get current statistics
- The topic would benefit from recent news or developments
- You want to add timely context or trending information"""

                    result = self.generate(
                        information_context=information_context,
                        hook_guidance=hook_guidance,
                        structure_template=structure_guidance,
                        evidence_guidance=evidence_guidance,
                        tool_instructions=tool_instructions
                    )
                    
                    # Process tool calls if any
                    tool_calls_made = []
                    enhanced_context = information_context
                    
                    print(f"[TweetGenerator] Checking for tool calls in result...")
                    
                    try:
                        import json
                        tool_calls_raw = result.tool_calls if hasattr(result, 'tool_calls') else "[]"
                        
                        # Handle empty or invalid tool calls
                        if not tool_calls_raw or tool_calls_raw.strip() == "":
                            print(f"[TweetGenerator] No tool calls found (empty or None)")
                            tool_calls_data = []
                        elif isinstance(tool_calls_raw, str):
                            # Clean and validate JSON string
                            tool_calls_raw = tool_calls_raw.strip()
                            if tool_calls_raw.startswith('[') and tool_calls_raw.endswith(']'):
                                try:
                                    tool_calls_data = json.loads(tool_calls_raw)
                                    print(f"[TweetGenerator] Parsed tool calls: {len(tool_calls_data)} calls found")
                                    if tool_calls_data:
                                        print(f"[TweetGenerator] Tool calls content: {tool_calls_raw[:200]}...")
                                except json.JSONDecodeError:
                                    print(f"[TweetGenerator] JSON parsing failed, trying regex extraction")
                                    # Try to extract JSON from text
                                    import re
                                    json_match = re.search(r'\[.*\]', tool_calls_raw, re.DOTALL)
                                    if json_match:
                                        tool_calls_data = json.loads(json_match.group())
                                        print(f"[TweetGenerator] Extracted {len(tool_calls_data)} tool calls via regex")
                                    else:
                                        print(f"[TweetGenerator] No valid JSON array found")
                                        tool_calls_data = []
                            else:
                                print(f"[TweetGenerator] Tool calls string doesn't look like JSON array: {tool_calls_raw[:100]}...")
                                tool_calls_data = []
                        elif isinstance(tool_calls_raw, list):
                            tool_calls_data = tool_calls_raw
                            print(f"[TweetGenerator] Tool calls already a list: {len(tool_calls_data)} calls")
                        else:
                            print(f"[TweetGenerator] Unexpected tool calls type: {type(tool_calls_raw)}")
                            tool_calls_data = []
                        
                        # Execute tool calls (limit to 2 for GEPA efficiency)
                        if tool_calls_data:
                            print(f"[TweetGenerator] Processing {len(tool_calls_data[:2])} tool calls...")
                        else:
                            print(f"[TweetGenerator] No valid tool calls to process")
                            
                        for i, tool_call in enumerate(tool_calls_data[:2], 1):
                            # Handle both formats: {"function": "search_web", "arguments": {...}} and {"tool_name": "search_web", "parameters": {...}}
                            function_name = None
                            arguments = None
                            
                            if isinstance(tool_call, dict):
                                if 'function' in tool_call and 'arguments' in tool_call:
                                    function_name = tool_call['function']
                                    arguments = tool_call['arguments']
                                elif 'function' in tool_call and 'parameters' in tool_call:
                                    function_name = tool_call['function']
                                    arguments = tool_call['parameters']
                                    print(f"[TweetGenerator] Converting function/parameters format to function/arguments")
                                elif 'tool_name' in tool_call and 'parameters' in tool_call:
                                    function_name = tool_call['tool_name']
                                    arguments = tool_call['parameters']
                                    print(f"[TweetGenerator] Converting tool_name/parameters format to function/arguments")
                                elif 'tool' in tool_call:
                                    # Handle {"tool": "search_web", "query": "...", "num_results": 3, ...} format
                                    function_name = tool_call['tool']
                                    # Extract all other keys as arguments
                                    arguments = {k: v for k, v in tool_call.items() if k != 'tool'}
                                    print(f"[TweetGenerator] Converting tool format to function/arguments")
                            
                            if function_name and arguments and function_name == "search_web" and isinstance(arguments, dict):
                                    print(f"[TweetGenerator] ✅ Executing tool call {i}: {function_name}")
                                    print(f"[TweetGenerator] Arguments: {arguments}")
                                    
                                    tool_result = self.tool_handler.execute_tool_call(function_name, arguments)
                                    tool_calls_made.append({
                                        'function': function_name,
                                        'arguments': arguments,
                                        'result': tool_result
                                    })
                                    
                                    # Enhance context with search results
                                    if not tool_result.get('error') and tool_result.get('results'):
                                        search_context = self._format_search_results(tool_result)
                                        enhanced_context = f"{information_context}\n\nSearch Results: {search_context}"
                                        print(f"[TweetGenerator] ✅ Enhanced context with {len(tool_result['results'])} search results")
                                    else:
                                        print(f"[TweetGenerator] ❌ Tool call failed or returned no results: {tool_result.get('error', 'No results')}")
                            else:
                                if function_name:
                                    print(f"[TweetGenerator] ❌ Skipping invalid tool call {i}: function='{function_name}', args_type={type(arguments)}")
                                else:
                                    print(f"[TweetGenerator] ❌ Skipping malformed tool call {i}: {tool_call}")
                    
                    except Exception as e:
                        print(f"[TweetGenerator] Tool call processing failed: {e}")
                        # Continue without tool calls
                    
                    # If we got enhanced context, regenerate
                    if enhanced_context != information_context and tool_calls_made:
                        print(f"[TweetGenerator] 🔄 Regenerating tweet with enhanced context ({len(tool_calls_made)} tool calls made)")
                        result = self.generate(
                            information_context=enhanced_context,
                            hook_guidance=hook_guidance,
                            structure_template=structure_guidance,
                            evidence_guidance=evidence_guidance,
                            tool_instructions="Use the search results provided in the context."
                        )
                        print(f"[TweetGenerator] ✅ Regeneration complete")
                    else:
                        if tool_calls_made:
                            print(f"[TweetGenerator] ⚠️  Tool calls made but context unchanged")
                        else:
                            print(f"[TweetGenerator] ℹ️  No tool calls made, using original generation")
                else:
                    result = self.generate(
                        information_context=information_context,
                        hook_guidance=hook_guidance,
                        structure_template=structure_guidance,
                        evidence_guidance=evidence_guidance
                    )
                    tool_calls_made = []
            
            # Ensure 280 character limit
            tweet = result.generated_tweet
            if len(tweet) > 280:
                tweet = tweet[:277] + "..."
            
            # Create variant with detailed metadata
            variant = {
                'tweet': tweet,
                'variant_id': f"v{i+1}",
                'params': {
                    'hook_style': self.hook_style,
                    'structure': self.structure,
                    'evidence_density': self.evidence_density,
                    'temperature': self.temperature,
                    'generation_seed': generation_seed,
                    'variant_seed': variant_seed
                },
                'tool_calls': tool_calls_made,
                'reasoning': getattr(result, 'reasoning', '') if self.enable_tool_calling else ''
            }
            candidates.append(variant)
            
            # Track variant stats
            self.generation_stats['total_variants'] += 1
        
        # Update generation stats
        self.generation_stats['total_generations'] += 1
        self.generation_stats['parameter_history'].append({
            'generation_id': self.generation_stats['total_generations'],
            'generation_seed': generation_seed,
            'num_variants': len(candidates),
            'hook_style': self.hook_style,
            'structure': self.structure,
            'evidence_density': self.evidence_density,
            'temperature': self.temperature
        })
        
        # Log generation stats
        print(f"\n[TweetGenerator] Generation {self.generation_stats['total_generations']}:")
        print(f"  Seed: {generation_seed}")
        print(f"  Variants: {len(candidates)}")
        print(f"  Total variants generated: {self.generation_stats['total_variants']}")
        
        # Return all candidates
        return dspy.Prediction(
            generated_tweet=candidates[0]['tweet'],  # Primary candidate
            all_candidates=candidates,  # All variants
            generation_stats=self.generation_stats  # Include stats
        )

class GEPATweetOptimizer:
    """Wrapper for official DSPy GEPA optimizer for tweet generation
    
    This optimizer uses GEPA (Guided Evolution with Prompt Adaptation) to evolve and optimize
    prompts for tweet generation. It includes features for tracking and saving intermediate
    prompts during the optimization process.
    
    Key Features:
    - Tracks and saves intermediate best prompts during optimization
    - Maintains a history of prompt evolution with scores and timestamps
    - Generates a summary of all intermediate prompts at the end
    - Saves prompts in an 'intermediate_prompts' directory within the log directory
    
    The intermediate prompts are saved whenever a new best score is achieved, allowing you
    to analyze the evolution of the prompt and potentially choose from different versions
    based on your specific needs.
    """
    
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
        
        This method runs the GEPA optimization process, evolving the prompt to improve
        tweet generation performance. During optimization, it saves intermediate prompts
        whenever a new best score is achieved.
        
        Args:
            student_module: DSPy module to optimize
            trainset: Training examples
            valset: Validation examples (if None, uses trainset)
            max_generations: Maximum full evaluations
            max_metric_calls: Maximum metric calls
            auto: Automatic configuration ('light', 'medium', 'heavy')
            track_stats: Whether to track detailed statistics
            log_dir: Directory for logs. Will contain:
                    - optimized_prompt.txt: Final best prompt
                    - intermediate_prompts/: Directory containing:
                        - prompt_[timestamp]_score_[score].txt: Individual prompt files
                        - summary.txt: Overview of all intermediate prompts
        
        Returns:
            Optimized DSPy module
            
        The intermediate prompts are saved in the log_dir/intermediate_prompts directory,
        with each file containing the prompt text, score, and improvement metrics. A
        summary file is also generated listing all intermediate prompts sorted by score.
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
        
        # Configure GEPA optimizer with version-safe core parameters
        print(f"\n[GEPATweetOptimizer] Starting GEPA optimization")
        
        # Define core configuration that works across versions
        core_config = {
            'metric': self.metric,
            'seed': 42,  # Always set for determinism
            'log_dir': log_dir or f"gepa_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'track_stats': track_stats
        }
        
        # Add optimization mode parameters
        if auto:
            core_config['auto'] = auto
            print(f"[GEPATweetOptimizer] Mode: {auto} (auto-configured)")
        elif max_metric_calls:
            core_config['max_metric_calls'] = max_metric_calls
            print(f"[GEPATweetOptimizer] Max metric calls: {max_metric_calls}")
        else:
            dataset_size = len(trainset) + len(valset)
            core_config['max_full_evals'] = max_generations
            print(f"[GEPATweetOptimizer] Using explicit generation limit: {max_generations} full evaluations")
            print(f"[GEPATweetOptimizer] This will run approximately {max_generations * dataset_size} metric calls")
        
        # Try to create GEPA with advanced features first
        try:
            advanced_config = {
                **core_config,
                'reflection_minibatch_size': 3,
                'candidate_selection_strategy': 'pareto',
                'reflection_lm': self.reflection_lm,
                'skip_perfect_score': True,
                'add_format_failure_as_feedback': True,
                'use_merge': True,
                'max_merge_invocations': 5,
                'failure_score': 0.0,
                'perfect_score': 1.0,
                'track_best_outputs': True
            }
            
            # Try creating with advanced config
            gepa_optimizer = GEPA(**advanced_config)
            print("[GEPATweetOptimizer] Using advanced GEPA configuration with Pareto selection and merging")
            
        except (TypeError, ValueError) as e:
            # Fall back to core config if advanced features aren't supported
            print(f"[GEPATweetOptimizer] Advanced features not supported, using core configuration: {str(e)}")
            try:
                gepa_optimizer = GEPA(**core_config)
                print("[GEPATweetOptimizer] Using core GEPA configuration")
            except (TypeError, ValueError) as e2:
                # Ultimate fallback for very old versions
                console.print(f"[yellow]Using legacy compatibility mode for GEPA: {e2}[/yellow]")
                gepa_optimizer = GEPA(
                    metric=self.metric,
                    max_bootstrapped_demos=max_generations,
                    max_labeled_demos=max_generations,
                    num_candidates=3
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
                    # Generate variants with baseline module
                    if hasattr(example, 'information_context'):
                        context = example.information_context
                    elif isinstance(example, dict) and 'information_context' in example:
                        context = example['information_context']
                    else:
                        # Fallback: use string representation
                        context = str(example)
                    
                    # Generate multiple variants
                    prediction = student_module(information_context=context)
                    
                    if hasattr(prediction, 'all_candidates'):
                        # Score all variants
                        variant_scores = self.metric.score_variants(
                            gold=example,
                            variants=prediction.all_candidates,
                            task_id=f"baseline_{i}"
                        )
                        # Use best variant's score
                        if variant_scores:
                            # Extract numeric score from first variant
                            score = float(variant_scores[0][0])  # First score from best variant
                        else:
                            score = 0.2
                    else:
                        # Fallback to single prediction scoring
                        result = self.metric(
                            gold=example,
                            pred=prediction,
                            trace=None,
                            pred_name=None,
                            pred_trace=None
                        )
                        score = float(result.score) if hasattr(result, 'score') else 0.2
                    
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
        
        # Import required modules
        import os
        from datetime import datetime
        
        # Capture GEPA's internal logs to extract scores AND evolved prompts
        class OptimizationCapture:
            def __init__(self, baseline=0.3, log_dir=None):
                self.best_score = 0.0
                self.avg_scores = []
                self.iteration_scores = []
                self.baseline = baseline  # Store baseline
                self.evolved_prompt = None
                self.capturing_prompt = False
                self.prompt_lines = []
                self.log_dir = log_dir
                self.intermediate_prompts = []  # List to store (timestamp, score, prompt) tuples
                
            def save_intermediate_prompt(self, prompt_text, score):
                """Save intermediate prompt with timestamp and score"""
                if not self.log_dir:
                    return
                    
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.intermediate_prompts.append((timestamp, score, prompt_text))
                
                # Create intermediate prompts directory
                intermediate_dir = os.path.join(self.log_dir, "intermediate_prompts")
                os.makedirs(intermediate_dir, exist_ok=True)
                
                # Save this prompt
                prompt_path = os.path.join(intermediate_dir, f"prompt_{timestamp}_score_{score:.3f}.txt")
                with open(prompt_path, 'w') as f:
                    f.write(f"=== INTERMEDIATE GEPA PROMPT ===\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Score: {score:.3f}\n")
                    f.write(f"Baseline: {self.baseline:.3f}\n")
                    f.write(f"Improvement: +{(score - self.baseline):.3f}\n\n")
                    f.write(prompt_text)
                
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
                        # If this is a new best score, save the prompt
                        if score > self.best_score:
                            self.best_score = score
                            # If we have a prompt being captured, save it
                            if self.prompt_lines:
                                current_prompt = "\n".join(self.prompt_lines)
                                self.save_intermediate_prompt(current_prompt, score)
                
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
        
        optimization_capture = OptimizationCapture(baseline=baseline_score, log_dir=log_dir)
        
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
                
                # Save summary of intermediate prompts
                if optimization_capture.intermediate_prompts:
                    intermediate_dir = os.path.join(log_dir, "intermediate_prompts")
                    summary_path = os.path.join(intermediate_dir, "summary.txt")
                    with open(summary_path, 'w') as f:
                        f.write("=== INTERMEDIATE PROMPTS SUMMARY ===\n\n")
                        f.write(f"Total intermediate prompts saved: {len(optimization_capture.intermediate_prompts)}\n")
                        f.write(f"Baseline score: {baseline_score:.3f}\n\n")
                        
                        # Sort by score
                        sorted_prompts = sorted(optimization_capture.intermediate_prompts, key=lambda x: x[1], reverse=True)
                        
                        for i, (timestamp, score, _) in enumerate(sorted_prompts, 1):
                            f.write(f"{i}. Timestamp: {timestamp} | Score: {score:.3f} | ")
                            f.write(f"Improvement: +{(score - baseline_score):.3f}\n")
                        
                        print(f"[GEPATweetOptimizer] Saved intermediate prompts summary to: {summary_path}")
                        print(f"[GEPATweetOptimizer] {len(optimization_capture.intermediate_prompts)} intermediate prompts saved")
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

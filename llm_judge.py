import dspy
from typing import Dict, List, Tuple
import json
import re

class TweetJudgeSignature(dspy.Signature):
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
- 2–3 sentences of feedback on what worked and what to improve.
    """
    
    information_sources: str = dspy.InputField(
        desc="The source information that was used to generate the tweet"
    )
    generated_tweet: str = dspy.InputField(
        desc="The tweet that was generated from the information"
    )
    original_tweet: str = dspy.InputField(
        desc="The original viral tweet for comparison"
    )
    evaluation: str = dspy.OutputField(
        desc="JSON string with scores and reasoning: {\"information_coverage\": {\"score\": X, \"reasoning\": \"...\"}, ...}"
    )

class LLMJudge(dspy.Module):
    """LLM-based judge for evaluating tweet quality"""
    
    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(TweetJudgeSignature)
        print("[LLMJudge] Initialized with evaluation criteria")
    
    def forward(self, information_sources: str, generated_tweet: str, original_tweet: str):
        """Evaluate a generated tweet"""
        
        result = self.judge(
            information_sources=information_sources,
            generated_tweet=generated_tweet,
            original_tweet=original_tweet
        )
        
        # Parse the evaluation JSON
        try:
            evaluation = json.loads(result.evaluation)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            print("[LLMJudge] Warning: Failed to parse evaluation JSON, using defaults")
            evaluation = self._create_default_evaluation()
        
        # Calculate overall score if not present
        if 'overall' not in evaluation:
            scores = [v.get('score', 5) for k, v in evaluation.items() if isinstance(v, dict) and 'score' in v]
            overall_score = sum(scores) / len(scores) if scores else 5
            evaluation['overall'] = {'score': overall_score, 'reasoning': 'Averaged from component scores'}
        
        return dspy.Prediction(
            evaluation=evaluation,
            overall_score=evaluation['overall']['score']
        )
    
    def _create_default_evaluation(self):
        """Create default evaluation structure"""
        return {
            'information_coverage': {'score': 5, 'reasoning': 'Unable to evaluate'},
            'style_match': {'score': 5, 'reasoning': 'Unable to evaluate'},
            'originality': {'score': 5, 'reasoning': 'Unable to evaluate'},
            'engagement_potential': {'score': 5, 'reasoning': 'Unable to evaluate'},
            'overall': {'score': 5, 'reasoning': 'Default evaluation due to parsing error'}
        }

class ComparativeJudge(dspy.Module):
    """Judge that compares multiple generated tweets"""
    
    def __init__(self):
        super().__init__()
        self.judge = LLMJudge()
        print("[ComparativeJudge] Initialized for multi-tweet comparison")
    
    def evaluate_batch(self, tweets_data: List[Dict]) -> List[Dict]:
        """Evaluate a batch of generated tweets
        
        Args:
            tweets_data: List of dicts with keys:
                - information_sources: str
                - generated_tweet: str  
                - original_tweet: str
                - metadata: dict (optional)
        
        Returns:
            List of evaluation results with scores and rankings
        """
        evaluations = []
        
        for i, tweet_data in enumerate(tweets_data):
            print(f"[ComparativeJudge] Evaluating tweet {i+1}/{len(tweets_data)}")
            
            result = self.judge(
                information_sources=tweet_data['information_sources'],
                generated_tweet=tweet_data['generated_tweet'],
                original_tweet=tweet_data['original_tweet']
            )
            
            eval_result = {
                'index': i,
                'generated_tweet': tweet_data['generated_tweet'],
                'evaluation': result.evaluation,
                'overall_score': result.overall_score,
                'metadata': tweet_data.get('metadata', {})
            }
            evaluations.append(eval_result)
        
        # Sort by overall score
        evaluations.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Add rankings
        for rank, eval_result in enumerate(evaluations, 1):
            eval_result['rank'] = rank
        
        return evaluations
    
    def get_best_tweet(self, evaluations: List[Dict]) -> Dict:
        """Get the best tweet from evaluations"""
        if not evaluations:
            return None
        return evaluations[0]  # Already sorted by score
    
    def get_statistics(self, evaluations: List[Dict]) -> Dict:
        """Calculate statistics from evaluations"""
        if not evaluations:
            return {}
        
        scores = [e['overall_score'] for e in evaluations]
        
        # Component scores
        component_scores = {}
        for component in ['information_coverage', 'style_match', 'originality', 'engagement_potential']:
            comp_scores = []
            for e in evaluations:
                if component in e['evaluation']:
                    comp_scores.append(e['evaluation'][component].get('score', 0))
            if comp_scores:
                component_scores[component] = {
                    'mean': sum(comp_scores) / len(comp_scores),
                    'max': max(comp_scores),
                    'min': min(comp_scores)
                }
        
        return {
            'overall': {
                'mean': sum(scores) / len(scores),
                'max': max(scores),
                'min': min(scores),
                'std': self._calculate_std(scores)
            },
            'components': component_scores,
            'total_evaluated': len(evaluations)
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

class PromptFeedback:
    """Generate feedback for prompt improvement based on evaluations"""
    
    def __init__(self):
        self.feedback_history = []
        print("[PromptFeedback] Initialized feedback generator")
    
    def analyze_failures(self, evaluations: List[Dict], threshold: float = 6.0) -> Dict:
        """Analyze poor performing tweets to identify patterns"""
        
        poor_tweets = [e for e in evaluations if e['overall_score'] < threshold]
        
        if not poor_tweets:
            return {'status': 'All tweets performed well', 'issues': []}
        
        # Analyze common issues
        issues = {
            'information_coverage': [],
            'style_match': [],
            'originality': [],
            'engagement_potential': []
        }
        
        for tweet in poor_tweets:
            eval_data = tweet['evaluation']
            for component, data in eval_data.items():
                if component != 'overall' and isinstance(data, dict):
                    if data.get('score', 10) < 6:
                        issues[component].append({
                            'score': data['score'],
                            'reasoning': data.get('reasoning', ''),
                            'tweet': tweet['generated_tweet'][:100] + '...'
                        })
        
        # Summarize issues
        summary = {}
        for component, issue_list in issues.items():
            if issue_list:
                summary[component] = {
                    'count': len(issue_list),
                    'avg_score': sum(i['score'] for i in issue_list) / len(issue_list),
                    'sample_reasoning': issue_list[0]['reasoning'] if issue_list else ''
                }
        
        return {
            'status': f'Found {len(poor_tweets)} tweets below threshold',
            'threshold': threshold,
            'issues': summary,
            'poor_tweet_count': len(poor_tweets),
            'total_tweets': len(evaluations)
        }
    
    def generate_improvement_suggestions(self, analysis: Dict) -> List[str]:
        """Generate specific suggestions for prompt improvement"""
        
        suggestions = []
        
        if 'issues' not in analysis or not analysis['issues']:
            return ["Current prompts are performing well. Consider minor refinements only."]
        
        for component, data in analysis['issues'].items():
            if component == 'information_coverage' and data['count'] > 2:
                suggestions.append(
                    "Improve information extraction: Add instructions to identify and incorporate "
                    "the most newsworthy or surprising elements from sources."
                )
            
            elif component == 'style_match' and data['count'] > 2:
                suggestions.append(
                    "Enhance viral style: Include examples of hook patterns, emotional triggers, "
                    "and engagement techniques used in viral tweets."
                )
            
            elif component == 'originality' and data['count'] > 2:
                suggestions.append(
                    "Boost originality: Instruct to add unique analysis, predictions, or "
                    "contrarian perspectives rather than just summarizing."
                )
            
            elif component == 'engagement_potential' and data['count'] > 2:
                suggestions.append(
                    "Increase engagement potential: Focus on creating controversy, asking questions, "
                    "or presenting surprising connections between ideas."
                )
        
        # Add general suggestion based on overall performance
        if analysis.get('poor_tweet_count', 0) > len(analysis.get('issues', {})) * 0.3:
            suggestions.append(
                "Consider adding few-shot examples of high-performing tweets to the prompt."
            )
        
        self.feedback_history.append({
            'analysis': analysis,
            'suggestions': suggestions
        })
        
        return suggestions

import dspy
from typing import Dict, List, Tuple
import json
import re

class TweetJudgeSignature(dspy.Signature):
    """You are a cognitive analysis expert evaluating tweet generation quality.

GOAL
Evaluate how well a generated tweet demonstrates the strategic thinking, audience psychology, and engagement mechanics shown in expert cognitive analysis examples.

INPUTS
- TWEET: {{TWEET}}
- CONTEXT (optional; the info/cognitive analysis the tweet is based on): {{CONTEXT}}
- ANALYSIS_EXAMPLES (how experts reason about strategy/audience/engagement): {{ANALYSIS_EXAMPLES}}

VALIDATION (hard gate)
If no valid tweet was generated—i.e., empty/whitespace, meta-text/instructions/commentary instead of a tweet, or a block that clearly labels itself as “analysis”, “instructions”, “variant list”, or similar—return:
{
  "strategic_understanding": 0,
  "audience_psychology": 0,
  "engagement_mechanics": 0,
  "cultural_fluency": 0,
  "execution_quality": 0,
  "raw_score": 0,
  "penalties": [],
  "penalty_total": 0,
  "score": 0,
  "feedback": "No tweet generated."
}
and stop.

SCORING (0–1 each)
Score only if a valid tweet exists.
1) strategic_understanding — Does the tweet show a clear strategic angle anchored in the analysis (positioning, reframing, or insight beyond surface-level)?
2) audience_psychology — Does it speak to likely motivations/priors of the target audience (identity, status, curiosity, outrage/hope, insider knowledge)?
3) engagement_mechanics — Hook quality, concreteness, emotional triggers, rhythm, social proof cues; avoids limp CTA clichés.
4) cultural_fluency — Timeliness, references, and tone aligned with current platform culture; avoids dated/awkward phrasing.
5) execution_quality — Clear, concise, coherent; ≤280 chars.

Compute:
raw_score = mean([strategic_understanding, audience_psychology, engagement_mechanics, cultural_fluency, execution_quality])

PENALTIES (deduct from final score; include any that apply)
Initialize penalty_total = 0 and an array penalties = [].
Apply the following additive deductions, with a short reason for each applied item:

A) Overlength (>280 characters): deduct 0.30
- Condition: character_count(TWEET) > 280
- Record: {"type":"overlength","weight":0.30,"reason":"Tweet exceeds 280 characters."}

B) Disallowed special characters: deduct 0.10 per category, up to 0.30 max
- Categories to check (any occurrence):
  1. Hashtags: "#" anywhere
  2. Emojis: presence of emoji characters
  3. Prohibited sequences: "--" or "**" or "*" (asterisk anywhere)
- For each present category add {"type":"special_characters","weight":0.10,"reason":"Found <category>."}
- Cap combined special-character penalties at 0.30.

C) Generic/formulaic: deduct 0.20
- Condition: tweet reads as template-y, clichéd, or boilerplate; shallow restatement without angle; empty CTA (“Thoughts?”) or listicle-y filler.
- Record: {"type":"generic_formulaic","weight":0.20,"reason":"Lacks strategic depth or uses formulaic phrasing."}

D) Repeats context verbatim/near-verbatim: deduct 0.30
- Condition: high overlap with CONTEXT (e.g., ≥70% phrase overlap or semantically identical restatement without novel angle).
- Record: {"type":"context_repetition","weight":0.30,"reason":"Repeats input/context without new framing."}

Compute:
penalty_total = sum(weights), clipped to [0, 0.90]
final score = max(0, min(1, raw_score - penalty_total))

OUTPUT (JSON ONLY)
Return strictly JSON with these keys:
{
  "strategic_understanding": <float 0-1>,
  "audience_psychology": <float 0-1>,
  "engagement_mechanics": <float 0-1>,
  "cultural_fluency": <float 0-1>,
  "execution_quality": <float 0-1>,
  "raw_score": <float 0-1>,
  "penalties": [
    {"type":"overlength","weight":0.30,"reason":"..."},
    {"type":"special_characters","weight":0.10,"reason":"Found hashtag."},
    {"type":"generic_formulaic","weight":0.20,"reason":"..."},
    {"type":"context_repetition","weight":0.30,"reason":"..."}
    // include only those applied; may be empty []
  ],
  "penalty_total": <float 0-0.90>,
  "score": <float 0-1>,  // final score after penalties
  "feedback": "<2–3 sentences on cognitive alignment and strategy; mention the most material strengths and the biggest penalty hits with 1 concrete fix.>"
}

ADDITIONAL RULES
- Reward strategic thinking beyond surface-level; favor audience insight and cultural fluency.
- Penalize if tweet is not ≤280 chars, contains hashtags/emojis or the sequences --, **, *, is generic/formulaic, or merely repeats CONTEXT.
- Do not include any text outside the JSON object in your response.

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
            scores = [v.get('score', 0.2) for k, v in evaluation.items() if isinstance(v, dict) and 'score' in v]
            overall_score = sum(scores) / len(scores) if scores else 0.2
            evaluation['overall'] = {'score': overall_score, 'reasoning': 'Averaged from component scores'}
        
        return dspy.Prediction(
            evaluation=evaluation,
            overall_score=evaluation['overall']['score']
        )
    
    def _create_default_evaluation(self):
        """Create default evaluation structure for cognitive analysis"""
        return {
            'strategic_understanding': {'score': 0.2, 'reasoning': 'Unable to evaluate'},
            'audience_psychology': {'score': 0.2, 'reasoning': 'Unable to evaluate'},
            'engagement_mechanics': {'score': 0.2, 'reasoning': 'Unable to evaluate'},
            'cultural_fluency': {'score': 0.2, 'reasoning': 'Unable to evaluate'},
            'execution_quality': {'score': 0.2, 'reasoning': 'Unable to evaluate'},
            'overall': {'score': 0.2, 'reasoning': 'Default evaluation due to parsing error'}
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

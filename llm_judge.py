import dspy
from typing import Dict, List, Tuple
import json
import re

def create_curriculum_signature(stage: str) -> type:
    """Create a dynamic signature based on curriculum stage"""
    
    # Base signature content
    base_content = """You are a cognitive analysis expert evaluating tweet generation quality.

GOAL
Evaluate how well a generated tweet demonstrates the strategic thinking, audience psychology, and engagement mechanics shown in expert cognitive analysis examples.

INPUTS
- TWEET: {{TWEET}}
- CONTEXT (optional; the info/cognitive analysis the tweet is based on): {{CONTEXT}}
- ANALYSIS_EXAMPLES (how experts reason about strategy/audience/engagement): {{ANALYSIS_EXAMPLES}}

VALIDATION (hard gate)
If no valid tweet was generated—i.e., empty/whitespace, meta-text/instructions/commentary instead of a tweet, or a block that clearly labels itself as "analysis", "instructions", "variant list", or similar—return:
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
Initialize penalty_total = 0 and an array penalties = []."""

    # Stage-specific penalty rules
    penalty_rules = {
        "basic": """
Apply the following additive deductions, with a short reason for each applied item:

A) Overlength (>280 characters): deduct 0.30
- Condition: character_count(TWEET) > 280
- Record: {"type":"overlength","weight":0.30,"reason":"Tweet exceeds 280 characters."}

B) Disallowed special characters: deduct 0.10 per category, up to 0.30 max
- Categories to check (any occurrence):
  1. Hashtags: "#" anywhere
  2. Emojis: presence of emoji characters
  3. Prohibited sequences: "--" or "**" or "*" (asterisk anywhere) or em dash anywhere
- For each present category add {"type":"special_characters","weight":0.10,"reason":"Found <category>."}
- Cap combined special-character penalties at 0.30.

CURRICULUM STAGE: BASIC (0-20% training progress)
Only apply penalties A and B in this training stage. Do NOT evaluate or penalize generic phrasing, context repetition, or contradictory patterns yet.""",
        
        "intermediate": """
Apply the following additive deductions, with a short reason for each applied item:

A) Overlength (>280 characters): deduct 0.30
- Condition: character_count(TWEET) > 280
- Record: {"type":"overlength","weight":0.30,"reason":"Tweet exceeds 280 characters."}

B) Disallowed special characters: deduct 0.10 per category, up to 0.30 max
- Categories to check (any occurrence):
  1. Hashtags: "#" anywhere
  2. Emojis: presence of emoji characters
  3. Prohibited sequences: "--" or "**" or "*" (asterisk anywhere) or em dash anywhere
- For each present category add {"type":"special_characters","weight":0.10,"reason":"Found <category>."}
- Cap combined special-character penalties at 0.30.

C) Generic/formulaic: deduct 0.20
- Condition: tweet reads as template-y, clichéd, or boilerplate; shallow restatement without angle; empty CTA ("Thoughts?") or listicle-y filler.
- Record: {"type":"generic_formulaic","weight":0.20,"reason":"Lacks strategic depth or uses formulaic phrasing."}

D) Repeats context verbatim/near-verbatim: deduct 0.30
- Condition: high overlap with CONTEXT (e.g., ≥70% phrase overlap or semantically identical restatement without novel angle).
- Record: {"type":"context_repetition","weight":0.30,"reason":"Repeats input/context without new framing."}

CURRICULUM STAGE: INTERMEDIATE (20-60% training progress)
Apply penalties A, B, C, and D in this training stage. Do NOT evaluate or penalize contradictory patterns yet.""",
        
        "advanced": """
Apply the following additive deductions, with a short reason for each applied item:

A) Overlength (>280 characters): deduct 0.30
- Condition: character_count(TWEET) > 280
- Record: {"type":"overlength","weight":0.30,"reason":"Tweet exceeds 280 characters."}

B) Disallowed special characters: deduct 0.10 per category, up to 0.30 max
- Categories to check (any occurrence):
  1. Hashtags: "#" anywhere
  2. Emojis: presence of emoji characters
  3. Prohibited sequences: "--" or "**" or "*" (asterisk anywhere) or em dash anywhere
- For each present category add {"type":"special_characters","weight":0.10,"reason":"Found <category>."}
- Cap combined special-character penalties at 0.30.

C) Generic/formulaic: deduct 0.20
- Condition: tweet reads as template-y, clichéd, or boilerplate; shallow restatement without angle; empty CTA ("Thoughts?") or listicle-y filler.
- Record: {"type":"generic_formulaic","weight":0.20,"reason":"Lacks strategic depth or uses formulaic phrasing."}

D) Repeats context verbatim/near-verbatim: deduct 0.30
- Condition: high overlap with CONTEXT (e.g., ≥70% phrase overlap or semantically identical restatement without novel angle).
- Record: {"type":"context_repetition","weight":0.30,"reason":"Repeats input/context without new framing."}

E) Contradictory phrasing pattern: deduct 0.15
- Condition: uses "X is not Y. It is actually Z" or similar contradictory patterns ("This isn't..., it's actually...", "That's not..., that's...")
- Record: {"type":"contradictory_phrasing","weight":0.15,"reason":"Uses contradictory 'X is not Y, it is actually Z' pattern."}

CURRICULUM STAGE: ADVANCED (60-100% training progress)
Apply all penalty categories A through E in this final training stage.""",
        
        "full": """
Apply the following additive deductions, with a short reason for each applied item:

A) Overlength (>280 characters): deduct 0.30
- Condition: character_count(TWEET) > 280
- Record: {"type":"overlength","weight":0.30,"reason":"Tweet exceeds 280 characters."}

B) Disallowed special characters: deduct 0.10 per category, up to 0.30 max
- Categories to check (any occurrence):
  1. Hashtags: "#" anywhere
  2. Emojis: presence of emoji characters
  3. Prohibited sequences: "--" or "**" or "*" (asterisk anywhere) or em dash anywhere
- For each present category add {"type":"special_characters","weight":0.10,"reason":"Found <category>."}
- Cap combined special-character penalties at 0.30.

C) Generic/formulaic: deduct 0.20
- Condition: tweet reads as template-y, clichéd, or boilerplate; shallow restatement without angle; empty CTA ("Thoughts?") or listicle-y filler.
- Record: {"type":"generic_formulaic","weight":0.20,"reason":"Lacks strategic depth or uses formulaic phrasing."}

D) Repeats context verbatim/near-verbatim: deduct 0.30
- Condition: high overlap with CONTEXT (e.g., ≥70% phrase overlap or semantically identical restatement without novel angle).
- Record: {"type":"context_repetition","weight":0.30,"reason":"Repeats input/context without new framing."}

E) Contradictory phrasing pattern: deduct 0.15
- Condition: uses "X is not Y. It is actually Z" or similar contradictory patterns ("This isn't..., it's actually...", "That's not..., that's...")
- Record: {"type":"contradictory_phrasing","weight":0.15,"reason":"Uses contradictory 'X is not Y, it is actually Z' pattern."}

NO CURRICULUM: Apply all penalty categories without restrictions."""
    }
    
    # Build full signature content
    full_content = f"""{base_content}

{penalty_rules.get(stage, penalty_rules["full"])}

Compute:
penalty_total = sum(weights), clipped to [0, 0.90]
final score = max(0, min(1, raw_score - penalty_total))

OUTPUT (JSON ONLY)
Return strictly JSON with these keys:
{{
  "strategic_understanding": <float 0-1>,
  "audience_psychology": <float 0-1>,
  "engagement_mechanics": <float 0-1>,
  "cultural_fluency": <float 0-1>,
  "execution_quality": <float 0-1>,
  "raw_score": <float 0-1>,
  "penalties": [
    // Only include penalties that are active in current curriculum stage
    // For basic stage: only overlength and special_characters
    // For intermediate stage: overlength, special_characters, generic_formulaic, context_repetition
    // For advanced/full stage: all penalty types
  ],
  "penalty_total": <float 0-0.90>,
  "score": <float 0-1>,  // final score after penalties
  "feedback": "<3–4 sentences on cognitive alignment and strategy. MANDATORY: For each penalty applied, provide specific actionable fix. Mention the most material strengths and explain how to address the biggest penalty hits with concrete examples.>"
}}

PENALTY FEEDBACK REQUIREMENTS
For each penalty applied, provide specific actionable feedback:

A) Overlength penalty → "Trim to X characters by removing Y and condensing Z"
B) Special characters penalty → "Remove hashtags/emojis and use text alternatives instead of emojis"
C) Generic/formulaic penalty → "Add specific angle like [example] instead of generic phrasing"
D) Context repetition penalty → "Reframe with fresh angle like [example] rather than restating the original"
E) Contradictory phrasing penalty → "Replace contradictory structure with direct statement like [example]"

ADDITIONAL RULES
- Reward strategic thinking beyond surface-level; favor audience insight and cultural fluency.
- MANDATORY: Include concrete examples in feedback for any penalties applied.
- AVOID CONTRADICTORY PATTERNS: Never use "X is not Y. It is actually Z" phrasing. Instead use direct, constructive language.
- Do not include any text outside the JSON object in your response.
    """
    
    # Create dynamic signature class
    class CurriculumTweetJudgeSignature(dspy.Signature):
        __doc__ = full_content
        
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
            desc="JSON string with scores and reasoning: {\"strategic_understanding\": X, \"penalties\": [...], ...}"
        )
    
    return CurriculumTweetJudgeSignature

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
  3. Prohibited sequences: "--" or "**" or "*" (asterisk anywhere) or em dash anywhere
- For each present category add {"type":"special_characters","weight":0.10,"reason":"Found <category>."}
- Cap combined special-character penalties at 0.30.

C) Generic/formulaic: deduct 0.20
- Condition: tweet reads as template-y, clichéd, or boilerplate; shallow restatement without angle; empty CTA (“Thoughts?”) or listicle-y filler.
- Record: {"type":"generic_formulaic","weight":0.20,"reason":"Lacks strategic depth or uses formulaic phrasing."}

D) Repeats context verbatim/near-verbatim: deduct 0.30
- Condition: high overlap with CONTEXT (e.g., ≥70% phrase overlap or semantically identical restatement without novel angle).
- Record: {"type":"context_repetition","weight":0.30,"reason":"Repeats input/context without new framing."}

E) Contradictory phrasing pattern: deduct 0.15
- Condition: uses "X is not Y. It is actually Z" or similar contradictory patterns ("This isn't..., it's actually...", "That's not..., that's...")
- Record: {"type":"contradictory_phrasing","weight":0.15,"reason":"Uses contradictory 'X is not Y, it is actually Z' pattern."}

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
    {"type":"context_repetition","weight":0.30,"reason":"..."},
    {"type":"contradictory_phrasing","weight":0.15,"reason":"Uses 'X is not Y, it is actually Z' pattern."}
    // include only those applied; may be empty []
  ],
  "penalty_total": <float 0-0.90>,
  "score": <float 0-1>,  // final score after penalties
  "feedback": "<3–4 sentences on cognitive alignment and strategy. MANDATORY: For each penalty applied, provide specific actionable fix. Mention the most material strengths and explain how to address the biggest penalty hits with concrete examples.>"
}

PENALTY FEEDBACK REQUIREMENTS
For each penalty applied, provide specific actionable feedback:

A) Overlength penalty → "Trim to X characters by removing Y and condensing Z"
B) Special characters penalty → "Remove hashtags/emojis and use text alternatives instead of emojis"
C) Generic/formulaic penalty → "Add specific angle like [example] instead of generic phrasing"
D) Context repetition penalty → "Reframe with fresh angle like [example] rather than restating the original"
E) Contradictory phrasing penalty → "Replace contradictory structure with direct statement like [example]"

ADDITIONAL RULES
- Reward strategic thinking beyond surface-level; favor audience insight and cultural fluency.
- Penalize if tweet is not ≤280 chars, contains hashtags/emojis or the sequences --, **, *, is generic/formulaic, or merely repeats CONTEXT.
- MANDATORY: Include concrete examples in feedback for any penalties applied.
- AVOID CONTRADICTORY PATTERNS: Never use "X is not Y. It is actually Z" phrasing. Instead use direct, constructive language.
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

class PenaltyEnforcerSignature(dspy.Signature):
    """You are a penalty enforcer for tweet evaluation.

GOAL
Check a generated tweet against penalty rules only. Compute penalties, then a final score = 1 - penalty_total. Provide concise, actionable feedback.
Do NOT judge strategy or reasoning quality.

INPUTS
- TWEET: {{TWEET}}
- CONTEXT: {{CONTEXT}}  // used only for repetition check

PENALTY RULES
Apply additive deductions. Each applied penalty MUST include an actionable fix in feedback. Cap total at 0.90.

1) Overlength (>280 characters) → -0.30
   Fix: "Trim to ≤280 by cutting [redundant phrase] and condensing [example]."

2) Special characters (max -0.30 combined)
   - Hashtags "#" → -0.10
   - Emojis → -0.10
   - Prohibited sequences: "--", "**", "*", or em dash "—" → -0.10
   Fix: "Remove hashtags/emojis; replace with plain text alternatives."

3) Generic / Formulaic phrasing → -0.20
   Condition: template-y, limp CTA ("Thoughts?"), or lacks fresh angle.
   Fix: "Add an absurd/funny fact or angle instead of a prompt. Example: 'Fun fact: the Pentagon buys more PowerPoints than victories.'"

4) Context Repetition → -0.30
   Condition: ≥70% overlap or near-verbatim restatement of CONTEXT.
   Fix: "Reframe with an absurd day-to-day metaphor. Example: 'Like a gym bro buying new outfits but never adding weight to the bar.'"

5) Contradictory phrasing pattern → -0.15
   Condition: "X is not Y. It is actually Z" or close variants.
   Fix: "Use direct phrasing. Example: 'The PR wins didn't translate to results.'"

6) Overwording / Ramble → -0.20
   Condition (any one triggers): more than 1 sentence; >2 commas; >22 words; or repeated clause links (", and" | ", but" | ", which" | "because" occurring ≥2).
   Fix: "Compress to a single sentence (12–18 words) with a concrete subject-verb-object; remove the secondary clause."

7) Weak Hook (low weight) → -0.03
   Condition: opener (first 3–8 words) uses clichés (^(Breaking:|PSA:|Hot take:|Thread:|Can we talk|We need to|In 20\\d\\d,|Let that sink in)) OR lacks a concrete noun/verb.
   Fix: "Replace with a vivid 3–8-word hook using a concrete noun or verb. Examples: 'Two logos, zero trophies.' / 'PowerPoint beats rockets, again.'"

SCORING (exact computation order; numerically stable)
1) Compute special_chars_raw = 0
   - If hashtag found → +0.10
   - If emoji found → +0.10
   - If any prohibited sequence ("--", "**", "*", "—") found → +0.10
   - special_chars = min(special_chars_raw, 0.30)

2) Sum all penalties before cap:
   penalty_total_raw =
       overlength(0/0.30) +
       special_chars +
       generic(0/0.20) +
       context_repetition(0/0.30) +
       contradictory(0/0.15) +
       overwording(0/0.20) +
       weak_hook(0/0.03)

3) Clip and round:
   penalty_total = min(0.90, round(penalty_total_raw + 1e-9, 2))

4) Final score:
   score = max(0.0, round(1.0 - penalty_total, 2))

OUTPUT (STRICT JSON ONLY)
{
  "penalties": [
    {"type":"overlength","weight":0.30,"reason":"..."},
    {"type":"special_characters","weight":0.10,"reason":"Found hashtag."},
    {"type":"generic_formulaic","weight":0.20,"reason":"..."},
    {"type":"context_repetition","weight":0.30,"reason":"..."},
    {"type":"contradictory_phrasing","weight":0.15,"reason":"..."},
    {"type":"overwording","weight":0.20,"reason":"Two sentences and >2 commas."},
    {"type":"weak_hook","weight":0.03,"reason":"Cliché opener and no concrete token."}
    // include only applied penalties; may be []
  ],
  "penalty_total": <float 0–0.90>,
  "score": <float 0–1>,
  "feedback": "<3–5 sentences. Mandatory: for EACH applied penalty, give a concrete, absurd/funny-leaning fix. Prioritize the biggest hits first (overlength/overwording), show a one-liner rewrite pattern (12–18 words), and give an example hook if 'weak hook' applied. If no penalties: 'No penalties. Tweet is compliant.'>"
}

RULES
- Always compute penalty_total and score.
- Always return feedback (even if 'No penalties...').
- Output ONLY the JSON object—no extra text.

    """
    
    generated_tweet: str = dspy.InputField(
        desc="The tweet that was generated to evaluate for penalties"
    )
    context: str = dspy.InputField(
        desc="The original context (used only for repetition check)"
    )
    penalty_evaluation: str = dspy.OutputField(
        desc="JSON string with penalty analysis: {\"penalties\": [...], \"penalty_total\": X, \"score\": Y, \"feedback\": \"...\"}"
    )

class PenaltyEnforcer(dspy.Module):
    """Penalty-focused judge that enforces mechanical rules for tweet compliance"""
    
    def __init__(self):
        super().__init__()
        self.enforcer = dspy.ChainOfThought(PenaltyEnforcerSignature)
        print("[PenaltyEnforcer] Initialized with mechanical penalty rules")
    
    def forward(self, generated_tweet: str = None, context: str = "", 
                information_sources: str = None, original_tweet: str = None, 
                post_time: str = None, audience_size: str = None, 
                platform_activity: str = None, trending_topics: str = None):
        """Evaluate a generated tweet for rule violations and penalties
        
        Args:
            generated_tweet: The tweet to evaluate
            context: Original context (used for repetition check)
            information_sources: Alternative name for context (GEPA compatibility)
            original_tweet: Original tweet (ignored by penalty enforcer)
            post_time: Post time (ignored by penalty enforcer)
            audience_size: Audience size (ignored by penalty enforcer)
            platform_activity: Platform activity (ignored by penalty enforcer)
            trending_topics: Trending topics (ignored by penalty enforcer)
        
        Returns:
            Prediction with penalty evaluation
        """
        
        # Handle parameter compatibility - use information_sources as context if provided
        if information_sources and not context:
            context = information_sources
        
        result = self.enforcer(
            generated_tweet=generated_tweet,
            context=context
        )
        
        # Parse the penalty evaluation JSON
        try:
            # Clean up common LLM output formatting issues
            penalty_json = result.penalty_evaluation.strip()
            
            # Remove markdown code blocks if present
            if penalty_json.startswith('```json'):
                penalty_json = penalty_json.replace('```json', '').replace('```', '').strip()
            elif penalty_json.startswith('```'):
                penalty_json = penalty_json.replace('```', '').strip()
            
            # Try to find JSON object if response has extra text
            import re
            json_match = re.search(r'\{.*\}', penalty_json, re.DOTALL)
            if json_match:
                penalty_json = json_match.group()
            
            evaluation = json.loads(penalty_json)
            
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            print(f"[PenaltyEnforcer] Warning: Failed to parse penalty JSON ({e}), using defaults")
            print(f"[PenaltyEnforcer] Raw output: {result.penalty_evaluation[:200]}...")
            evaluation = self._create_default_penalty_evaluation()
        
        # Ensure required fields exist
        if 'penalty_total' not in evaluation:
            evaluation['penalty_total'] = 0.0
        if 'score' not in evaluation:
            evaluation['score'] = 1.0
        if 'penalties' not in evaluation:
            evaluation['penalties'] = []
        if 'feedback' not in evaluation:
            evaluation['feedback'] = f"Penalty evaluation completed. Score: {evaluation['score']}"
        
        return dspy.Prediction(
            penalty_evaluation=evaluation,
            penalty_total=evaluation['penalty_total'],
            score=evaluation['score'],
            overall_score=evaluation['score'],  # Add overall_score for GEPA compatibility
            penalties=evaluation['penalties'],
            evaluation=evaluation,  # Add evaluation for training pipeline compatibility
            feedback=evaluation['feedback']  # Add feedback for GEPA compatibility
        )
    
    def _create_default_penalty_evaluation(self):
        """Create default penalty evaluation structure"""
        return {
            'penalties': [],
            'penalty_total': 0.0,
            'score': 1.0,
            'feedback': 'Unable to evaluate penalties due to parsing error.'
        }

class LLMJudge(dspy.Module):
    """LLM-based judge for evaluating tweet quality with curriculum penalty learning"""
    
    def __init__(self, enable_curriculum: bool = False, total_generations: int = None):
        super().__init__()
        self.judge = dspy.ChainOfThought(TweetJudgeSignature)
        self.enable_curriculum = enable_curriculum
        self.total_generations = total_generations or 100  # Default for safety
        self.current_generation = 0
        
        if enable_curriculum:
            print(f"[LLMJudge] Initialized with curriculum penalty learning over {self.total_generations} generations")
        else:
            print("[LLMJudge] Initialized with standard evaluation criteria")
    
    def set_training_progress(self, current_generation: int):
        """Update current training generation for curriculum learning"""
        self.current_generation = current_generation
        
    def _get_curriculum_stage(self) -> str:
        """Determine which curriculum stage we're in based on training progress"""
        if not self.enable_curriculum:
            return "full"  # No curriculum, use all penalties
            
        progress = self.current_generation / self.total_generations if self.total_generations > 0 else 1.0
        
        if progress <= 0.2:
            return "basic"      # 0-20%: overlength + special characters
        elif progress <= 0.6:
            return "intermediate"  # 20-60%: + generic + context repetition  
        else:
            return "advanced"   # 60-100%: all penalties
    
    def forward(self, information_sources: str, generated_tweet: str, original_tweet: str, 
                training_generation: int = None):
        """Evaluate a generated tweet with curriculum-based penalty learning"""
        
        # Update training progress if provided
        if training_generation is not None:
            self.set_training_progress(training_generation)
        
        # Get current curriculum stage
        stage = self._get_curriculum_stage()
        
        # Use curriculum-based dynamic signature
        if self.enable_curriculum:
            curriculum_signature = create_curriculum_signature(stage)
            judge = dspy.ChainOfThought(curriculum_signature)
            print(f"[LLMJudge] Generation {self.current_generation}, Stage: {stage}")
        else:
            # Use standard static signature for non-curriculum mode
            judge = self.judge
        
        result = judge(
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

import asyncio
import os
import json
import aiohttp
import time
import argparse
from typing import Dict, List, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

class TweetGenerator:
    def __init__(self, max_retries: int = 3, retry_delay: int = 2):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate_tweet(self, prompt: str, article: Dict[str, str], article_index: int) -> Dict:
        """Generate a tweet using OpenRouter's API with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                if not self.session:
                    raise ValueError("Session not initialized. Use 'async with' context manager.")

                # Prepare API request
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/eternis-dev/generate-tweets-gepa",
                    "X-Title": "Generate Tweets GEPA"
                }

                # Combine prompt with article for context
                full_context = f"{prompt}\n\nARTICLE CONTENT:\n{article['content'][:500]}"
                
                # Prepare request payload
                payload = {
                    "model": "moonshotai/kimi-k2ß",
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": full_context}
                    ],
                    "temperature": 1.0
                }

                # Make API request
                async with self.session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"API request failed with status {response.status}: {error_text}")
                    
                    result = await response.json()
                    tweet = result['choices'][0]['message']['content'].strip()
                    
                
                    
                    return {
                        "article_index": article_index,
                        "article_type": article.get("type", "unknown"),
                        "article_source": article.get("source", "unknown"),
                        "generated_tweet": tweet,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "success"
                    }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                return {
                    "article_index": article_index,
                    "article_type": article.get("type", "unknown"),
                    "article_source": article.get("source", "unknown"),
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "error"
                }

async def process_articles(articles: List[Dict], prompt: str, batch_size: int = 5) -> List[Dict]:
    """Process articles in parallel batches"""
    results = []
    
    async with TweetGenerator() as generator:
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            tasks = [
                generator.generate_tweet(
                    prompt=prompt,
                    article=article,
                    article_index=i + idx
                )
                for idx, article in enumerate(batch)
            ]
            
            # Process batch with progress bar
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Add a small delay between batches to avoid rate limits
            if i + batch_size < len(articles):
                await asyncio.sleep(1)
    
    return results

def parse_args():
    """Parse command line arguments and environment variables"""
    # First check environment variables
    default_batch_size = int(os.getenv('TWEET_BATCH_SIZE', '40'))
    default_input_file = os.getenv('TWEET_INPUT_FILE', 'articles.json')
    
    parser = argparse.ArgumentParser(description='Generate tweets from articles in parallel')
    parser.add_argument('--batch-size', type=int, default=default_batch_size,
                       help=f'Number of articles to process in parallel (default: {default_batch_size}, '
                            'can also be set via TWEET_BATCH_SIZE env var). '
                            'Increase for faster processing but watch for rate limits.')
    parser.add_argument('--input-file', type=str, default=default_input_file,
                       help=f'Input JSON file containing articles (default: {default_input_file}, '
                            'can also be set via TWEET_INPUT_FILE env var)')
    
    args = parser.parse_args()
    
    # Validate batch size
    if args.batch_size < 1:
        parser.error('Batch size must be at least 1')
    
    return args

async def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load articles from JSON file
        with open(args.input_file, 'r') as f:
            articles = json.load(f)
        
        print(f"\nLoaded {len(articles)} articles from articles.json")
        
        # Load the prompt
        prompt = """Signature: Viral, CTA-first tweet generation from multi-source evidence (meme-capable)

Task
Given multiple information sources (research papers, case studies, surveys, repos, etc.), generate a single viral-style tweet (and optionally a short thread) that:
- Opens with a punchy CTA or meme-y quote and exactly one link.
- Uses 1–2 killer, credible stats in plain English.
- Fits ≤280 characters per tweet.
- Matches a minimal, high-velocity social style (not a policy brief).

Inputs
- information_context: array of objects with:
  - content: string (core details; findings, stats, dates, quotes; optional code snippets)
  - source: string (e.g., MIT, Pew Research Center, Kela, GitHub)
  - type: string (e.g., research_paper, case_study, survey/statistics, github_repo, report, study, market_analysis, implementation, search_query)

Output format
- reasoning: 2–5 bullets describing:
  - Which core insight(s) you selected (cite source names).
  - Why those support the CTA.
  - Any wording choices to avoid jargon and keep ≤280 chars.
- generated_tweet: the single main tweet (≤280 chars).
- Optional:
  - thread: array of 2–3 tweets (each ≤280 chars). Include the single link only in T1.
  - alt_versions: up to 2 alternative single-tweet options with different hooks (each ≤280 chars).

Style and constraints
- Hook: CTA-first or meme-first. Start with a caps hook or short playful quote.
  Examples: QUICK SIGN HERE:, READ THIS:, WATCH THIS:, or a cold-open quote (“Then he said, ‘I have to work on Tuesday.’”).
- Exactly one link in the main tweet. If no link is provided in inputs, insert [link] as a placeholder. Prefer official sources (.org, .edu, .gov, or official GitHub).
- Keep it ultra-brief and scannable. Prefer 1 stat in the main tweet; move extra proof to a thread.
- Tone: minimal, high-velocity, meme-capable, conversational. Short lines are fine. 0–1 emoji max (optional) and 0–1 hashtag max (optional).
- Avoid jargon and unfamiliar acronyms:
  - Replace “sybil resistance” with “proof-of-personhood” in general. If accuracy requires (e.g., for Proof of Humanity), use “one human, one payout—no means testing” or “sybil-resistant” sparingly.
  - Replace “pp” with “points.”
  - Avoid method details (2SLS, instruments, Merkle trees) and special symbols; use plain words (“led to,” “rose by”).
- Calibrate precisely; do not over-claim:
  - Use exact figures when available (e.g., “45%,” “12 points”).
  - Attribute results to context/time (e.g., “In Stockton…”, “Kela’s 2017–2018 trial…”).
  - Do not generalize a single study to all contexts; say “in [place/study]” or “on average.”
  - If support is split, say “public is split,” not “most people.”
  - If a control-group figure is not provided, say “vs a smaller change in control,” not a specific number.
- If technical implementation is relevant, mention it in plain language (“one person, one payout”; “run a city e‑petition”; “model your UBI”) and move details to T2/T3 in a thread.
- Ensure factual accuracy and character limits for every tweet produced.
- If inputs imply a target audience or locale (e.g., Canadian PM candidates), mirror that salience and address directly.

Content selection strategy
1) Skim inputs and extract the 1–2 strongest, quantifiable takeaways that directly support the CTA (e.g., sign a petition, read a report, pilot a program).
2) Prefer stats that are clear and compelling in one line:
   - Labor market effects of automation/robots.
   - Effects of unconditional/guaranteed cash on work, volatility, and well-being.
   - Public opinion figures that signal momentum or division worth action.
   - Simple, operational implementation hooks (proof-of-personhood; petition platforms; policy modeling; mobile money delivery).
3) Translate technical language into plain English. Drop methods; keep context and numbers.
4) Compose a main tweet with:
   - A short, caps-forward CTA or a meme-y quote opener.
   - Exactly one link.
   - One killer stat and one plain-language insight max.
   - ≤280 characters (ideally 150–200); push extra proof to a thread.
5) If more evidence/tooling is crucial, add a 2–3 tweet thread:
   - T1: CTA + link (+ at most 1 stat).
   - T2: 1–2 proof points with precise but brief numbers.
   - T3: Implementation/tool in simple words or an extra credibility anchor.

Meme/virality guidance
- Hooks that work: cold-open quotes, playful call-outs, minimalist CAP lines, a single well-placed emoji (🫵😂, 🍟).
- Keep the main tweet short; move extra stats to a thread.
- Favor cadence and punch over full naming. E.g., “Stockton’s $500/mo pilot,” not a long institutional title.
- You can echo a joke/metaphor (“fries in the bag”) then pivot to a stat + CTA.
- Use line breaks for rhythm if helpful.

Accuracy guardrails and reference anchors (use as needed; keep context precise)
- Robots and Jobs (Acemoglu & Restrepo; US 1990–2007; revised 2020; IFR data):
  - 360k–670k fewer jobs nationally (1990–2007) attributable to robots.
  - Each additional robot per 1,000 workers: employment-to-pop down ~0.2–0.34 points; wages down ~0.36–0.74%.
  - Context: commuting zones; instrumented; controls include China import exposure.
- Finland Basic Income (Kela, 2017–2018; €560/month; 2,000 unemployed):
  - No adverse aggregate employment effects.
  - In 2018, recipients worked modestly more (~5–6 additional days).
  - Well-being up: life satisfaction, lower stress, more trust.
- Alaska Permanent Fund (Jones & Marinescu; 1990s–2015):
  - Universal annual cash did not reduce total employment.
  - Part-time employment rose modestly (~1–2 points).
  - Demand shifts: non-tradables up; tradables down; net near zero on total employment.
- J-PAL cash transfer evidence (160+ RCTs/reviews of UCTs):
  - Little/no reduction in adult labor supply on average.
  - Gains in food security, assets, planning, psychological well-being; predictability matters.
- Stockton SEED ($500/month; 125 recipients; 2019–2020):
  - +12-point increase in full-time employment in year 1 (control had a smaller gain).
  - Income volatility down ~46%; better mental health; spending on essentials (~37% food; <1% alcohol/tobacco).
- Pew public opinion:
  - 2017 (automation framing): 48% favor, 52% oppose a basic income for workers displaced by automation.
  - 2020 ($1,000/month UBI framing): ~45% support, ~54% oppose; strong partisan and age splits.
- GiveDirectly Kenya UBI ops (2017–; M‑Pesa):
  - >99% successful delivery after retries/helpdesk.
  - Admin costs often ~10–13% of transfer value; >85–90 cents per $1 reaching recipients.
  - No sustained local price inflation observed in monitoring.
- Proof-of-personhood systems:
  - World ID: “one person, one payout, privacy-preserving” (uses zero-knowledge; no raw biometrics on-chain).
  - Proof of Humanity (PoH): on-chain video + vouching + challenges. Avoid saying “privacy-preserving”; safer: “one human, one payout—no means testing” or “sybil-resistant registry.”
- Decidim (open-source petitions): “run a city e‑petition with verified signatures.”
- PolicyEngine (OpenFisca-based): “model UBI costs and who benefits.”
- Crypto UBI implementations:
  - GoodDollar (ERC-20 G$; yield-funded; daily claim; identity checks; Ethereum + low-fee sidechains). Plain wording: “open-source crypto UBI; daily claims funded by staking yield.”
  - Circles UBI (Gnosis/xDai): “each person mints their own token; trust links route value.”

Templates you may adapt
- Minimalist CTA + stat:
  - QUICK SIGN HERE: [link] Stockton’s $500/mo saw full-time jobs up 12 points in year one. Cash didn’t cut work across 160+ studies. Let’s test it.
- Meme-aligned hook:
  - FRIES IN THE BAG 🍟 Stockton’s $500/mo: full-time jobs +12 points. Translation: cash stabilizes. Pilot it: [link]
- Proof-of-personhood:
  - UBI = AI money. Proof-of-personhood = one person, one payout. Pilot it: [link]
- Hybrid myth vs fact:
  - “Cash kills work.” Myth. In Stockton: +12 points full-time. Finland: no drop in work, stress down. READ: [link]
- Union/leverage frame:
  - PAY FLOOR = STRIKE POWER. Alaska’s annual cash didn’t cut jobs; part-time rose ~1–2 points. Make UBI part of the platform: [link]
- Duo framing (GBI + UBI dividend):
  - GBI = a floor to end poverty. UBI = a dividend from our common wealth. Mincome: hospitalizations −8.5%. Alaska: no overall job loss. Perfect duo—pilot it: [link]

What to avoid
- Dense phrasing, long clauses, or multiple stats crammed into one tweet.
- Jargon (“sybil,” “2SLS,” Merkle trees) in the main tweet.
- Overstating support/effects; keep numbers precise and contextual.
- Multiple links or embedded code in any tweet.
- Claiming privacy for PoH; use accurate wording as above.

Link selection
- Prefer a single authoritative link tied to the CTA (official report, .org/.edu/.gov, or official GitHub).
- If inputs include a pamphlet or campaign link, use that over secondary sources.
- If no link is provided, insert [link] as a placeholder.

Validation checklist before finalizing
- Main tweet:
  - Starts with a CTA or meme-y quote.
  - Contains exactly one link.
  - ≤280 characters (ideally 150–200).
  - Uses at most 1 stat and plain wording.
  - No jargon; “points” not “pp”; no arrows/special symbols.
  - Claims calibrated to context/time/place; privacy phrasing accurate to the system named.
- Thread (if present):
  - Link only in T1; each tweet ≤280 chars.
  - T2–T3 add 1–2 precise stats or simple implementation hooks.
- Output keys present: reasoning; generated_tweet; optional thread; optional alt_versions.

General strategy (from prior feedback)
- Lead with a memorable hook; keep the stat count low in the main tweet.
- Use line breaks and caps for cadence; end with a clear CTA (“Pilot it,” “Sign here,” “Read this”).
- If a control group figure isn’t verified in inputs, write “control had a smaller gain.”
- Prefer “one human, one payout—no means testing” when describing identity-gated payouts; only claim “privacy-preserving” for systems that actually provide it (e.g., World ID’s ZK design).
- If the audience or country context is implied (e.g., Canada, PM candidates), say it explicitly for relevance.

Example closers you can reuse
- “Cash doesn’t kill work—it kills chaos.”
- “Pilot it.”
- “Receipts inside.”
- “Perfect duo—pilot it.”
"""
        print("\nStarting parallel tweet generation...")
        print("=" * 50)
        
        # Process articles in parallel with progress tracking
        with tqdm(total=len(articles), desc="Generating tweets") as pbar:
            results = []
            batch_results = await process_articles(articles, prompt, batch_size=args.batch_size)
            results.extend(batch_results)
            pbar.update(len(batch_results))
        
        # Save results to JSON file
        output_file = f"generated_tweets_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_articles": len(articles),
                    "successful_generations": len([r for r in results if r["status"] == "success"]),
                    "failed_generations": len([r for r in results if r["status"] == "error"])
                },
                "results": results
            }, f, indent=2)
        
        print("\nResults saved to:", output_file)
        print("=" * 50)
        print("\nSummary:")
        print(f"Total articles processed: {len(articles)}")
        print(f"Successful generations: {len([r for r in results if r['status'] == 'success'])}")
        print(f"Failed generations: {len([r for r in results if r['status'] == 'error'])}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())





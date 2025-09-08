"""Database utilities for tweet generation workflow."""

import os
from typing import Dict, Any, List
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Get a database connection."""
    # Get database URL from environment
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    try:
        # Connect with SSL required (for Supabase)
        conn = psycopg2.connect(
            db_url,
            sslmode='require',
            connect_timeout=10,
            gssencmode='disable'  # Disable GSSAPI encryption to force simple SSL
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"[Database] Connection error: {str(e)}")
        raise

def create_workflow_record(topic: str) -> int:
    """Create initial workflow record and return its ID."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Create empty arrays using ARRAY constructor
            cur.execute("""
                INSERT INTO tweet_generation 
                (created_at, topic, articles, exp_result, tweet_drafts)
                VALUES (NOW(), %s, ARRAY[]::json[], ARRAY[]::json[], ARRAY[]::json[])
                RETURNING id
            """, (topic,))
            return cur.fetchone()[0]

def update_exploration_results(workflow_id: int, topic: str, articles: List[Dict], exp_result: Dict):
    """Update exploration results in the database."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Convert lists to JSONB arrays
            articles_json = [Json(article) for article in articles]
            exp_result_json = [Json(exp_result)]
            
            cur.execute("""
                UPDATE tweet_generation
                SET topic = %s,
                    articles = %s::json[],
                    exp_result = %s::json[]
                WHERE id = %s
            """, (topic, articles_json, exp_result_json, workflow_id))

def update_prompt(workflow_id: int, prompt: str):
    """Update generated prompt in the database."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE tweet_generation
                SET prompt = %s
                WHERE id = %s
            """, (prompt, workflow_id))

def update_tweet_drafts(workflow_id: int, tweet_drafts: List[Dict]):
    """Update generated tweet drafts in the database."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Convert list to JSONB array
            drafts_json = [Json(draft) for draft in tweet_drafts]
            
            cur.execute("""
                UPDATE tweet_generation
                SET tweet_drafts = %s::json[]
                WHERE id = %s
            """, (drafts_json, workflow_id))

def get_workflow_status(workflow_id: int) -> Dict[str, Any]:
    """Get current status of a workflow."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT created_at, topic, articles, exp_result, prompt, tweet_drafts
                FROM tweet_generation
                WHERE id = %s
            """, (workflow_id,))
            row = cur.fetchone()
            if row:
                return {
                    "id": workflow_id,
                    "created_at": row[0],
                    "topic": row[1],
                    "articles": row[2],
                    "exp_result": row[3],
                    "prompt": row[4],
                    "tweet_drafts": row[5]
                }
            return None
"""
FAKE-SHA Backend - Supabase Client

Initializes the Supabase client from environment variables.
Returns None if Supabase is not configured (URL or KEY missing),
so the rest of the app can run without a database.

Environment variables:
    SUPABASE_URL  - Your Supabase project URL (e.g. https://xxx.supabase.co)
    SUPABASE_KEY  - Your Supabase anon/service role key (never commit this!)

Usage:
    client = get_supabase_client()
    if client:
        client.table("analysis_records").insert({...}).execute()
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_supabase_client: Optional[object] = None


def get_supabase_client():
    """
    Get the Supabase client, or None if not configured.
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()

    if not url or not key:
        logger.debug(
            "Supabase not configured: SUPABASE_URL or SUPABASE_KEY missing. "
            "Analysis records will not be stored."
        )
        return None

    try:
        from supabase import create_client

        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as e:
        logger.warning("Failed to initialize Supabase client: %s", e)
        return None

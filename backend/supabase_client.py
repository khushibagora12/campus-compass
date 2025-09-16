# supabase_client.py

import os
from supabase import create_client, Client

# This is the public client, safe for some operations
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# NEW: This is the powerful admin client for backend-only operations
service_key: str = os.environ.get("SUPABASE_SERVICE_KEY")
supabase_admin: Client = create_client(url, service_key)
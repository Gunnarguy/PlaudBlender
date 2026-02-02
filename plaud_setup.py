#!/usr/bin/env python3
"""
PlaudBlender Setup & Authentication Script

Run this first to:
1. Set up your environment
2. Authenticate with Plaud OAuth
3. Test your connection
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def print_banner():
    """Print welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧠 PlaudBlender - Voice Transcript Knowledge Graph          ║
║                                                               ║
║   Transform your Plaud recordings into a searchable,          ║
║   visual knowledge graph with AI-powered insights.            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def check_env():
    """Check environment configuration."""
    print("\n📋 Checking environment configuration...\n")

    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("❌ No .env file found!")
        print("   Creating template .env file...")
        create_env_template()
        return False

    load_dotenv()

    required_vars = {
        "PLAUD_CLIENT_ID": os.getenv("PLAUD_CLIENT_ID"),
        "PLAUD_CLIENT_SECRET": os.getenv("PLAUD_CLIENT_SECRET"),
    }

    optional_vars = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "QDRANT_URL": os.getenv("QDRANT_URL", "http://localhost:6333"),
    }

    all_required_set = True

    print("Required (Plaud API):")
    for var, value in required_vars.items():
        if value:
            print(f"  ✅ {var}: {'*' * 8}...{value[-4:]}")
        else:
            print(f"  ❌ {var}: Not set")
            all_required_set = False

    print("\nOptional (AI Processing):")
    for var, value in optional_vars.items():
        if value:
            print(f"  ✅ {var}: {'*' * 8}...{value[-4:]}")
        else:
            print(f"  ⚠️  {var}: Not set (needed for AI features)")

    return all_required_set

def create_env_template():
    """Create template .env file."""
    template = """# PlaudBlender Configuration
# ========================

# === PLAUD API (Required) ===
# Get these from: https://platform.plaud.ai/developer/portal
PLAUD_CLIENT_ID=your_client_id_here
PLAUD_CLIENT_SECRET=your_client_secret_here
PLAUD_REDIRECT_URI=http://localhost:8080/callback

# === AI / SEARCH (Optional, enables most 'knowledge graph' features) ===
# Google Gemini - https://ai.google.dev/
GEMINI_API_KEY=your_gemini_key_here

# Qdrant Vector Database - https://qdrant.tech/
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=  # Only needed for Qdrant Cloud

# OpenAI Responses (optional; enables in-app Chat + MCP server)
OPENAI_API_KEY=your_openai_key_here
OPENAI_DEFAULT_MODEL=gpt-4.1
# OPENAI_BASE_URL=https://api.openai.com/v1

# Notion (optional; enables direct Notion sync features)
# NOTION_TOKEN=
# NOTION_DATABASE_ID=
"""

    env_file = Path(__file__).parent / ".env"
    with open(env_file, 'w') as f:
        f.write(template)

    print(f"\n✅ Created .env template at: {env_file}")
    print("\n📝 Next steps:")
    print("   1. Go to https://platform.plaud.ai/developer/portal")
    print("   2. Create a new OAuth App")
    print("   3. Copy Client ID and Client Secret to .env")
    print("   4. Run this script again")

def authenticate_plaud():
    """Run Plaud OAuth authentication flow."""
    print("\n🔐 Starting Plaud authentication...\n")

    try:
        from src.plaud_oauth import PlaudOAuthClient

        client = PlaudOAuthClient()

        if client.is_authenticated:
            print("✅ Already authenticated with Plaud!")
            return client

        client.authenticate_interactive()
        return client

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        return None

def test_connection(oauth_client):
    """Test Plaud API connection."""
    print("\n🧪 Testing Plaud API connection...\n")

    try:
        from src.plaud_client import PlaudClient

        client = PlaudClient(oauth_client)

        # Get user info
        print("Fetching user info...")
        user = client.get_user()
        print(f"  ✅ Logged in as: {user.get('email', user.get('name', 'Unknown'))}")

        # List recent recordings
        print("\nFetching recent recordings...")
        recordings = client.list_recordings(page=1, page_size=10)
        print(f"  ✅ Found {len(recordings)} recordings")

        if recordings:
            print("\n  Recent recordings:")
            for rec in recordings[:5]:
                title = (rec.get("name") or rec.get("title") or "Untitled")[:40]
                rec_id = rec.get('id', 'unknown')[:8]
                print(f"    📝 {title} ({rec_id}...)")

        return True

    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        return False

def main():
    """Main setup flow."""
    print_banner()

    # Step 1: Check environment
    if not check_env():
        print("\n⚠️  Please configure your .env file and run again.")
        return

    # Step 2: Authenticate
    oauth = authenticate_plaud()
    if not oauth:
        return

    # Step 3: Test connection
    if test_connection(oauth):
        print("\n" + "="*60)
        print("🎉 Setup complete! You're ready to use PlaudBlender.")
        print("="*60)
        print("\nNext steps:")
        print("  • Run: streamlit run chronos_app.py  - Launch the Chronos UI")
        print(
            "  • Run: python scripts/chronos_pipeline.py --full - Full ingest → process → index pipeline"
        )
        print("  • Run: python verify_integration.py - Developer feature wiring smoke test")
        print("\nFor help: python plaud_setup.py --help")

if __name__ == "__main__":
    main()

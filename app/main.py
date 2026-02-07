"""Chronos - Glorious Knowledge Graph UI.

Run with: python -m app.main
Or: python app/main.py
"""

import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from dash import Dash
import dash_bootstrap_components as dbc

from app.layout import create_layout

# Import callbacks to register them
import app.callbacks  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> Dash:
    """Create and configure the Dash application."""

    # Create Dash app with Bootstrap theme
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.SLATE,  # Dark theme
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
        ],
        suppress_callback_exceptions=True,
        title="Chronos - Knowledge Timeline",
        update_title="Chronos | Loading...",
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    )

    # Set layout
    app.layout = create_layout()

    return app


# Create the app instance
app = create_app()
server = app.server  # For WSGI deployment


def main():
    """Run the development server."""
    logger.info("=" * 60)
    logger.info("  ⏳ CHRONOS - Knowledge Timeline")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  Starting Dash server...")
    logger.info("  Open http://localhost:8050 in your browser")
    logger.info("")
    logger.info("=" * 60)

    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050,
    )


if __name__ == "__main__":
    main()

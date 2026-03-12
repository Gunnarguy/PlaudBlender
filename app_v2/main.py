"""Chronos App v2 - Recording-Centric UI

Run with: python -m app_v2.main
"""

import logging
from dash import Dash
from flask_compress import Compress

from app_v2.layout import create_layout
from app_v2.callbacks import register_all_callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Dash:
    """Create and configure the Dash app."""
    app = Dash(
        __name__,
        title="Chronos",
        assets_folder="assets",
        suppress_callback_exceptions=True,
        eager_loading=False,
    )

    # Enable gzip/brotli compression on all responses
    app.server.config["COMPRESS_ALGORITHM"] = ["br", "gzip"]
    app.server.config["COMPRESS_MIN_SIZE"] = 500
    Compress(app.server)

    # Set layout
    app.layout = create_layout()

    # Register callbacks
    register_all_callbacks(app)

    # Start auto-sync service in background
    try:
        from src.plaud_auto_sync import get_auto_sync

        auto_sync = get_auto_sync()
        auto_sync.start()
        logger.info("Auto-sync service started in background")
    except Exception as e:
        logger.warning(f"Could not start auto-sync: {e}")

    logger.info("Chronos app v2 initialized")
    return app


def main():
    """Run the app."""
    app = create_app()

    logger.info("Starting Chronos v2 at http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()

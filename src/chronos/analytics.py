"""Chronos analytics service for temporal pattern analysis.

Provides aggregation and insight generation for day-of-week queries
like "What happens on Mondays?" and "What do I think about on Thursdays?"
"""

import logging
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

from sqlalchemy.orm import Session

from src.database.chronos_repository import (
    get_chronos_events_by_day,
    get_chronos_events_by_date_range,
)
from src.chronos.qdrant_client import ChronosQdrantClient
from src.chronos.embedding_service import ChronosEmbeddingService

logger = logging.getLogger(__name__)


class ChronosAnalytics:
    """Analytics service for temporal pattern queries."""

    def __init__(self, db_session: Session):
        """Initialize analytics service.

        Args:
            db_session: SQLAlchemy session
        """
        self.db = db_session
        self.qdrant = ChronosQdrantClient()

    def analyze_day_of_week(self, day: str, limit: int = 1000) -> Dict[str, Any]:
        """Analyze all events for a specific day of week.

        Args:
            day: Day name (Monday, Tuesday, etc.)
            limit: Max events to analyze

        Returns:
            Dict with quantitative and qualitative insights
        """
        logger.info(f"Analyzing {day} patterns...")

        # Fetch events from Qdrant
        events = self.qdrant.get_events_by_day(day, limit=limit)

        if not events:
            return {
                "day": day,
                "total_events": 0,
                "message": f"No events found for {day}",
            }

        # Extract payloads
        payloads = [e["payload"] for e in events]

        # Quantitative analysis
        categories = [p.get("category") for p in payloads]
        category_counts = Counter(categories)

        hours = [
            p.get("hour_of_day") for p in payloads if p.get("hour_of_day") is not None
        ]
        hour_distribution = Counter(hours)

        sentiments = [
            p.get("sentiment") for p in payloads if p.get("sentiment") is not None
        ]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Extract keywords
        all_keywords = []
        for p in payloads:
            keywords = p.get("keywords", [])
            if keywords:
                all_keywords.extend(keywords)
        top_keywords = Counter(all_keywords).most_common(10)

        # Peak activity hour
        peak_hour = (
            hour_distribution.most_common(1)[0][0] if hour_distribution else None
        )

        return {
            "day": day,
            "total_events": len(events),
            "category_distribution": dict(category_counts),
            "top_category": (
                category_counts.most_common(1)[0][0] if category_counts else None
            ),
            "hour_distribution": dict(hour_distribution),
            "peak_hour": peak_hour,
            "average_sentiment": round(avg_sentiment, 2),
            "top_keywords": [{"keyword": k, "count": c} for k, c in top_keywords],
            "sentiment_range": {
                "min": round(min(sentiments), 2) if sentiments else None,
                "max": round(max(sentiments), 2) if sentiments else None,
            },
        }

    def compare_days(self, days: List[str]) -> Dict[str, Any]:
        """Compare patterns across multiple days.

        Args:
            days: List of day names to compare

        Returns:
            Dict with comparative analysis
        """
        logger.info(f"Comparing days: {days}")

        results = {}
        for day in days:
            results[day] = self.analyze_day_of_week(day)

        # Find differences
        comparison = {
            "days": days,
            "individual_analysis": results,
            "differences": {},
        }

        # Compare top categories
        categories_by_day = {
            day: data.get("top_category") for day, data in results.items()
        }
        comparison["differences"]["top_category"] = categories_by_day

        # Compare sentiment
        sentiment_by_day = {
            day: data.get("average_sentiment") for day, data in results.items()
        }
        comparison["differences"]["average_sentiment"] = sentiment_by_day

        return comparison

    def get_activity_heatmap(self, days: List[str]) -> Dict[str, Any]:
        """Generate activity heatmap data for visualization.

        Args:
            days: List of days to include

        Returns:
            Dict with heatmap data (day x hour matrix)
        """
        heatmap = {}

        for day in days:
            analysis = self.analyze_day_of_week(day)
            heatmap[day] = analysis.get("hour_distribution", {})

        return {
            "days": days,
            "heatmap": heatmap,
            "description": "Event count by day and hour",
        }

    def generate_insight_summary(self, day: str) -> str:
        """Generate natural language summary of day patterns.

        Args:
            day: Day name

        Returns:
            str: Human-readable summary
        """
        analysis = self.analyze_day_of_week(day)

        if analysis["total_events"] == 0:
            return f"No data available for {day}."

        # Build narrative
        summary_parts = []

        summary_parts.append(
            f"On {day}s, you typically have {analysis['total_events']} recorded events."
        )

        if analysis.get("top_category"):
            summary_parts.append(
                f"Your primary focus is '{analysis['top_category']}' activities."
            )

        if analysis.get("peak_hour") is not None:
            hour = analysis["peak_hour"]
            period = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
            summary_parts.append(f"Peak activity occurs around {hour}:00 ({period}).")

        sentiment = analysis.get("average_sentiment", 0)
        if sentiment > 0.3:
            mood = "optimistic and energized"
        elif sentiment < -0.3:
            mood = "challenged or frustrated"
        else:
            mood = "neutral and focused"
        summary_parts.append(
            f"Overall emotional tone is {mood} (sentiment: {sentiment:.2f})."
        )

        if analysis.get("top_keywords"):
            top_3 = [kw["keyword"] for kw in analysis["top_keywords"][:3]]
            summary_parts.append(f"Recurring themes: {', '.join(top_3)}.")

        return " ".join(summary_parts)

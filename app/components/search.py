"""Search Component."""

from dash import html, dcc


def create_search_component() -> html.Div:
    """Create the search bar component.

    Returns:
        Dash HTML Div containing the search bar
    """
    return html.Div(
        className="search-container",
        children=[
            html.Div(
                className="search-wrapper",
                children=[
                    html.Span("🔍", className="search-icon"),
                    dcc.Input(
                        id="search-input",
                        type="text",
                        placeholder="Search your memories... (e.g., 'meeting with Sarah about the project')",
                        className="search-input",
                        debounce=True,  # Wait for user to stop typing
                    ),
                    html.Button(
                        "Search",
                        id="search-btn",
                        className="btn btn-primary search-btn",
                    ),
                ],
            ),
            # Search results dropdown
            html.Div(
                id="search-results",
                className="search-results",
                style={"display": "none"},
            ),
        ],
    )


def create_search_result(event: dict, score: float) -> html.Div:
    """Create a single search result item.

    Args:
        event: Event dictionary
        score: Similarity score

    Returns:
        Search result element
    """
    event_id = event.get("id", "")
    narrative = event.get("narrative", "")[:150]
    category = event.get("category", "general")
    timestamp = event.get("event_timestamp", "")

    # Format date
    date_str = ""
    if timestamp:
        try:
            from datetime import datetime

            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                dt = datetime.fromtimestamp(timestamp)
            date_str = dt.strftime("%b %d, %Y")
        except Exception:
            pass

    return html.Div(
        className="search-result",
        id={"type": "search-result", "id": event_id},
        children=[
            html.Div(
                className="result-header",
                children=[
                    html.Span(date_str, className="result-date"),
                    html.Span(f"{score:.0%}", className="result-score"),
                    html.Span(category, className=f"result-category cat-{category}"),
                ],
            ),
            html.P(
                narrative + ("..." if len(event.get("narrative", "")) > 150 else ""),
                className="result-text",
            ),
        ],
    )

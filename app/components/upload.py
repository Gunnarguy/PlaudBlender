"""Upload Modal Component."""

from dash import html, dcc


def create_upload_modal() -> html.Div:
    """Create the upload modal component.

    Returns:
        Dash HTML Div containing the upload modal
    """
    return html.Div(
        id="upload-modal",
        className="modal",
        style={"display": "none"},
        children=[
            html.Div(
                className="modal-overlay",
                id="modal-overlay",
            ),
            html.Div(
                className="modal-content",
                children=[
                    # Modal header
                    html.Div(
                        className="modal-header",
                        children=[
                            html.H3("⬆️ Upload Recording"),
                            html.Button(
                                "×",
                                id="modal-close-btn",
                                className="modal-close",
                            ),
                        ],
                    ),
                    # Modal body
                    html.Div(
                        className="modal-body",
                        children=[
                            # Drag and drop zone
                            dcc.Upload(
                                id="upload-zone",
                                className="upload-zone",
                                children=[
                                    html.Div(
                                        [
                                            html.Div("🎤", className="upload-icon"),
                                            html.H4("Drag & Drop Audio File"),
                                            html.P("or click to browse"),
                                            html.P(
                                                "Supports: MP3, M4A, WAV, OGG, FLAC",
                                                className="upload-formats",
                                            ),
                                        ]
                                    ),
                                ],
                                multiple=False,
                                accept="audio/*,.mp3,.m4a,.wav,.ogg,.flac",
                            ),
                            # Upload progress
                            html.Div(
                                id="upload-progress",
                                className="upload-progress",
                                style={"display": "none"},
                                children=[
                                    html.Div(
                                        className="progress-bar",
                                        children=[
                                            html.Div(
                                                id="progress-fill",
                                                className="progress-fill",
                                                style={"width": "0%"},
                                            ),
                                        ],
                                    ),
                                    html.P(id="progress-text", children="Uploading..."),
                                ],
                            ),
                            # Upload status
                            html.Div(
                                id="upload-status",
                                className="upload-status",
                            ),
                            # Options
                            html.Div(
                                className="upload-options",
                                children=[
                                    html.Label(
                                        [
                                            dcc.Checklist(
                                                id="upload-process-immediately",
                                                options=[
                                                    {
                                                        "label": " Process immediately after upload",
                                                        "value": "yes",
                                                    }
                                                ],
                                                value=["yes"],
                                                className="checkbox",
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Modal footer
                    html.Div(
                        className="modal-footer",
                        children=[
                            html.Button(
                                "Cancel",
                                id="upload-cancel-btn",
                                className="btn btn-secondary",
                            ),
                            html.Button(
                                "Process",
                                id="upload-process-btn",
                                className="btn btn-primary",
                                disabled=True,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

"""Lightweight GUI package used by the test suite.

The original PlaudBlender GUI evolved over time. The current repo focuses on the
Chronos pipeline (Qdrant-first), but the unit tests expect a basic importable
`gui` package tree.

These modules are intentionally minimal and avoid hard dependencies on a display
server (e.g., Tkinter) so they can be imported in headless test runs.
"""

"""Admin-only operations shared by the CLI scripts and the HTTP API.

Everything in this package is CPU-safe: no GPU dependency (``torch``,
``flagembedding``, ``docling``) is imported at module import time.
"""

from __future__ import annotations

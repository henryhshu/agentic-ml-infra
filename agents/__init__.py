from __future__ import annotations

"""Agentic monitoring layer for the ML training platform.

Toggled via PipelineConfig.enable_agentic or --agentic CLI flag.
When disabled, all agent operations become no-ops.
"""
import os

def is_agentic_enabled() -> bool:
    """Check if agentic monitoring is enabled via env var fallback."""
    return os.environ.get('AGENTIC_MODE', '').lower() in ('1', 'true', 'yes')

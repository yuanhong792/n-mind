"""N-MIND V3 bounded autonomy facade.

This package is deliberately orchestration-only: it never performs optimizer
steps and never enables network access by itself.
"""

from .facade import V3Facade

__all__ = ["V3Facade"]

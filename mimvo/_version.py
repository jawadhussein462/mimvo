"""The installed Mimvo version, in a module with no imports.

Reports stamp this onto their output. Keeping it here lets the models
import it without importing the whole package.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]

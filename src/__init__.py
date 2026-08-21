"""
Agentic Video Analytics Engine MVP
"""

import os
import logging

# Suppress HuggingFace transformers internal docstring warnings & hub warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

__version__ = "0.1.0"

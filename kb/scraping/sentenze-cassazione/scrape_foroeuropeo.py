"""Legacy entrypoint retained for backward compatibility.

The implementation moved to ``src/scrape_sentenze.py``. Importing this module
re-exports the new classes so older automation keeps working until it migrates
to the new entrypoint.
"""

from src.scrape_sentenze import ForoEuropeoScraper, run_standalone

__all__ = ["ForoEuropeoScraper", "run_standalone"]

"""Project configuration package.

* :mod:`config.settings` - environment driven runtime settings (secrets, paths,
  model, research limits).
* :mod:`config.tuning` - the YAML tuning document (heuristics, thresholds,
  prompt budgets, output templates, search feed template).
* ``config/*.yaml`` - the human editable configuration itself.
"""

from config.settings import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_SOURCES_PATH,
    LOGGER_NAME,
    PROJECT_ROOT,
    Settings,
    configure_logging,
    get_settings,
)
from config.tuning import (
    DEFAULT_TUNING_PATH,
    Tuning,
    TuningError,
    get_tuning,
    load_tuning,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUTS_DIR",
    "DEFAULT_SOURCES_PATH",
    "DEFAULT_TUNING_PATH",
    "LOGGER_NAME",
    "PROJECT_ROOT",
    # settings
    "Settings",
    # tuning
    "Tuning",
    "TuningError",
    "configure_logging",
    "get_settings",
    "get_tuning",
    "load_tuning",
]
"""aeon.config — W4 first-run configuration + schema + preflight."""
from aeon.config.schema import (
    validate_config_file, validate_config_dict, atomic_write_user_config,
    load_user_config, migrate_user_config, USER_CONFIG_SCHEMA_VERSION,
)
from aeon.config.preflight import run_preflight, PreflightVerdict, PreflightResult

__all__ = ["validate_config_file", "validate_config_dict",
           "atomic_write_user_config", "load_user_config", "migrate_user_config",
           "USER_CONFIG_SCHEMA_VERSION", "run_preflight", "PreflightVerdict",
           "PreflightResult"]

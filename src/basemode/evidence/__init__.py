"""Durable, shared evidence about model endpoints.

Unlike :mod:`basemode.health`, this database is an append-only experimental
record.  Verification evidence is never aged out.  Applications such as Loom
may publish aggregate corpus observations here without exposing their text or
tree structure.

This package is split by concern: :mod:`.schema` owns the connection and
migrations, :mod:`.store` is the write path, :mod:`.query` is the read path,
and :mod:`.importers` folds in external/legacy evidence sources. Everything
public is re-exported here so ``basemode.evidence`` behaves exactly as it did
as a single module.
"""

from __future__ import annotations

from ._util import classify_text_endpoint
from .importers import (
    import_annotations,
    import_live_catalog_cache,
    import_provider_health_jsonl,
    import_rejected_registry,
    import_sweep_jsonl,
    import_verified_registry,
)
from .query import (
    current_status,
    enforce_text_only_and_supersede_obsolete_failures,
    excluded_non_text_models,
    get_model_rating,
    list_model_ratings,
    recheck_statuses,
    transient_recheck_models,
)
from .schema import _DB_FILE as _DB_FILE  # re-exported for test monkeypatching
from .schema import (
    ACCOUNT_FAILURES,
    PERSISTENT_RECHECK_DELAY,
    RECHECK_DELAYS,
    SCHEMA_VERSION,
    TRANSIENT_FAILURES,
    connect,
)
from .store import (
    add_annotation,
    ensure_endpoint,
    finish_run,
    publish_corpus_observations,
    record_attempt,
    record_catalog_observation,
    record_probe_result,
    resume_run,
    set_model_rating,
    start_run,
)

__all__ = [
    "ACCOUNT_FAILURES",
    "PERSISTENT_RECHECK_DELAY",
    "RECHECK_DELAYS",
    "SCHEMA_VERSION",
    "TRANSIENT_FAILURES",
    "add_annotation",
    "classify_text_endpoint",
    "connect",
    "current_status",
    "enforce_text_only_and_supersede_obsolete_failures",
    "ensure_endpoint",
    "excluded_non_text_models",
    "finish_run",
    "get_model_rating",
    "import_annotations",
    "import_live_catalog_cache",
    "import_provider_health_jsonl",
    "import_rejected_registry",
    "import_sweep_jsonl",
    "import_verified_registry",
    "list_model_ratings",
    "publish_corpus_observations",
    "recheck_statuses",
    "record_attempt",
    "record_catalog_observation",
    "record_probe_result",
    "resume_run",
    "set_model_rating",
    "start_run",
    "transient_recheck_models",
]

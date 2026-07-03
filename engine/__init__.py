"""SimNorge Modul 5 — spørremotor.

Poller Claude per persona-instans (med personaens konkrete holdningsrealisering
som myk, ikke-forhandlingsbar kontekst), cacher, og aggregerer vektet.
"""

from .backend import (
    LLMBackend, ClaudeCLIBackend, APIBackend, MockBackend, BackendError,
)
from .prompt import QuestionSpec, build_prompt, TRUST_POLICE, TRUST_POLITICAL_SYSTEM
from .geo import (
    GeoContext, SENT_MARKER, geo_line, income_tier, lowinc_tier,
    load_centrality, national_median_income,
)
from .engine import (
    PollOutput, expand_by_item, poll, parse_response, DEFAULT_MODEL,
    weighted_mean, weighted_std, level_means, weighted_position_distribution,
)

__all__ = [
    "LLMBackend", "ClaudeCLIBackend", "APIBackend", "MockBackend", "BackendError",
    "QuestionSpec", "build_prompt", "TRUST_POLICE", "TRUST_POLITICAL_SYSTEM",
    "GeoContext", "SENT_MARKER", "geo_line", "income_tier", "lowinc_tier",
    "load_centrality", "national_median_income",
    "PollOutput", "expand_by_item", "poll", "parse_response", "DEFAULT_MODEL",
    "weighted_mean", "weighted_std", "level_means", "weighted_position_distribution",
]

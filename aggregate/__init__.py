"""SimNorge Modul 6 — aggregering.

Vektet opprulling persona → kommune → fylke → nasjonalt med usikkerhetsbånd.
"""

from .rollup import Distribution, bootstrap_distribution, rollup, fylke_of

__all__ = ["Distribution", "bootstrap_distribution", "rollup", "fylke_of"]

"""SimNorge Modul 8 — spørretaben.

Interaktivt UI over hele kjeden: fritt spørsmål + geografi -> tolket
spesifikasjon -> persona-polling -> vektet aggregat med usikkerhet og
ærlighetsmetadata. Server: `.venv/bin/python -m app.server`.
"""

from .qspec import InterpretedQuestion, QSpecError, interpret_question
from .service import AskResult, SimNorgeService

__all__ = [
    "InterpretedQuestion", "QSpecError", "interpret_question",
    "AskResult", "SimNorgeService",
]

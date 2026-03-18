"""Entity extraction and resolution logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz

from openrag.models.processing import EntityType

if TYPE_CHECKING:
    from openrag.models.processing import EntityCandidate


class EntityExtractor:
    """Extracts, resolves, and deduplicates entities from content blocks.

    Uses a two-stage approach:
    1. spaCy NER for standard types (PERSON, ORG, GPE).
    2. Fuzzy matching (rapidfuzz) for entity resolution/deduplication.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self._nlp: Any = None
        self._model_name = spacy_model

    def _ensure_nlp(self) -> None:
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self._model_name)
            except OSError:
                # Fallback or error? For now, let's try to download it if allowed,
                # but in this environment we'll assume it's pre-installed or we'll fail gracefully.
                raise RuntimeError(
                    f"spaCy model '{self._model_name}' not found. "
                    f"Run `python -m spacy download {self._model_name}`."
                ) from None

    def extract_from_text(self, text: str) -> list[dict[str, Any]]:
        """Extract basic entities using spaCy NER."""
        if not text:
            return []

        self._ensure_nlp()
        doc = self._nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "name": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        return entities

    def resolve(
        self,
        candidates: list[EntityCandidate],
        threshold: float = 85.0,
    ) -> list[dict[str, Any]]:
        """Resolve and deduplicate entity candidates.

        Groups candidates with similar names (fuzzy match) and returns
        a list of canonical entities.
        """
        if not candidates:
            return []

        # Sort by name to help group similar ones
        sorted_cands = sorted(candidates, key=lambda c: c.name.lower())
        
        resolved: list[dict[str, Any]] = []
        
        for cand in sorted_cands:
            found = False
            for existing in resolved:
                # Fuzzy match canonical name
                score = fuzz.ratio(cand.canonical_name, existing["canonical_name"])
                if score >= threshold:
                    # Same entity: merge? For now just skip or update confidence
                    existing["mentions"].append(cand.name)
                    existing["confidence"] = max(existing["confidence"], cand.confidence)
                    found = True
                    break
            
            if not found:
                resolved.append({
                    "name": cand.name,
                    "canonical_name": cand.canonical_name,
                    "type": cand.entity_type,
                    "confidence": cand.confidence,
                    "mentions": [cand.name],
                })
                
        return resolved

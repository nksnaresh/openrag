"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class BaseEmbeddingAdapter(ABC):
    """Abstract interface for all embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for a list of strings.

        Args:
            texts: List of strings to encode.

        Returns:
            List of numpy arrays (dimension matches the provider's model).
        """

    @abstractmethod
    def dimension(self) -> int:
        """Return the output dimension of this embedding model."""

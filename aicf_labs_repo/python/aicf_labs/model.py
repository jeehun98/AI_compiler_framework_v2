"""Base model declaration API."""

from abc import ABC, abstractmethod


class Model(ABC):
    """A declarative model representation with no execution behavior."""

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable structural summary."""

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return self.summary()

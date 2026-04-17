from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from ..params import GenerationParams


class ContinuationStrategy(ABC):
    name: str

    @abstractmethod
    def stream(self, prefix: str, params: GenerationParams) -> AsyncGenerator[str, None]: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

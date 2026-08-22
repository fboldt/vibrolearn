from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PreparedData:
    X: Any
    y: np.ndarray
    domains: np.ndarray | None = None


class DataAdapter(ABC):

    @abstractmethod
    def prepare(
        self,
        registers,
        experimental_setup,
        training=False
    ) -> PreparedData:
        """
        Converte registros de aquisição na representação
        esperada pelo método.
        """
        pass
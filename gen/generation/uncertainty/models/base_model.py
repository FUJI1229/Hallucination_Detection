from abc import ABC, abstractmethod
from typing import List, Text


class BaseModel(ABC):

    stop_sequences: List[Text]

    @abstractmethod
    def predict(self, input_data, temperature):
        pass

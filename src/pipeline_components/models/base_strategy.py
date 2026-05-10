from typing import Protocol


class BaseModelStrategy(Protocol):
    """
    Serves as blueprint for enforcing this strategy during development
    """

    def create_model(self, trial) -> object:
        pass

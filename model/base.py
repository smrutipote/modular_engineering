from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, data) -> None:
        """Train the model using the Data object."""
        ...

    @abstractmethod
    def predict(self, X_test) -> None:
        """Run predictions on test data."""
        ...

    @abstractmethod
    def print_results(self, data) -> None:
        """Print classification report."""
        ...
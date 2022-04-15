from abc import ABC, abstractmethod


class Component(ABC):
    """
    The base Component class declares common operations for both simple and
    complex objects of a composition.
    """

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError


class Composite(Component):
    """
    The Composite class represents the complex components that may have
    children.
    In our case, HilbertMap and LocalHilbertMapCollection are Composites.
    """

    def update(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError


class Leaf(Component):
    """
    The Leaf class represents the end objects of a composition. A leaf can't
    have any children.
    In our case, GlobalModel and LocalHilbertMap are Leafs.
    """

    def update(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

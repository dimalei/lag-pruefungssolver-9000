import numpy as np
from core import normalize


class Line:
    def __init__(self, aufpunkt: np.ndarray, richtungsvektor: np.ndarray):
        if not aufpunkt.shape == richtungsvektor.shape:
            raise ValueError("Dimensions don't match.")
        self.aufpunkt = aufpunkt
        self.richtungsvektor = richtungsvektor

    def is_on_line(self, vector: np.ndarray):
        if vector.shape != self.aufpunkt.shape:
            return False
        distance = self.distance_to(vector)
        return bool(np.isclose(distance, 0))

    def distance_to(self, vector: np.ndarray):
        ap = vector - self.aufpunkt
        area = np.cross(ap, self.richtungsvektor)
        distance = np.linalg.norm(area) / np.linalg.norm(self.richtungsvektor)
        return distance

    def get_dimensions(self) -> int:
        return self.aufpunkt.shape[0]

    def __str__(self) -> str:
        return f"Aufpunkt: {self.aufpunkt}, Richtungsvektor {self.richtungsvektor}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Line):
            return False
        same_dim = self.get_dimensions() == other.get_dimensions()
        same_dir = np.allclose(
            normalize(self.richtungsvektor), normalize(other.richtungsvektor)
        )
        on_line = self.is_on_line(other.aufpunkt)

        return same_dim and bool(same_dir) and on_line


if __name__ == "__main__":

    l1 = Line(np.array([0, 0, 0]), np.array([1, 0, 0]))
    p1 = np.array((0, 2, 0))
    dist = l1.distance_to(p1)

    print(dist)

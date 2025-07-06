import numpy as np

from . import core as co
from IPython.display import display, Math
from math import isclose
from itertools import combinations

ABS_TOLERANCE = 1e-10


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return v / norm


class Line:
    """This is a Line in parameterform and koordinatenform"""

    def __init__(self, aufpunkt: np.ndarray, richtungsvektor: np.ndarray):
        if not aufpunkt.shape == richtungsvektor.shape:
            raise ValueError("Dimensions don't match.")
        self.aufpunkt = aufpunkt
        self.richtungsvektor = richtungsvektor
        self.line = self.get_coordinate_form()

    def get_coordinate_form(self) -> np.ndarray:
        if self.richtungsvektor.shape[0] == 2:
            normal = (-self.richtungsvektor[1], self.richtungsvektor[0])
        else:
            normal = co.perpendicular_vector(self.richtungsvektor)
        constant = np.dot(self.aufpunkt, normal)
        return np.array((*normal, constant))

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

    def intersect(self, other: "Line"):
        """does not work!!"""
        if self == other:
            return self
        if np.allclose(
            normalize(self.richtungsvektor), normalize(other.richtungsvektor)
        ):
            return None
        lgs = np.stack((self.line, other.line))
        return co.lgs_particular_solution(lgs)

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

    def display(self):
        display(
            Math(
                co._latex_vector(self.aufpunkt, self.get_dimensions())
                + "+"
                + co._latex_directional_vector(self.richtungsvektor)
            )
        )

    @classmethod
    def intersect_all(cls, a: "Line", b: "Line", *others: "Line"):
        """does not work!!"""
        lines = [a, b, *others]
        intersections = combinations(lines, 2)
        points = []

        for a, b in intersections:
            intersection = a.intersect(b)
            if isinstance(intersection, np.ndarray):
                add = True
                for point in points:
                    if np.allclose(intersection, point):
                        add = False
                        break
                if add:
                    points.append(intersection)

        return points


class Plane:
    """
    This is a n-Dimensional Plane in Coordinate Form
                    1x + 2y + 3z = 5
    Plane(np.array((1,   2,   3,   5)))
    """

    def __init__(self, inhomogen_plane: np.ndarray) -> None:
        self.plane = inhomogen_plane
        self.normal = inhomogen_plane[:-1]
        self.constant = inhomogen_plane[-1]
        if isclose(np.linalg.norm(self.normal), 0):
            print("warning: normal is null")

    @classmethod
    def from_parameterform(
        cls,
        aufpunkt: np.ndarray,
        direction_u: np.ndarray,
        direction_v: np.ndarray,
    ):
        coefficients = np.cross(direction_u, direction_v).tolist()
        constant = np.dot(aufpunkt, coefficients)
        full_plane = np.append(coefficients, constant)
        return cls(full_plane)

    @property
    def dimensions(self) -> int:
        return self.normal.shape[0]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, np.ndarray):
            other = Plane(other)
        if isinstance(other, Plane) and self.dimensions == other.dimensions:
            return self.is_parallel(other) and isclose(
                self.distance_to_point(other.point_on_plane()), 0, abs_tol=ABS_TOLERANCE
            )
        return False

    def is_parallel(self, other: "Plane"):
        self_normal = normalize(self.normal)
        other_normal = normalize(other.normal)
        return np.allclose(self_normal, other_normal) or np.allclose(
            self_normal, -other_normal
        )

    def distance_to_point(self, vector: np.ndarray):
        # hesse normalenform
        if vector.shape[0] != self.normal.shape[0]:
            raise ValueError("Vectors are not in the same dimension.")
        numerator = np.matmul(self.normal, vector) - self.constant
        denominator = np.linalg.norm(self.normal)
        return abs(numerator / denominator)

    def point_on_plane(self) -> np.ndarray:
        # generate a point on the plane with the first pivot found
        for i, val in enumerate(self.normal):
            if val != 0:
                pivot_value = self.constant / val
                # force float array, otherwise this fails
                vector = np.zeros_like(self.normal, dtype=float)
                vector[i] = pivot_value
                return vector
        raise ValueError("Normal vector is zero.")

    def __str__(self) -> str:
        out = ""
        axis = ["x", "y", "z", "u", "v"]
        for i, coefficient in enumerate(self.normal):
            out += str(coefficient) + axis[i]
            if i < len(self.normal) - 1:
                out += "+"
        out += f"={self.constant}"
        return out

    def intersect(self, other: "Plane"):
        """Computes the Schnittmenge of 2 Planes

        Args:
            other (Plane): Another Plane within the same dimensions.

        Return:
            Plane: itslef if intersect with a duplicate
            Line: if an intersection exists
            None: if planes are paralell
        """
        if not self.dimensions == other.dimensions:
            raise ValueError("Planes are not in the same dimensions.")
        if self == other:
            # print("planes are the same")
            return self
        if self.is_parallel(other):
            # print("planes are paralell")
            return None

        lgs = np.stack((self.plane, other.plane))

        # co.solve_general_solution(lgs)
        return self.intersecting_line(other)

    def intersecting_line(self, other: "Plane") -> Line:
        """Create a Line by intersecting 2 non-paralell planes."""

        lgs = np.stack((self.plane, other.plane))
        aufpunkt = co.lgs_particular_solution(lgs)
        null_space = co.lgs_null_space(lgs)

        return Line(aufpunkt, null_space[0])

    @classmethod
    def intersect_all(cls, a: "Plane", b: "Plane", *others: "Plane"):
        planes = [a, b, *others]
        pairs = combinations(planes, 2)

        lines = []

        for i, two_planes in enumerate(pairs):
            print(f"Intersection {i+1}#:")
            a, b = two_planes
            intersection = a.intersect(b)

            if isinstance(intersection, Plane):
                print("identical")

            elif isinstance(intersection, Line):
                print(f"Line: {intersection}")
                lines.append(intersection)
            elif intersection == None:
                print(f"None. Planes are Paralell")

        if len(planes) > 1:
            lgs = np.stack([p.plane for p in planes])
            co.solve_general_solution(lgs)

        return lines


if __name__ == "__main__":
    pass

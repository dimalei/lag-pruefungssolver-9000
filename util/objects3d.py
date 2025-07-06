class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Point3D):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False

    def tule(self) -> tuple:
        return (self.x, self.y, self.z)


# class Line3 ():
#     __init(self, )

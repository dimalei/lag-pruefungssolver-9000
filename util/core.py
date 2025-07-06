import numpy as np

import sympy as sym
import scipy.linalg as sp
from random import random, randint
from typing import Iterable, Callable
import math
from IPython.display import display, Math


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return v / norm


def perpendicular_vector(vector: np.ndarray):
    if np.allclose(vector, 0):
        raise ValueError("zero vector has no well-defined perpendicular vectors")

    # pick an arbitrary vector not parallel to v
    if vector[0] != 0 or vector[1] != 0:
        # if x or y is nonzero, use [-y, x, 0]
        return np.array([-vector[1], vector[0], 0])
    else:
        # if v is along z-axis, use [1, 0, 0]
        return np.array([1, 0, 0])


def angle_between_vectors(v1: list, v2: list):
    """
    Computes the angle between 2 Vectors.
    Prints the direction of the angle for R² vectors as a Bonus
    Parameters:
        v1 (list): Vector 1
        v2 (list): vector 2
    Returns:
        (float): Angle between vectors in rad
    """
    if len(v1) == 2:
        v1_0 = v1 + [0]
        v2_0 = v2 + [0]
        if np.cross(v1_0, v2_0)[2] > 0:
            print("CCW turn")
        else:
            print("CW turn")
    return np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) * np.linalg.norm(v1))


def distance_to_line(Punkt: list, Aufpunkt: list, v: list):
    """
    Computes the distance from a point to a Line in R³
    Parameters:
        Punkt (list): Point to calculate the distance from
        Aufpunkt (list): Aufpunkt of the line
        v (list): directional vector 1
    Returns:
        int: Distance from the Point to the line
    """
    AP = np.array(Punkt) - np.array(Aufpunkt)
    area = np.cross(AP, v)
    distance = area / np.linalg.norm(np.array(v))

    return distance


def distance_to_plane(Punkt: list, Aufpunkt: list, v: list, w: list):
    """
    Computes the distance from a point to a Plane in R³ in 'Parameterform'
    Parameters:
        Punkt (list): Point to calculate the distance from
        Aufpunkt (list): Aufpunkt of the plane
        v (list): directional vector 1
        w (list): directional vector 2
    Returns:
        int: absolute Distance from the Point to the Plane
    """
    EP = np.array(Punkt) - np.array(Aufpunkt)
    paralellogramm_fl = np.linalg.norm(np.cross(v, w))
    paralellopid_vol = np.dot(EP, np.cross(v, w))  # spatprodukt
    distance = paralellopid_vol / paralellogramm_fl

    return abs(distance.item())


def distance_to_plane_hesse(Punkt: list, Koordinatenform: list):
    """
    Computes the distance from a Point to a Plane with Hesse-Normalenform
    Parameters:
        Punkt (list): Point to measure distance to
        Koordinatenform [list]: Homogen part of the Koordinatenform (without the = 0) as coeffitients
    Returns:
        float: Distance plane - point
    """
    koeffizienten = [
        Koordinatenform[:-1]
    ]  # konstante ignorieren, horizontale koeffizientenmatrix
    eq_top = np.matmul(koeffizienten, Punkt) + Koordinatenform[-1]
    eq_bottom = np.sqrt(sum([x**2 for x in Koordinatenform[:-1]]))
    return abs(eq_top / eq_bottom)


def plane_normal_to_coordinates(Aufpunkt: list, Normalenvektor: list):
    """
    Computes the Koordinatenform from the Normal Form of a plane
    Parameters:
        Aufpunkt (list): Aufpunkt of the plane
        Normalenvektor (list): Normalenvektor of the plane
    Returns:
        list: The homogenous verion of the coordinate form
    """
    konstante = -np.dot(Aufpunkt, Normalenvektor).item()
    koodrdinatenform = Normalenvektor + [konstante]
    return koodrdinatenform


def _find_variable_linear_combination(coeffs: list, variable_values: list, target):
    """
    resolves the value of a variable in a equation.
    Parameters:
        coeffs (list): coeffitient values
        variable_values (list): a list of variable values, 'None' marks the one to be found
        target: the target sum of the linear combination
    Returns:
        list: the completed variable values.
    """
    known_sum = sum(c * v for c, v in zip(coeffs, variable_values) if v is not None)
    # Solve for the missing variable
    missing_value = (target - known_sum) / coeffs[variable_values.index(None)]
    return [x if x != None else missing_value for x in variable_values]


def solve_plane_coordinates_to_parameter(Koordinatenform: list):
    """
    Nimmt eine homogene form einer Ebene und gibt die Koordinatenform zurück
    Aufpunkt + Richtungsvektoren
    """
    coeffs = Koordinatenform[:-1]
    target = -Koordinatenform[-1]
    pivot_index = 0
    for i, num in enumerate(coeffs):
        if num != 0:
            pivot_index = i
            break

    Aufpunkt = [0, 0]
    Aufpunkt.insert(pivot_index, None)  # type: ignore
    Aufpunkt = _find_variable_linear_combination(coeffs, Aufpunkt, target)

    v_1 = [1, 0]
    v_1.insert(pivot_index, None)  # type: ignore
    v_1 = _find_variable_linear_combination(coeffs, v_1, target)
    v1 = np.array(Aufpunkt) - np.array(v_1)

    v_2 = [0, 1]
    v_2.insert(pivot_index, None)  # type: ignore
    v_2 = _find_variable_linear_combination(coeffs, v_2, target)
    v2 = np.array(Aufpunkt) - np.array(v_2)

    out = (
        _latex_vector([], 3)
        + "="
        + _latex_vector(Aufpunkt)
        + "+"
        + _latex_directional_vector(v1, 0)
        + "+"
        + _latex_directional_vector(v2, 1)
    )
    display(Math(out))


def solve_plane_normal_to_coordinate(Aufpunkt: list, Normalenvektor: list):
    normalenform = plane_normal_to_coordinates(Aufpunkt, Normalenvektor)
    coeffs = normalenform[:-1]
    out = ""
    for i, coef in enumerate(coeffs):
        if i > 0:
            out += "+"
        out += f"{coef} x_{i}"
    out += f"{normalenform[-1]}=0"
    display(Math(out))


def solve_plane_parameter_to_coordinate(Aufpunkt: list, v1: list, v2: list):
    coeffs = np.cross(v1, v2).tolist()
    constant = np.dot(Aufpunkt, coeffs)
    out = ""
    for i, coef in enumerate(coeffs):
        if i > 0:
            out += "+"
        out += f"{coef} x_{i}"
    out += f"={constant}"
    display(Math(out))


def eliminate(Aa_in, tolerance=np.finfo(float).eps * 10.0, fix=False, verbos=0):
    # eliminates first row
    # assumes Aa is np.array, As.shape>(1,1)
    Aa = Aa_in
    Nn = len(Aa)
    # Mm = len(Aa[0,:])
    if Nn < 2:
        return Aa
    else:
        if not fix:
            prof = np.argsort(np.abs(Aa_in[:, 0]))
            Aa = Aa[prof[::-1]]
        if np.abs(Aa[0, 0]) > tolerance:
            el = np.eye(Nn)
            el[0:Nn, 0] = -Aa[:, 0] / Aa[0, 0]
            el[0, 0] = 1.0 / Aa[0, 0]
            if verbos > 50:
                print("Aa \n", Aa)
                print("el \n", el)
                print("pr \n", np.matmul(el, Aa))
            return np.matmul(el, Aa)
        else:
            return Aa


def _FirstNonZero(lis):
    return next((i for i, x in enumerate(lis) if x), len(lis) - 1)


def _SortRows(Aa):
    inx = np.array(list(map(_FirstNonZero, Aa)))
    # print('inx: ',inx,inx.argsort())
    return Aa[inx.argsort()]


def mrref(Aa_in: np.ndarray, verbos=0):
    """
    Computes a modified row-reduced echelon form (RREF) of a matrix.

    Parameters:
        Aa_in (np.ndarray): The input matrix to be row reduced.
        verbos (int): Verbosity level for debug printing (default: 0).

    Returns:
        np.ndarray: Matrix in modified RREF form.
    """
    Aa = Aa_in * 1.0
    Nn = len(Aa)
    kklist = np.arange(0, Nn - 1)

    # print("kklist", kklist)

    for kk in kklist:
        Aa[kk:, kk:] = eliminate(Aa[kk:, kk:], verbos=verbos - 1)
    Aa = _SortRows(Aa)
    Aa = np.flipud(Aa)

    for kkh in kklist:
        kk = _FirstNonZero(Aa[kkh, :])
        Aa[kkh::, kk::] = eliminate(Aa[kkh::, kk::], fix=True, verbos=verbos - 1)
    return np.flipud(Aa)


def mnull(Aa_in, leps=np.finfo(float).eps * 10, verbos=0):
    """
    Nimmt die Koeffizientenmatrix von homogenen Gleichungsysteme und gibt die
    Richtungsvektoren zurück (bzw. Richtungsvektoren des Nullraums)

    Parameters:
        Aa_in (np.ndarray): Input matrix whose null space is to be computed.
        leps (float): Threshold below which values are considered zero (default: 10×machine epsilon).
        verbos (int): Verbosity level for debug printing (default: 0).

    Returns:
        np.ndarray: Richtungsvektor aus der Homogenen Gleichung.
                    A matrix whose columns form a basis for the null space of `Aa_in`.

    """
    Aa = mrref(Aa_in)
    Aa = Aa[list(map(np.linalg.norm, Aa)) > leps]  # extract non-zero linies
    mpiv = np.array(Aa[0] * 0, dtype=bool)
    jj = 0  # setup mask, indicating pivot-variables
    for ro in Aa > leps:
        for x in ro[jj:]:
            if x:
                mpiv[jj] = True
                jj = jj + 1
                break

    jj = 0
    la = Aa[:, mpiv]
    veo = []
    for jj in np.argwhere(mpiv == False):
        ve = np.linalg.lstsq(la, -Aa[:, jj], rcond=None)[0]
        vel = np.zeros((len(mpiv)))
        vel[mpiv] = ve[:, 0]
        vel[jj] = 1
        veo.append(vel)

    opt = np.array(veo).T
    if verbos > 10:
        print(Aa.shape, opt.shape)
        print("Test: ", np.matmul(Aa, opt))
    return opt


def is_null(vector) -> bool:
    """
    Checks if a vector is a null vector
    Parameters:
        vector (Iterable, Number): vector to be checked
    Returns:
        bool: true if input is a null-vector
    """

    # scalar to list
    if not isinstance(vector, Iterable):
        vector = [vector]

    # check vector null
    for number in vector:
        try:
            if number != 0:
                return False
        except:
            # return false input is a matrix
            return False
    return True


def random_vector(dimension: int, R=True) -> list:
    """
    Generates a vector with random values,
    Parameters:
        dimension (int): Defines the dimension of the vector
        R (bool): use Real numbres if true, integers of false
    Returns:
        list: a random vector
    """
    if R:
        return [random() for i in range(dimension)]
    return [randint(-99999, 99999) for i in range(dimension)]


def _latex_vector(vector=[], dimension=1):
    """
    returns a String that can be rendered by katex as a vector.
    Parameters:
        components (Number): value of that component, null if variable (renderd as x_1, x_2)
        dimension (int): number of components of the latex_vector
    Returns:
        String: latex formatted vector (pmatrix)
    """
    # cast to list if components is np.narray
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()

    if dimension > len(vector):
        # fill up None components if missing
        vector += [None] * (dimension - len(vector))

    # write latex string
    out = "\\begin{pmatrix}"
    for i, component in enumerate(vector):
        out += (
            f"x_{i+1}\\\\"
            if component == None
            else f"{component:.3f}".rstrip("0").rstrip(
                "."
            )  # don't print unnecassary decimals
            + "\\\\"
        )
    out += "\\end{pmatrix}"
    return out


def _latex_directional_vector(vector, index=0):
    """
    returns a String that renders latex vectors with variable scalars
    Parameters:
        vector (Iterable): the vector that is printed
        index (int): defines the name of the scalar
    Returns:
        String: latex formatted vector (pmatrix) with symbolic scalar
    """
    scalars = [
        r"\lambda",
        r"\mu",
        r"\gamma",
    ]
    index = index % len(scalars)
    return scalars[index] + r"\cdot" + _latex_vector(vector)


def _latex_null_space(lgs: np.ndarray, verbos=0):
    """
    Returns the null space formatted to be rendered in latex
    Parameters:
        lgs (np.ndarray): homogenous lgs (no constants)
        verbos (int): print debug info if >0
    Returns:
        String: latex formated null-space
    """
    richtungsvektoren = mnull(lgs)
    if verbos > 0:
        print(richtungsvektoren)

    if is_null(richtungsvektoren):
        return _latex_vector([0] * lgs.shape[1])

    anzahl_richtungsvektoren = richtungsvektoren.shape[1]

    out = ""
    for i in range(anzahl_richtungsvektoren):

        out += ("+" if i != 0 else "") + _latex_directional_vector(
            richtungsvektoren[:, i], i
        )

    return out


def lgs_null_space(lgs: np.ndarray):
    richtungsvektoren = mnull(lgs)

    if is_null(richtungsvektoren):
        return np.zeros(lgs.shape[1])

    anzahl_richtungsvektoren = richtungsvektoren.shape[1]

    nullspace = []
    for i in range(anzahl_richtungsvektoren):
        nullspace.append(richtungsvektoren[:-1, i])

    return nullspace


def _latex_particular_solution(lgs: np.ndarray, verbos=0):
    """
    Calulates a particular solution to a LGS (Aufpunkt)
    Parameters:
        lgs (np.ndarray): inhomogenous LGS
        verbos (int): print debug info if >0
    Returns:
        String: latex formatted partiular solution (vector)
    """
    # löse das inhomogene lgs
    solved = mrref(lgs)
    if verbos > 0:
        print(f"Solved:\n{solved}")

    m, n_plus_1 = solved.shape
    n = n_plus_1 - 1  # number of variables
    x = np.zeros(n)

    # Identify pivot columns
    row = 0
    for col in range(n):
        if row < m and abs(solved[row, col]) > 1e-10:
            x[col] = solved[row, -1]  # Assign RHS value to pivot variable
            row += 1
        # else: free variable -> stay 0

    return _latex_vector(x)


def lgs_particular_solution(lgs: np.ndarray) -> np.ndarray:

    solved = mrref(lgs)

    dimensions, n_plus_1 = solved.shape
    no_variables = n_plus_1 - 1  # number of variables
    solution = np.zeros(no_variables)

    # Identify pivot columns
    row = 0
    for col in range(no_variables):
        if row < dimensions and abs(solved[row, col]) > 1e-10:
            solution[col] = solved[row, -1]  # Assign RHS value to pivot variable
            row += 1
        # else: free variable -> stay 0

    return solution


def display_vector(vector=[], dimension=1):
    display(Math(_latex_vector(vector, dimension)))


def display_matrix(matrix: np.ndarray):

    out = r"\begin{bmatrix}"

    for row in matrix:
        for i, column in enumerate(row):
            if i > 0:
                out += "&"
            out += f"{column:.3f}".rstrip("0").rstrip(".")
        out += r"\\"
    out += r"\end{bmatrix}"

    display(Math(out))


def display_coefficient_matrix(matrix: np.ndarray):

    out = r"\begin{vmatrix*}[r]"

    for row in matrix:
        for i, column in enumerate(row):
            if i > 0:
                out += "&"
            out += f"{column:.3f}".rstrip("0").rstrip(".") + f"x_{i+1}"
        out += r"\\"
    out += r"\end{vmatrix*}"

    display(Math(out))


def solve_general_solution(lgs: np.ndarray, verbos=0):
    """
    Displays the general solution of intersecting planes, rendered in beautyful latex.
    Parameters:
        lgs (np.ndarray): An inhomogenous LGS (includes the right side of the equation)
        verbos (int): prints debug info if >0
    """

    # check intersecting

    # the lgs is expeted to have constants,
    # thus dimensions is with minus constants
    dimensionen = lgs.shape[1] - 1
    out = _latex_vector([], dimensionen) + "="
    out += _latex_particular_solution(lgs, verbos) + "+"

    # make lgs homogen, and get null space
    out += _latex_null_space(lgs[:, :-1])

    display(Math(out))


def solve_null_space(lgs: np.ndarray, verbos=0):
    """
    Displays the null-space of intersecting planes, rendered in beautyful latex.
    Parameters:
        lgs (np.ndarray): A inhomogenous LGS
        verbos (int): prints debug info if >0
    """
    dimensionen = lgs.shape[1]
    out = _latex_vector([], dimensionen) + "="
    out += _latex_null_space(lgs, verbos)

    display(Math(out))


def solve_check_linearity(input_dimensions: int, transformation: Callable):
    """
    Checks if a transformation is linear or not by applying random real values.
    Prints results directly into console.

    Parameters:
        input_dimensions (int): expected input dimensions of the transformation
        transformation (Callable): lambda expression representing the transformation
    """
    DECIMAL_PRECISION = 8

    # check null-vector
    null_vector = np.array([0 for i in range(input_dimensions)])
    result = transformation(null_vector)

    has_null = is_null(result)

    print(
        f"{"✔️" if has_null else "❌"} Null Vector:  {result} is{"" if has_null else " NOT"} null"
    )

    # check homogenity
    random_vector_v = np.array([random() for i in range(input_dimensions)])
    random_scalar = random()

    a = transformation(np.multiply(random_scalar, random_vector_v))
    b = np.multiply(random_scalar, transformation(random_vector_v))

    is_homogen = np.allclose(
        np.round(a, DECIMAL_PRECISION), np.round(b, DECIMAL_PRECISION)
    )

    print(
        f"{"✔️" if is_homogen else "❌"} Homogenity:   L(lambda * v) = {a} =?= lambda * L(v) = {b}"
    )

    # check additivity
    random_vector_w = np.array([random() for i in range(input_dimensions)])

    a = transformation(np.add(random_vector_v, random_vector_w))
    b = np.add(transformation(random_vector_v), transformation(random_vector_w))
    is_additiv = np.allclose(
        np.round(a, DECIMAL_PRECISION), np.round(b, DECIMAL_PRECISION)
    )

    print(
        f"{"✔️" if is_additiv else "❌"} Additivity:   L(v + w) = {a} =?= L(v) + L(w) = {b}"
    )

    # print summary
    if has_null and is_homogen and is_additiv:
        print("✔️ Transformation is linear")
    else:
        print("❌ Transformation is NOT linear")


def solve_linear_combination(*vectors: list):
    """
    Checks if input vectors are linearly depended (co-planar)
    and prints the linear combination to create the null-vector

    Parameters:
        *vectors (list): Input vectors. Must be of the same dimesnion.

    """
    # assemble matrix from vectors
    if not isinstance(vectors, np.ndarray):
        try:
            matrix = np.array(vectors)
        except ValueError:
            print("Vektoren haben nicht die gleiche Dimension!")
            return
    # unless it's already a matrix
    else:
        matrix = vectors

    if matrix.size <= 1:
        print("Mehr als 1 Vektor nötig!")
        return

    # generate LGS
    # print(lgs)
    rows, cols = matrix.shape
    matrix = np.concatenate([matrix, np.eye(rows, cols)], axis=1)
    # print(lgs)

    # eliminate components
    result = mrref(matrix)

    # extract last vector
    last_row = result[-1]
    last_row_coefficients = last_row[:cols]

    # check if null vector
    if is_null(last_row_coefficients):
        print("✔️ Linear abhängig")
        # extract combination
        combination = last_row[cols:]
        for index, vector in enumerate(combination):
            print(f"{chr(ord("a") + index)}:  {int(vector)}")
        return

    print("❌ nicht Linear abhängig")


# testing
if __name__ == "__main__":

    pass

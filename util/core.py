import numpy as np

# symbolisches Rechnen mit u,v,x,y,z
# from sympy import *
import sympy as sym
import scipy.linalg as sp
from random import random, randint
from typing import Iterable, Callable
import math
from IPython.display import display, Math

DECIMAL_PRECISION = 10


# eigene Funktionen
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


def FirstNonZero(lis):
    return next((i for i, x in enumerate(lis) if x), len(lis) - 1)


def SortRows(Aa):
    inx = np.array(list(map(FirstNonZero, Aa)))
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
    Aa = SortRows(Aa)
    Aa = np.flipud(Aa)

    for kkh in kklist:
        kk = FirstNonZero(Aa[kkh, :])
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


def linear_combination(*vectors: list):
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


def check_linearity(input_dimensions: int, transformation: Callable):
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


def is_null(vector) -> bool:

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
    if R:
        return [random() for i in range(dimension)]
    return [randint(-99999, 99999) for i in range(dimension)]


def latex_vector(dimension: int, components=[]):
    """
    returns a String that can be rendered by katex as a vector.
    Parameters:
        dimension (int): number of components of the latex_vector
        components (Number): value of that component, null if variable (renderd as x_1, x_2)

    Returns:
        String
    """
    # cast to list if components is np.narray
    if isinstance(components, np.ndarray):
        components = components.tolist()

    # fill up None components if missing
    components += [None] * (dimension - len(components))
    # write latex string
    out = "\\begin{pmatrix}"
    for i in range(dimension):
        out += (
            f"x_{i+1}\\\\"
            if components[i] == None
            else f"{components[i]:.3f}".rstrip("0").rstrip(
                "."
            )  # don't print unnecassary decimals
            + "\\\\"
        )
    out += "\\end{pmatrix}"
    return out


def latex_directional_vector(dimension: int, components, index=0):
    """
    returns a String that renders latex vectors with variable scalars
    Parameters:
        index (int): defines the name of the scalar
    """
    scalars = [
        r"\lambda",
        r"\mu",
        r"\gamma",
    ]
    index = index % len(scalars)
    return scalars[index] + r"\cdot" + latex_vector(dimension, components)


def null_space(lgs: np.ndarray, verbos=0):

    dimensionen = lgs.shape[1]
    richtungsvektoren = mnull(lgs)
    if verbos > 0:
        print(richtungsvektoren)

    if is_null(richtungsvektoren):
        print(f"Triviale Lösung: {richtungsvektoren}")
        return

    anzahl_richtungsvektoren = richtungsvektoren.shape[1]

    out = latex_vector(dimensionen) + "="
    for i in range(anzahl_richtungsvektoren):

        out += ("+" if i != 0 else "") + latex_directional_vector(
            dimensionen, richtungsvektoren[:, i], i
        )

    display(Math(out))


# testing
if __name__ == "__main__":

    pass

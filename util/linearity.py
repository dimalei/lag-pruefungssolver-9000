from typing import Callable
import numpy as np
import sympy as sp
import core as co


sp.Symbol


def linearity(transformation: Callable, input_dimensions: int):
    """
    Checks if a transformation is linear using sybolic phyton.
    Prints results directly into console.

    Parameters:
        transformation (Callable): lambda expression representing the transformation
        *symbols (sp.Symbol): sympi symbols
    """

    u, v = generate_symbol_vectors(input_dimensions)

    # additivity L(u + v) == L(u) + L(v)
    lhs_add = transformation(*(u + v))
    rhs_add = transformation(*u) + transformation(*v)
    additivity = sp.simplify(lhs_add - rhs_add)

    is_additive = False
    try:
        sum = recursive_sum(additivity)
        if sum == 0:
            is_additive = True
    except:
        is_additive = False

    print(f"{"Is additive. ✅" if is_additive else "Is NOT additive. ❌"}")
    print(additivity)

    # Homogeneity: L(c*u) == c*L(u)
    c = sp.Symbol("c", real=True)

    lhs_hom = transformation(*(c * u))
    rhs_hom = c * transformation(*u)
    homogeneity = sp.simplify(lhs_hom - rhs_hom)

    is_homogenous = False
    try:
        sum = recursive_sum(homogeneity)
        if sum == 0:
            is_homogenous = True
    except:
        is_homogenous = False

    print(f"{"Is homogenous. ✅" if is_homogenous else "Is NOT homogenous. ❌"}")
    print(homogeneity)

    print(
        f"{"Linearity checks out. 📐" if is_additive and is_homogenous else " Linearity does not check out. 🍆"}"
    )


def generate_symbol_vectors(dimensions: int):
    """generate symbolic vector pair in n-dimensions"""
    dimension_names = "xyzuvwabcdefghijklmnopqrst"

    if dimensions > len(dimension_names):
        raise ValueError(
            f"Cannot generate more than {len(dimension_names)} pairs with this function."
        )

    vector_u = []
    vector_v = []
    for i in range(dimensions):
        letter = dimension_names[i]
        s1, s2 = sp.symbols(f"{letter}1 {letter}2")
        vector_u.append(s1)
        vector_v.append(s2)
    return (sp.Matrix(vector_u), sp.Matrix(vector_v))


def recursive_sum(input_list: list):
    total = 0
    for elem in input_list:
        if isinstance(elem, list):
            total += recursive_sum(elem)
        else:
            total += elem
    return total


# testing
if __name__ == "__main__":
    phi = sp.symbols("phi", real=True)
    transformation = lambda x, y: sp.Matrix([sp.cos(phi) * x, sp.sin(phi) * y])
    linearity(transformation, 2)

    transformation2 = lambda x, y: x**2
    linearity(transformation2, 2)

    transformation3 = lambda x, y: x + 1
    linearity(transformation3, 2)

import sympy as sp
from IPython.display import display, Math
import math


def real_and_imaginary_part(expression: sp.Expr):
    simplified = sp.simplify(expression)
    display(Math(r"\text{Simplified: }" + sp.latex(simplified)))

    # real and imaginary parts
    real_part = sp.re(simplified)
    imag_part = sp.im(simplified)

    # display with latex
    display(Math(r"\text{Re(z): }" + sp.latex(real_part)))
    display(Math(r"\text{Im(z): }" + sp.latex(imag_part)))


def polar_to_cartesian(norm, angle):
    """
    Zeigt Polarkoordinaten in Reell und Imaginärteil. Beispiel:
        norm = math.sqrt(2)
        phi = (7 * math.pi) / 36
        im.polar_to_cartesian(norm, phi)

    Args:
        norm (Number): Die Norm (länge) oder |z|
        angle (Number): Winkel in RAD
    """
    # satz von euler
    coordinates = norm * (sp.cos(angle) + sp.I * sp.sin(angle))
    real_and_imaginary_part(coordinates)


def cartesian_to_polar(expression: sp.Expr):
    """
    Gibt polarkoordinaten zurück aus real & imaginärteil
        expr = sp.Rational(2,3)* sp.E ** (sp.I * 3 * sp.pi/2)
        im.cartesian_to_polar(expr)
    """
    simplified = sp.simplify(expression)
    # real and imaginary parts
    real_part = sp.re(simplified)
    imag_part = sp.im(simplified)

    norm = sp.simplify(sp.sqrt(real_part**2 + imag_part**2))  # type: ignore
    angle = sp.simplify(sp.atan2(imag_part, real_part))

    norm_symbol = "|z|"
    phi_symbol = r"\varphi"

    display(Math(f"{norm_symbol}= {sp.latex(norm)} , {phi_symbol} = {sp.latex(angle)}"))

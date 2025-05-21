import numpy as np

# symbolisches Rechnen mit u,v,x,y,z
# from sympy import *
import sympy as sym
import scipy.linalg as sp
import pprint


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


def mrref(Aa_in, verbos=0):
    Aa = Aa_in * 1.0
    Nn = len(Aa)
    kklist = np.arange(0, Nn - 1)
    # print('kklist', kklist)
    for kk in kklist:
        Aa[kk:, kk:] = eliminate(Aa[kk:, kk:], verbos=verbos - 1)
    Aa = SortRows(Aa)
    Aa = np.flipud(Aa)
    # for kk in kklist:
    for kkh in kklist:
        kk = FirstNonZero(Aa[kkh, :])
        Aa[kkh::, kk::] = eliminate(Aa[kkh::, kk::], fix=True, verbos=verbos - 1)
    return np.flipud(Aa)


def mnull(Aa_in, leps=np.finfo(float).eps * 10, verbos=0):
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


def ebenen_schneiden(planes: np.ndarray):
    pass


def linearkombination_vektoren(*vectors: list):
    try:
        lgs = np.array(vectors)
    except ValueError:
        print("Vektoren haben nicht die gleiche Dimension!")
        return

    if lgs.size <= 1:
        print("Mehr als 1 Vektor nötig!")
        return

    # generate LGS
    # print(lgs)
    rows, cols = lgs.shape
    lgs = np.concatenate([lgs, np.eye(rows, cols)], axis=1)
    # print(lgs)

    # eliminate components
    result = mrref(lgs)

    # extract last vector
    last_row = result[-1]
    last_row_coefficients = last_row[:cols]

    # check if null vector
    if is_null(last_row_coefficients):
        print("Linear abhängig ✔️")
        # extract combination
        combination = last_row[cols:]
        for index, vector in enumerate(combination):
            print(f"{chr(ord("a") + index)}:  {int(vector)}")
        return

    print("nicht Linear abhängig ❌")


def is_null(vector: list) -> bool:
    for number in vector:
        if number != 0:
            return False
    return True


# def pprint_dict(message: dict):
#     for property in message.keys():
#         print(f"{property}:")

#         pass


# testing
if __name__ == "__main__":
    a = [3, 2, 0]
    b = [0, 4, 3]
    c = [3, 10, 12]

    linearkombination_vektoren(a, b, c)

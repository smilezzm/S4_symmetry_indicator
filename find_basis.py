# Compute the Hilbert basis with PyNormaliz in the Normaliz environment.
# R3/R4 and X3/X4 have C2 eigenvalues -i/+i; the scaled R coefficients are +2/-2.

from pathlib import Path

import numpy as np
from PyNormaliz import Cone

N_ROWS = 20
CR = np.zeros((8, N_ROWS), dtype=int)
CR[0, 0] = CR[0, 1] = 1
CR[0, 12] = CR[0, 13] = -1
CR[1, 2] = CR[1, 3] = 1
CR[1, 14] = CR[1, 15] = -1
CR[2, 4] = CR[2, 5] = 1
CR[2, 8] = CR[2, 9] = -1
CR[3, 6] = CR[3, 7] = 1
CR[3, 10] = CR[3, 11] = -1
CR[4, 16] = 1
CR[4, 18] = -1
CR[5, 17] = 1
CR[5, 19] = -1
CR[6, 0:4] = 1
CR[6, 16] = CR[6, 17] = -1
CR[7, 4:8] = 1
CR[7, 16] = CR[7, 17] = -1

# for z_{4S} indicator
P_1 = np.array([0,0,0,0,1.5,-0.5,-1.5,0.5,0,0,0,0,1.5,-0.5,-1.5,0.5,1.0,-1.0,0,0])
# for \delta_{2S} indicator
P_2 = np.array([-1,0,1,0,1,0,-1,0,-1,0,1,0,1,0,-1,0,0,0,0,0], dtype=int)
P_1 = (2 * P_1).astype(int)

def monoid_hilbert_basis(CR, P1, P2, mods=(8, 2)):
    """Hilbert basis of {B >= 0 : CR·B = 0, P1·B ≡ 0 (mod 8), P2·B ≡ 0 (mod 2)}."""
    CR = np.atleast_2d(np.asarray(CR, dtype=int))
    n  = CR.shape[1]

    equations    = CR.tolist()                              # CR·B = 0
    inequalities = np.eye(n, dtype=int).tolist()            # B_i >= 0  (makes the cone pointed)
    congruences  = [list(map(int, P1)) + [int(mods[0])],    # P1·B ≡ 0 (mod 8)
                    list(map(int, P2)) + [int(mods[1])]]    # P2·B ≡ 0 (mod 2)

    C = Cone(equations=equations,
             inequalities=inequalities,
             congruences=congruences)
    H = np.array(C.HilbertBasis(), dtype=int)               # one generator per row
    return H

H = monoid_hilbert_basis(CR, P_1, P_2)
assert np.all(H >= 0) and np.all(H @ CR.T == 0)
assert np.all(H @ P_1 % 8 == 0) and np.all(H @ P_2 % 2 == 0)
print("Hilbert basis: ", H.shape[0], "generators, dim", H.shape[1])

out_path = Path(__file__).resolve().parent / "hilbert_basis.npy"
np.save(out_path, H)
print("Saved Hilbert basis to", out_path)

"""Minimal S4-only axion-insulator tight-binding model and its numerical diagnostics.

Basis (A up, A dn, B up, B dn); tau = orbital, sigma = spin.

    H(k) = eps(k) tau_x s0 + u sin kx tau_y sx + u sin ky tau_y sy + v sin kz tau_y sz
           + g(k) tau_z s0 + m_z tau_0 sz
    eps  = eps0 - cos kx - cos ky + 2 cos kz
    g    = eta (cos kx - cos ky) + kappa sin kz          (S4-odd form factor)
    U_S4 = tau_x (x) exp(-i pi/4 sz),  S4 k = (ky, -kx, -kz),  U_S4^4 = -1

For u = v = 1 the spectrum is closed-form:
    E = +-sqrt((rho +- m_z)^2 + sin^2 kx + sin^2 ky),  rho = sqrt(eps^2 + sin^2 kz + g^2)

The optional ``zeta`` term  zeta sin kx sin ky tau_y s0  is S4-symmetric and only used to push the
occupied-band Weyl nodes off the kz = pi plane in one diagnostic.
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------------------------------------------------------- matrices
s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"0": s0, "x": sx, "y": sy, "z": sz}

TXS0, TYSX, TYSY, TYSZ = np.kron(sx, s0), np.kron(sy, sx), np.kron(sy, sy), np.kron(sy, sz)
TZS0, T0SZ, TYS0 = np.kron(sz, s0), np.kron(s0, sz), np.kron(sy, s0)

U_S4 = np.kron(sx, np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)]))
U_C2 = U_S4 @ U_S4

DEFAULT = dict(eps0=2.4, u=1.0, v=1.0, m_z=0.3, eta=0.5, kappa=0.5, zeta=0.0)

K4 = {"G": (0, 0, 0), "Z": (0, 0, np.pi), "M": (np.pi, np.pi, 0), "A": (np.pi, np.pi, np.pi)}
K2 = {"X": (np.pi, 0, 0), "Y": (0, np.pi, 0), "R": (np.pi, 0, np.pi), "T": (0, np.pi, np.pi)}


# ----------------------------------------------------------------------------- Hamiltonians
def hamiltonian(K, p=DEFAULT):
    """Vectorised Bloch Hamiltonian: K of shape (..., 3) -> (..., 4, 4)."""
    K = np.asarray(K, dtype=float)
    kx, ky, kz = K[..., 0], K[..., 1], K[..., 2]
    eps = p["eps0"] - np.cos(kx) - np.cos(ky) + 2 * np.cos(kz)
    g = p["eta"] * (np.cos(kx) - np.cos(ky)) + p["kappa"] * np.sin(kz)
    sh = kx.shape + (1, 1)
    return (eps.reshape(sh) * TXS0
            + (p["u"] * np.sin(kx)).reshape(sh) * TYSX
            + (p["u"] * np.sin(ky)).reshape(sh) * TYSY
            + (p["v"] * np.sin(kz)).reshape(sh) * TYSZ
            + g.reshape(sh) * TZS0
            + p["m_z"] * T0SZ
            + (p.get("zeta", 0.0) * np.sin(kx) * np.sin(ky)).reshape(sh) * TYS0)


def hamiltonian_old(K, eps0=2.4, m_z=0.2, delta=0.2):
    """Eq. (12) of the original manuscript (kept only for the symmetry comparison)."""
    K = np.asarray(K, dtype=float)
    kx, ky, kz = K[..., 0], K[..., 1], K[..., 2]
    eps = eps0 - np.cos(kx) - np.cos(ky) + 2 * np.cos(kz)
    sh = kx.shape + (1, 1)
    return (eps.reshape(sh) * TXS0 + np.sin(kx).reshape(sh) * TYSX + np.sin(ky).reshape(sh) * TYSY
            + np.sin(kz).reshape(sh) * TYSZ + m_z * T0SZ + (delta * np.sin(kz)).reshape(sh) * TYS0)


def closed_form_bands(K, p=DEFAULT):
    K = np.asarray(K, dtype=float)
    kx, ky, kz = K[..., 0], K[..., 1], K[..., 2]
    eps = p["eps0"] - np.cos(kx) - np.cos(ky) + 2 * np.cos(kz)
    g = p["eta"] * (np.cos(kx) - np.cos(ky)) + p["kappa"] * np.sin(kz)
    rho = np.sqrt(eps**2 + np.sin(kz)**2 + g**2)
    sp = np.sin(kx)**2 + np.sin(ky)**2
    Em, Ep = np.sqrt((rho - p["m_z"])**2 + sp), np.sqrt((rho + p["m_z"])**2 + sp)
    return np.sort(np.stack([-Ep, -Em, Em, Ep], axis=-1), axis=-1)


def s4_k(k):
    return np.array([k[1], -k[0], -k[2]])


def k_grid(nk):
    g = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    return g, np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1)


# ----------------------------------------------------------------------------- symmetry scan
def _generate_oh():
    """The 48 signed-permutation matrices of the cubic holohedry O_h."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            R = np.zeros((3, 3), dtype=int)
            R[np.arange(3), perm] = signs
            ops.append(R)
    return ops


def _axis_label(v):
    """(canonical axis name, sign) for an integer direction vector v = sign * canonical.

    Cardinal axes are named 'x', 'y', 'z'; the others by their integer components, e.g. '(110)',
    '(1-10)', '(011)', '(111)', '(1-11)'."""
    v = np.asarray(v, dtype=int)
    v = v // np.gcd.reduce(np.abs(v))
    sign = 1 if v[np.nonzero(v)[0][0]] > 0 else -1
    m = v * sign
    if np.abs(m).sum() == 1:
        return "xyz"[int(np.argmax(m != 0))], sign
    return "(" + "".join("0" if c == 0 else ("1" if c > 0 else "-1") for c in m) + ")", sign


def _proper_name(R):
    """Name of a proper rotation (det = +1) of O_h: E, C2<axis>, C3<axis><sign>, C4<axis><sign>."""
    if np.array_equal(R, np.eye(3, dtype=int)):
        return "E"
    trace = int(np.trace(R))
    if trace == -1:                                    # 180 degrees: the axis is the +1 eigenvector
        S = R + np.eye(3, dtype=int)
        return "C2" + _axis_label(S[:, int(np.argmax(np.abs(S).sum(axis=0)))])[0]
    # 90 degrees (trace 1) or 120 degrees (trace 0): w = 2 sin(theta) n gives axis *and* sense
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    lab, sign = _axis_label(w)
    return ("C4" if trace == 1 else "C3") + lab + ("+" if sign > 0 else "-")


def _name_op(R):
    """Label of any of the 48 elements of O_h.

    Improper elements are named through the proper rotation -R = P R:
    -R = E -> P;  -R = C2(n) -> 'M(n)', the mirror whose *normal* is n;  -R = C4 -> S4;  -R = C3 -> S6.
    The S4 sign follows the C4 it is built from, so the S_4 of the model, k -> (ky, -kx, -kz), is 'S4z+'."""
    R = np.asarray(R, dtype=int)
    if int(round(np.linalg.det(R))) == 1:
        return _proper_name(R)
    name = _proper_name(-R)
    if name == "E":
        return "P"
    if name.startswith("C2"):
        return "M" + name[2:]
    return ("S4" if name.startswith("C4") else "S6") + name[2:]


def find_symmetry_rep(R, antiunitary, hfun, ks):
    """Solve H(Rk) U - U H(k)^(*) = 0 for a constant 4x4 U over many k. Returns (nullity, U)."""
    rows = []
    for k in ks:
        Hk = hfun(k).conj() if antiunitary else hfun(k)
        rows.append(np.kron(hfun(R @ k), np.eye(4)) - np.kron(np.eye(4), Hk.T))
    M = np.vstack(rows)
    _, sv, Vh = np.linalg.svd(M)
    nullity = int(np.sum(sv < 1e-8 * sv.max()))
    if nullity == 0:
        return 0, None
    U = Vh[-1].conj().reshape(4, 4)      # M = U S Vh, so the null vector is the conjugate of a row of Vh
    return nullity, U * 2 / np.linalg.norm(U)


def magnetic_point_group(hfun, n_k=40, seed=0, verbose=False, return_details=False):
    """Brute-force scan over the full cubic holohedry: 48 R in O_h x {unitary, antiunitary} = 96 candidates.

    R is the *complete* action on momentum in both cases, so an antiunitary hit at R is the operation
    (-R) T.  Since O_h contains -1, letting R run over all of O_h in both lines exhausts O_h + O_h T.
    A candidate is accepted only when its null space is one dimensional *and* its generator is unitary;
    anything with a nonzero but inadmissible null space is reported and rejected.

    hfun : callable k(3,) -> H (4x4).  Returns the list of symmetry labels found (and, optionally,
    a list of dicts with R, the antiunitary flag, the nullity and the representation matrix U)."""
    rng = np.random.default_rng(seed)
    ks = rng.uniform(-np.pi, np.pi, (n_k, 3))
    found, details = [], []
    for R in _generate_oh():
        for anti in (False, True):
            nullity, U = find_symmetry_rep(R, anti, hfun, ks)
            if not nullity:
                continue
            label = _name_op(-R) + "T" if anti else _name_op(R)
            unit_err = np.abs(U @ U.conj().T - np.eye(4)).max()
            admissible = (nullity == 1) and (unit_err < 1e-10)
            if verbose or not admissible:
                print(f"    {label:12s} {'antiunitary' if anti else 'unitary    '} "
                      f"nullity={nullity} unitarity err={unit_err:.1e}"
                      + ("" if admissible else "   REJECTED: no one-dimensional unitary null space"))
            if admissible:
                found.append(label)
                details.append(dict(label=label, R=R.copy(), antiunitary=anti, nullity=nullity,
                                    U=U, unitarity_error=unit_err))
    return (found, details) if return_details else found


# ----------------------------------------------------------------------------- spectrum, gaps, nodes
def bands_on_grid(p, nk=40):
    _, K = k_grid(nk)
    return K, np.linalg.eigvalsh(hamiltonian(K, p))


def _refined_min(fun, K, val, n_start=4):
    flat = val.reshape(-1)
    idx = np.argsort(flat)[:n_start]
    best = (flat[idx[0]], K.reshape(-1, 3)[idx[0]])
    for i in idx:
        r = minimize(fun, K.reshape(-1, 3)[i], method="Nelder-Mead",
                     options=dict(xatol=1e-8, fatol=1e-12, maxiter=600))
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return best


def min_gap(p, nk=40):
    """(value, k) of the minimal bulk gap E3-E2, refined from the grid minimum."""
    K, E = bands_on_grid(p, nk)
    return _refined_min(lambda k: np.diff(np.linalg.eigvalsh(hamiltonian(k, p)))[1], K, E[..., 2] - E[..., 1])


def min_occupied_splitting(p, nk=40):
    """(value, k) of the minimal splitting E2-E1 between the two occupied bands."""
    K, E = bands_on_grid(p, nk)
    return _refined_min(lambda k: np.diff(np.linalg.eigvalsh(hamiltonian(k, p)))[0], K, E[..., 1] - E[..., 0])


def band_chirality(p, k0, band=0, r=0.02, n=24):
    """Berry flux (in units of 2pi) of one band through a small cube around k0."""
    total = 0.0
    for axis in range(3):
        for sgn in (+1, -1):
            a, b = [i for i in range(3) if i != axis]
            g = np.linspace(-r, r, n + 1)   # n x n plaquettes tile each whole face, so the six faces close
            KA, KB = np.meshgrid(g, g, indexing="ij")
            K = np.zeros((n + 1, n + 1, 3))
            K[..., a] = k0[a] + KA
            K[..., b] = k0[b] + KB
            K[..., axis] = k0[axis] + sgn * r
            u = np.linalg.eigh(hamiltonian(K, p))[1][..., band]
            link = lambda X, Y: np.einsum("...i,...i->...", X.conj(), Y)
            u0, ua, ub, uab = u[:-1, :-1], u[1:, :-1], u[:-1, 1:], u[1:, 1:]
            F = np.angle(link(u0, ua) * link(ua, uab) / (link(ub, uab) * link(u0, ub)))
            total += sgn * (1 if (a, b) in [(1, 2), (2, 0), (0, 1)] else -1) * F.sum()
    return total / (2 * np.pi)


# ----------------------------------------------------------------------------- symmetry data
# label j <-> S4 eigenvalue exp(-i j pi/2):  n^{1/2}: e^{-i pi/4}, n^{-1/2}: e^{+i pi/4},
#                                            n^{3/2}: e^{-i 3pi/4}, n^{-3/2}: e^{+i 3pi/4}
S4_LABELS = {"1/2": -0.25, "-1/2": 0.25, "3/2": -0.75, "-3/2": 0.75}   # phase / pi
C2_LABELS = {"1/2": -0.5, "-1/2": 0.5}


def symmetry_counts(k, Uop, labels, p, n_occ=2):
    w, V = np.linalg.eigh(hamiltonian(np.array(k, float), p))
    occ = V[:, :n_occ]
    ph = np.angle(np.linalg.eigvals(occ.conj().T @ Uop @ occ)) / np.pi
    counts = {lab: 0 for lab in labels}
    for x in ph:
        counts[min(labels, key=lambda L: abs(((x - labels[L]) + 1) % 2 - 1))] += 1
    return counts


def indicators(p):
    """(z4S, d2S, z2), K4 counts, R counts  -- formulas of MSG 81.33 (P-4)."""
    n = {K: symmetry_counts(k, U_S4, S4_LABELS, p) for K, k in K4.items()}
    nR = symmetry_counts(K2["R"], U_C2, C2_LABELS, p)
    z4S = (-0.5 * n["Z"]["1/2"] + 0.5 * n["Z"]["-1/2"] - 1.5 * n["Z"]["3/2"] + 1.5 * n["Z"]["-3/2"]
           - 0.5 * n["A"]["1/2"] + 0.5 * n["A"]["-1/2"] - 1.5 * n["A"]["3/2"] + 1.5 * n["A"]["-3/2"]
           + nR["1/2"] - nR["-1/2"])
    d2S = (-n["Z"]["3/2"] + n["Z"]["-3/2"] - n["A"]["3/2"] + n["A"]["-3/2"]
           + n["G"]["3/2"] - n["G"]["-3/2"] + n["M"]["3/2"] - n["M"]["-3/2"])
    z2 = sum((n[K]["1/2"] - n[K]["-3/2"]) / 2 for K in K4)
    return (int(round(z4S)) % 4, int(round(d2S)) % 2, int(round(z2)) % 2), n, nR


def chern_plane(p, axis, val, nk=40, bands=(0, 1)):
    """Fukui-Hatsugai-Suzuki Chern number of the given bands on the plane k_axis = val."""
    g = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    a, b = [i for i in range(3) if i != axis]
    K = np.zeros((nk, nk, 3))
    KA, KB = np.meshgrid(g, g, indexing="ij")
    K[..., a] = KA
    K[..., b] = KB
    K[..., axis] = val
    U = np.linalg.eigh(hamiltonian(K, p))[1][..., list(bands)]
    Ua, Ub = np.roll(U, -1, axis=0), np.roll(U, -1, axis=1)
    Uab = np.roll(Ua, -1, axis=1)
    link = lambda X, Y: np.linalg.det(np.swapaxes(X.conj(), -1, -2) @ Y)
    F = np.angle(link(U, Ua) * link(Ua, Uab) / (link(Ub, Uab) * link(U, Ub)))
    return int(round(F.sum() / (2 * np.pi)))


def all_chern(p, nk=40, bands=(0, 1)):
    return {f"C_{'xyz'[ax]}({lab})": chern_plane(p, ax, v, nk, bands)
            for ax in range(3) for lab, v in (("0", 0.0), ("pi", np.pi))}


# ----------------------------------------------------------------------------- sewing matrix & 2P3
REFS = {
    "A-orb": np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=complex),
    "B-orb": np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=complex),
    "tau_x=-1": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=complex) / np.sqrt(2),
    "tau_x=+1": np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=complex) / np.sqrt(2),
    "mixed": np.array([[1, 0], [0, 1], [0.5j, 0.3], [0.2, -0.4j]], dtype=complex),
}


def projector_frames(p, nk, ref=None, verbose=True):
    """Global smooth periodic occupied frame U_occ = P R (R^+ P R)^{-1/2} on the k-grid."""
    _, K = k_grid(nk)
    V = np.linalg.eigh(hamiltonian(K, p))[1]
    Vocc = V[..., :2]
    P = Vocc @ np.swapaxes(Vocc.conj(), -1, -2)
    if ref is None:
        best = None
        for name, R in REFS.items():
            W = P @ R
            m = np.linalg.eigvalsh(np.swapaxes(W.conj(), -1, -2) @ W).min()
            if best is None or m > best[0]:
                best = (m, name, R)
        ref = best[2]
        if verbose:
            print(f"    reference frame '{best[1]}' (min overlap eigenvalue {best[0]:.3f})")
    W = P @ ref
    ge, gv = np.linalg.eigh(np.swapaxes(W.conj(), -1, -2) @ W)
    ginv = gv @ (ge[..., None] ** -0.5 * np.swapaxes(gv.conj(), -1, -2))
    return W @ ginv, ge.min()


def sewing_matrix_grid(U):
    """B(k) = U(S4 k)^+ U_S4 U(k) on the periodic grid, S4 index map (ix,iy,iz)->(iy,-ix,-iz)."""
    nk = U.shape[0]
    ix, iy, iz = np.meshgrid(np.arange(nk), np.arange(nk), np.arange(nk), indexing="ij")
    U_s4 = U[iy % nk, (-ix) % nk, (-iz) % nk]
    return np.swapaxes(U_s4.conj(), -1, -2) @ U_S4 @ U


def det_one(B):
    """Continuous half-phase reduction B -> exp(-i arg det B / 2) B (requires zero det winding)."""
    phase = np.angle(np.linalg.det(B))
    for a in range(3):
        phase = np.unwrap(phase, axis=a)
    return np.exp(-0.5j * phase)[..., None, None] * B


def det_winding(B):
    d = np.linalg.det(B)
    return max(np.abs(np.angle(np.roll(d, -1, axis=a) / d).sum(axis=a) / (2 * np.pi)).max() for a in range(3))


def two_p3_from_B(B, nk):
    """-1/(24 pi^2) sum eps^{ijk} Tr[(B^+ d_i B)(B^+ d_j B)(B^+ d_k B)] with periodic central differences."""
    dk = 2 * np.pi / nk
    d = lambda X, a: (np.roll(X, -1, axis=a) - np.roll(X, 1, axis=a)) / (2 * dk)
    Bd = np.swapaxes(B.conj(), -1, -2)
    A = [Bd @ d(B, a) for a in range(3)]
    tr = lambda X, Y, Z: np.trace(X @ Y @ Z, axis1=-2, axis2=-1)
    lev = (tr(A[0], A[1], A[2]) + tr(A[1], A[2], A[0]) + tr(A[2], A[0], A[1])
           - tr(A[0], A[2], A[1]) - tr(A[2], A[1], A[0]) - tr(A[1], A[0], A[2]))
    return (-(1 / (24 * np.pi**2)) * lev.sum() * dk**3), A


def two_p3(p, nk=40, verbose=True):
    """2P3 from the projector-smooth-frame sewing matrix. Returns (value, min overlap, det winding)."""
    U, min_overlap = projector_frames(p, nk, verbose=verbose)
    B = sewing_matrix_grid(U)
    wind = det_winding(B)
    val, _ = two_p3_from_B(det_one(B), nk)
    if verbose:
        print(f"    nk={nk}: |det B|-1 max = {np.abs(np.abs(np.linalg.det(B)) - 1).max():.1e}, "
              f"det-B winding = {wind:.1e}, 2P3 = {val.real:+.4f}{val.imag:+.1e}j")
    return val.real, min_overlap, wind


def _unitary_part(M):
    u, _, vh = np.linalg.svd(M)
    return u @ vh


def parallel_transport_frames(p, nk):
    """The 'stitched' gauge of the old notebook: sequential parallel transport x-line -> y-lines -> z-lines."""
    _, K = k_grid(nk)
    U = np.linalg.eigh(hamiltonian(K, p))[1][..., :2].copy()
    align = lambda u, ref: u @ _unitary_part(ref.conj().T @ u).conj().T
    for ix in range(1, nk):
        U[ix, 0, 0] = align(U[ix, 0, 0], U[ix - 1, 0, 0])
    for ix in range(nk):
        for iy in range(1, nk):
            U[ix, iy, 0] = align(U[ix, iy, 0], U[ix, iy - 1, 0])
    for ix in range(nk):
        for iy in range(nk):
            for iz in range(1, nk):
                U[ix, iy, iz] = align(U[ix, iy, iz], U[ix, iy, iz - 1])
    return U


def two_p3_parallel_transport(p, nk=24):
    """2P3 evaluated in the stitched parallel-transport gauge (demonstrates why it gives ~0)."""
    U = parallel_transport_frames(p, nk)
    B = sewing_matrix_grid(U)
    val, A = two_p3_from_B(B, nk)
    # the seam kz = -pi enters the central difference at layers 1 and nk-1 (directly, or via the S4 image -iz)
    Az_bulk = np.abs(A[2][:, :, 2:-1]).max()
    Az_seam = np.abs(A[2][:, :, [1, -1]]).max()
    return val.real, Az_bulk, Az_seam


# ----------------------------------------------------------------------------- Wilson loop, plots
def wilson_z_spectrum(p, nk_line=60, nkz=60):
    path = [(0, 0), (np.pi, 0), (np.pi, np.pi), (0, np.pi), (0, 0)]
    labels = [r"$\Gamma$", "X", "M", "Y", r"$\Gamma$"]
    pts = []
    for (a, b), (c, d) in zip(path[:-1], path[1:]):
        for t in np.linspace(0, 1, nk_line, endpoint=False):
            pts.append((a + t * (c - a), b + t * (d - b)))
    pts.append(path[-1])
    kz = np.linspace(-np.pi, np.pi, nkz, endpoint=False)
    out = np.zeros((len(pts), 2))
    for i, (kx, ky) in enumerate(pts):
        K = np.stack([np.full(nkz, kx), np.full(nkz, ky), kz], axis=-1)
        U = np.linalg.eigh(hamiltonian(K, p))[1][..., :2]
        Wl = np.eye(2, dtype=complex)
        for j in range(nkz):
            Wl = Wl @ (U[j].conj().T @ U[(j + 1) % nkz])
        out[i] = np.sort(np.angle(np.linalg.eigvals(Wl)) / (2 * np.pi))
    return out, [i * nk_line for i in range(5)], labels


def band_path(pts=None, labels=None, n=80):
    if pts is None:
        pts = [(0, 0, 0), (np.pi, 0, 0), (np.pi, np.pi, 0), (np.pi, np.pi, np.pi), (0, 0, np.pi), (0, 0, 0),
               (np.pi, 0, np.pi), (np.pi, np.pi, np.pi)]
        labels = [r"$\Gamma$", "X", "M", "A", "Z", r"$\Gamma$", "R", "A"]
    ks, ticks, x = [], [0.0], 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.array(a, float), np.array(b, float)
        ks.append(np.linspace(a, b, n, endpoint=False))
        x += np.linalg.norm(b - a)
        ticks.append(x)
    ks.append(np.array([pts[-1]], float))
    ks = np.vstack(ks)
    xs = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ks, axis=0), axis=1))])
    return ks, xs, ticks, labels


def set_style():
    """Publication style. Energies are in units of the hopping u (= v = 1)."""
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save_figure(fig, stem):
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".pdf")


def dos(p, nk=64, sigma=0.05, n_e=800):
    """Gaussian-broadened DOS per unit cell: int rho(E) dE = 4 (number of bands)."""
    _, E = bands_on_grid(p, nk)
    e = np.linspace(E.min() - 6 * sigma, E.max() + 6 * sigma, n_e)
    rho = np.exp(-0.5 * ((e[:, None] - E.reshape(-1)[None, :]) / sigma) ** 2).sum(axis=1)
    rho /= np.sqrt(2 * np.pi) * sigma * E.reshape(-1).size / 4
    return e, rho


def _panel_label(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")


def plot_bands_dos(p, nk_dos=64, sigma=0.05):
    """(a) bands along Gamma-X-M-A-Z-Gamma-R-A, (b) broadened DOS sharing the energy axis."""
    set_style()
    ks, xs, ticks, labels = band_path()
    E = np.linalg.eigvalsh(hamiltonian(ks, p))
    e, rho = dos(p, nk_dos, sigma)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True, gridspec_kw=dict(width_ratios=[2.6, 1], wspace=0.06))
    for n in range(4):
        ax[0].plot(xs, E[:, n], color="C0" if n < 2 else "C3", lw=1.4)
    for t in ticks:
        ax[0].axvline(t, color="0.6", lw=0.6)
    ax[0].axhline(0, color="0.6", lw=0.6, ls="--")
    ax[0].set_xticks(ticks, labels)
    ax[0].set_xlim(xs[0], xs[-1])
    ax[0].set_ylabel(r"Energy $E/u$")
    ax[0].plot([], [], color="C0", lw=1.4, label="occupied")
    ax[0].plot([], [], color="C3", lw=1.4, label="unoccupied")
    ax[0].legend(loc="lower right", frameon=False)
    ax[1].fill_betweenx(e, 0, rho, color="0.75", lw=0)
    ax[1].plot(rho, e, color="k", lw=0.8)
    ax[1].axhline(0, color="0.6", lw=0.6, ls="--")
    ax[1].set_xlim(0, rho.max() * 1.08)
    ax[1].set_xlabel(r"DOS  (states / cell / $u$)")
    ax[1].tick_params(labelleft=False)
    _panel_label(ax[0], "(a)")
    _panel_label(ax[1], "(b)")
    return fig


def plot_wilson(p, nk_line=80, nkz=80):
    """Eigenphases of the Wilson loop along kz on the path Gamma-X-M-Y-Gamma."""
    set_style()
    ph, ticks, labels = wilson_z_spectrum(p, nk_line, nkz)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(ph))
    ax.plot(x, ph[:, 0], "o", ms=2.2, color="C0", mew=0)
    ax.plot(x, ph[:, 1], "o", ms=2.2, color="C1", mew=0)
    for t in ticks:
        ax.axvline(t, color="0.6", lw=0.6)
    for y in (-0.5, 0.0, 0.5):
        ax.axhline(y, color="0.6", lw=0.6, ls=":")
    ax.set_xticks(ticks, labels)
    ax.set_xlim(0, len(ph) - 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_ylabel(r"Wilson-loop phase $\theta_{1,2}/2\pi$")
    return fig


# ----------------------------------------------------------------------------- exact phase diagram
_KZ = np.linspace(-np.pi, np.pi, 4001)
_LINES = [(0, 0), (np.pi, np.pi), (np.pi, 0), (0, np.pi)]


def gapless(p):
    """Exact criterion: the gap closes iff rho(kz) = m_z somewhere on one of the four C2 lines."""
    for kx, ky in _LINES:
        eps = p["eps0"] - np.cos(kx) - np.cos(ky) + 2 * np.cos(_KZ)
        g = p["eta"] * (np.cos(kx) - np.cos(ky)) + p["kappa"] * np.sin(_KZ)
        rho = np.sqrt(eps**2 + np.sin(_KZ)**2 + g**2)
        if rho.min() <= p["m_z"] <= rho.max():
            return True
    return False


def analytic_gap(p, nk=32):
    _, K = k_grid(nk)
    E = closed_form_bands(K, p)
    return (E[..., 2] - E[..., 1]).min()


def phase_diagram(xname, xs, yname, ys, base, mark=None):
    """Exact phase diagram: (a) bulk gap, (b) phases (Weyl semimetal / trivial / axion z2=1)."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    set_style()
    gapmap = np.zeros((len(xs), len(ys)))
    z2map = np.zeros_like(gapmap, dtype=int)
    okmap = np.zeros_like(gapmap, dtype=bool)
    sm = np.zeros_like(gapmap, dtype=bool)
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            p = dict(base)
            p[xname] = xv
            p[yname] = yv
            sm[i, j] = gapless(p)
            gapmap[i, j] = 0.0 if sm[i, j] else analytic_gap(p)
            (z4, d2, z2), _, _ = indicators(p)
            z2map[i, j] = z2
            okmap[i, j] = (z4 == 0 and d2 == 0)
    names = {"eps0": r"$\epsilon_0/u$", "m_z": r"$m_z/u$", "eta": r"$\eta/u$", "kappa": r"$\kappa/u$"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(wspace=0.32))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    im = ax[0].pcolormesh(X, Y, gapmap, shading="nearest", cmap="viridis", vmin=0, rasterized=True)
    cb = fig.colorbar(im, ax=ax[0], pad=0.02)
    cb.set_label(r"bulk gap $\Delta/u$")
    # phases: 0 = gapless, 1 = trivial insulator, 2 = axion insulator, 3 = other insulator
    phase = np.where(sm, 0, np.where(~okmap, 3, np.where(z2map == 1, 2, 1)))
    cmap = ListedColormap(["0.35", "#c6dbef", "#e6550d", "#ffd92f"])
    ax[1].pcolormesh(X, Y, phase, shading="nearest", cmap=cmap, vmin=-0.5, vmax=3.5, rasterized=True)
    handles = [Patch(color="0.35", label="Weyl semimetal"), Patch(color="#c6dbef", label=r"trivial ($z_2=0$)"),
               Patch(color="#e6550d", label=r"axion ($z_2=1$)")]
    if (phase == 3).any():
        handles.append(Patch(color="#ffd92f", label=r"$z_{4S}$ or $\delta_{2S}\neq0$"))
    ax[1].legend(handles=handles, loc="upper center", frameon=True, framealpha=0.9, ncol=1, fontsize=8)
    if mark is not None:
        for a in ax:
            a.plot(*mark, marker="*", ms=9, color="w", mec="k", mew=0.7)
    for a in ax:
        a.set_xlabel(names.get(xname, xname))
        a.set_ylabel(names.get(yname, yname))
    _panel_label(ax[0], "(a)")
    _panel_label(ax[1], "(b)")
    return fig, (gapmap, z2map, okmap, sm)


# ----------------------------------------------------------------------------- robust smooth frame, direct CS integral
def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _line_transport(PL, v0, axis):
    """Parallel transport of a unit section v0 of the line bundle with projector PL along one grid axis.

    PL : projector array (..., 4, 4) with the transport direction at `axis`; v0 : section on slice 0 of `axis`.
    Returns the transported section (not yet periodic) and the Berry phase mismatch at the periodic seam
    (an array over the transverse indices)."""
    PL = np.moveaxis(PL, axis, 0)
    nk = PL.shape[0]
    out = np.empty(PL.shape[:-1], dtype=complex)
    out[0] = v0
    for j in range(1, nk):
        out[j] = _unit(np.einsum("...ab,...b->...a", PL[j], out[j - 1]))
    back = _unit(np.einsum("...ab,...b->...a", PL[0], out[-1]))
    phi = np.angle(np.einsum("...a,...a->...", out[0].conj(), back))
    return np.moveaxis(out, 0, axis), phi


def smooth_frames(p, nk, seed=0, n_try=8, verbose=False):
    """Global smooth periodic frame of the occupied bundle, valid whenever all plane Chern numbers vanish.

    e1(k) = P(k) r1 / |P(k) r1| for a generic constant r1 (a generic section of a rank-2 bundle over T^3 vanishes
            only on a codimension-4 set, i.e. nowhere);
    e2(k) = unit section of the complementary line bundle L = occ - e1, built by parallel transport along kx
            (one line), ky (the plane kz=-pi) and kz (the bulk).  At every periodic seam the abelian Berry phase
            is unwrapped in the transverse variables (possible because all Chern numbers of L vanish) and removed
            by a linear phase twist, so the result is smooth and periodic on the grid.
    (The constant-reference projector frame of `projector_frames` needs a nowhere-vanishing section of the
    *line* bundle L from a constant reference, which generically fails on curves; here that section is built
    by transport instead.)
    Returns U (nk,nk,nk,4,2) and a dict of diagnostics."""
    _, K = k_grid(nk)
    V = np.linalg.eigh(hamiltonian(K, p))[1][..., :2]
    P = V @ np.swapaxes(V.conj(), -1, -2)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_try):
        r1 = rng.normal(size=4) + 1j * rng.normal(size=4)
        s = P @ (r1 / np.linalg.norm(r1))
        m = np.linalg.norm(s, axis=-1).min()
        if best is None or m > best[0]:
            best = (m, s)
    e1 = _unit(best[1])
    PL = P - e1[..., :, None] * e1[..., None, :].conj()
    r2 = rng.normal(size=4) + 1j * rng.normal(size=4)
    v0 = _unit(PL[0, 0, 0] @ r2)
    j = np.arange(nk)
    # line kx  (iy = iz = 0)
    line, phx = _line_transport(PL[:, 0, 0], v0, 0)
    line = line * np.exp(-1j * phx * j / nk)[:, None]
    # plane kz = -pi (iz = 0): transport along ky starting from the line
    plane, phy = _line_transport(PL[:, :, 0], line, 1)
    phy = np.unwrap(phy)
    seam_y = abs(phy[-1] - phy[0]) / (2 * np.pi)                   # ~0 iff Chern number of L on kz=-pi vanishes
    plane = plane * np.exp(-1j * phy[:, None] * j[None, :] / nk)[..., None]
    # bulk: transport along kz starting from the plane
    e2, phz = _line_transport(PL, plane, 2)
    phz = np.unwrap(np.unwrap(phz, axis=0), axis=1)
    seam_z = max(np.abs(phz[-1] - phz[0]).max(), np.abs(phz[:, -1] - phz[:, 0]).max()) / (2 * np.pi)
    e2 = e2 * np.exp(-1j * phz[..., None] * j / nk)[..., None]
    U = np.stack([e1, e2], axis=-1)
    step = max(np.abs(np.roll(U, -1, axis=a) - U).max() for a in range(3))
    diag = dict(min_section=best[0], seam_y=seam_y, seam_z=seam_z, max_step=step,
                orth=np.abs(np.swapaxes(U.conj(), -1, -2) @ U - np.eye(2)).max())
    if verbose:
        print(f"    smooth frame nk={nk}: min|P r1| = {best[0]:.3f}, seam windings (y,z) = ({seam_y:.1e}, {seam_z:.1e}), "
              f"max grid step = {step:.3f}, orthonormality err = {diag['orth']:.1e}")
    return U, diag


def chern_simons_p3(U, nk):
    """Direct Chern-Simons integral in the smooth periodic gauge U:
       P3 = 1/(8 pi^2) int eps^{ijk} Tr[ A_i d_j A_k + (2/3) A_i A_j A_k ],   A_i = U^+ d_i U  (anti-Hermitian),
    i.e. the standard -1/(8 pi^2) int eps Tr[a da - (2i/3) a a a] with the Hermitian Berry connection a = iA.
    Defined mod 1; equals theta/2pi.  Periodic central differences."""
    dk = 2 * np.pi / nk
    d = lambda X, a: (np.roll(X, -1, axis=a) - np.roll(X, 1, axis=a)) / (2 * dk)
    Ud = np.swapaxes(U.conj(), -1, -2)
    A = [Ud @ d(U, a) for a in range(3)]
    tr = lambda X: np.trace(X, axis1=-2, axis2=-1)
    tot = 0.0
    for (i, j, k, s) in ((0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1), (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)):
        tot = tot + s * tr(A[i] @ d(A[k], j) + (2 / 3) * A[i] @ A[j] @ A[k])
    return (tot.sum() * dk**3 / (8 * np.pi**2)).real


def fold_mod2(x):
    """Distance of x from the nearest even integer, folded into [0, 1]: 0 <-> even, 1 <-> odd."""
    return np.abs(((np.asarray(x, dtype=float) + 1) % 2) - 1)


def two_p3_smooth(p, nk=24, seed=0, direct=True, verbose=False):
    """2P3 from the S4 sewing-matrix winding in the robust smooth frame; optionally also the direct CS integral.
    Returns dict(winding=W[B], cs=2*P3_direct, diag=...)."""
    U, diag = smooth_frames(p, nk, seed=seed, verbose=verbose)
    B = sewing_matrix_grid(U)
    diag["det_winding"] = det_winding(B)
    val, _ = two_p3_from_B(det_one(B), nk)
    out = dict(winding=val.real, diag=diag)
    if direct:
        out["cs"] = 2 * chern_simons_p3(U, nk)
    if verbose:
        msg = f"    nk={nk}: 2P3 (sewing-matrix winding) = {val.real:+.4f}"
        if direct:
            msg += f",  2P3 (direct Chern-Simons) = {out['cs']:+.4f}"
        print(msg)
    return out


def p3_phase_map(xname, xs, yname, ys, base, nks=(24, 36, 48, 64), step_tol=0.6, seed=0, direct=True,
                 chern_nk=24, verbose=True):
    """2P3 on a parameter grid (insulating points with vanishing plane Chern numbers only).

    The k-grid is refined through `nks` until the largest change of the smooth frame between neighbouring
    grid points is below `step_tol` (near a phase boundary the projector is sharp and needs a finer grid).
    Returns dict of arrays over (xs, ys): gapless, z2, chern_ok, winding (W[B]), cs (direct 2P3),
    nk (grid used), max_step, min_section."""
    import time
    shape = (len(xs), len(ys))
    out = {k: np.full(shape, np.nan) for k in ("winding", "cs", "min_section", "max_step")}
    out["z2"] = np.full(shape, -1, dtype=int)
    out["nk"] = np.zeros(shape, dtype=int)
    out["gapless"] = np.zeros(shape, dtype=bool)
    out["chern_ok"] = np.zeros(shape, dtype=bool)
    t0, n_done = time.time(), 0
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            p = dict(base)
            p[xname] = xv
            p[yname] = yv
            out["gapless"][i, j] = gapless(p)
            if out["gapless"][i, j]:
                continue
            out["z2"][i, j] = indicators(p)[0][2]
            out["chern_ok"][i, j] = all(v == 0 for v in all_chern(p, nk=chern_nk).values())
            if not out["chern_ok"][i, j]:
                continue
            for nk in nks:
                r = two_p3_smooth(p, nk=nk, seed=seed, direct=direct)
                if r["diag"]["max_step"] <= step_tol:
                    break
            out["winding"][i, j] = r["winding"]
            out["min_section"][i, j] = r["diag"]["min_section"]
            out["max_step"][i, j] = r["diag"]["max_step"]
            out["nk"][i, j] = nk
            if direct:
                out["cs"][i, j] = r["cs"]
            n_done += 1
        if verbose and (i % max(1, len(xs) // 6) == 0 or i == len(xs) - 1):
            print(f"    {xname} = {xv:+.2f}: {n_done} insulating points done, {time.time() - t0:.0f} s")
    return out


def p3_map_report(res, tol=0.3):
    """Compare the folded 2P3 values with z2 on every insulating point with vanishing Chern numbers.
    A point is 'resolved' if its folded value is within `tol` of 0 or 1, and a 'mismatch' if the nearer
    integer differs from z2.  Returns a dict per method."""
    ins = ~res["gapless"]
    ok = ins & res["chern_ok"]
    stats = {}
    for key in ("winding", "cs"):
        if np.all(np.isnan(res[key])):
            continue
        f = fold_mod2(res[key])[ok]
        z = res["z2"][ok]
        resolved = np.minimum(f, 1 - f) <= tol
        stats[key] = dict(n_points=int(ok.sum()), n_unresolved=int((~resolved).sum()),
                          n_mismatch=int((resolved & (np.round(f) != z)).sum()),
                          worst_resolved=float(np.abs(f[resolved] - z[resolved]).max()),
                          n_nonzero_chern=int((ins & ~res["chern_ok"]).sum()),
                          nk_used=dict(zip(*[list(map(int, a)) for a in np.unique(res["nk"][ok], return_counts=True)])))
    return stats


def plot_p3_map(xname, xs, yname, ys, res, mark=None, direct=True):
    """(a) z2 indicator, (b) 2P3 from the S4 sewing-matrix winding, (c) 2P3 from the direct CS integral.
    2P3 panels show the value folded mod 2 (0 = even, 1 = odd); gapless points are grey."""
    set_style()
    names = {"eps0": r"$\epsilon_0/u$", "m_z": r"$m_z/u$", "eta": r"$\eta/u$", "kappa": r"$\kappa/u$"}
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    panels = [(r"$z_2$ indicator", np.where(res["gapless"], np.nan, res["z2"]).astype(float)),
              (r"$2P_3$, $S_4$ sewing-matrix winding (mod 2)", fold_mod2(res["winding"]))]
    if direct:
        panels.append((r"$2P_3$, direct Chern-Simons integral (mod 2)", fold_mod2(res["cs"])))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.3 * len(panels) + 0.6, 3.0), gridspec_kw=dict(wspace=0.3))
    ax = np.atleast_1d(ax)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("0.35")
    for a, (title, Z) in zip(ax, panels):
        im = a.pcolormesh(X, Y, np.ma.masked_invalid(Z), shading="nearest", cmap=cmap, vmin=0, vmax=1, rasterized=True)
        a.set_title(title, fontsize=8.5)
        a.set_xlabel(names.get(xname, xname))
        a.set_ylabel(names.get(yname, yname))
        if mark is not None:
            a.plot(*mark, marker="*", ms=9, color="w", mec="k", mew=0.7)
    fig.colorbar(im, ax=ax.tolist(), pad=0.02, fraction=0.03)
    for a, lab in zip(ax, "abc"):
        _panel_label(a, f"({lab})")
    return fig


# ----------------------------------------------------------------------------- constant-reference (projector) frame on the phase diagram
def two_p3_projector(p, nk=24, direct=True, verbose=False):
    """Same diagnostics as `two_p3_smooth`, but with the constant-reference projector frame of Sec. 5,
    U = P R (R^+ P R)^{-1/2}, with R chosen among REFS to maximise the minimal overlap eigenvalue.
    Returns dict(winding=W[B], cs=2*P3_direct, diag=dict(min_overlap, max_step, det_winding))."""
    U, min_overlap = projector_frames(p, nk, verbose=verbose)
    U = np.nan_to_num(U)                       # a vanishing overlap gives NaN; keep going and let the value be garbage
    step = max(np.abs(np.roll(U, -1, axis=a) - U).max() for a in range(3))
    B = sewing_matrix_grid(U)
    val, _ = two_p3_from_B(det_one(B), nk)
    out = dict(winding=val.real, diag=dict(min_overlap=min_overlap, max_step=step, det_winding=det_winding(B)))
    if direct:
        out["cs"] = 2 * chern_simons_p3(U, nk)
    if verbose:
        msg = f"    nk={nk}: 2P3 (sewing-matrix winding) = {val.real:+.4f}"
        if direct:
            msg += f",  2P3 (direct Chern-Simons) = {out['cs']:+.4f}"
        print(msg + f",  max grid step = {step:.3f}")
    return out


def p3_phase_map_projector(xname, xs, yname, ys, base, nks=(24, 36, 48, 64), step_tol=0.6, direct=True,
                           chern_nk=24, verbose=True):
    """Like `p3_phase_map`, but every insulating point is evaluated with the constant-reference projector
    frame (`two_p3_projector`).  Extra output: min_overlap (smallest eigenvalue of R^+ P R over the grid)."""
    import time
    shape = (len(xs), len(ys))
    out = {k: np.full(shape, np.nan) for k in ("winding", "cs", "min_overlap", "max_step")}
    out["z2"] = np.full(shape, -1, dtype=int)
    out["nk"] = np.zeros(shape, dtype=int)
    out["gapless"] = np.zeros(shape, dtype=bool)
    out["chern_ok"] = np.zeros(shape, dtype=bool)
    t0, n_done = time.time(), 0
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            p = dict(base)
            p[xname] = xv
            p[yname] = yv
            out["gapless"][i, j] = gapless(p)
            if out["gapless"][i, j]:
                continue
            out["z2"][i, j] = indicators(p)[0][2]
            out["chern_ok"][i, j] = all(v == 0 for v in all_chern(p, nk=chern_nk).values())
            if not out["chern_ok"][i, j]:
                continue
            for nk in nks:
                r = two_p3_projector(p, nk=nk, direct=direct)
                if r["diag"]["max_step"] <= step_tol:
                    break
            out["winding"][i, j] = r["winding"]
            out["min_overlap"][i, j] = r["diag"]["min_overlap"]
            out["max_step"][i, j] = r["diag"]["max_step"]
            out["nk"][i, j] = nk
            if direct:
                out["cs"][i, j] = r["cs"]
            n_done += 1
        if verbose and (i % max(1, len(xs) // 6) == 0 or i == len(xs) - 1):
            print(f"    {xname} = {xv:+.2f}: {n_done} insulating points done, {time.time() - t0:.0f} s")
    return out


def plot_overlap_map(xname, xs, yname, ys, res, mark=None):
    """Minimal overlap eigenvalue of the constant-reference frame (log scale) over the parameter grid."""
    from matplotlib.colors import LogNorm
    set_style()
    names = {"eps0": r"$\epsilon_0/u$", "m_z": r"$m_z/u$", "eta": r"$\eta/u$", "kappa": r"$\kappa/u$"}
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.where(res["gapless"], np.nan, np.maximum(res["min_overlap"], 1e-6))
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.35")
    im = ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), shading="nearest", cmap=cmap, norm=LogNorm(1e-4, 1), rasterized=True)
    fig.colorbar(im, ax=ax, pad=0.02, label=r"min eigenvalue of $R^\dagger P_{\rm occ}R$")
    ax.set_xlabel(names.get(xname, xname))
    ax.set_ylabel(names.get(yname, yname))
    if mark is not None:
        ax.plot(*mark, marker="*", ms=9, color="w", mec="k", mew=0.7)
    return fig

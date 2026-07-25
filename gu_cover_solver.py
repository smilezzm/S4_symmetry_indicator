"""Solve GU = h + EV for many RHS vectors h via persistent HiGHS MIP models."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import highspy
import numpy as np
from scipy import sparse

# Per-worker state (process pool on Windows imports this module in child processes).
_g_h: highspy.Highs | None = None
_g_row_idx: np.ndarray | None = None
_g_n_rows: int = 0
_g_nU: int = 0


def build_highs_solver(A: np.ndarray, n_rows: int, n_var: int) -> tuple[highspy.Highs, np.ndarray]:
    """Build one persistent HiGHS MIP: min sum(x) s.t. A x = rhs, x >= 0 integer."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("log_to_console", False)

    A_csc = sparse.csc_matrix(A)
    row_lower = np.zeros(n_rows, dtype=np.float64)
    row_upper = np.zeros(n_rows, dtype=np.float64)
    col_lower = np.zeros(n_var, dtype=np.float64)
    col_upper = np.full(n_var, h.getInfinity(), dtype=np.float64)
    col_cost = np.ones(n_var, dtype=np.float64)
    integrality = np.ones(n_var, dtype=np.int32)

    status = h.passModel(
        n_var,
        n_rows,
        A_csc.nnz,
        0,
        highspy.MatrixFormat.kColwise,
        highspy.HessianFormat.kTriangular,
        highspy.ObjSense.kMinimize,
        0.0,
        col_cost,
        col_lower,
        col_upper,
        row_lower,
        row_upper,
        A_csc.indptr.astype(np.int32),
        A_csc.indices.astype(np.int32),
        A_csc.data.astype(np.float64),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
        integrality,
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS passModel failed: {status}")

    return h, np.arange(n_rows, dtype=np.int32)


def solve_one_warm(
    h_solver: highspy.Highs,
    row_idx: np.ndarray,
    n_rows: int,
    nU: int,
    rhs: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Update row bounds (RHS) and re-solve; returns (ok, U, V)."""
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    h_solver.changeRowsBounds(n_rows, row_idx, rhs, rhs)
    h_solver.run()
    if h_solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return False, None, None
    x = np.rint(h_solver.getSolution().col_value).astype(int)
    return True, x[:nU], x[nU:]


def _init_worker(A: np.ndarray, n_rows: int, n_var: int, nU: int) -> None:
    global _g_h, _g_row_idx, _g_n_rows, _g_nU
    _g_h, _g_row_idx = build_highs_solver(A, n_rows, n_var)
    _g_n_rows, _g_nU = n_rows, nU


def _solve_chunk(items: list[tuple[int, np.ndarray]]):
    return [
        (idx, *solve_one_warm(_g_h, _g_row_idx, _g_n_rows, _g_nU, rhs))
        for idx, rhs in items
    ]


def solve_all(
    H: np.ndarray,
    A: np.ndarray,
    nU: int,
    *,
    n_workers: int | None = None,
    chunk_size: int = 40,
    parallel: str = "process",
    progress_every: int = 200,
    t0: float | None = None,
):
    """Solve GU = h + EV for every row h in H.

    parallel: "process" | "thread" | "serial"
      - process: one persistent HiGHS model per worker (recommended)
      - thread: same, but thread pool (use in Jupyter on Windows if process pool fails)
    """
    H = np.asarray(H, dtype=int)
    n_rows, n_var = A.shape
    nV = n_var - nU
    nH = len(H)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 8)
    if t0 is None:
        t0 = time.time()

    feasible = np.zeros(nH, dtype=bool)
    U_all = np.zeros((nH, nU), dtype=int)
    V_all = np.zeros((nH, nV), dtype=int)
    failures: list[int] = []
    done = 0

    def _store(batch):
        nonlocal done
        for idx, ok, U, V in batch:
            if ok:
                feasible[idx] = True
                U_all[idx] = U
                V_all[idx] = V
            else:
                failures.append(idx)
        done += len(batch)
        if done % progress_every == 0 or done == nH:
            print(
                f"  {done}/{nH} done, {feasible.sum()} feasible, "
                f"{time.time() - t0:.1f}s"
            )

    items = list(enumerate(H))
    chunks = [items[i : i + chunk_size] for i in range(0, nH, chunk_size)]

    if parallel == "serial" or n_workers <= 1:
        solver, row_idx = build_highs_solver(A, n_rows, n_var)
        batch = []
        for idx, rhs in items:
            batch.append((idx, *solve_one_warm(solver, row_idx, n_rows, nU, rhs)))
            if len(batch) >= chunk_size:
                _store(batch)
                batch = []
        if batch:
            _store(batch)
        return feasible, U_all, V_all, failures

    Executor = ProcessPoolExecutor if parallel == "process" else ThreadPoolExecutor
    with Executor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(A, n_rows, n_var, nU),
    ) as pool:
        for fut in as_completed(pool.submit(_solve_chunk, ch) for ch in chunks):
            _store(fut.result())

    return feasible, U_all, V_all, failures

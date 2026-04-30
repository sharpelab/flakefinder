"""Regression tests for the air|film|SiO₂|Si transfer-matrix reflectance.

Runs as `uv run python scripts/test_hbn_contrast.py` (or via pytest).

Pins down the absorbing-film case where an earlier formulation used
`np.conj(exp(1j*β))` for the negative-phase terms — equivalent to
`exp(-1j*β)` only for real β, silently dropping round-trip absorption
when β had an imaginary part.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from hbn_contrast import _SI_DATA, _SIO2_DATA, _interp_index, _reflectance_at_angle
from hbn_contrast_widget import compute_rg


def _exact_3layer_R(
    lamb: np.ndarray,
    n1: complex | np.ndarray,
    t_film_nm: float,
    t_oxide_nm: float,
) -> np.ndarray:
    """Closed-form normal-incidence R(λ) for air|film|SiO₂|Si.

    Computed by composing two single-film transfer matrices analytically,
    using exp(-2j*β) factors directly (no np.conj shortcut), so it stays
    correct for complex n_film.
    """
    n0 = 1.0
    n2 = _interp_index(lamb, _SIO2_DATA)  # real
    n3 = _interp_index(lamb, _SI_DATA)  # complex
    if not isinstance(n1, np.ndarray):
        n1 = np.full_like(lamb, n1, dtype=np.complex128)

    # Normal incidence — cos(θ) = 1 in every medium.
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    r01 = (n0 - n1) / (n0 + n1)

    beta1 = 2 * np.pi * n1 * t_film_nm / lamb
    beta2 = 2 * np.pi * n2 * t_oxide_nm / lamb

    # SiO₂/Si stack reflection coefficient (looking down from inside film).
    r_23_eff = (r12 + r23 * np.exp(-2j * beta2)) / (1.0 + r12 * r23 * np.exp(-2j * beta2))
    # Add the air/film interface on top.
    r_total = (r01 + r_23_eff * np.exp(-2j * beta1)) / (1.0 + r01 * r_23_eff * np.exp(-2j * beta1))
    return (r_total * np.conj(r_total)).real


def _cli_R_pol(n1: complex, t_film_nm: float, t_oxide_nm: float, lamb: np.ndarray, pol: str) -> np.ndarray:
    """Drive the CLI's _reflectance_at_angle at normal incidence."""

    def n_fn(_lam: np.ndarray) -> np.ndarray:
        return np.full_like(_lam, n1, dtype=np.complex128)

    return _reflectance_at_angle(
        lamb, n_fn, n_layers=1, t_oxide_nm=t_oxide_nm, theta0=0.0, pol=pol, layer_thickness_nm=t_film_nm
    )


# --- Cases -----------------------------------------------------------------

LAMB = np.linspace(400.0, 700.0, 31)


def _check(label: str, n1: complex, t_film: float, t_oxide: float, tol: float = 1e-6) -> None:
    R_exact = _exact_3layer_R(LAMB, n1, t_film, t_oxide)
    # At normal incidence s and p degenerate; CLI s-pol matches the textbook formula.
    R_cli_s = _cli_R_pol(n1, t_film, t_oxide, LAMB, "s")
    R_cli_p = _cli_R_pol(n1, t_film, t_oxide, LAMB, "p")
    err_s = float(np.max(np.abs(R_cli_s - R_exact)))
    err_p = float(np.max(np.abs(R_cli_p - R_exact)))
    status = "OK" if max(err_s, err_p) < tol else "FAIL"
    print(f"  [{status}] {label:30s} n={n1}  err_s={err_s:.3e}  err_p={err_p:.3e}")
    assert err_s < tol, f"{label}: s-pol max err {err_s:.3e} > {tol:.3e}"
    assert err_p < tol, f"{label}: p-pol max err {err_p:.3e} > {tol:.3e}"


def _check_widget_matches_cli() -> None:
    """compute_rg integrates over RGB bandpasses; cross-check the underlying
    R(λ) by computing the widget's reflectance for a single film and comparing
    against _reflectance_at_angle (which has been validated against the exact
    formula in _check above)."""

    # Widget compute_rg stacks layer_thickness * n_layers — drive a single
    # layer of t_film thickness at na=0.
    t_film = 12.0
    t_oxide = 90.0
    n1 = 2.6 - 1.3j

    # Widget's compute_rg returns (r_contrast, g_contrast, t_films) — derived
    # quantities, not R(λ).  Easiest way to verify the widget's transfer
    # matrix is sound: re-use compute_rg and check that its 1-layer contrast
    # matches a contrast computed from the exact-formula R(λ) folded through
    # the same RGB bandpasses + illuminant.
    from hbn_contrast_widget import _IMX183_GREEN_LIT, _IMX183_RED_LIT, lamb

    r_arr, g_arr, t_arr = compute_rg(n1, t_oxide, na=0.0, layer_thickness_nm=t_film, max_layers=1)
    r_widget, g_widget = float(r_arr[0]), float(g_arr[0])

    # Exact reference path: closed-form R for film and bare-substrate.
    R_film = _exact_3layer_R(lamb, n1, t_film, t_oxide)
    R_sub = _exact_3layer_R(lamb, n1, 0.0, t_oxide)
    V_flake_r = float(np.trapezoid(R_film * _IMX183_RED_LIT, lamb))
    V_flake_g = float(np.trapezoid(R_film * _IMX183_GREEN_LIT, lamb))
    V_sub_r = float(np.trapezoid(R_sub * _IMX183_RED_LIT, lamb))
    V_sub_g = float(np.trapezoid(R_sub * _IMX183_GREEN_LIT, lamb))
    r_ref = (V_flake_r - V_sub_r) / V_sub_r
    g_ref = (V_flake_g - V_sub_g) / V_sub_g

    err_r = abs(r_widget - r_ref)
    err_g = abs(g_widget - g_ref)
    status = "OK" if max(err_r, err_g) < 1e-6 else "FAIL"
    print(f"  [{status}] widget compute_rg (1-layer)   err_r={err_r:.3e}  err_g={err_g:.3e}")
    assert err_r < 1e-6, f"widget R contrast err {err_r:.3e}"
    assert err_g < 1e-6, f"widget G contrast err {err_g:.3e}"


def main() -> int:
    print("CLI _reflectance_at_angle vs exact closed-form R(λ):")
    # Transparent baseline (k=0): the conj trick used to be correct here, so
    # this case should always pass even on the buggy implementation.
    _check("transparent  n=2.10", complex(2.10, 0.0), 10.0, 90.0)
    _check("transparent  n=2.40", complex(2.40, 0.0), 20.0, 285.0)
    # Absorbing cases — these would fail with the np.conj formulation.
    _check("absorbing    n=2.6-1.3j", complex(2.6, -1.3), 10.0, 90.0)
    _check("absorbing    n=2.6-1.3j", complex(2.6, -1.3), 30.0, 90.0)
    _check("strongly abs n=3.0-2.5j", complex(3.0, -2.5), 15.0, 90.0)
    _check("graphite-ish n=2.6-1.3j", complex(2.6, -1.3), 6.4, 89.46)

    print("\nWidget compute_rg vs closed-form-derived contrast:")
    _check_widget_matches_cli()

    print("\nAll regression checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

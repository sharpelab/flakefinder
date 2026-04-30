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

from hbn_contrast import (
    _SI_DATA,
    _SIO2_DATA,
    _interp_index,
    _na_gauss_legendre,
    _reflectance_at_angle,
    reflectance,
)
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


def _check_against_tmm() -> None:
    """Cross-check our R(λ, θ) against the `tmm` package across a panel of cases.

    `tmm.coh_tmm` is an independently-maintained reference implementation
    that is the de-facto standard in optics literature.  Two convention
    notes:

    1. tmm uses ``n + i·k`` for absorbing media (positive imag = absorbing);
       our code uses ``n - i·k``.  Pass ``np.conj(n)`` when crossing over.
    2. tmm wants ``[inf, …, inf]`` thicknesses for the semi-infinite cap
       layers; we feed in finite SiO₂ + the air/Si caps.
    """
    import tmm

    cases = [
        # (label, n_film, t_film, t_oxide, lam, theta, pol)
        ("transparent normal", complex(2.10, 0.0), 10.0, 90.0, 550.0, 0.0, "s"),
        ("transparent oblique-s", complex(2.10, 0.0), 20.0, 285.0, 632.8, 0.30, "s"),
        ("transparent oblique-p", complex(2.10, 0.0), 20.0, 285.0, 632.8, 0.30, "p"),
        ("absorbing normal", complex(2.6, -1.3), 10.0, 90.0, 550.0, 0.0, "s"),
        ("absorbing thick", complex(2.6, -1.3), 30.0, 90.0, 633.0, 0.0, "s"),
        ("strongly abs normal", complex(3.0, -2.5), 15.0, 90.0, 500.0, 0.0, "s"),
        ("absorbing oblique-s", complex(2.6, -1.3), 12.0, 90.0, 600.0, 0.40, "s"),
        ("absorbing oblique-p", complex(2.6, -1.3), 12.0, 90.0, 600.0, 0.40, "p"),
        ("very thick absorber", complex(2.6, -1.3), 500.0, 90.0, 550.0, 0.0, "s"),  # round-trip ≈ 0
        ("graphite-ish at 6.4nm", complex(2.6, -1.3), 6.4, 89.46, 550.0, 0.0, "s"),
        ("graphite-ish at 13.7nm", complex(2.6, -1.3), 13.7, 90.0, 700.0, 0.0, "s"),
    ]

    for label, n_film, t_film, t_oxide, lam_nm, theta, pol in cases:
        # Our path — single-wavelength array.
        lamb = np.array([lam_nm])
        n2 = complex(_interp_index(lamb, _SIO2_DATA)[0])
        n3 = complex(_interp_index(lamb, _SI_DATA)[0])

        def n_fn(_lam: np.ndarray, _nf: complex = n_film) -> np.ndarray:
            return np.full_like(_lam, _nf, dtype=np.complex128)

        R_ours = _reflectance_at_angle(
            lamb,
            n_fn,
            n_layers=1,
            t_oxide_nm=t_oxide,
            theta0=theta,
            pol=pol,
            layer_thickness_nm=t_film,
        )[0]

        # tmm path — flip imag sign to match its n+ik convention.
        n_list = [1.0, np.conj(n_film), np.conj(n2), np.conj(n3)]
        d_list = [np.inf, t_film, t_oxide, np.inf]
        R_tmm = tmm.coh_tmm(pol, n_list, d_list, theta, lam_nm)["R"]

        err = float(abs(R_ours - R_tmm))
        status = "OK" if err < 1e-12 else "FAIL"
        print(
            f"  [{status}] {label:24s} pol={pol} θ={theta:.2f}  R_ours={R_ours:.6f}  R_tmm={R_tmm:.6f}  err={err:.2e}"
        )
        assert err < 1e-12, f"{label}: |R_ours - R_tmm| = {err:.3e} > 1e-12"


def _check_na_quadrature_weights() -> None:
    """`_na_gauss_legendre` weights should integrate sin(θ)cos(θ) over [0, θ_max].

    Analytical: ∫₀^θ_max sin(θ) cos(θ) dθ = ½ sin²(θ_max) = ½ NA² (since
    θ_max = arcsin(NA)).  This sanity-checks the basic quadrature
    construction independent of any reflectance.
    """
    for na in (0.10, 0.25, 0.50, 0.85):
        thetas, weights = _na_gauss_legendre(na)
        # The weights already absorb sin·cos·dθ — summing approximates the
        # integral of f(θ)=1 weighted by sin(θ)cos(θ).
        integral = float(weights.sum())
        expected = 0.5 * na * na
        err = abs(integral - expected)
        status = "OK" if err < 1e-12 else "FAIL"
        print(f"  [{status}] NA={na:.2f}  Σweights={integral:.9f}  ½ sin²θ_max={expected:.9f}  err={err:.2e}")
        assert err < 1e-12, f"NA={na}: weight sum {integral:.6e} vs expected {expected:.6e}"


def _check_na_continuity() -> None:
    """As NA→0, reflectance(NA) should approach the NA=0 single-angle value.

    Quantitatively, the leading correction is O(NA²) (the cone-averaged R
    differs from on-axis R by a quadratic in θ_max).  Test: at very small
    NA the difference is dominated by the |R'(θ)| · NA² term, so for
    NA=0.005 we expect agreement to a few parts in 1e6 for any smooth R.
    """
    lamb = np.array([550.0])

    def n_fn(_lam: np.ndarray) -> np.ndarray:
        return np.full_like(_lam, complex(2.6, -1.3), dtype=np.complex128)

    R0 = float(reflectance(lamb, n_fn, n_layers=1, t_oxide_nm=90.0, na=0.0, layer_thickness_nm=10.0)[0])
    R_eps = float(reflectance(lamb, n_fn, n_layers=1, t_oxide_nm=90.0, na=0.005, layer_thickness_nm=10.0)[0])
    err = abs(R_eps - R0)
    status = "OK" if err < 5e-6 else "FAIL"
    print(f"  [{status}] R(NA=0)={R0:.9f}  R(NA=0.005)={R_eps:.9f}  Δ={err:.2e}  (should be O(NA²)≈3e-5)")
    assert err < 5e-6, f"NA=0.005 should match NA=0 to ~1e-5; got {err:.3e}"


def _check_na_node_convergence() -> None:
    """Increasing the number of Gauss-Legendre nodes shouldn't change the answer."""
    lamb = np.array([550.0])

    def n_fn(_lam: np.ndarray) -> np.ndarray:
        return np.full_like(_lam, complex(2.6, -1.3), dtype=np.complex128)

    R_8 = float(reflectance(lamb, n_fn, 1, 90.0, na=0.25, n_angles=8, layer_thickness_nm=10.0)[0])
    R_20 = float(reflectance(lamb, n_fn, 1, 90.0, na=0.25, n_angles=20, layer_thickness_nm=10.0)[0])
    R_40 = float(reflectance(lamb, n_fn, 1, 90.0, na=0.25, n_angles=40, layer_thickness_nm=10.0)[0])
    err_8_40 = abs(R_8 - R_40)
    err_20_40 = abs(R_20 - R_40)
    status = "OK" if err_20_40 < 1e-13 else "FAIL"
    print(f"  [{status}] R(8 nodes)={R_8:.12f}  R(20)={R_20:.12f}  R(40)={R_40:.12f}")
    print(f"        |8−40|={err_8_40:.2e}  |20−40|={err_20_40:.2e}")
    assert err_20_40 < 1e-13, f"20-node integration diverges from 40-node: {err_20_40:.3e}"


def _check_na_cone_against_tmm() -> None:
    """Manual-tmm cone average vs `reflectance(na=…)`.

    Pulls the same Gauss-Legendre nodes/weights, calls tmm at each angle
    for both polarizations, and computes the weighted average ourselves.
    Compares to the CLI's `reflectance(na=…)` output, which integrates
    via the same nodes through `_reflectance_at_angle`.

    This isolates the NA-averaging path from the per-angle reflectance
    path (which the previous test panel already verified vs tmm).
    """
    import tmm

    cases = [
        # (label, n_film, t_film, t_oxide, lam, na)
        ("transparent NA=0.25", complex(2.10, 0.0), 20.0, 285.0, 550.0, 0.25),
        ("absorbing NA=0.25", complex(2.6, -1.3), 12.0, 90.0, 600.0, 0.25),
        ("absorbing NA=0.50", complex(2.6, -1.3), 12.0, 90.0, 600.0, 0.50),
        ("strongly abs NA=0.85", complex(3.0, -2.5), 15.0, 90.0, 500.0, 0.85),
        ("graphite NA=0.25", complex(2.6, -1.3), 6.4, 90.0, 550.0, 0.25),
    ]

    for label, n_film, t_film, t_oxide, lam_nm, na in cases:
        lamb = np.array([lam_nm])
        n2 = complex(_interp_index(lamb, _SIO2_DATA)[0])
        n3 = complex(_interp_index(lamb, _SI_DATA)[0])

        def n_fn(_lam: np.ndarray, _nf: complex = n_film) -> np.ndarray:
            return np.full_like(_lam, _nf, dtype=np.complex128)

        R_ours = float(reflectance(lamb, n_fn, 1, t_oxide, na=na, layer_thickness_nm=t_film)[0])

        # Manual tmm averaging — same nodes/weights, same s/p convention.
        thetas, weights = _na_gauss_legendre(na)
        n_list = [1.0, np.conj(n_film), np.conj(n2), np.conj(n3)]
        d_list = [np.inf, t_film, t_oxide, np.inf]
        R_sum = 0.0
        for theta, w in zip(thetas, weights, strict=True):
            R_s = tmm.coh_tmm("s", n_list, d_list, theta, lam_nm)["R"]
            R_p = tmm.coh_tmm("p", n_list, d_list, theta, lam_nm)["R"]
            R_sum += 0.5 * (R_s + R_p) * w
        R_tmm = R_sum / weights.sum()

        err = abs(R_ours - R_tmm)
        status = "OK" if err < 1e-12 else "FAIL"
        print(f"  [{status}] {label:24s}  R_ours={R_ours:.9f}  R_tmm-avg={R_tmm:.9f}  err={err:.2e}")
        assert err < 1e-12, f"{label}: |R_ours - R_tmm-avg| = {err:.3e} > 1e-12"


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

    print("\nCLI _reflectance_at_angle vs `tmm` package (oblique + s/p included):")
    _check_against_tmm()

    print("\nNA-cone quadrature weight integral (Σweights = ½ sin²θ_max):")
    _check_na_quadrature_weights()

    print("\nNA → 0 continuity (small-NA average matches NA=0 single point):")
    _check_na_continuity()

    print("\nNA quadrature node-count convergence:")
    _check_na_node_convergence()

    print("\nNA-cone average vs manual tmm cone average (same nodes/weights):")
    _check_na_cone_against_tmm()

    print("\nAll regression checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

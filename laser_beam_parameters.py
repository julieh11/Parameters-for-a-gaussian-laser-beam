import math
import itertools
import numpy as np

# this code is based on the methods described in this paper "Determination of waist parameters of a Gaussian beam" by Shojiro Nemoto from 1986!
# Inpurt data of beam diameters 1/e^2 (mm) 

z_mm = np.array([4, 25, 30, 48, 70, 100, 125, 160, 175, 220, 250, 300, 315, 425, 450, 550, 675, 950], dtype=float)
dx_mm = np.array([2.2737, 2.3375, 2.3566, 2.4021, 2.49, 2.6055, 2.7333, 2.9137, 2.9783, 3.1979, 3.346, 3.5827, 3.653, 4.1274, 4.2072, 4.4505, 4.9107, 6.088], dtype=float)
dy_mm = np.array([1.2487, 1.2932, 1.3065, 1.3547, 1.427, 1.4894, 1.5648, 1.6624, 1.6956, 1.8408, 1.9166, 2.055, 2.1008, 2.3968, 2.4499, 2.8852, 3.0871, 3.8003], dtype=float)


LAM_MM = 0.663e-3 # 663 nm in mm


def gaussian_diameter_mm(z_mm, z0_mm, w0_mm, lam_mm=LAM_MM):
    """Ideal Gaussian diameter,  w0_mm is the waist radius."""
    k = 2.0 * math.pi / lam_mm
    return 2.0 * w0_mm * np.sqrt(1.0 + ((z_mm - z0_mm) / (k * w0_mm * w0_mm)) ** 2)


def pairwise_nemoto_solutions(z1, d1, z2, d2, lam_mm=LAM_MM, tol=1e-9):
    """
    Pairwise two-point inversion
    returns candidate solutions (z0_mm, w0_mm, branch_id).
    diameters are converted to radii.
    """
    s1 = d1 / 2.0
    s2 = d2 / 2.0
    k = 2.0 * math.pi / lam_mm

    r = ((z1 - z2) / (k * s1 * s2)) ** 2
    if r > 1.0:
        return []

    p = s1 / s2 + s2 / s1
    q = s1 / s2 - s2 / s1
    sbar = math.sqrt(s1 * s2)

    solutions = []

    # Two branches from the paper
    for branch_id, branch_sign in [("+", +1), ("-", -1)]:
        inside = r * (p + branch_sign * 2.0 * math.sqrt(max(0.0, 1.0 - r))) / (q * q + 4.0 * r)
        if inside <= 0.0:
            continue

        w0 = sbar * math.sqrt(inside)  # waist radius

        # Recover z0 by matching the two point equations
        matches = []
        for sgn1 in (-1, +1):
            z0_1 = z1 + sgn1 * k * w0 * math.sqrt(max(s1 * s1 - w0 * w0, 0.0))
            for sgn2 in (-1, +1):
                z0_2 = z2 + sgn2 * k * w0 * math.sqrt(max(s2 * s2 - w0 * w0, 0.0))
                if abs(z0_1 - z0_2) <= tol * max(1.0, abs(z0_1), abs(z0_2)):
                    matches.append(0.5 * (z0_1 + z0_2))

        # Deduplicate
        for z0 in sorted(set(round(v, 12) for v in matches)):
            solutions.append((float(z0), float(w0), branch_id))

    return solutions


def estimate_axis(z_mm, d_mm, lam_mm=LAM_MM):
    """
    Computes pairwiseestimates for one axis and aggregate them robustly.
      - Generates both algebraic branches for every pair.
      -  Keeps the candidate that best fits the full dataset for that pair.
      -  Aggregates those pairwise best candidates with inverse-RMS^2 weights.
    """
    z_mm = np.asarray(z_mm, dtype=float)
    d_mm = np.asarray(d_mm, dtype=float)

    kept = []
    for i, j in itertools.combinations(range(len(z_mm)), 2):
        sols = pairwise_nemoto_solutions(z_mm[i], d_mm[i], z_mm[j], d_mm[j], lam_mm=lam_mm)
        if not sols:
            continue

        best = None
        best_rms = float("inf")
        for z0, w0, branch_id in sols:
            pred = gaussian_diameter_mm(z_mm, z0, w0, lam_mm=lam_mm)
            rms = float(np.sqrt(np.mean((pred - d_mm) ** 2)))
            if rms < best_rms:
                best_rms = rms
                best = (z0, w0, branch_id, rms, i, j)

        if best is not None:
            kept.append(best)

    if not kept:
        raise RuntimeError("Couldn't find valid pairwise solutions :(")

    kept = np.array(kept, dtype=object)

    z0_vals = kept[:, 0].astype(float)
    w0_vals = kept[:, 1].astype(float)
    rms_vals = kept[:, 3].astype(float)

    weights = 1.0 / np.maximum(rms_vals, 1e-12) ** 2
    z0_est = float(np.average(z0_vals, weights=weights))
    w0_est = float(np.average(w0_vals, weights=weights))

    pred = gaussian_diameter_mm(z_mm, z0_est, w0_est, lam_mm=lam_mm)
    fit_rms = float(np.sqrt(np.mean((pred - d_mm) ** 2)))

    zR = math.pi * w0_est * w0_est / lam_mm
    theta_half = lam_mm / (math.pi * w0_est)

    return {
        "z0_mm": z0_est,
        "w0_mm": w0_est,
        "rayleigh_range_mm": zR,
        "half_angle_rad": theta_half,
        "fit_rms_mm": fit_rms,
        "pairwise_candidates": kept,
        "predicted_diameter_mm": pred,
    }


if __name__ == "__main__":
    x = estimate_axis(z_mm, dx_mm)
    y = estimate_axis(z_mm, dy_mm)

    print("X axis:")
    print(f"  z0 = {x['z0_mm']:.3f} mm")
    print(f"  w0 = {x['w0_mm']:.5f} mm")
    print(f"  zR = {x['rayleigh_range_mm']:.3f} mm")
    print(f"  half-angle divergence = {x['half_angle_rad']*1e3:.3f} mrad")
    print(f"  RMS error = {x['fit_rms_mm']:.4f} mm")
    print(x['predicted_diameter_mm'])

    print("\nY axis:")
    print(f"  z0 = {y['z0_mm']:.3f} mm")
    print(f"  w0 = {y['w0_mm']:.5f} mm")
    print(f"  zR = {y['rayleigh_range_mm']:.3f} mm")
    print(f"  half-angle divergence = {y['half_angle_rad']*1e3:.3f} mrad")
    print(f"  RMS error = {y['fit_rms_mm']:.4f} mm")
    print(y['predicted_diameter_mm'])
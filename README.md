# Guaranteed-bounds-iterative-error-CG-FFT-homogenization

Supplementary notebooks to the paper

M. Ladecký, J. Papež, P. Tichý, L. Pastewka,
*Guaranteed Bounds on the Iterative Error of CG-Accelerated FFT-Based Homogenization*.

One self-contained notebook per test problem of Section 6. Each notebook contains the finite element
operators (linear triangles on a regular grid), the discrete Green's operator preconditioner and the
solves; only the PCG with the error bounds and estimates is in a separate file.

| file | content |
|---|---|
| `P1_square_inclusion.ipynb` | problem P1, square inclusion; exact effective conductivity of Obnosov |
| `P2_circular_inclusions.ipynb` | problem P2, 58 circles with the two rotated anisotropic phases $\kappa\,\mathrm{diag}(1,2)$ and $\kappa\,\mathrm{diag}(2,1)$ |
| `P3_phase_field.ipynb` | problem P3, phase field from topology optimization, $\mathbf{A}=\rho\,\mathbf{I}+(1-\rho)\,\kappa\,\mathbf{I}$ |
| `pcg_error_estimates.py` | PCG (Algorithm 1 of the paper) with the trivial bounds, the Gauss–Radau (GR) upper bound and the PT lower bound and upper estimate with the adaptive delay (Appendix B) |
| `p2_circles_256.npz` | phases of P2 on the grid $256^2$ (0 matrix, 1 and 2 the two inclusion phases) |
| `p3_density_256.npz` | density $\rho$ of P3 on the grid $256^2$ |

## Content of each notebook

Every notebook has the same structure: element library, operators, material and geometry, system setup,
Green's preconditioner, solve, and the experiments of Section 6 for its problem:

1. iterative error, residual, bounds and estimates for $\kappa=10^{-3}$ (Section 6.1, Figure 2), for P1 and P2 next to the opposite contrast $\kappa=10^{3}$;
2. efficiency for $\kappa=10^{-1},10^{-2},10^{-3}$ (Section 6.1, Figure 3);
   for P1 and P2 also for highly conducting inclusions, $\kappa=10^{1},10^{2},10^{3}$ (Plot 2b, supplementary to Figure 3);
3. fixed and adaptive delay, with the minimal delay $d_{\min}=10$ (Section 6.2, Figure 4);
   additional experiment: the tolerance $\tau=0.5,0.25,0.1,0.05$ of the adaptive delay (Section 6.2), each value a separate PCG run;
4. stopping criteria with $\mathit{tol}_\mathrm{CG}=10^{-8}$ (Section 6.3, Figure 5);
5. discretization and total error on the grids $256^2$, $512^2$ and $1024^2$ (Section 6.4, Figure 6).

The figures use the line styles, markers and axis limits of the paper.

## Differences from the paper

- The computations use the grid $256^2$ instead of $1024^2$ (except the discretization study), so the
  numbers differ slightly from the paper. The data of P2 and P3 are given on $256^2$ in the paper too, and
  on $1024^2$ the notebooks reproduce the paper's runs (P1 and P3 to about $10^{-11}$; P2 to about $10^{-12}$
  up to iteration 30, and to about $10^{-4}$ in the last iterations, where finite-precision effects of CG
  on the plateaus differ between implementations).
- For P2 and P3, the effective conductivity in the discretization study is the discrete solution on
  $4096^2$ from the paper's runs (a $4096^2$ solve takes too long for a notebook).
- Each monitored run stops when the squared residual, the trivial upper bound, the GR upper bound and the
  PT upper estimate are all below $10^{-10}$.

## Running

```
pip install -r requirements.txt
jupyter lab
```

The notebooks are stored with their outputs. Run times on a workstation: P1 about 20 s, P2 about 2 min,
P3 about 4 min.

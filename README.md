# Stern-Gerlach Effects on the Nab Neutron Beam

Numerical simulations for evaluating Stern-Gerlach (SG) effects on the incoming neutron beam in the **Nab experiment**. The code tracks ensembles of neutrons through a precomputed magnetic-field map and is used to study three questions relevant to the beam systematic:

1. How strongly can the SG force broaden or displace the neutron beam?
2. Can SG deflections combined with the Nab collimators produce unintended neutron polarization?
3. Can SG acceleration or deceleration along the beam direction measurably change the neutron capture/decay flux in the spectrometer?

The repository contains the simulation utilities and analysis scripts used for the collaboration study *Stern-Gerlach Effect on Nab's Neutron Beam*.

---


## Simulation model

### Coordinate system

The simulation uses

- `+x`: neutron beam direction,
- `+y`: beam-left,
- `+z`: vertically upward from the beam guide.

Neutrons are initialized at `x = -1.19 m`. Typical Monte Carlo runs draw initial positions uniformly over

- `y in [-0.03, 0.03] m`,
- `z in [-0.035, 0.035] m` relative to the beam center,

with a vertical beam-center offset of `0.13189 m` applied to match the field-map coordinate system.

Initial neutron wavelengths may be fixed for controlled studies or sampled uniformly from **2 A to 25 A**. Wavelengths are converted to longitudinal velocities with the de Broglie relation

$$
v_x = \frac{h}{m_n \lambda}.
$$

### Magnetic-field data

The simulation uses a precomputed Opera magnetic-field map so that the passive magnetic shielding is included. The scripts currently load

```text
SG z-adjusted_m.npy
```

as the field input.

The expected array layout is

```text
[x, y, z, Bx, By, Bz, |B|]
```

with positions in meters and magnetic field in tesla. The current indexing assumes regularly ordered transverse field slices and a **5 mm** spatial step. `Support.find_nearest_points()` uses the neutron's current `x` slice and selects the nearest `(y, z)` field point within that slice.

> **Important:** The field-map ordering and grid spacing are part of the numerical method. If a different Opera export is used, verify its units, ordering, transverse grid dimensions, and spacing before running the simulation.

### Stern-Gerlach force

For the adiabatic approximation, the neutron magnetic moment is assumed to remain aligned or anti-aligned with the local magnetic field, giving

$$
\mathbf{F} = \pm \mu_n \nabla |\mathbf{B}|.
$$

The magnetic-field gradient is calculated with centered finite differences on the field grid.

For non-adiabatic calculations, each neutron carries a spin-direction unit vector $\hat{\sigma}$ and the full component-wise force is evaluated as

$$
F_i = \mu_n \left(
\hat{\sigma}_x \frac{\partial B_x}{\partial i}
+ \hat{\sigma}_y \frac{\partial B_y}{\partial i}
+ \hat{\sigma}_z \frac{\partial B_z}{\partial i}
\right).
$$

Random spin directions are sampled uniformly on the unit sphere.

### Particle propagation

The field grid is divided into discrete slices along `x`. For each neutron, the code solves for the time required to reach the next slice under the local SG force. The neutron position and velocity are then updated using constant-acceleration kinematics over that interval:

$$
\mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_n \Delta t
+ \frac{1}{2}\frac{\mathbf{F}}{m_n}\Delta t^2,
$$

$$
\mathbf{v}_{n+1} = \mathbf{v}_n + \frac{\mathbf{F}}{m_n}\Delta t.
$$

Neutrons that leave the usable field-map region or intersect a modeled collimator are removed from subsequent dynamics while preserving vectorized NumPy operations for the surviving ensemble.

### Spin propagation

In the non-adiabatic treatment, spins precess about the local field with angular frequency

$$
\omega_L = \gamma_n |\mathbf{B}|.
$$

For each propagation step, the spin vector is updated with Rodrigues' rotation formula using the local field direction as the rotation axis and

$$
\phi = \gamma_n |\mathbf{B}|\Delta t \pmod{2\pi}.
$$

---

## Repository contents

| File | Purpose |
| --- | --- |
| `Support.py` | Physical constants and shared utilities: random spin sampling, wavelength-to-velocity conversion, field-point lookup, vectorized dot/cross products, Opera table conversion, and synthetic test-field generation. |
| `Perp Spread Histogram.py` | Adiabatic beam-spread study. Computes the maximum displacement transverse to the beam and produces the perpendicular-displacement histogram. |
| `z-dependence on spread.py` | Controlled 5 A study of vertical SG deflection for several initial `z` positions. |
| `y-dependence on spread.py` | Sweeps the initial `y` position to quantify the lateral-position dependence of the vertical deflection. |
| `Adiabaticity Tests(2).py` | Non-adiabatic spin propagation and spin-field angle tracking used to test the adiabatic approximation. |
| `Polarization Checks(1).py` | Propagates neutrons through the modeled collimators and evaluates the surviving beam polarization as a function of Monte Carlo sample size. |
| `Capture Flux Check(1).py` | Tracks SG-induced changes in longitudinal neutron velocity for the capture-flux upper-bound calculation. |
| `Main(20260909-234202).py` | Development/integration driver containing the common simulation structure and diagnostic code. |
| `NeutronClass.py` | Experimental object-oriented neutron container. The production analysis scripts currently use vectorized NumPy arrays instead. |

The analysis files are research scripts rather than a packaged command-line application; simulation parameters are set near the top of each script.

---

## Requirements

- Python **3.9+**
- NumPy
- SciPy
- Matplotlib

A minimal environment can be created with

```bash
python -m venv .venv
python -m pip install numpy scipy matplotlib
```

For long-term reproducibility, pin the package versions used for the final analysis in a `requirements.txt` or environment file.

---

## Running the analyses

Place `SG z-adjusted_m.npy` in the working directory used by the scripts, or update the hard-coded field path in the relevant file.

Examples:

```bash
python "Perp Spread Histogram.py"
python "z-dependence on spread.py"
python "y-dependence on spread.py"
python "Adiabaticity Tests(2).py"
python "Polarization Checks(1).py"
python "Capture Flux Check(1).py"
```

The most frequently changed controls are defined near the beginning of each analysis script, including:

```python
N = 10000
spin_orientation = 'random'
gravity = False
x0 = -1.19
ymin, ymax = -0.03, 0.03
zmin, zmax = -0.035, 0.035
lambdamin, lambdamax = 2e-10, 25e-10
```

Some studies override these defaults with fixed initial positions or wavelengths. Read the configuration block at the top of a script before launching a large run.

### Reproducibility note

Monte Carlo initial conditions are generated with NumPy's random-number generator and the current scripts do not set a fixed random seed. Exact numerical values will therefore vary slightly from run to run. Set a seed explicitly when bitwise-reproducible samples are required.

---

## Collimator model

The polarization analysis includes four collimators within the simulated flight region. Their modeled positions and apertures are:

| Collimator | x position (m) | Height (m) | Width (m) |
| --- | ---: | ---: | ---: |
| C2 | -1.029 | 0.070 | 0.064 |
| E1 | -0.558 | 0.070 | 0.064 |
| E2 | -0.372 | 0.070 | 0.054 |
| E3 | -0.202 | 0.070 | 0.054 |

C1 is upstream of the simulated starting position and is therefore not explicitly propagated in these scripts.

The centered E2/E3 apertures geometrically remove approximately 10% of the initially uniform beam area even without an SG force. The relevant systematic question is whether the SG displacement makes that removal spin-dependent; within the centered geometry studied here, no significant induced polarization was observed.

---

## Numerical validation and limitations

The propagation code was checked against a custom magnetic-field case with an analytic solution; the collaboration study reports agreement of the simulated forces, times, and positions with the analytic calculation.

Important limitations of the current implementation include:

- **Finite field resolution.** Gradients and nearest-field-point assignment are limited by the 5 mm Opera grid.
- **Piecewise-constant force.** The force is held constant while each neutron travels from one `x` slice to the next.
- **Nearest-neighbor field lookup.** No spatial interpolation is currently performed within a slice.
- **Monte Carlo statistics.** Large ensembles require substantial memory because neutron states and intermediate quantities are stored in vectorized arrays. Runs beyond approximately `N = 10^6` became impractical in the configuration used for the collaboration study.
- **Centered collimators.** The reported polarization result applies to the modeled centered geometry. Collimator misalignment is a natural follow-up systematic study.
- **Research-code structure.** Several scripts contain analysis-specific diagnostics or commented plotting blocks. Inspect a script before using it as a general-purpose simulation driver.

---

## Analysis provenance

The methodology and results represented by these scripts are documented in the Nab collaboration technical write-up:

> **Skylar Clymer, _Stern-Gerlach Effect on Nab's Neutron Beam_, September 9, 2026.**

The same write-up is intended to form the basis of the corresponding thesis chapter. If the document is approved for distribution with the repository, a copy can be placed under `docs/` and linked here; otherwise the repository can remain code-only and the write-up can be distributed through the appropriate Nab collaboration channel.

---

## Citation / reuse

If these scripts or their numerical results are reused in Nab analysis, please reference this repository together with the associated Stern-Gerlach technical write-up or the corresponding thesis chapter.

For questions about the implementation, field-map conventions, or analysis assumptions, contact the repository author through the Nab collaboration or the GitHub repository.

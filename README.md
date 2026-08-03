# ember

An 'Enhanced Multi-Block solvER' for turbomachinery computational fluid
dynamics, written by [James Brind](https://jamesbrind.uk) of the [Whittle
Laboratory](https://whittle.eng.cam.ac.uk) at the University of Cambridge.
Solves the compressible Reynolds-averaged Navier-Stokes equations on
multi-block structured meshes, using an evolution of the fast and robust Denton
algorithms. Pre- and post-processing is handled through a numpy-like Python
interface, while the heavy computations run through compiled Fortran for speed.

[See the manual for full documentation](https://ember-cfd.org)

## Installation

ember requires Python 3.12 or newer. Install it from the Python Package Index
under the distribution name `ember-cfd`, not the import name `ember`:

```bash
pip install ember-cfd
```

Precompiled wheels are published for Linux (x86_64), so no Fortran compiler is
needed there. On other platforms pip falls back to building from source, which
requires a Fortran toolchain. See the
[installation guide](https://ember-cfd.org/en/latest/install.html) for more information on source builds and performance tuning the build for your CPU.

## Performance tuning for a new CPU

The hot kernels carry blocking constants tuned against a specific cache
hierarchy and vector width. They are correct on any machine — every one of
them is bitwise-neutral, affecting only the order in which work is done — but
a value tuned for one CPU can cost tens of percent on another. Re-sweep them
when moving to a new architecture, and do it **in the real build** (`make
compile`, which compiles the whole `_fortran/` tree with `-flto
-fwhole-program`); a standalone single-file compile has inverted results here
before.

| constant | where | what bounds it |
| --- | --- | --- |
| `IRS_BJ = 32` | `_fortran/residual.f90` | **Vector-chain count, then nothing.** Sets how many independent lanes the i-solve's recurrence runs over. Below 32 it goes latency-bound (16 costs +19%); above 32 it is flat (64 and 128 are within ±2%), so the tile spilling L1d no longer matters. The least fussy of these constants — but only because the transpose is blocked; it was a sharp L1d-capacity optimum before that. |
| `IRS_TB = 8` | `_fortran/residual.f90` | **SIMD lane count** (8 for AVX2, 16 for AVX-512). Edge of the transpose block, so one staged row is exactly one vector load. Steeply optimal — 4 and 16 are both large regressions on AVX2. |
| `IRS_W = 64` | `_fortran/residual.f90` | **L2 capacity.** The i-strip carried through the fused j+k solves is `IRS_W*(nj-1)*(nk-1)*4` bytes — 917 KB on a 273×65×57 block, inside a 2 MB L2. Also sets the vector loop length, so it trades off in two directions at once. |
| `_KB_SLAB = 8` | `grid.py` | **L2/L3 capacity.** Depth of the k-slab that `set_residual` and `set_visc_force` stream their nodal working set in. Re-sweep if blocks grow past ~1M cells. |
| `njp` pad rule | `grid.py` | **Page size.** Pads the rolling face-flow plane by one j-row when `ni*nj` is a multiple of 1024 (= 4096-byte page ÷ 4-byte float), to dodge a 4K-aliasing penalty at power-of-two plane sizes. |

Build flags matter as much as the constants:

- `EMBER_MARCH` defaults to `-march=haswell` for portable wheels. Set
  `EMBER_MARCH="-march=native -mtune=native"` when building for one machine.
- The pinned GCC inline budgets in `setup.py`
  (`inline-unit-growth`, `large-function-growth`) are worth **−53% serial** on
  `set_residual` — without them its face helpers are simply not inlined. Do
  not drop them. Both are needed; `inline-unit-growth` alone is worth only
  2.3% but is necessary for the other to bind.
- **Do not enable PGO** (`EMBER_PGO=use`). On this code it is a **+169%
  regression** — profile data makes GCC un-inline exactly the helpers whose
  inlining is worth the 53%.

`bench/README.md` documents the benchmarking harness and the protocol these
numbers were measured under, including several ways of measuring them that
produced confidently wrong answers.

## Example usage

Solve the flow through a straight annular duct:

```python
import numpy as np

import ember.block
import ember.fluid
import ember.grid
import ember.patch
import ember.solver

# Generate coordinates for a straight annular duct
ni, nj, nk = 25, 17, 17
L = 0.1  # Span [m]
r_hub = 0.45  # Hub radius [m]
Nb = 60  # Number of blades [-]
pitch = 2 * np.pi / Nb  # Theta periodicity [rad]
x = np.linspace(0.0, 2*L, ni)
r = np.linspace(0.0, L, nj) + r_hub
t = np.linspace(-pitch, pitch, nk) / 2
xrt = np.stack(np.meshgrid(x, r, t, indexing="ij"), axis=-1)

# Allocate a block and set up geometry
block = ember.block.Block(shape=(ni, nj, nk))
block.set_xrt(xrt)
block.set_Nb(Nb)

# Set working fluid to a perfect gas
fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
block.set_fluid(fluid)

# Define inlet boundary conditions at i=0 face
# Fixed stagnation pressure and temperature, no swirl
Po1 = 1e5  # [Pa]
To1 = 300.0  # [K]
block.patches["inlet"] = ember.patch.InletPatch(i=0)
block.patches["inlet"].set_Po_To(Po1, To1)
block.patches["inlet"].set_Alpha(0.0)
block.patches["inlet"].set_Beta(0.0)

# Define outlet boundary conditions at i=-1 face
# Fixed static pressure
P2 = 0.9e5  # [Pa]
block.patches["outlet"] = ember.patch.OutletPatch(i=-1)
block.patches["outlet"].set_P(P2)

# Initial conditions: uniform axial flow
block.set_P_T(P2, To1)
Vx_guess = 100.0  # [m/s]
block.set_Vx(Vx_guess)
block.set_Vr(0.0)
block.set_Vt(0.0)

# Create a single-block grid, set wall distance for turbulence model
grid = ember.grid.Grid([block])
grid.calculate_wdist()

# Choose solver settings and run
solver = ember.solver.Solver(n_step=500, cfl=3.0, n_stage=4, n_levels=3)
solver.run(grid)

```

The [example gallery](https://ember-cfd.org/en/latest/auto_examples/index.html) works
through the block interface, and demonstrates the capabilities of the processing interface.

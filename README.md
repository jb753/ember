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
block.patches["inlet"].set_Po_To_Alpha_Beta(Po1, To1, 0.0, 0.0)

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

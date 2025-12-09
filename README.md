# 3D Linear Wave Equation Solver

High-performance C++ solver for the 3D linear wave equation using deal.II finite element library and Newmark-β time integration.

## Problem Formulation

Solves the second-order wave equation:

```
∂²u/∂t² - Δu = f(x,t)  in Ω × (0,T]
u = g(x,t)              on ∂Ω × (0,T]
u(x,0) = u₀(x)          in Ω
∂u/∂t(x,0) = v₀(x)      in Ω
```

where:
- `Ω = [0,1]³` is the computational domain (unit cube)
- `u(x,t)` is the displacement field
- `f(x,t)` is the forcing term
- `g(x,t)` are Dirichlet boundary conditions
- `u₀(x)`, `v₀(x)` are initial displacement and velocity

## Features

- **Newmark-β time integration** in second-order form (predictor-corrector)
- **Continuous Lagrange finite elements** on tetrahedral meshes (simplex elements)
- **MPI parallelization** using Trilinos sparse linear algebra
- **Manufactured solution verification** with convergence analysis
- **Energy conservation monitoring** with automatic warnings
- **Flexible mesh generation**: built-in cube or external Gmsh meshes
- **Performance profiling** with timing breakdowns
- **Output**: ParaView-compatible VTU files + CSV time-series data

## Dependencies

### Required
- **deal.II** ≥ 9.3.1 (with Trilinos support)
- **CMake** ≥ 3.12
- **MPI** (e.g., OpenMPI, MPICH)
- **Trilinos** (for parallel linear algebra)
- **Boost** ≥ 1.72.0

### Optional
- **Gmsh** (for custom mesh generation)
- **Python** 3.x with matplotlib, pandas (for plotting)
- **ParaView** (for visualization)

## Building

### Standard Build

```bash
cd waveEquation
mkdir build && cd build
cmake ..
make -j4
```

### With Custom deal.II Path

```bash
cmake -DDEAL_II_DIR=/path/to/dealii ..
make -j4
```

### Build Targets

- `wave-solver`: Basic solver with manufactured solution
- `wave-convergence`: Spatial and temporal convergence studies
- `wave-dispersion`: Dispersion/dissipation analysis

## Usage

### Basic Solver

```bash
mpirun -np 4 ./wave-solver
```

**Default parameters:**
- Mesh: Built-in cube, 10 subdivisions
- Polynomial degree: 2 (P2 elements)
- Time step: Δt = 0.005
- Final time: T = 1.0
- Newmark parameters: β = 0.25, γ = 0.5 (average acceleration method)
- Output frequency: Every 10 timesteps

**Output:**
- `output-*.vtu`: Displacement, velocity, acceleration fields
- `results/time_series.csv`: Time, energy, errors vs time

### Convergence Study

```bash
mpirun -np 4 ./wave-convergence
```

Performs two convergence studies:

1. **Temporal convergence** (fixed h, varying Δt):
   - Δt ∈ {0.04, 0.02, 0.01, 0.005, 0.0025}
   - Verifies second-order accuracy in time

2. **Spatial convergence** (fixed Δt, varying h):
   - N ∈ {5, 10, 15, 20} subdivisions
   - Verifies optimal FEM convergence rates

**Output:**
- `results/convergence_temporal.csv`
- `results/convergence_spatial.csv`

**Visualization:**
```bash
cd scripts
python plot_convergence.py
```

### Dispersion/Dissipation Analysis

```bash
mpirun -np 4 ./wave-dispersion
```

Simulates Gaussian wave packet propagation for 5 seconds to analyze:
- Numerical dispersion (phase errors)
- Numerical dissipation (energy decay)
- Long-time stability

**Output:**
- `results/time_series.csv`: Energy vs time
- `output-*.vtu`: Wave propagation snapshots

**Visualization:**
```bash
cd scripts
python plot_dispersion.py
```

## Numerical Method

### Newmark-β Time Integration

The solver uses the Newmark-β method in **second-order form** (direct discretization of acceleration):

**Predictor:**
```
u_pred = u^n + Δt·v^n + Δt²·(0.5-β)·a^n
```

**Solve for acceleration:**
```
M·a^{n+1} = f^{n+1} - K·u_pred
```

**Corrector:**
```
v^{n+1} = v^n + Δt·[(1-γ)·a^n + γ·a^{n+1}]
u^{n+1} = u^n + Δt·v^n + Δt²·[(0.5-β)·a^n + β·a^{n+1}]
```

**Default parameters:** β = 0.25, γ = 0.5 (average acceleration method)
- Second-order accurate in time
- Unconditionally stable
- Mild numerical damping

See `docs/design_notes.md` for detailed algorithm discussion.

### Spatial Discretization

- **Finite elements:** Continuous Lagrange (P_r) on tetrahedra
- **Quadrature:** Gauss-Legendre with r+1 points (exact for mass/stiffness)
- **Assembly:** Standard FEM with FEValues
- **Boundary conditions:** Strong enforcement via `MatrixTools::apply_boundary_values`

### Linear Solver

- **Method:** Conjugate Gradient (CG)
- **Preconditioner:** SSOR (default) or AMG (commented, for large problems)
- **Tolerance:** 10⁻⁶ relative to RHS norm
- **System:** M·a = RHS (mass matrix, constant across timesteps)

To enable AMG preconditioner for better scaling, uncomment in `Wave.cpp`:
```cpp
// Uncomment these lines in solve_time_step():
// TrilinosWrappers::PreconditionAMG preconditioner;
// TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
// amg_data.smoother_sweeps = 2;
// amg_data.aggregation_threshold = 0.02;
// preconditioner.initialize(lhs_matrix, amg_data);
```

## Manufactured Solution

For verification, the solver uses:

```
u_ex(x,t) = sin(ω·t)·sin(2π·x)·sin(3π·y)·sin(4π·z)
```

where ω = 5π. This yields:

```
v_ex = ∂u_ex/∂t = ω·cos(ω·t)·sin(2π·x)·sin(3π·y)·sin(4π·z)
a_ex = ∂²u_ex/∂t² = -ω²·sin(ω·t)·sin(2π·x)·sin(3π·y)·sin(4π·z)
f = a_ex - Δu_ex = (-ω² + 29π²)·u_ex
```

Initial conditions are derived from exact solution at t=0.

## Expected Results

### Convergence Rates

**Temporal (Newmark-β with β=0.25, γ=0.5):**
- L2(u): Order ~ 2.0
- H1(u): Order ~ 2.0

**Spatial (P2 elements):**
- L2(u): Order ~ 3.0
- H1(u): Order ~ 2.0

### Energy Conservation

For undamped wave equation, total energy E(t) = ½∫(|v|² + |∇u|²)dx should be conserved.

**Expected behavior:**
- β = 0.25, γ = 0.5: Mild dissipation (~0.1-1% drift over T=1)
- Energy warnings triggered if drift > 1%

## File Structure

```
waveEquation/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── src/
│   ├── Wave.hpp             # Wave class declaration
│   ├── Wave.cpp             # Implementation
│   ├── wave-solver.cpp      # Basic driver
│   ├── wave-convergence.cpp # Convergence study
│   └── wave-dispersion.cpp  # Dispersion analysis
├── mesh/
│   └── generated_mesh.vtk   # Auto-generated mesh (if using built-in)
├── scripts/
│   ├── mesh-cube.geo        # Gmsh geometry script
│   ├── plot_convergence.py  # Convergence plotting
│   └── plot_dispersion.py   # Dispersion plotting
├── docs/
│   └── design_notes.md      # Algorithm details
└── results/
    ├── time_series.csv      # Time-series data
    ├── convergence_*.csv    # Convergence data
    └── *.png                # Generated plots
```

## Performance Tips

1. **For small problems (< 10⁴ DoFs):** Use SSOR preconditioner (default)
2. **For large problems (> 10⁵ DoFs):** Enable AMG preconditioner
3. **For long simulations:** Increase `output_frequency` to reduce I/O
4. **For convergence studies:** Use smaller timesteps (Δt ~ h² for stability)

## Troubleshooting

### Compilation Errors

**Error:** `deal.II not found`
```bash
cmake -DDEAL_II_DIR=/path/to/dealii/install ..
```

**Error:** `Trilinos not found`
- Ensure deal.II was compiled with Trilinos support
- Check `cmake` output for Trilinos location

### Runtime Errors

**Linear solver does not converge:**
- Reduce time step Δt
- Try AMG preconditioner
- Check mesh quality

**Energy drift > 1%:**
- Normal for β ≠ 0 (implicit methods have slight dissipation)
- Reduce Δt for better accuracy
- Try β = 0 for explicit method (requires Δt < CFL limit)

## References

1. K.-J. Bathe, *Finite Element Procedures*, 2nd ed., 2014
2. T. J. R. Hughes, *The Finite Element Method*, Dover, 2000
3. deal.II documentation: https://www.dealii.org/
4. Newmark, N. M., "A Method of Computation for Structural Dynamics", *ASCE J. Eng. Mech. Div.*, 1959

## License

This project follows the same license as deal.II (LGPL 2.1+).

## Author

Created as part of NMPDE course project, following coding patterns from heat equation labs.

## Acknowledgments

- deal.II developers for excellent documentation
- Course instructors for heat solver examples
- Trilinos team for scalable linear algebra

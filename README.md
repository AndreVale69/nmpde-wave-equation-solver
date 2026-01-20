# Wave Equation Solver (deal.II)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Build system](https://img.shields.io/badge/build-CMake-informational)
![MPI](https://img.shields.io/badge/parallel-MPI-blueviolet)
![Visualization](https://img.shields.io/badge/visualization-VTK%20%2F%20ParaView-orange)

A compact **2D wave equation** solver using the **finite element method** with **deal.II**.
Built for teaching/experimentation (PoliMi *Numerical Methods for PDEs*), but structured like a small reproducible research code.

- Spatial discretization with deal.II FE spaces (MPI-parallel).
- Time integration via **theta $\theta$ schemes** (plus alternative schemes, see parameters).
- VTK output (`.vtu/.pvtu`) for ParaView.
- Manufactured Solution (MMS) mode with optional **error history CSV** export.
- End-to-end tooling: **Tkinter** parameter GUI + **Streamlit** error comparison report.

## Table of Contents
- [Overview](#overview)
- [Problem Description](#problem-description)
- [End-to-end workflow](#end-to-end-workflow)
- [Quick start (build & run)](#quick-start-build--run)
  - [60-second quickstart](#60-second-quickstart)
  - [Prerequisites](#prerequisites)
  - [Build](#build)
  - [Run](#run)
- [Create a `.prm` file with the GUI](#create-a-prm-file-with-the-gui)
- [Inputs: parameter file & meshes](#inputs-parameter-file--meshes)
- [Parameter reference](#parameter-reference)
  - [Problem](#problem)
  - [Boundary condition](#boundary-condition)
  - [Mesh](#mesh)
  - [Time](#time)
  - [Output](#output)
- [Outputs](#outputs)
- [Visualize / compare MMS errors (Streamlit)](#visualize--compare-mms-errors-streamlit)
- [Source layout](#source-layout)
- [Troubleshooting](#troubleshooting)

## Overview

This project solves the second-order wave equation in 2D, with Dirichlet boundary conditions and user-specified initial data.
The implementation focuses on clarity and reproducibility for teaching and experimentation with:
- Spatial discretization via deal.II finite element spaces.
- Time integration based on theta-schemes (including explicit/implicit variants).
- Exporting results in VTK format for visualization (VTU / PVTU).
- Manufactured Solution (MMS) runs that can export error histories to CSV.

---

## Problem Description

Consider the wave equation in 2D [1, 2]:

<p align="center">
    <img src="_static/problem_equation.png" alt="Wave Equation" width="200"/>
</p>

Implement a finite element solver for problem above. Discuss the choice of the time and space discretization methods,
the properties of the chosen method (especially in terms of numerical dissipation and dispersion, see [1, 2])
and the computational and algorithmic aspects of the solver.

[1]: A. Quarteroni. Numerical models for differential problems, volume 2. Springer, 2017.

[2]: S. Salsa and G. Verzini. Partial differential equations in action: from modelling to theory, volume 147. Springer Nature, 2022.

---

## End-to-end workflow

### 1) Create a `.prm` file (Tkinter GUI, no dependencies)

The repository ships a tiny GUI to generate and edit deal.II-style parameter files:
- Script: [`tools/prm_gui.py`](./tools/prm_gui.py)
- Dependencies: **none** beyond Python's standard library (Tkinter)

You'll produce a `.prm` file with standard blocks like:
- `subsection Problem ... end`
- `subsection Mesh ... end`
- `subsection Output ... end`

The GUI prevents typos and keeps values consistent with the solver's expected ranges/options.

https://github.com/user-attachments/assets/1694f8b2-eb7c-4430-a62c-5489dca0a6cf

### 2) Run the solver + choose where to save errors

The solver executable is built as `nm4pde` (see `CMakeLists.txt`).

Run it with an explicit parameter file path (recommended):
- physical run: [`parameters/wave.prm`](./parameters/wave.prm)
- MMS run (writes error history if enabled): [`parameters/mms.prm`](./parameters/mms.prm)

For MMS, set:
- `Output.compute_error = true`
- `Output.error_file = <path/to.csv>`

VTK output is controlled by:
- `Output.vtk_directory = <output_dir>`

> [!NOTE]
> The code includes interactive prompts for output paths when run in a TTY.
> In batch/non-interactive environments (e.g., redirected stdin), prompts are skipped.
> For reproducible runs, set `error_file` and `vtk_directory` explicitly in the `.prm`.

### 3) Visualize/compare error histories (Streamlit)

The repo includes a Streamlit app to compare one or more MMS error CSVs:
- App: [`tools/plot/mms_viewer.py`](./tools/plot/mms_viewer.py)
- Container build context: [`tools/plot/`](./tools/plot)

Easily run it via Docker (preferred):

```bash
docker build -t nm4pde-mms-viewer tools/plot
docker run --rm -p 8501:8501 nm4pde-mms-viewer
```

Or directly with Python (install dependencies from `tools/plot/requirements.txt`):

```bash
# Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r tools/plot/requirements.txt

# Run the Streamlit app
streamlit run tools/plot/mms_viewer.py
```

Then open `http://localhost:8501` in your browser and upload one or more CSV files generated by the solver.

Here's a video demo of the streamlit viewer in action:

https://github.com/user-attachments/assets/e847a78d-4053-47a8-b00b-c880151faad9

---

## Quick start (build & run)

### 60-second quickstart

This is the shortest end-to-end path (GUI $\to$ MMS run $\to$ Streamlit):

1. Create a parameter file (or start from the example):
  - easiest: run the GUI and save a new `.prm`
  - or use the example: `parameters/mms.prm`
2. Build and run an MMS case (produces VTK + a CSV error history if enabled).
3. Start the Streamlit viewer and upload the generated CSV.

Example commands (from repo root):
```bash
cmake --preset nm4pde-lab
cmake --build build/nm4pde-lab

# (optional) generate/edit a prm file with the GUI
python3 tools/prm_gui.py

# MMS run
mpirun -np 4 ./build/nm4pde-lab/nm4pde parameters/mms.prm

# viewer (Docker)
docker build -t nm4pde-mms-viewer tools/plot
docker run --rm -p 8501:8501 nm4pde-mms-viewer
```

---

### Prerequisites

**Required**
- A working **deal.II** installation.
- An MPI implementation (e.g. OpenMPI/MPICH): provides `mpirun` + an MPI C++ compiler wrapper.
- CMake + a C++17 compiler.
- Ninja (recommended, but not required).

**Optional (nice to have)**
- [ParaView](https://www.paraview.org/) (to open `.vtu/.pvtu`).
- [`gmsh`](https://gmsh.info/) (only needed if you want to mesh `.geo` files at runtime).
- [Docker](https://www.docker.com/) (for the Streamlit viewer container).

> [!NOTE]
> This repository contains optional "mk" helpers and a small set of shared CMake fragments that are used by
> the Politecnico di Milano course infrastructure.
> These are intended for students/staff who use the Politecnico `mk` tool and the course's common CMake files
> (mk: https://github.com/pcafrica/mk and the course common CMake fragment
> at https://github.com/michelebucelli/nmpde-labs-aa-25-26/blob/main/common/cmake-common.cmake).
> If you are a standard user or just want to build with a normal CMake workflow,
> you do not need `mk` nor the course-specific common CMake files; the instructions above are sufficient.
>
> If the project fails to configure because it tries to include those course files,
> open `CMakeLists.txt` and comment out or remove the lines that include or reference the course fragments
> (look for lines that call `include(...)`, `find_package(...)`, or `add_subdirectory(...)`
> pointing at the course/common cmake files).
> After removing those references the project will configure and build with a regular CMake setup.


### Build

This repo provides CMake presets (`CMakePresets.json`). If your environment matches, use presets.

Example (local toolchain preset):
```bash
cmake --preset nm4pde-lab
cmake --build build/nm4pde-lab
```

The build outputs the executable in:
- `build/<presetName>/nm4pde`

### Run

Physical run (default example parameter file):
```bash
mpirun -np 4 ./build/nm4pde-lab/nm4pde parameters/wave.prm
```

MMS run (exports VTK + error CSV if enabled):
```bash
mpirun -np 4 ./build/nm4pde-lab/nm4pde parameters/mms.prm
```

---

## Create a `.prm` file with the GUI

**Strongly recommended:** use the Tkinter GUI instead of writing parameter files by hand.

It's the most reliable way to produce an input file that the solver will accept:
- prevents typos in subsection / key names
- keeps values consistent with the solver's expected ranges/options
- faster iteration when exploring many MMS cases

Run:
```bash
python3 tools/prm_gui.py
```

Typical flow:
1. Pick `Problem.type` (`physical`, `mms`, or `expr`).
2. Choose a mesh (`Mesh.mesh_file`) and polynomial degree (`Mesh.degree`).
3. Set time stepping (`Time.T`, `Time.dt`, `Time.theta`, `Time.scheme`).
4. Set output:
  - VTK output folder: `Output.vtk_directory`
  - (MMS only) enable CSV export: `Output.compute_error = true` and choose `Output.error_file`
5. Save the file and run the solver with it.

> [!NOTE]
> Under the hood we use deal.II's `ParameterHandler`. While this project ships examples as classic text **`.prm`**,
> deal.II can also read/write parameters in formats like **XML** and **JSON**.
> If you prefer those for tooling (or want to auto-generate configs), see the official documentation:
> https://www.dealii.org/current/doxygen/deal.II/classParameterHandler.html

---

## Inputs: parameter file & meshes

- `parameters/wave.prm`: physical run defaults.
- `parameters/mms.prm`: forced MMS example (includes error CSV output settings).
- `mesh/`: Gmsh geometry and mesh files.

The solver accepts `.geo` and `.msh` inputs:
- `.msh`: loaded directly
- `.geo`: meshed at runtime (requires `gmsh` available)

---

## Parameter reference

Parameters are parsed via deal.II `ParameterHandler` (see `include/parameters.hpp`).
They are grouped into subsections.

### Problem
- `type` (selection): `physical | mms | expr`
- `u_exact_expr`, `v_exact_expr`, `f_exact_expr` (strings, MMS)
- `u0_expr`, `v0_expr`, `f_expr` (strings, expression-based)
- `mu_expr` (string): coefficient (default `1`)

### Boundary condition
- `type` (selection): `zero | mms | expr`
- `g_expr` (string): Dirichlet value for displacement (used if `expr`)
- `v_expr` (string): Dirichlet value for velocity (used if `expr`)

### Mesh
- `mesh_file` (string): path to `.geo` or `.msh`
- `degree` (int >= 1): polynomial degree

### Time
- `T` (double): final time
- `dt` (double): time step
- `theta` (double in [0,1]): theta parameter (used by theta scheme)
- `scheme` (selection): `theta | central | newmark`

### Output
- `every` (int >= 1): write VTK every N steps
- `compute_error` (bool): if `true`, export error CSV (only meaningful for MMS)
- `error_file` (string): CSV output path (used if `compute_error=true`)
- `vtk_directory` (string): directory for `.vtu/.pvtu` output files

---

## Outputs

### VTK output (ParaView)
The solver writes:
- `output_XXX.0.vtu`
- `output_XXX.pvtu`

The location is controlled by `Output.vtk_directory`.

### MMS error history (CSV)
If running an MMS problem (`Problem.type = mms`) and `Output.compute_error = true`, rank 0 writes an extended error history CSV.

The CSV includes columns like:
- `step,time,error_u,error_v,...`

The output path is controlled by `Output.error_file`.

---

## Visualize / compare MMS errors (Streamlit)

A Streamlit viewer is provided in `tools/plot`.

### Purpose
The Streamlit app visualizes and compares error histories from MMS runs.
Upload one or more CSV files (generated by the solver with `Output.compute_error = true`).

### Expected CSV columns
The CSV files should contain:
- `step`: time step number
- `time`: simulation time
- `error_u`, `error_v`, ...: error metrics for comparison

### Option A: Docker (recommended)
From the repository root:
```bash
docker build -t nm4pde-mms-viewer tools/plot
docker run --rm -p 8501:8501 nm4pde-mms-viewer
```

Then open `http://localhost:8501` and upload one or more CSV files (from `Output.error_file`).

### Option B: Run locally (Python)
`tools/plot/requirements.txt` lists the dependencies.
Install them in your Python environment, then run the Streamlit app:
```bash
pip install -r tools/plot/requirements.txt
streamlit run tools/plot/mms_viewer.py
```

---

## Source layout

Top-level:
- `CMakeLists.txt`: build instructions
- `CMakePresets.json`: preset configurations
- `mesh/`: meshes and Gmsh scripts
- `parameters/`: example parameter files
- `src/`: C++ source code
- `tools/`: PRM GUI and Streamlit plotting

Key files:
- `src/main.cpp`: entry point
- `src/Wave.cpp`, `include/Wave.hpp`: main PDE solver class
- `include/parameters.hpp`: parameter parsing and defaults

---

## Troubleshooting

- **`.geo` mesh fails**: install `gmsh` or use a pre-generated `.msh`.
- **No error CSV produced**:
  - ensure `Problem.type = mms`
  - set `Output.compute_error = true`
  - set a valid `Output.error_file` path
- **Output paths are "weird" when running from build/**:
  - paths in `.prm` are interpreted relative to the current working directory.
  - prefer running from repo root or use absolute paths.

[#1]: https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=837285&__pj0=0&__pj1=5147aa88ed0802458f409c0048df93c8
[#2]: https://www.paraview.org/

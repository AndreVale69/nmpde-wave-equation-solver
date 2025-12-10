# Project 2: Wave Equation Solver using deal.II

## Table of Contents

- [Overview](#overview)
- [Problem Description](#problem-description)
- [Professor Information](#professor-information)

---

## Overview

This project implements a finite element solver for the 2D wave equation using the deal.II library.
The solver addresses the wave equation with specified boundary and initial conditions,
allowing for the simulation of wave propagation in a defined domain.

---

## Problem Description

Consider the wave equation in 2D [1, 2]:

$$
\begin{cases}
    \dfrac{\partial^2 u}{\partial t^2} - \Delta u = f & \text{in } \Omega, \\[.3em]
    u = g & \text{on } \partial \Omega, \\[.3em]
    u(t = 0) = u_0 & \text{in } \Omega, \\[.3em]
    \dfrac{\partial u}{\partial t} = u_1 & \text{in } \Omega.
\end{cases}
$$

Implement a finite element solver for problem above. Discuss the choice of the time and space discretization methods,
the properties of the chosen method (especially in terms of numerical dissipation and dispersion, see [1, 2])
and the computational and algorithmic aspects of the solver. 

\[1\]: A. Quarteroni. Numerical models for differential problems, volume 2. Springer, 2017.

\[2\]: S. Salsa and G. Verzini. Partial differential equations in action: from modelling to theory, volume 147. Springer Nature, 2022.

---

## Professor Information

### Organizing the source code
Please place all your sources into the `src` folder.

Binary files must not be uploaded to the repository (including executables).

Mesh files should not be uploaded to the repository. If applicable, upload `gmsh` scripts with suitable instructions to generate the meshes (and ideally a Makefile that runs those instructions). If not applicable, consider uploading the meshes to a different file sharing service, and providing a download link as part of the building and running instructions.

### Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The executable will be created into `build`, and can be executed through
```bash
$ ./executable-name
```
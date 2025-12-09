# Mesh Files

This directory contains mesh files for the wave equation solver.

## Generating Meshes

### Using Gmsh

Generate meshes with different refinement levels:

```bash
# Coarse mesh (lc = 0.2)
gmsh -3 ../scripts/mesh-cube.geo -setnumber lc 0.2 -o mesh-cube-5.msh

# Medium mesh (lc = 0.1)
gmsh -3 ../scripts/mesh-cube.geo -setnumber lc 0.1 -o mesh-cube-10.msh

# Fine mesh (lc = 0.05)
gmsh -3 ../scripts/mesh-cube.geo -setnumber lc 0.05 -o mesh-cube-20.msh

# Very fine mesh (lc = 0.025)
gmsh -3 ../scripts/mesh-cube.geo -setnumber lc 0.025 -o mesh-cube-40.msh
```

### Using Built-in Generator

Alternatively, the solver can generate cube meshes internally by setting `mesh_file_name = ""` in the driver programs.

## File Naming Convention

- `mesh-cube-N.msh`: Cube mesh with approximately N elements per direction
- `generated_mesh.vtk`: Automatically generated mesh from solver (for visualization)

## Notes

- All meshes use tetrahedral (simplex) elements
- Boundary faces are tagged 0-5 (bottom, top, front, right, back, left)
- Generated meshes are written to this directory by the solver

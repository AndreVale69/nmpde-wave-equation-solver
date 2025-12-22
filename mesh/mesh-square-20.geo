// Gmsh script for 2D square mesh with triangular elements
// Usage: gmsh -2 mesh-square-20.geo

// Mesh size parameter
h = 1.0 / 20; // 20 elements per side

// Define the square domain [0,1] x [0,1]
Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};

// Define lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Define surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Physical groups for boundary conditions
Physical Line(0) = {4}; // left (x=0)
Physical Line(1) = {2}; // right (x=1)
Physical Line(2) = {1}; // bottom (y=0)
Physical Line(3) = {3}; // top (y=1)
Physical Surface(0) = {1}; // domain

// Use triangular elements
Mesh.ElementOrder = 1;
Mesh.Algorithm = 6; // Frontal-Delaunay for 2D

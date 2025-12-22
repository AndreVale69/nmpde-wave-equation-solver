// Gmsh mesh script for unit cube with tetrahedral elements
// Usage: gmsh -3 mesh-cube.geo -o mesh-cube-N.msh
// where N is the characteristic length parameter

// Mesh size parameter (can be overridden from command line)
If (!Exists(lc))
  lc = 0.1;
EndIf

// Define cube vertices
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};
Point(5) = {0, 0, 1, lc};
Point(6) = {1, 0, 1, lc};
Point(7) = {1, 1, 1, lc};
Point(8) = {0, 1, 1, lc};

// Define edges
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Define faces
Curve Loop(1) = {1, 2, 3, 4};    // Bottom (z=0)
Plane Surface(1) = {1};

Curve Loop(2) = {5, 6, 7, 8};    // Top (z=1)
Plane Surface(2) = {2};

Curve Loop(3) = {1, 10, -5, -9}; // Front (y=0)
Plane Surface(3) = {3};

Curve Loop(4) = {2, 11, -6, -10}; // Right (x=1)
Plane Surface(4) = {4};

Curve Loop(5) = {3, 12, -7, -11}; // Back (y=1)
Plane Surface(5) = {5};

Curve Loop(6) = {4, 9, -8, -12};  // Left (x=0)
Plane Surface(6) = {6};

// Define volume
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// Physical tags for boundaries
Physical Surface("bottom", 0) = {1};
Physical Surface("top", 1) = {2};
Physical Surface("front", 2) = {3};
Physical Surface("right", 3) = {4};
Physical Surface("back", 4) = {5};
Physical Surface("left", 5) = {6};

// Physical tag for volume
Physical Volume("domain", 0) = {1};

// Mesh algorithm: Delaunay for tetrahedra
Mesh.Algorithm3D = 1;

// Mesh recombination (keep tetrahedral)
Mesh.RecombineAll = 0;

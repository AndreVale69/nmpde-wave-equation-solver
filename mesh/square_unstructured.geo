SetFactory("OpenCASCADE");

L = 1.0;
h = 0.02;   // target element size (0.02 -> ~2500 triangles-ish)

Point(1) = {0, 0, 0, h};
Point(2) = {L, 0, 0, h};
Point(3) = {L, L, 0, h};
Point(4) = {0, L, 0, h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

Physical Surface("domain") = {1};
Physical Curve("boundary") = {1,2,3,4};

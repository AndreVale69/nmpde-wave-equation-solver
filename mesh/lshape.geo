SetFactory("OpenCASCADE");

h = 0.02;

Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};
Point(5) = {0.5, 0.5, 0, h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,1};

Curve Loop(1) = {1,2,3,4,5};
Plane Surface(1) = {1};

Physical Surface("domain") = {1};
Physical Curve("boundary") = {1,2,3,4,5};

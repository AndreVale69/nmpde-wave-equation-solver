SetFactory("OpenCASCADE");

L = 1.0;
Nx = 50;  // increase to 100 for even finer

Point(1) = {0, 0, 0, 1.0};
Point(2) = {L, 0, 0, 1.0};
Point(3) = {L, L, 0, 1.0};
Point(4) = {0, L, 0, 1.0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

// Transfinite (structured) discretization
Transfinite Curve {1,2,3,4} = Nx+1 Using Progression 1;
Transfinite Surface {1};

Physical Surface("domain") = {1};
Physical Curve("boundary") = {1,2,3,4};

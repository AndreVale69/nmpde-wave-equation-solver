#include "Wave.hpp"

// Main function.
int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const std::string  mesh_file_name = "../mesh/mesh-square.msh";
    const unsigned int degree         = 1;

    const double T      = 1.0;
    const double deltat = 0.01;
    const double theta  = 1;

    Wave problem(mesh_file_name, degree, T, deltat, theta, TimeScheme::Theta);

    problem.setup();
    problem.solve();

    return 0;
}

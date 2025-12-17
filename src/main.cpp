#include "Wave.hpp"

// Main function.
int main(int argc, char *argv[]) {
    // read wave_prm from command line if provided
    std::string wave_prm = "../wave.prm";
    if (argc > 1) {
        wave_prm = argv[1];
    } else {
        std::cout << "No parameter file provided. Run with mpirun -np <nprocs> ./wave "
                     "<parameter_file>"
                  << std::endl;
        std::cout << "Using default parameter file: " << wave_prm << std::endl;
    }

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    Wave problem(wave_prm);

    problem.setup();
    problem.solve();

    return 0;
}

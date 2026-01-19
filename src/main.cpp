#include "Wave.hpp"

// Main function.
int main(int argc, char *argv[]) {
    // read wave_prm from command line if provided
    std::string wave_prm;

    if (argc > 1) {
        wave_prm = argv[1];
    } else {
        /**
         * Project root directory.
         * 1. Get canonical path of argv[0] (executable);
         *    argv[0] is /.../root/build/nm4pde-lab/nm4pde
         * 2. First parent: /.../root/build/nm4pde-lab
         * 3. Second parent: /.../root/build
         * 4. Third parent: /.../root
         */
        const std::filesystem::path root =
                std::filesystem::canonical(argv[0]).parent_path().parent_path().parent_path();
        wave_prm = (root / "parameters" / "wave.prm").string();
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

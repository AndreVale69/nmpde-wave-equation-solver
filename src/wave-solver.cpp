#include "Wave.hpp"

/**
 * @brief Basic wave equation solver driver.
 * 
 * Solves the wave equation with manufactured solution for verification.
 */
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  try
    {
      // Problem parameters
      const std::string  mesh_file_name = "";  // Empty = use built-in cube
      const unsigned int N_subdivisions = 10;   // 10 subdivisions per direction
      const unsigned int degree         = 2;    // Polynomial degree
      const double       T              = 1.0;  // Final time
      const double       deltat         = 0.005; // Time step
      const double       beta           = 0.25; // Newmark beta (average acceleration)
      const double       gamma          = 0.5;  // Newmark gamma
      const bool         use_manufactured = true;
      const unsigned int output_frequency = 10; // Output every 10 steps

      Wave problem(mesh_file_name,
                   N_subdivisions,
                   degree,
                   T,
                   deltat,
                   beta,
                   gamma,
                   use_manufactured,
                   output_frequency);

      problem.setup();
      problem.solve();

      // Print final errors
      if (mpi_rank == 0 && use_manufactured)
        {
          const double eL2_u = problem.compute_error(VectorTools::L2_norm, false);
          const double eH1_u = problem.compute_error(VectorTools::H1_norm, false);
          const double eL2_v = problem.compute_error(VectorTools::L2_norm, true);

          std::cout << "===============================================" << std::endl;
          std::cout << "Final errors:" << std::endl;
          std::cout << "  L2 error (displacement): " << std::scientific << eL2_u
                    << std::endl;
          std::cout << "  H1 error (displacement): " << std::scientific << eH1_u
                    << std::endl;
          std::cout << "  L2 error (velocity):     " << std::scientific << eL2_v
                    << std::endl;
          std::cout << "===============================================" << std::endl;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}

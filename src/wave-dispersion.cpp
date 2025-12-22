#include "Wave.hpp"

/**
 * @brief Dispersion and dissipation analysis driver.
 * 
 * Analyzes numerical dispersion and dissipation using Gaussian wave packet.
 */
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  try
    {
      // Problem parameters for dispersion/dissipation study
      const std::string  mesh_file_name = "";  // Use built-in cube
      const unsigned int N_subdivisions = 20;   // Finer mesh for wave propagation
      const unsigned int degree         = 2;    // Polynomial degree
      const double       T              = 5.0;  // Longer simulation time
      const double       deltat         = 0.01; // Time step
      const double       beta           = 0.25; // Average acceleration
      const double       gamma          = 0.5;
      const bool         use_manufactured = false; // Use Gaussian wave packet
      const unsigned int output_frequency = 20;    // Output every 20 steps

        {
          std::cout << "\n"
                    << "======================================================\n"
                    << "DISPERSION AND DISSIPATION ANALYSIS\n"
                    << "======================================================\n"
                    << std::endl;
          std::cout << "Simulating Gaussian wave packet propagation" << std::endl;
          std::cout << "  Duration: " << T << " s" << std::endl;
          std::cout << "  Time step: " << deltat << std::endl;
          std::cout << "  Mesh subdivisions: " << N_subdivisions << std::endl;
          std::cout << "  Newmark parameters: beta = " << beta << ", gamma = " << gamma
                    << std::endl;
          std::cout << std::endl;
        }

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

        {
          std::cout << "\n"
                    << "======================================================\n"
                    << "Analysis complete!\n"
                    << "======================================================\n"
                    << std::endl;
          std::cout << "\nResults saved to:" << std::endl;
          std::cout << "  - ../results/time_series.csv (energy and error vs time)"
                    << std::endl;
          std::cout << "  - output-*.vtu (displacement, velocity fields)" << std::endl;
          std::cout << "\nUse ../scripts/plot_dispersion.py to visualize results"
                    << std::endl;
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

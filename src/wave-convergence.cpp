#include "Wave.hpp"

#include <fstream>
#include <iomanip>

/**
 * @brief Convergence study driver for wave equation solver.
 * 
 * Performs spatial and temporal convergence studies with manufactured solution.
 */
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  try
    {
      const double T     = 1.0;  // Final time
      const double beta  = 0.25; // Newmark beta
      const double gamma = 0.5;  // Newmark gamma

      // ========================================================================
      // Temporal convergence study (fixed spatial resolution)
      // ========================================================================
      if (mpi_rank == 0)
        {
          std::cout << "\n"
                    << "======================================================\n"
                    << "TEMPORAL CONVERGENCE STUDY\n"
                    << "======================================================\n"
                    << std::endl;
        }

      const unsigned int degree_temporal = 2;
      const unsigned int N_spatial       = 10; // Fixed coarse mesh

      const std::vector<double> deltat_vector = {0.04, 0.02, 0.01, 0.005, 0.0025};

      std::vector<double> errors_L2_u_temporal;
      std::vector<double> errors_H1_u_temporal;
      std::vector<double> errors_L2_v_temporal;

      for (const auto &deltat : deltat_vector)
        {
          Wave problem("", N_spatial, degree_temporal, T, deltat, beta, gamma, true, 0);

          problem.setup();
          problem.solve();

          errors_L2_u_temporal.push_back(
            problem.compute_error(VectorTools::L2_norm, false));
          errors_H1_u_temporal.push_back(
            problem.compute_error(VectorTools::H1_norm, false));
          errors_L2_v_temporal.push_back(
            problem.compute_error(VectorTools::L2_norm, true));
        }

      // Print temporal convergence results
      if (mpi_rank == 0)
        {
          std::ofstream convergence_file("../results/convergence_temporal.csv");
          convergence_file << "dt,eL2_u,order_L2_u,eH1_u,order_H1_u,eL2_v,order_L2_v"
                           << std::endl;

          std::cout << std::string(80, '=') << std::endl;
          std::cout << "Temporal Convergence (degree = " << degree_temporal
                    << ", N = " << N_spatial << ")" << std::endl;
          std::cout << std::string(80, '-') << std::endl;

          for (unsigned int i = 0; i < deltat_vector.size(); ++i)
            {
              std::cout << std::scientific << "dt = " << std::setw(8)
                        << std::setprecision(4) << deltat_vector[i];

              std::cout << " | eL2(u) = " << std::setw(10) << std::setprecision(4)
                        << errors_L2_u_temporal[i];

              double order_L2 = 0.0;
              if (i > 0)
                {
                  order_L2 = std::log(errors_L2_u_temporal[i] /
                                      errors_L2_u_temporal[i - 1]) /
                             std::log(deltat_vector[i] / deltat_vector[i - 1]);
                  std::cout << " (p=" << std::fixed << std::setprecision(2)
                            << std::setw(4) << order_L2 << ")";
                }
              else
                {
                  std::cout << " (p= -- )";
                }

              std::cout << std::scientific << " | eH1(u) = " << std::setw(10)
                        << std::setprecision(4) << errors_H1_u_temporal[i];

              double order_H1 = 0.0;
              if (i > 0)
                {
                  order_H1 = std::log(errors_H1_u_temporal[i] /
                                      errors_H1_u_temporal[i - 1]) /
                             std::log(deltat_vector[i] / deltat_vector[i - 1]);
                  std::cout << " (p=" << std::fixed << std::setprecision(2)
                            << std::setw(4) << order_H1 << ")";
                }
              else
                {
                  std::cout << " (p= -- )";
                }

              std::cout << std::endl;

              convergence_file << deltat_vector[i] << "," << errors_L2_u_temporal[i]
                               << "," << order_L2 << "," << errors_H1_u_temporal[i]
                               << "," << order_H1 << "," << errors_L2_v_temporal[i]
                               << "," << 0.0 << std::endl;
            }

          std::cout << std::string(80, '=') << std::endl;
          convergence_file.close();
        }

      // ========================================================================
      // Spatial convergence study (fixed time step)
      // ========================================================================
      if (mpi_rank == 0)
        {
          std::cout << "\n"
                    << "======================================================\n"
                    << "SPATIAL CONVERGENCE STUDY\n"
                    << "======================================================\n"
                    << std::endl;
        }

      const double       deltat_spatial = 0.001; // Small timestep for spatial study
      const unsigned int degree_spatial = 2;

      const std::vector<unsigned int> N_vector = {5, 10, 15, 20};

      std::vector<double> h_vector;
      std::vector<double> errors_L2_u_spatial;
      std::vector<double> errors_H1_u_spatial;

      for (const auto &N : N_vector)
        {
          Wave problem("", N, degree_spatial, T, deltat_spatial, beta, gamma, true, 0);

          problem.setup();
          problem.solve();

          h_vector.push_back(1.0 / static_cast<double>(N));
          errors_L2_u_spatial.push_back(
            problem.compute_error(VectorTools::L2_norm, false));
          errors_H1_u_spatial.push_back(
            problem.compute_error(VectorTools::H1_norm, false));
        }

      // Print spatial convergence results
      if (mpi_rank == 0)
        {
          std::ofstream convergence_file("../results/convergence_spatial.csv");
          convergence_file << "h,N,eL2_u,order_L2_u,eH1_u,order_H1_u" << std::endl;

          std::cout << std::string(80, '=') << std::endl;
          std::cout << "Spatial Convergence (degree = " << degree_spatial
                    << ", dt = " << deltat_spatial << ")" << std::endl;
          std::cout << std::string(80, '-') << std::endl;

          for (unsigned int i = 0; i < N_vector.size(); ++i)
            {
              std::cout << "N = " << std::setw(3) << N_vector[i] << " (h = "
                        << std::scientific << std::setprecision(4) << h_vector[i] << ")";

              std::cout << " | eL2(u) = " << std::setw(10) << std::setprecision(4)
                        << errors_L2_u_spatial[i];

              double order_L2 = 0.0;
              if (i > 0)
                {
                  order_L2 =
                    std::log(errors_L2_u_spatial[i] / errors_L2_u_spatial[i - 1]) /
                    std::log(h_vector[i] / h_vector[i - 1]);
                  std::cout << " (p=" << std::fixed << std::setprecision(2)
                            << std::setw(4) << order_L2 << ")";
                }
              else
                {
                  std::cout << " (p= -- )";
                }

              std::cout << std::scientific << " | eH1(u) = " << std::setw(10)
                        << std::setprecision(4) << errors_H1_u_spatial[i];

              double order_H1 = 0.0;
              if (i > 0)
                {
                  order_H1 =
                    std::log(errors_H1_u_spatial[i] / errors_H1_u_spatial[i - 1]) /
                    std::log(h_vector[i] / h_vector[i - 1]);
                  std::cout << " (p=" << std::fixed << std::setprecision(2)
                            << std::setw(4) << order_H1 << ")";
                }
              else
                {
                  std::cout << " (p= -- )";
                }

              std::cout << std::endl;

              convergence_file << h_vector[i] << "," << N_vector[i] << ","
                               << errors_L2_u_spatial[i] << "," << order_L2 << ","
                               << errors_H1_u_spatial[i] << "," << order_H1 << std::endl;
            }

          std::cout << std::string(80, '=') << std::endl;
          convergence_file.close();

          std::cout << "\nConvergence data written to:" << std::endl;
          std::cout << "  - ../results/convergence_temporal.csv" << std::endl;
          std::cout << "  - ../results/convergence_spatial.csv" << std::endl;
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

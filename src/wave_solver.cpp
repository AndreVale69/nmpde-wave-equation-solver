#include "Wave.hpp"
#include <iomanip>
#include <cmath>

/**
 * @brief Driver per l'equazione delle onde con analisi di convergenza.
 * * Esegue simulazioni multiple con diversi passi temporali (dt) per verificare
 * l'ordine di convergenza dello schema alle differenze centrate.
 */
int main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
      const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

      // --- Parametri del Problema ---
      const std::string  mesh_file_name   = "../mesh/mesh-square-20.msh";
      const unsigned int degree           = 1;     // Grado polinomiale FE
      const double       T                = 0.5;   // Tempo finale
      const bool         use_manufactured = true;
      const unsigned int output_frequency = 0;     // 0 = output solo finale (per velocità)

      // Vettore dei passi temporali per lo studio di convergenza
      const std::vector<double> deltat_vector = {0.05, 0.025, 0.0125, 0.00625};
      
      std::vector<double> errors_L2_u;
      std::vector<double> errors_H1_u;
      std::vector<double> errors_L2_v;

      if (mpi_rank == 0)
        {
          std::cout << "==========================================================" << std::endl;
          std::cout << " Solving Wave Equation (Centered Differences)" << std::endl;
          std::cout << " Mesh: " << mesh_file_name << " | Degree: " << degree << std::endl;
          std::cout << "==========================================================" << std::endl;
        }

      // --- Ciclo di Convergenza ---
      for (const auto &dt : deltat_vector)
        {
          if (mpi_rank == 0)
            std::cout << "\nRunning with dt = " << dt << "..." << std::endl;

          WaveEquation problem(mesh_file_name,
                               degree,
                               T,
                               dt,
                               use_manufactured,
                               output_frequency);

          problem.setup();
          problem.solve();

          // Calcolo degli errori (u in L2 e H1, v in L2)
          errors_L2_u.push_back(problem.compute_error(VectorTools::L2_norm, false));
          errors_H1_u.push_back(problem.compute_error(VectorTools::H1_norm, false));
          errors_L2_v.push_back(problem.compute_error(VectorTools::L2_norm, true));
        }

      // --- Analisi dei Risultati (solo su Rank 0) ---
      if (mpi_rank == 0)
        {
          std::cout << "\n" << std::string(85, '=') << std::endl;
          std::cout << std::setw(10) << "dt" 
                    << std::setw(18) << "L2 Error (u)" << std::setw(10) << "Order"
                    << std::setw(18) << "H1 Error (u)" << std::setw(10) << "Order"
                    << std::setw(18) << "L2 Error (v)" << std::endl;
          std::cout << std::string(85, '-') << std::endl;

          for (unsigned int i = 0; i < deltat_vector.size(); ++i)
            {
              std::cout << std::scientific << std::setprecision(4)
                        << std::setw(10) << deltat_vector[i]
                        << std::setw(18) << errors_L2_u[i];

              // Calcolo ordine L2 per spostamento
              if (i > 0)
                {
                  double p = std::log(errors_L2_u[i] / errors_L2_u[i - 1]) / 
                             std::log(deltat_vector[i] / deltat_vector[i - 1]);
                  std::cout << std::fixed << std::setprecision(2) << std::setw(10) << p;
                }
              else
                std::cout << std::setw(10) << "---";

              std::cout << std::scientific << std::setprecision(4)
                        << std::setw(18) << errors_H1_u[i];

              // Calcolo ordine H1 per spostamento
              if (i > 0)
                {
                  double p = std::log(errors_H1_u[i] / errors_H1_u[i - 1]) / 
                             std::log(deltat_vector[i] / deltat_vector[i - 1]);
                  std::cout << std::fixed << std::setprecision(2) << std::setw(10) << p;
                }
              else
                std::cout << std::setw(10) << "---";

              std::cout << std::scientific << std::setprecision(4)
                        << std::setw(18) << errors_L2_v[i] << std::endl;
            }
          std::cout << std::string(85, '=') << std::endl;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}
#include "Wave.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>

/**
 * @brief Driver per lo studio di convergenza (Spaziale e Temporale).
 * Verifica che l'errore decresca correttamente rispetto a dt e h.
 */
int main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
      const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

      const double T = 1.0; // Tempo finale fisso per tutti i test

      // ========================================================================
      // 1. STUDIO DI CONVERGENZA TEMPORALE (Mesh fissa, dt variabile)
      // ========================================================================
      if (mpi_rank == 0)
        {
          std::cout << "\n======================================================" << std::endl;
          std::cout << " TEMPORAL CONVERGENCE STUDY" << std::endl;
          std::cout << "======================================================" << std::endl;
        }

      const unsigned int degree_temporal = 2;
      const unsigned int N_spatial        = 20; // Mesh abbastanza fine da non dominare l'errore
      const std::vector<double> deltat_vector = {0.005, 0.0025, 0.00125, 0.000625};

      std::vector<double> errors_L2_u_temporal;
      std::vector<double> errors_H1_u_temporal;

      for (const auto &dt : deltat_vector)
        {
          // Avviamo il problema (manufactured_solution = true, output_freq = 0 per velocità)
          WaveEquation problem("../mesh/mesh-square-20.msh", degree_temporal, T, dt, true, 0);
          
          // Nota: se la classe usa N_subdivisions internamente per la mesh built-in:
          // problem.set_subdivisions(N_spatial); 
          
          problem.setup();
          problem.solve();

          errors_L2_u_temporal.push_back(problem.compute_error(VectorTools::L2_norm, false));
          errors_H1_u_temporal.push_back(problem.compute_error(VectorTools::H1_norm, false));
        }

      // Stampa Risultati Temporali (solo Rank 0)
      if (mpi_rank == 0)
        {
          std::cout << std::left << std::setw(10) << "dt" 
                    << std::setw(15) << "L2_Error" << std::setw(10) << "Order" << std::endl;
          for (unsigned int i = 0; i < deltat_vector.size(); ++i)
            {
              std::cout << std::scientific << std::setprecision(4) << std::setw(10) << deltat_vector[i]
                        << std::setw(15) << errors_L2_u_temporal[i];
              if (i > 0)
                {
                  double p = std::log(errors_L2_u_temporal[i] / errors_L2_u_temporal[i-1]) /
                             std::log(deltat_vector[i] / deltat_vector[i-1]);
                  std::cout << std::fixed << std::setprecision(2) << std::setw(10) << p;
                }
              std::cout << std::endl;
            }
        }

      // ========================================================================
      // 2. STUDIO DI CONVERGENZA SPAZIALE (dt piccolo, Mesh variabile)
      // ========================================================================
      if (mpi_rank == 0)
        {
          std::cout << "\n======================================================" << std::endl;
          std::cout << " SPATIAL CONVERGENCE STUDY" << std::endl;
          std::cout << "======================================================" << std::endl;
        }

      const double dt_spatial = 0.0005; // dt molto piccolo per "isolare" l'errore spaziale
      const unsigned int degree_spatial = 2;
      const std::vector<unsigned int> N_vector = {8, 16, 32}; // Suddivisioni mesh

      std::vector<double> errors_L2_u_spatial;
      std::vector<double> h_values;

      for (const auto &N : N_vector)
        {
          // Qui carichi mesh diverse o passi N al costruttore
          WaveEquation problem("../mesh/mesh-square-20.msh", degree_spatial, T, dt_spatial, true, 0);
          
          problem.setup(); // Assicurati che setup() generi la mesh con N suddivisioni
          problem.solve();

          h_values.push_back(1.0 / N);
          errors_L2_u_spatial.push_back(problem.compute_error(VectorTools::L2_norm, false));
        }

      // Stampa Risultati Spaziali (solo Rank 0)
      if (mpi_rank == 0)
        {
          std::cout << std::left << std::setw(10) << "h" 
                    << std::setw(15) << "L2_Error" << std::setw(10) << "Order" << std::endl;
          for (unsigned int i = 0; i < h_values.size(); ++i)
            {
              std::cout << std::scientific << std::setprecision(4) << std::setw(10) << h_values[i]
                        << std::setw(15) << errors_L2_u_spatial[i];
              if (i > 0)
                {
                  double p = std::log(errors_L2_u_spatial[i] / errors_L2_u_spatial[i-1]) /
                             std::log(h_values[i] / h_values[i-1]);
                  std::cout << std::fixed << std::setprecision(2) << std::setw(10) << p;
                }
              std::cout << std::endl;
            }
          std::cout << "======================================================" << std::endl;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << "Error: " << exc.what() << std::endl;
      return 1;
    }

  return 0;
}
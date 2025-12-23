#include "Wave.hpp"
#include <iomanip>

/**
 * @brief Driver per l'analisi di dispersione e dissipazione.
 * * Simula la propagazione di un pacchetto d'onda gaussiano su tempi lunghi
 * per verificare la stabilità e la qualità numerica dello schema.
 */
int main(int argc, char *argv[])
{
  try
    {
      // Inizializzazione MPI
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
      const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

      // --- Parametri per lo studio di Dispersione/Dissipazione ---
      // Nota: usiamo una mesh fitta e un tempo finale lungo (T=5.0) 
      // per vedere l'accumulo degli errori di fase.
      const std::string  mesh_file_name   = "../mesh/mesh-square-20.msh"; 
      const unsigned int degree           = 2;     // Grado 2 consigliato per onde
      const double       T                = 5.0;   // Simulazione lunga
      const double       deltat           = 0.002;  // Passo temporale
      const bool         use_manufactured = false; // false = usa pacchetto Gaussiano (IC)
      const unsigned int output_frequency = 20;    // Output ogni 20 step

      if (mpi_rank == 0)
        {
          std::cout << "======================================================" << std::endl;
          std::cout << " DISPERSION AND DISSIPATION ANALYSIS (MPI)" << std::endl;
          std::cout << "======================================================" << std::endl;
          std::cout << " Simulating Gaussian wave packet propagation" << std::endl;
          std::cout << " Mesh: " << mesh_file_name << std::endl;
          std::cout << " Duration (T): " << T << " s" << std::endl;
          std::cout << " Time step (dt): " << deltat << std::endl;
          std::cout << " FE Degree: " << degree << std::endl;
          std::cout << " Centered Differences Scheme (Beta=0, Gamma=0.5 equivalent)" << std::endl;
          std::cout << "======================================================" << std::endl;
        }

      // Inizializzazione del problema
      // Nota: abbiamo rimosso beta/gamma perché lo schema Centered Differences è fisso
      WaveEquation problem(mesh_file_name,
                           degree,
                           T,
                           deltat,
                           use_manufactured,
                           output_frequency);

      problem.setup();
      
      // Esecuzione del solutore
      // Internamente scriverà i file .vtu e i log di energia in results.csv
      problem.solve();

      if (mpi_rank == 0)
        {
          std::cout << "\nAnalysis complete!" << std::endl;
          std::cout << "Results saved to:" << std::endl;
          std::cout << "  - results.csv (Energy vs Time)" << std::endl;
          std::cout << "  - output-*.vtu (Visualizzazione in ParaView)" << std::endl;
          std::cout << "\nMonitora l'energia in results.csv per verificare la dissipazione." << std::endl;
          std::cout << "======================================================" << std::endl;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << "\n\nException: " << exc.what() << "\nAborting!\n";
      return 1;
    }
  catch (...)
    {
      std::cerr << "\n\nUnknown exception!\nAborting!\n";
      return 1;
    }

  return 0;
}
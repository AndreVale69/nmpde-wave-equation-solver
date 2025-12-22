#ifndef NM4PDE_PARAMETERS_HPP
#define NM4PDE_PARAMETERS_HPP

#include <deal.II/base/parameter_handler.h>
#include <filesystem>

#include "time_scheme.hpp"

/**
 * @brief Structure to hold simulation parameters.
 */
struct Parameters {
    /**
     * @brief Mesh file path. Defaults to a structured square mesh in ../mesh/ directory.
     */
    std::string mesh_file = kDefaultMeshFile;

    /**
     * @brief T indicates the final time of the simulation
     * (i.e., the time at which the simulation ends).
     */
    double T = 1.0;
    /**
     * @brief dt indicates the time step size
     * (i.e., the increment in time between two consecutive time steps).
     */
    double dt = 0.01;
    /**
     * @brief theta parameter for the theta time integration scheme.
     * Ranges from 0 (explicit) to 1 (implicit).
     */
    double theta = 1.0;

    /**
     * @brief Polynomial degree of the finite element basis functions.
     */
    unsigned int degree = 1;

    /**
     * @brief Output frequency: output_every indicates how often
     * (in terms of time steps) the solution is written to output files.
     */
    unsigned int output_every = 1;

    // -------------------- Study: Dissipation --------------------
    bool enable_dissipation_study = false;

    // Write dissipation CSV every N time steps
    unsigned int dissipation_every = 1;

    // Output CSV file (written by rank 0)
    std::string dissipation_csv = "dissipation.csv";

    // -------------------- Study: Modal --------------------
    bool enable_modal_study = false;
    unsigned int modal_every = 1;
    std::string modal_csv = "modal.csv";

    unsigned int modal_k = 1; // sin(k*pi*x)
    unsigned int modal_l = 1; // sin(l*pi*y)


    /**
     * @brief Time integration scheme to use.
     */
    TimeScheme scheme = TimeScheme::Theta;

    /**
     * @brief Constructor: reads parameters from the specified input file.
     * @param filename Input file name.
     */
    explicit Parameters(const std::string &filename)
        : mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0) {
        pcout << "Reading parameters from file: " << filename << std::endl;
        dealii::ParameterHandler prm;
        declare(prm);
        parse(prm, filename);
    }

private:
    /**
     * @brief This MPI process.
     */
    const unsigned int mpi_rank;

    /**
     * @brief Parallel output stream.
     */
    dealii::ConditionalOStream pcout;

    /**
     * @brief Default mesh file path.
     * Uses a structured square mesh located in the ../mesh/ directory.
     */
    inline static const std::string kDefaultMeshFile =
            (std::filesystem::path("..") / "mesh" / "square_structured.geo").string();
    /**
     * @brief Selection string for time scheme parameter.
     * Lists all available time integration schemes.
     * Used for parameter validation in deal.II's ParameterHandler.
     */
    inline static const std::string kSelectionTimeScheme =
            to_string(TimeScheme::Theta) + "|" + to_string(TimeScheme::CentralDifference) + "|" +
            to_string(TimeScheme::Newmark);

    /**
     * @brief Declare parameters in the given ParameterHandler.
     * @param prm ParameterHandler object to declare parameters in.
     */
    static void declare(dealii::ParameterHandler &prm) {
        prm.enter_subsection("Mesh");
        {
            prm.declare_entry("mesh_file", kDefaultMeshFile, dealii::Patterns::Anything());
            prm.declare_entry("degree", "1", dealii::Patterns::Integer(1));
        }
        prm.leave_subsection();

        prm.enter_subsection("Time");
        {
            prm.declare_entry("T", "1.0", dealii::Patterns::Double(0.0));
            prm.declare_entry("dt", "0.01", dealii::Patterns::Double(0.0));
            prm.declare_entry("theta", "1.0", dealii::Patterns::Double(0.0, 1.0));
            prm.declare_entry("scheme", "theta", dealii::Patterns::Selection(kSelectionTimeScheme));
        }
        prm.leave_subsection();

        prm.enter_subsection("Output");
        {
            prm.declare_entry("every", "1", dealii::Patterns::Integer(1));
        }
        prm.leave_subsection();

        prm.enter_subsection("Study");
        {
            prm.declare_entry("enable_dissipation", "false", dealii::Patterns::Bool());
            prm.declare_entry("dissipation_every", "1", dealii::Patterns::Integer(1));
            prm.declare_entry("dissipation_csv", "dissipation.csv", dealii::Patterns::Anything());
            prm.declare_entry("enable_modal", "false", dealii::Patterns::Bool());
            prm.declare_entry("modal_every", "1", dealii::Patterns::Integer(1));
            prm.declare_entry("modal_csv", "modal.csv", dealii::Patterns::Anything());
            prm.declare_entry("modal_k", "1", dealii::Patterns::Integer(1));
            prm.declare_entry("modal_l", "1", dealii::Patterns::Integer(1));

        }
        prm.leave_subsection();

    }

    /**
     * @brief Parse parameters from the given input file.
     * @param prm ParameterHandler object to populate.
     * @param filename Input file name to read parameters from.
     */
    void parse(dealii::ParameterHandler &prm, const std::string &filename) {
        prm.parse_input(filename);

        prm.enter_subsection("Mesh");
        {
            mesh_file = prm.get("mesh_file");
            degree    = prm.get_integer("degree");
        }
        prm.leave_subsection();

        prm.enter_subsection("Time");
        {
            T      = prm.get_double("T");
            dt     = prm.get_double("dt");
            theta  = prm.get_double("theta");
            scheme = time_scheme_from_string(prm.get("scheme"));
        }
        prm.leave_subsection();

        prm.enter_subsection("Output");
        {
            output_every = prm.get_integer("every");
        }
        prm.leave_subsection();

        prm.enter_subsection("Study");
        {
            enable_dissipation_study = prm.get_bool("enable_dissipation");
            dissipation_every        = prm.get_integer("dissipation_every");
            dissipation_csv          = prm.get("dissipation_csv");
            enable_modal_study = prm.get_bool("enable_modal");
            modal_every        = prm.get_integer("modal_every");
            modal_csv          = prm.get("modal_csv");
            modal_k            = prm.get_integer("modal_k");
            modal_l            = prm.get_integer("modal_l");

        }
        prm.leave_subsection();
    }
};

#endif // NM4PDE_PARAMETERS_HPP

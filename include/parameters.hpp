#ifndef NM4PDE_PARAMETERS_HPP
#define NM4PDE_PARAMETERS_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <filesystem>
#include <iostream>
#include <unistd.h>

#include "enum/boundary_type.hpp"
#include "enum/problem_type.hpp"
#include "enum/time_scheme.hpp"
#include "problem_functions.hpp"

/**
 * @brief Structure to hold simulation parameters.
 */
template<int dim>
struct Parameters {
    /**
     * @brief Problem-related parameters.
     */
    struct Problem {
        /**
         * @brief Type of problem to solve.
         */
        ProblemType type = ProblemType::Physical;
        /**
         * @brief Initial displacement expression (for expression-based problems).
         */
        std::string u0_expr = "x*(1-x)*y*(1-y)";
        /**
         * @brief Exact displacement expression (for MMS problems).
         */
        std::string u_exact_expr = ManufacturedSolution<dim>().get_expression();
        /**
         * @brief Initial velocity expression (for expression-based problems).
         */
        std::string v0_expr = "0";
        /**
         * @brief Exact velocity expression (for MMS problems).
         */
        std::string v_exact_expr = ManufacturedVelocity<dim>().get_expression();
        /**
         * @brief Forcing term expression (for expression-based problems).
         */
        std::string f_expr = "0";
        /**
         * @brief Exact forcing term expression (for MMS problems).
         */
        std::string f_exact_expr = ManufacturedForcing<dim>().get_expression();
        /**
         * @brief Additional term expression (for expression-based problems).
         */
        std::string mu_expr = "1";
    } problem;

    /**
     * @brief Boundary condition-related parameters.
     */
    struct BoundaryCondition {
        /**
         * @brief Type of boundary condition to apply.
         */
        BoundaryType type = BoundaryType::Zero;

        /**
         * @brief Dirichlet for displacement: g = h(x,y,t)
         */
        std::string g_expr = "0";

        /**
         * @brief Dirichlet for velocity: v = h(x,y,t)
         */
        std::string v_expr = "0";
    } boundary_condition;

    /**
     * @brief Mesh-related parameters.
     */
    struct Mesh {
        /**
         * @brief Mesh file path. Defaults to a structured square mesh in ../mesh/ directory.
         */
        std::string mesh_file = kDefaultMeshFile;
        /**
         * @brief Polynomial degree of the finite element basis functions.
         */
        unsigned int degree = 1;
    } mesh;

    /**
     * @brief Time-related parameters.
     */
    struct Time {
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
         * @brief Time integration scheme to use.
         */
        TimeScheme scheme = TimeScheme::Theta;
    } time;

    /**
     * @brief Output-related parameters.
     */
    struct Output {
        /**
         * @brief Output frequency: output_every indicates how often
         * (in terms of time steps) the solution is written to output files.
         */
        unsigned int output_every = 1;

        /**
         * @brief Whether to compute/save the error history.
         * Note: this is only meaningful for MMS problems; if set to true
         * for non-MMS problems it will be ignored (and a warning printed).
         */
        bool compute_error = false;

        /**
         * @brief Path to the CSV file where the error history will be saved.
         * Used only when compute_error is true (and the problem type is MMS).
         */
        std::string error_history_file = kDefaultErrorFile;

        /**
         * @brief Directory where VTK (.vtu/.pvtu) output files will be written.
         */
        std::string vtk_output_directory = kDefaultVTKDir;
    } output;

    /**
     * @brief Constructor: reads parameters from the specified input file.
     * @param filename Input file name.
     */
    explicit Parameters(const std::string &filename)
        : mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0) {
        pcout << "Reading parameters from file: " << filename << std::endl;
        ParameterHandler prm;
        declare(prm);
        parse(prm, filename);
    }

    /**
     * @brief Initialize the problem functions (mu, forcing term, initial conditions).
     * @tparam dim Spatial dimension.
     * @param mu FunctionParser for the mu coefficient.
     * @param boundary_g FunctionParser for the boundary condition g.
     * @param boundary_v FunctionParser for the boundary condition for the velocity.
     * @param forcing_term FunctionParser for the forcing term.
     * @param u_0 FunctionParser for the initial displacement.
     * @param v_0 FunctionParser for the initial velocity.
     */
    void initialize_problem(FunctionParser<dim>            &mu,
                            std::unique_ptr<Function<dim>> &boundary_g,
                            std::unique_ptr<Function<dim>> &boundary_v,
                            FunctionParser<dim>            &forcing_term,
                            FunctionParser<dim>            &u_0,
                            FunctionParser<dim>            &v_0) {
        const std::string                   vars      = "x,y,t";
        const std::map<std::string, double> constants = {
                {"pi", numbers::PI}, {"PI", numbers::PI}, {"Pi", numbers::PI}};

        // mu coefficient
        {
            pcout << "    Initializing the mu coefficient" << std::endl;
            mu.initialize(vars, problem.mu_expr, constants, true);
        }

        // Forcing term
        {
            pcout << "    Initializing the forcing term" << std::endl;
            if (problem.type == ProblemType::MMS) {
                forcing_term.initialize(vars, problem.f_exact_expr, constants, true);
            } else if (problem.type == ProblemType::Expr) {
                forcing_term.initialize(vars, problem.f_expr, constants, true);
            } else {
                forcing_term.initialize(vars, "0", constants, true);
            }
        }

        // Boundary condition g
        {
            pcout << "    Initializing the boundary condition g" << std::endl;
            if (boundary_condition.type == BoundaryType::Zero) {
                boundary_g = std::make_unique<BoundaryGZero<dim>>();
            } else if (boundary_condition.type == BoundaryType::MMS) {
                auto fp = std::make_unique<FunctionParser<dim>>();
                fp->initialize(vars, problem.u_exact_expr, constants, true);
                boundary_g = std::move(fp);
            } else {
                auto fp = std::make_unique<FunctionParser<dim>>();
                fp->initialize(vars, boundary_condition.g_expr, constants, true);
                boundary_g = std::move(fp);
            }
        }

        // Boundary condition for the velocity
        {
            pcout << "    Initializing the boundary condition for the velocity" << std::endl;
            if (boundary_condition.type == BoundaryType::Zero) {
                boundary_v = std::make_unique<BoundaryVZero<dim>>();
            } else if (boundary_condition.type == BoundaryType::MMS) {
                auto fp = std::make_unique<FunctionParser<dim>>();
                fp->initialize(vars, problem.v_exact_expr, constants, true);
                boundary_v = std::move(fp);
            } else {
                auto fp = std::make_unique<FunctionParser<dim>>();
                fp->initialize(vars, boundary_condition.v_expr, constants, true);
                boundary_v = std::move(fp);
            }
        }

        // Initial condition
        {
            pcout << "    Initializing the initial condition" << std::endl;
            if (problem.type == ProblemType::MMS) {
                u_0.initialize(vars, problem.u_exact_expr, constants, true);
            } else if (problem.type == ProblemType::Expr) {
                u_0.initialize(vars, problem.u0_expr, constants, true);
            } else {
                u_0.initialize(vars, "0", constants, true);
            }
        }

        // Initial condition for the velocity
        {
            pcout << "    Initializing the initial velocity" << std::endl;
            if (problem.type == ProblemType::MMS) {
                v_0.initialize(vars, problem.v_exact_expr, constants, true);
            } else if (problem.type == ProblemType::Expr) {
                v_0.initialize(vars, problem.v0_expr, constants, true);
            } else {
                v_0.initialize(vars, "0", constants, true);
            }
        }
    }

private:
    /**
     * @brief This MPI process.
     */
    const unsigned int mpi_rank;

    /**
     * @brief Parallel output stream.
     */
    ConditionalOStream pcout;

    /**
     * @brief Default mesh file path.
     * Uses a structured square mesh located in the ../mesh/ directory.
     */
    inline static const std::string kDefaultMeshFile =
            (std::filesystem::path("..") / "mesh" / "square_structured.geo").string();

    /**
     * @brief Default error history CSV file path.
     */
    inline static const std::string kDefaultErrorFile =
            (std::filesystem::path("build") / "error_history.csv").string();

    /**
     * @brief Default VTK output directory.
     */
    inline static const std::string kDefaultVTKDir = std::filesystem::path("build").string();

    /**
     * @brief Selection string for time scheme parameter.
     * Lists all available time integration schemes.
     * Used for parameter validation in deal.II's ParameterHandler.
     */
    inline static const std::string kSelectionTimeScheme =
            to_string(TimeScheme::Theta) + "|" + to_string(TimeScheme::CentralDifference) + "|" +
            to_string(TimeScheme::Newmark);

    /**
     * @brief Selection string for problem type parameter.
     * Lists all available problem types.
     * Used for parameter validation in deal.II's ParameterHandler.
     */
    inline static const std::string kSelectionProblemType = to_string(ProblemType::Physical) + "|" +
                                                            to_string(ProblemType::MMS) + "|" +
                                                            to_string(ProblemType::Expr);

    /**
     * @brief Selection string for boundary condition type parameter.
     * Lists all available boundary condition types.
     * Used for parameter validation in deal.II's ParameterHandler.
     */
    inline static const std::string kBoundaryConditionType = to_string(BoundaryType::Zero) + "|" +
                                                             to_string(BoundaryType::MMS) + "|" +
                                                             to_string(BoundaryType::Expr);

    /**
     * @brief Declare parameters in the given ParameterHandler.
     * @param prm ParameterHandler object to declare parameters in.
     */
    static void declare(ParameterHandler &prm) {
        prm.enter_subsection("Problem");
        {
            prm.declare_entry(
                    "type",
                    "physical",
                    Patterns::Selection(kSelectionProblemType),
                    "Type of problem to solve: 'physical' for physical problem, "
                    "'mms' for manufactured solution, 'expr' for expression-based problem.");

            // MMS-based problem entries
            prm.declare_entry("u_exact_expr",
                              ManufacturedSolution<dim>().get_expression(),
                              Patterns::Anything(),
                              "Exact initial displacement expression (for MMS problems). For "
                              "example, u_0(x) = u_{ex}(x,0).");
            prm.declare_entry("v_exact_expr",
                              ManufacturedVelocity<dim>().get_expression(),
                              Patterns::Anything(),
                              "Exact initial velocity expression (for MMS problems). For example, "
                              "v_0(x) = du_{ex}/dt(x,0).");
            prm.declare_entry("f_exact_expr",
                              ManufacturedForcing<dim>().get_expression(),
                              Patterns::Anything(),
                              "Exact forcing term expression (for MMS problems).");

            // Expression-based problem entries
            prm.declare_entry("u0_expr",
                              "x*(1-x)*y*(1-y)",
                              Patterns::Anything(),
                              "Initial displacement expression (for expression-based problems).");
            prm.declare_entry("v0_expr",
                              "0",
                              Patterns::Anything(),
                              "Initial velocity expression (for expression-based problems).");
            prm.declare_entry("f_expr",
                              "0",
                              Patterns::Anything(),
                              "Forcing term expression (for expression-based problems).");
            prm.declare_entry("mu_expr",
                              "1",
                              Patterns::Anything(),
                              "Additional term expression (for expression-based problems).");
        }
        prm.leave_subsection();

        prm.enter_subsection("Boundary condition");
        {
            prm.declare_entry(
                    "type",
                    to_string(BoundaryType::Zero),
                    Patterns::Selection(kBoundaryConditionType),
                    "Type of boundary condition to apply: 'zero' for homogeneous Dirichlet, "
                    "'mms' for manufactured solution, 'expr' for expression-based boundary "
                    "condition.");
            prm.declare_entry("g_expr",
                              "0",
                              Patterns::Anything(),
                              "Dirichlet boundary for displacement u=g(x,y,t) (if type is expr).");
            prm.declare_entry("v_expr",
                              "0",
                              Patterns::Anything(),
                              "Dirichlet boundary for velocity v=h(x,y,t) (if type is expr).");
        }
        prm.leave_subsection();

        prm.enter_subsection("Mesh");
        {
            prm.declare_entry("mesh_file",
                              kDefaultMeshFile,
                              Patterns::Anything(),
                              "Path to the mesh file or .geo file.");
            prm.declare_entry("degree",
                              "1",
                              Patterns::Integer(1),
                              "Polynomial degree of the finite element basis functions.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Time");
        {
            prm.declare_entry("T", "1.0", Patterns::Double(0.0), "Final time of the simulation.");
            prm.declare_entry("dt", "0.01", Patterns::Double(0.0), "Time step size.");
            prm.declare_entry("theta",
                              "1.0",
                              Patterns::Double(0.0, 1.0),
                              "Theta parameter for the theta time integration scheme.");
            prm.declare_entry("scheme",
                              to_string(TimeScheme::Theta),
                              Patterns::Selection(kSelectionTimeScheme),
                              "Time integration scheme to use: '" + to_string(TimeScheme::Theta) +
                                      "', '" + to_string(TimeScheme::CentralDifference) +
                                      "', or '" + to_string(TimeScheme::Newmark) + "'.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Output");
        {
            prm.declare_entry(
                    "every", "1", Patterns::Integer(1), "Output frequency in time steps.");
            prm.declare_entry(
                    "compute_error",
                    "false",
                    Patterns::Bool(),
                    "Whether to compute and save the error history (useful for MMS problems). "
                    "If set to true and the problem is not MMS, this option will be ignored.");
            prm.declare_entry("error_file",
                              kDefaultErrorFile,
                              Patterns::Anything(),
                              "CSV file path where error history will be saved (used if "
                              "compute_error is true and problem is MMS).");
            prm.declare_entry("vtk_directory",
                              kDefaultVTKDir,
                              Patterns::Anything(),
                              "Directory where VTK (.vtu/.pvtu) output files will be written.");
        }
        prm.leave_subsection();
    }

    /**
     * @brief Parse parameters from the given input file.
     * @param prm ParameterHandler object to populate.
     * @param filename Input file name to read parameters from.
     */
    void parse(ParameterHandler &prm, const std::string &filename) {
        prm.parse_input(filename);

        prm.enter_subsection("Problem");
        {
            problem.type = problem_type_from_string(prm.get("type"));

            // MMS-based problem entries
            problem.u_exact_expr = prm.get("u_exact_expr");
            problem.v_exact_expr = prm.get("v_exact_expr");
            problem.f_exact_expr = prm.get("f_exact_expr");

            // Expression-based problem entries
            problem.u0_expr = prm.get("u0_expr");
            problem.v0_expr = prm.get("v0_expr");
            problem.f_expr  = prm.get("f_expr");
            problem.mu_expr = prm.get("mu_expr");
        }
        prm.leave_subsection();

        prm.enter_subsection("Boundary condition");
        {
            boundary_condition.type   = boundary_type_from_string(prm.get("type"));
            boundary_condition.g_expr = prm.get("g_expr");
            boundary_condition.v_expr = prm.get("v_expr");
        }
        prm.leave_subsection();


        prm.enter_subsection("Mesh");
        {
            mesh.mesh_file = prm.get("mesh_file");
            mesh.degree    = prm.get_integer("degree");
        }
        prm.leave_subsection();

        prm.enter_subsection("Time");
        {
            time.T      = prm.get_double("T");
            time.dt     = prm.get_double("dt");
            time.theta  = prm.get_double("theta");
            time.scheme = time_scheme_from_string(prm.get("scheme"));
        }
        prm.leave_subsection();

        prm.enter_subsection("Output");
        {
            output.output_every         = prm.get_integer("every");
            output.compute_error        = prm.get_bool("compute_error");
            output.error_history_file   = prm.get("error_file");
            output.vtk_output_directory = prm.get("vtk_directory");
        }
        prm.leave_subsection();

        // Validate compute_error option based on problem type
        if (output.compute_error && problem.type != ProblemType::MMS) {
            pcout << "Warning: 'compute_error' was requested but the problem type is not MMS. "
                  << "compute_error will be disabled (error history only available for MMS)."
                  << std::endl;
            output.compute_error = false;
        }

        // Interactive prompts (only on rank 0 and only if stdin is a TTY).
        // TTY: isatty() checks whether the file descriptor refers to a terminal, i.e., whether
        // the program is run in an interactive terminal session. If not (e.g., input redirection
        // from a file or non-interactive environment), we skip the prompts.
        // We ask the user whether to compute error (only meaningful for MMS problems),
        // and where to save the CSV and the VTK output directory. Answers from rank 0
        // are broadcast to all other ranks.
        try {
            if (mpi_rank == 0 && isatty(fileno(stdin))) {
                // Ask about computing error (only if MMS)
                if (problem.type == ProblemType::MMS) {
                    pcout << "Do you want to compute/save the MMS error history? (y/n) ["
                          << (output.compute_error ? "y" : "n") << "] : ";
                    std::string ans;
                    std::getline(std::cin, ans);
                    if (!ans.empty()) {
                        if (ans == "y" || ans == "Y" || ans == "1")
                            output.compute_error = true;
                        else
                            output.compute_error = false;
                    }

                    // Ask where to save the CSV error history (allow empty to keep current)
                    pcout << "CSV file for error history [press Enter to keep '"
                          << output.error_history_file << "'] : ";
                    std::string csv_path;
                    std::getline(std::cin, csv_path);
                    if (!csv_path.empty())
                        output.error_history_file = csv_path;
                } else {
                    pcout << "Note: error computation is only available for MMS problems.\n";
                }

                // Ask where to write VTK output directory (allow empty to keep current):
                pcout << "VTK output directory [press Enter to keep '"
                      << output.vtk_output_directory << "'] : ";
                std::string vtk_dir;
                std::getline(std::cin, vtk_dir);
                if (!vtk_dir.empty())
                    output.vtk_output_directory = vtk_dir;
            }

            // Broadcast the (possibly updated) choices to all ranks
            Utilities::MPI::broadcast(MPI_COMM_WORLD, output.compute_error, 0);
            Utilities::MPI::broadcast(MPI_COMM_WORLD, output.error_history_file, 0);
            Utilities::MPI::broadcast(MPI_COMM_WORLD, output.vtk_output_directory, 0);
        } catch (std::exception &e) {
            pcout << "Warning: interactive prompts skipped due to exception: " << e.what()
                  << std::endl;
        }

        // Inform the user where outputs will be written
        pcout << "VTK output directory: " << output.vtk_output_directory << std::endl;
        try {
            // Create VTK output directory if it does not exist
            if (!std::filesystem::exists(output.vtk_output_directory)) {
                std::filesystem::create_directories(output.vtk_output_directory);
                pcout << "Created VTK output directory: " << output.vtk_output_directory
                      << std::endl;
            }
        } catch (const std::exception &e) {
            pcout << "Warning: could not create VTK output directory '"
                  << output.vtk_output_directory << "': " << e.what() << std::endl;
        }

        // Inform about error history file (if applicable)
        if (output.compute_error) {
            pcout << "Error history will be saved to: " << output.error_history_file << std::endl;
        }
    }
};

#endif // NM4PDE_PARAMETERS_HPP

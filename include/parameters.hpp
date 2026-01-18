#ifndef NM4PDE_PARAMETERS_HPP
#define NM4PDE_PARAMETERS_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

#include "enum/boundary_type.hpp"
#include "enum/convergence_type.hpp"
#include "enum/problem_type.hpp"
#include "enum/time_scheme.hpp"
#include "functions/problem_functions.hpp"

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
         * @brief Whether to run a convergence study.
         *
         * This option is meaningful if and only if the problem type is MMS.
         * If enabled for non-MMS problems it will be disabled (and a warning printed).
         *
         * When enabled in MMS mode, a convergence type must be provided.
         */
        bool convergence_study = false;

        /**
         * @brief Convergence study type (time or space).
         * Only used when convergence_study is true.
         */
        ConvergenceType convergence_type = ConvergenceType::Time;

        /**
         * @brief Optional path to a CSV file where the convergence table will be written.
         *
         * This option is meaningful if and only if convergence_study is true.
         * If provided while convergence_study is false, it will be ignored (and cleared).
         */
        std::string convergence_csv = "";

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

    struct Study {
        /**
         * @brief Enable dissipation study.
         */
        bool enable_dissipation_study = false;

        /**
         * @brief Dissipation study output frequency.
         */
        unsigned int dissipation_every = 1;

        /**
         * @brief Dissipation study CSV output file name.
         */
        std::string dissipation_csv = "dissipation.csv";

        /**
         * @brief Enable modal study. If enabled, the program tracks
         * the evolution of a specific mode defined by (k,l).
         */
        bool enable_modal_study = false;

        /**
         * @brief Modal study output frequency.
         */
        unsigned int modal_every = 1;

        /**
         * @brief Modal study CSV output file name.
         */
        std::string modal_csv = "modal.csv";

        /**
         * @brief Mode indices (k,l) to track in the modal study.
         */
        unsigned int modal_k = 1; // sin(k*pi*x)

        /**
         * @brief Mode indices (k,l) to track in the modal study.
         */
        unsigned int modal_l = 1; // sin(l*pi*y)
    } study;

    /**
     * @brief Constructor: reads parameters from the specified input file.
     * @param filename Input file name.
     */
    explicit Parameters(const std::string &filename)
        : mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , prm(ParameterHandler()) {
        pcout << "Reading parameters from file: " << filename << std::endl;
        declare();
        parse(filename);
    }

    /**
     * @brief Initialize the problem functions (mu, forcing term, initial conditions).
     * @tparam dim Spatial dimension.
     * @param mu FunctionParser for the mu coefficient.
     * @param boundary_g Function for the boundary condition g.
     * @param boundary_v Function for the boundary condition for the velocity.
     * @param forcing_term FunctionParser for the forcing term.
     * @param u_0 FunctionParser for the initial displacement.
     * @param v_0 FunctionParser for the initial velocity.
     */
    void initialize_problem(FunctionParser<dim>            &mu,
                            std::unique_ptr<Function<dim>> &boundary_g,
                            std::unique_ptr<Function<dim>> &boundary_v,
                            FunctionParser<dim>            &forcing_term,
                            FunctionParser<dim>            &u_0,
                            FunctionParser<dim>            &v_0) const {
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

    /**
     * @brief Write the current parameter values back to a deal.II-style .prm file.
     *
     * This implementation uses deal.II's ParameterHandler printing facilities.
     * It first syncs the internal ParameterHandler (`prm`) with the current struct values,
     * then prints the full parameter tree.
     *
     * @param filepath Path to the output .prm file.
     */
    void write_back_to_file(const std::string &filepath) {
        const std::filesystem::path out_path(filepath);
        try {
            if (out_path.has_parent_path()) {
                std::filesystem::create_directories(out_path.parent_path());
            }
        } catch (const std::exception &e) {
            pcout << "Warning: could not create directory for prm output '" << filepath
                  << "': " << e.what() << std::endl;
        }

        // Sync current values into the internal ParameterHandler.
        // The ParameterHandler already has all entries declared in the constructor.
        update();

        std::ofstream out(filepath);
        if (!out) {
            throw std::runtime_error("Cannot open prm output file for writing: " + filepath);
        }

        prm.print_parameters(out, ParameterHandler::OutputStyle::PRM);
        out.flush();
        if (!out) {
            throw std::runtime_error("Failed while writing prm output file: " + filepath);
        }

        pcout << "Wrote parameters to: " << filepath << std::endl;
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
     * @brief Internal ParameterHandler instance.
     */
    ParameterHandler prm;

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
     * @brief Selection string for convergence study type parameter.
     */
    inline static const std::string kSelectionConvergenceType =
            to_string(ConvergenceType::Time) + "|" + to_string(ConvergenceType::Space);

    /**
     * @brief Declare parameters in the given ParameterHandler.
     */
    void declare() {
        prm.enter_subsection("Problem");
        {
            prm.declare_entry("type",
                              to_string(ProblemType::Physical),
                              Patterns::Selection(kSelectionProblemType),
                              "Type of problem to solve: '" + to_string(ProblemType::Physical) +
                                      "' for physical problem, '" + to_string(ProblemType::MMS) +
                                      "' for manufactured solution, '" +
                                      to_string(ProblemType::Expr) +
                                      "' for expression-based problem.");

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
                    "Type of boundary condition to apply: '" + to_string(BoundaryType::Zero) +
                            "' for homogeneous Dirichlet, '" + to_string(BoundaryType::MMS) +
                            "' for manufactured solution, '" + to_string(BoundaryType::Expr) +
                            "' for expression-based boundary condition.");
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

            prm.declare_entry(
                    "convergence_study",
                    "false",
                    Patterns::Bool(),
                    "Whether to run a convergence study (only meaningful for MMS problems). "
                    "If set to true and the problem is not MMS, this option will be ignored.");

            prm.declare_entry("convergence_type",
                              to_string(ConvergenceType::Time),
                              Patterns::Selection(kSelectionConvergenceType),
                              "Convergence study type (only if convergence_study is true): '" +
                                      to_string(ConvergenceType::Time) + "' or '" +
                                      to_string(ConvergenceType::Space) + "'.");

            prm.declare_entry("convergence_csv",
                              "",
                              Patterns::Anything(),
                              "Optional CSV file path where the convergence table will be saved "
                              "(used only if "
                              "convergence_study is true). Leave empty to disable CSV output.");

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

        prm.enter_subsection("Study");
        {
            prm.declare_entry("enable_dissipation", "false", Patterns::Bool());
            prm.declare_entry("dissipation_every", "1", Patterns::Integer(1));
            prm.declare_entry("dissipation_csv", "dissipation.csv", Patterns::Anything());
            prm.declare_entry("enable_modal", "false", Patterns::Bool());
            prm.declare_entry("modal_every", "1", Patterns::Integer(1));
            prm.declare_entry("modal_csv", "modal.csv", Patterns::Anything());
            prm.declare_entry("modal_k", "1", Patterns::Integer(1));
            prm.declare_entry("modal_l", "1", Patterns::Integer(1));
        }
        prm.leave_subsection();
    }

    /**
     * @brief Parse parameters from the given input file.
     * @param filename Input file name to read parameters from.
     */
    void parse(const std::string &filename) {
        // Read the parameter file
        prm.parse_input(filename);
        // Update the struct members based on the parsed values
        update();
    }

    /**
     * @brief Update the struct members based on the internal ParameterHandler values.
     */
    void update() {
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
            output.convergence_study    = prm.get_bool("convergence_study");
            output.convergence_type     = convergence_type_from_string(prm.get("convergence_type"));
            output.convergence_csv      = prm.get("convergence_csv");
            output.error_history_file   = prm.get("error_file");
            output.vtk_output_directory = prm.get("vtk_directory");
        }
        prm.leave_subsection();

        prm.enter_subsection("Study");
        {
            study.enable_dissipation_study = prm.get_bool("enable_dissipation");
            study.dissipation_every        = prm.get_integer("dissipation_every");
            study.dissipation_csv          = prm.get("dissipation_csv");
            study.enable_modal_study       = prm.get_bool("enable_modal");
            study.modal_every              = prm.get_integer("modal_every");
            study.modal_csv                = prm.get("modal_csv");
            study.modal_k                  = prm.get_integer("modal_k");
            study.modal_l                  = prm.get_integer("modal_l");
        }
        prm.leave_subsection();

        // Validate compute_error option based on problem type
        if (output.compute_error && problem.type != ProblemType::MMS) {
            pcout << "Warning: 'compute_error' was requested but the problem type is not MMS. "
                  << "compute_error will be disabled (error history only available for MMS)."
                  << std::endl;
            output.compute_error = false;
        }

        // Validate convergence study option based on problem type
        if (output.convergence_study && problem.type != ProblemType::MMS) {
            pcout << "Warning: 'convergence_study' was requested but the problem type is not MMS. "
                  << "convergence_study will be disabled (convergence study only available for "
                     "MMS)."
                  << std::endl;
            output.convergence_study = false;
        }

        // If convergence study is enabled, we must have a valid type.
        // Patterns::Selection should already prevent invalid strings, but keep a clear message.
        if (output.convergence_study) {
            if (output.convergence_type != ConvergenceType::Time &&
                output.convergence_type != ConvergenceType::Space) {
                throw std::runtime_error(
                        "Invalid convergence_type while convergence_study is enabled. Expected '" +
                        to_string(ConvergenceType::Time) + "' or '" +
                        to_string(ConvergenceType::Space) + "'.");
            }
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

                // Ask about running convergence study
                if (problem.type == ProblemType::MMS) {
                    pcout << "Do you want to run a convergence study? (y/n) ["
                          << (output.convergence_study ? "y" : "n") << "] : ";
                    std::string ans_conv;
                    std::getline(std::cin, ans_conv);
                    if (!ans_conv.empty()) {
                        if (ans_conv == "y" || ans_conv == "Y" || ans_conv == "1")
                            output.convergence_study = true;
                        else
                            output.convergence_study = false;
                    }

                    if (output.convergence_study) {
                        pcout << "Convergence type (time/space) ["
                              << to_string(output.convergence_type) << "] : ";
                        std::string conv_type;
                        std::getline(std::cin, conv_type);
                        if (!conv_type.empty()) {
                            // normalize
                            for (auto &ch: conv_type)
                                ch = static_cast<char>(::tolower(ch));
                            output.convergence_type = convergence_type_from_string(conv_type);
                        }

                        // Ask optional CSV path for convergence table
                        pcout << "CSV file for convergence table (optional) [press Enter to keep '"
                              << output.convergence_csv << "'] : ";
                        std::string conv_csv;
                        std::getline(std::cin, conv_csv);
                        if (!conv_csv.empty())
                            output.convergence_csv = conv_csv;
                    } else {
                        // if disabled, keep it empty
                        output.convergence_csv.clear();
                    }
                } else {
                    pcout << "Note: convergence study is only available for MMS problems.\n";
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
            Utilities::MPI::broadcast(MPI_COMM_WORLD, output.convergence_study, 0);
            Utilities::MPI::broadcast(MPI_COMM_WORLD, output.error_history_file, 0);
            Utilities::MPI::broadcast(MPI_COMM_WORLD, output.vtk_output_directory, 0);
            Utilities::MPI::broadcast(MPI_COMM_WORLD, output.convergence_csv, 0);

            // convergence_type: broadcast as string to avoid dealing with enum serialization
            {
                std::string conv_str = to_string(output.convergence_type);
                Utilities::MPI::broadcast(MPI_COMM_WORLD, conv_str, 0);
                output.convergence_type = convergence_type_from_string(conv_str);
            }
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

        if (output.convergence_study) {
            pcout << "Convergence study enabled. Type: " << output.convergence_type << std::endl;
            if (!output.convergence_csv.empty())
                pcout << "Convergence table CSV will be saved to: " << output.convergence_csv
                      << std::endl;
        }
    }
};

#endif // NM4PDE_PARAMETERS_HPP

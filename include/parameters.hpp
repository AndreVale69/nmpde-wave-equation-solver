#ifndef NM4PDE_PARAMETERS_HPP
#define NM4PDE_PARAMETERS_HPP

#include <deal.II/base/parameter_handler.h>
#include <filesystem>

#include "enum/boundary_type.hpp"
#include "enum/problem_type.hpp"
#include "enum/time_scheme.hpp"
#include "problem_functions.hpp"

/**
 * @brief Structure to hold simulation parameters.
 */
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
         * @brief Angular frequency for the manufactured solution (MMS).
         */
        double mms_omega = 1.0;
        /**
         * @brief Initial displacement expression (for expression-based problems).
         */
        std::string u0_expr = "x*(1-x)*y*(1-y)";
        /**
         * @brief Initial velocity expression (for expression-based problems).
         */
        std::string v0_expr = "0";
        /**
         * @brief Forcing term expression (for expression-based problems).
         */
        std::string f_expr = "0";
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
         * @brief Expression for the boundary condition (if type is Expr).
         */
        std::string expr = "0";
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
     * @param forcing_term FunctionParser for the forcing term.
     * @param u_0 FunctionParser for the initial displacement.
     * @param v_0 FunctionParser for the initial velocity.
     */
    template<int dim>
    void initialize_problem(FunctionParser<dim>            &mu,
                            std::unique_ptr<Function<dim>> &boundary_g,
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
                ManufacturedForcing<dim> manufactured_forcing(problem.mms_omega);
                forcing_term.initialize(
                        vars, manufactured_forcing.get_expression(), constants, true);
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
                boundary_g = std::make_unique<ManufacturedSolution<dim>>(problem.mms_omega);
            } else {
                auto fp = std::make_unique<FunctionParser<dim>>();
                fp->initialize(vars, boundary_condition.expr, constants, true);
                boundary_g = std::move(fp);
            }
        }

        // Initial condition
        {
            pcout << "    Initializing the initial condition" << std::endl;
            if (problem.type == ProblemType::MMS) {
                ManufacturedSolution<dim> manufactored_solution(problem.mms_omega);
                u_0.initialize(vars, manufactored_solution.get_expression(), constants, true);
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
                ManufacturedVelocity<dim> manufactured_velocity(problem.mms_omega);
                v_0.initialize(vars, manufactured_velocity.get_expression(), constants, true);
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

            prm.declare_entry("mms_omega",
                              "1.0",
                              Patterns::Double(0.0),
                              "Angular frequency for the manufactured solution (MMS).");

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
            prm.declare_entry("type", "zero", Patterns::Selection(kBoundaryConditionType));
            prm.declare_entry("expr", "0", Patterns::Anything());
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
            prm.declare_entry(
                    "scheme",
                    "theta",
                    Patterns::Selection(kSelectionTimeScheme),
                    "Time integration scheme to use: 'theta', 'central_difference', or 'newmark'.");
        }
        prm.leave_subsection();

        prm.enter_subsection("Output");
        {
            prm.declare_entry(
                    "every", "1", Patterns::Integer(1), "Output frequency in time steps.");
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
            const std::string type = prm.get("type");
            if (type == "physical")
                problem.type = ProblemType::Physical;
            else if (type == "mms")
                problem.type = ProblemType::MMS;
            else
                problem.type = ProblemType::Expr;

            problem.mms_omega = prm.get_double("mms_omega");

            problem.u0_expr         = prm.get("u0_expr");
            problem.v0_expr         = prm.get("v0_expr");
            problem.f_expr          = prm.get("f_expr");
            problem.mu_expr         = prm.get("mu_expr");
        }
        prm.leave_subsection();

        prm.enter_subsection("Boundary condition");
        {
            const std::string t = prm.get("type");
            if (t == "zero")
                boundary_condition.type = BoundaryType::Zero;
            else if (t == "mms")
                boundary_condition.type = BoundaryType::MMS;
            else
                boundary_condition.type = BoundaryType::Expr;

            boundary_condition.expr = prm.get("expr");
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
            output.output_every = prm.get_integer("every");
        }
        prm.leave_subsection();
    }
};

#endif // NM4PDE_PARAMETERS_HPP

#ifndef HEAT_HPP
#define HEAT_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "enum/time_scheme.hpp"
#include "functions/mms_functions.hpp"
#include "parameters.hpp"
#include "time_integrator/time_integrator.hpp"

using namespace dealii;

class TimeIntegrator;

// Class representing the non-linear diffusion problem.
class Wave {
public:
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 2;

    // Constructor. We provide the final time, time step Delta t and theta method
    // parameter as constructor arguments.
    explicit Wave(const std::string &parameters_file)
        : parameters_file(parameters_file)
        , parameters(std::make_shared<Parameters<dim>>(parameters_file))
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , T(parameters->time.T)
        , mesh_file_name(parameters->mesh.mesh_file)
        , r(parameters->mesh.degree)
        , output_every(parameters->output.output_every)
        , deltat(parameters->time.dt)
        , time_scheme(parameters->time.scheme)
        , mesh(MPI_COMM_WORLD) {
        // If the user provided a .geo file, try to generate a .msh mesh file
        // using gmsh automatically.
        process_mesh_input();
    }

    // Alternative constructor that takes already created Parameters object.
    explicit Wave(std::shared_ptr<const Parameters<dim>> parameters)
        : parameters(std::move(parameters))
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , T(this->parameters->time.T)
        , mesh_file_name(this->parameters->mesh.mesh_file)
        , r(this->parameters->mesh.degree)
        , output_every(this->parameters->output.output_every)
        , deltat(this->parameters->time.dt)
        , time_scheme(this->parameters->time.scheme)
        , mesh(MPI_COMM_WORLD) {
        AssertThrow(this->parameters, ExcMessage("Wave: parameters pointer is null"));
        AssertThrow(!mesh_file_name.empty(), ExcMessage("Wave: mesh_file_name empty"));
        // If the user provided a .geo file, try to generate a .msh mesh file
        // using gmsh automatically.
        process_mesh_input();
    }

    // Parameter file name.
    std::string parameters_file;

    // Initialization.
    void setup();

    // Solve the problem or run convergence studies. It depends on the parameters.
    void solve();

protected:
    // Compute discrete energy: 0.5*(v^T M v + u^T K u)
    double compute_energy(const TrilinosWrappers::MPI::Vector &u_owned,
                          const TrilinosWrappers::MPI::Vector &v_owned) const;

    // Build a map of zero Dirichlet boundary conditions.
    std::map<types::global_dof_index, double> build_zero_dirichlet_map() const;

    // Assemble the mass and stiffness matrices.
    void assemble_matrices();

    // Assemble the right-hand side of the problem.
    void assemble_rhs(const double &time, TrilinosWrappers::MPI::Vector &F_out);

    // Apply Dirichlet boundary conditions.
    void make_dirichlet_constraints(const double &time, AffineConstraints<> &constraints) const;

    // Apply Dirichlet boundary conditions to the velocity.
    void make_velocity_dirichlet_constraints(double time, AffineConstraints<> &constraints) const;

    // Output.
    void output(const unsigned int &time_step) const;

    // Solve the problem (internal).
    void do_solve();

    // Convergence studies.
    void convergence() const;

    // Error statistics structure.
    struct ErrorStatistics {
        double mean;
        double median;
        double std;
        double rms;
        double sum;
        double min;
        double max;
        // Indices (position in the time/error history vectors) where min/max occur
        size_t idx_min;
        size_t idx_max;
    };

    // Error norms structure.
    struct ErrorNorms {
        double u_L2;
        double u_H1;
        double v_L2;
    };

    // Compute error norms at the given time.
    ErrorNorms compute_error_norms(const double time);

    // Compute error statistics: min, max, mean, std, rms (root-mean-square).
    static ErrorStatistics compute_error_statistics(const std::vector<double> &errors);

    // Print a summary of the errors computed during the simulation.
    void print_error_summary() const;

    // Estimate the order of convergence given two errors and mesh sizes.
    static double estimate_order(double e1, double e2, double h1, double h2);

    // Time convergence structure.
    struct TimeConvRow {
        double     dt;
        ErrorNorms err;

        // Observed convergence orders between this row and the previous one.
        // (First row has no previous row -> left as NaN.)
        double q_uL2 = std::numeric_limits<double>::quiet_NaN();
        double q_uH1 = std::numeric_limits<double>::quiet_NaN();
        double q_vL2 = std::numeric_limits<double>::quiet_NaN();
    };

    // Run time convergence study.
    static std::vector<TimeConvRow> run_time_convergence(const std::string         &prm_base,
                                                         const std::vector<double> &dts);

    // Write time convergence results to CSV.
    void write_time_convergence_csv(const std::string              &filename,
                                    const std::vector<TimeConvRow> &rows) const;

    // Space convergence structure.
    struct SpaceConvRow {
        double      h;
        std::string mesh;
        ErrorNorms  err;

        // Observed spatial convergence orders between this row and the previous one.
        double p_uL2 = std::numeric_limits<double>::quiet_NaN();
        double p_uH1 = std::numeric_limits<double>::quiet_NaN();
        double p_vL2 = std::numeric_limits<double>::quiet_NaN();
    };

    // Run space convergence study.
    std::vector<SpaceConvRow>
    run_space_convergence(const std::string                                 &prm_base,
                          const std::vector<std::pair<std::string, double>> &meshes,
                          double                                             dt_small) const;

    // Write space convergence results to CSV.
    void write_space_convergence_csv(const std::string               &filename,
                                     const std::vector<SpaceConvRow> &rows) const;

    // Solve the problem and get the final errors.
    ErrorNorms solve_and_get_final_errors();

    // Problem parameters. ////////////////////////////////////////////////////////

    // Parameters object.
    std::shared_ptr<const Parameters<dim>> parameters;

    // MPI parallel. /////////////////////////////////////////////////////////////

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    // Problem definition. ///////////////////////////////////////////////////////

    // mu coefficient.
    FunctionParser<dim> mu;

    // Boundary condition g.
    std::unique_ptr<Function<dim>> boundary_g;

    // Boundary condition for the velocity.
    std::unique_ptr<Function<dim>> boundary_v;

    // Boundary IDs where Dirichlet BCs are applied.
    std::set<types::boundary_id> boundary_ids;

    // Forcing term.
    FunctionParser<dim> forcing_term;

    // Forcing vectors at time n.
    TrilinosWrappers::MPI::Vector forcing_n;

    // Forcing vectors at time n+1.
    TrilinosWrappers::MPI::Vector forcing_np1;

    // Initial condition.
    FunctionParser<dim> u_0;

    // Initial condition for the velocity.
    FunctionParser<dim> v_0;

    // Final time.
    const double T;

    // Discretization. ///////////////////////////////////////////////////////////

    /**
     * Mesh file name (input). Can be a .msh or .geo file. If it is a .geo file,
     * gmsh will be called to generate a .msh mesh file.
     */
    std::string mesh_file_name;

    // Polynomial degree.
    const unsigned int r;

    // Output frequency (in time steps).
    const unsigned int output_every = 10;

    // Time step.
    const double deltat;

    // Time integration scheme.
    const TimeScheme time_scheme;

    // Time integrator.
    std::unique_ptr<TimeIntegrator> time_integrator;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    // Mass matrix M
    TrilinosWrappers::SparseMatrix mass_matrix;

    // Stiffness matrix A.
    TrilinosWrappers::SparseMatrix stiffness_matrix;

    // Right-hand side vector in the linear system.
    TrilinosWrappers::MPI::Vector system_rhs;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::Vector solution_owned;

    // Displacement (including ghost elements) = u^n.
    TrilinosWrappers::MPI::Vector solution;

    // Velocity (without ghost elements) = v^n.
    TrilinosWrappers::MPI::Vector velocity_owned;

    // Velocity (including ghost elements).
    TrilinosWrappers::MPI::Vector velocity;

    // Error history (filled during solve when using MMS).
    std::vector<double> time_history;
    std::vector<double> error_u_L2_history;
    std::vector<double> error_u_H1_history;
    std::vector<double> error_v_L2_history;

    /**
     * Process the mesh input file. If it is a .geo file, attempt to generate
     * a .msh mesh file using gmsh and update the mesh_file_name accordingly.
     */
    void process_mesh_input();
};

#endif // HEAT_HPP

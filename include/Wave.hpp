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
#include <utility>
#include <vector>

#include "enum/time_scheme.hpp"
#include "mms_functions.hpp"
#include "parameters.hpp"
#include "time_integrator.hpp"

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
        : parameters(Parameters(parameters_file))
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , T(parameters.time.T)
        , mesh_file_name(parameters.mesh.mesh_file)
        , r(parameters.mesh.degree)
        , output_every(parameters.output.output_every)
        , deltat(parameters.time.dt)
        , theta(parameters.time.theta)
        , time_scheme(parameters.time.scheme)
        , mesh(MPI_COMM_WORLD) {
        // If the user provided a .geo file, try to generate a .msh mesh file
        // using gmsh automatically.
        process_mesh_input();
    }

    // Initialization.
    void setup();

    // Solve the problem.
    void solve();

protected:
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

    // Compute errors with respect to the exact solution at the given time.
    std::pair<double, double> compute_errors(double time);

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

    // Compute error statistics: min, max, mean, std, rms (root-mean-square).
    static ErrorStatistics compute_error_statistics(const std::vector<double> &errors);

    // Print a summary of the errors computed during the simulation.
    void print_error_summary() const;

    // Problem parameters. ////////////////////////////////////////////////////////

    // Parameters object.
    Parameters parameters;

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

    // Theta parameter of the theta method.
    const double theta;

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
    // error_u_history[k] and error_v_history[k] correspond to time_history[k].
    std::vector<double> error_u_history;
    std::vector<double> error_v_history;
    std::vector<double> time_history;

    /**
     * Process the mesh input file. If it is a .geo file, attempt to generate
     * a .msh mesh file using gmsh and update the mesh_file_name accordingly.
     */
    void process_mesh_input();
};

#endif // HEAT_HPP

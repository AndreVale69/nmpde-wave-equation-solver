#ifndef HEAT_HPP
#define HEAT_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
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

#include <fstream>
#include <iostream>
#include <utility>

#include <cstdlib>
#include <filesystem>

#include "parameters.hpp"
#include "time_integrator.hpp"
#include "time_scheme.hpp"

using namespace dealii;

class TimeIntegrator;

// Class representing the non-linear diffusion problem.
class Wave {
public:
    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 2;

    // Function for the mu coefficient.
    class FunctionMu : public Function<dim> {
    public:
        double value(const Point<dim> & /*p*/,
                     const unsigned int /*component*/ = 0) const override {
            return 1;
        }
    };

    // Function for the forcing term.
    class ForcingTerm : public Function<dim> {
    public:
        double value(const Point<dim> & /*p*/,
                     const unsigned int /*component*/ = 0) const override {
            return 0.0;
        }
    };

    // Function for the initial condition (single eigenmode, compatible with u=0 on boundary).
    class FunctionU0 : public Function<dim> {
    public:
        double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override {
            const double pi = numbers::PI;
            // 1st mode in 2D square-like domain: sin(pi x) sin(pi y)
            return std::sin(pi * p[0]) * std::sin(pi * p[1]);
        }
    };


    // Function for the initial velocity.
    class FunctionV0 : public Function<dim> {
    public:
        double value(const Point<dim> & /*p*/,
                     const unsigned int /*component*/ = 0) const override {
            return 0.0;
        }
    };


    // Constructor. We provide the final time, time step Delta t and theta method
    // parameter as constructor arguments.
    explicit Wave(const std::string &parameters_file)
        : parameters(Parameters(parameters_file))
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , T(parameters.T)
        , mesh_file_name(parameters.mesh_file)
        , r(parameters.degree)
        , output_every(parameters.output_every)
        , deltat(parameters.dt)
        , theta(parameters.theta)
        , time_scheme(parameters.scheme)
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

    // Output.
    void output(const unsigned int &time_step) const;

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
    FunctionMu mu;

    // Forcing term.
    ForcingTerm forcing_term;

    // Forcing vectors at time n.
    TrilinosWrappers::MPI::Vector forcing_n;

    // Forcing vectors at time n+1.
    TrilinosWrappers::MPI::Vector forcing_np1;

    // Initial condition.
    FunctionU0 u_0;

    // Initial condition for the velocity.
    FunctionV0 v_0;

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

    /**
     * Process the mesh input file. If it is a .geo file, attempt to generate
     * a .msh mesh file using gmsh and update the mesh_file_name accordingly.
     */
    void process_mesh_input();
};

#endif // HEAT_HPP

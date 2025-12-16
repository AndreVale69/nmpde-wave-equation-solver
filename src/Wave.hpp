#ifndef WAVE_HPP
#define WAVE_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

using namespace dealii;

/**
 * @brief Class for solving the 3D linear wave equation using Newmark-beta time integration.
 * 
 * Solves: ∂²u/∂t² - Δu = f(x,t) in Ω
 *         u = g(x,t) on ∂Ω
 *         u(x,0) = u₀(x), ∂u/∂t(x,0) = v₀(x)
 * 
 * Uses Newmark-β method in second-order form with predictor-corrector updates.
 */
class Wave
{
public:
  /// Physical dimension (2D)
  static constexpr unsigned int dim = 2;

  /**
   * @brief Exact solution for manufactured solution tests.
   * 
   * u_ex(x,t) = sin(ω·t)·sin(2π·x)·sin(3π·y)·sin(4π·z)
   * where ω = 5π
   */

  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution()
      : Function<dim>()
      , omega(5.0 * M_PI)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return std::sin(omega * get_time()) * std::sin(2.0 * M_PI * p[0]) *
             std::sin(3.0 * M_PI * p[1]); // 2D: removed z-component
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> grad;
      const double   time_part = std::sin(omega * get_time());

      grad[0] = time_part * 2.0 * M_PI * std::cos(2.0 * M_PI * p[0]) *
                std::sin(3.0 * M_PI * p[1]); // 2D: removed z-component

      grad[1] = time_part * std::sin(2.0 * M_PI * p[0]) * 3.0 * M_PI *
                std::cos(3.0 * M_PI * p[1]); // 2D: removed z-component

      return grad;
    }

  protected:
    const double omega;
  };

  /**
   * @brief Exact velocity (time derivative of exact solution).
   * 
   * v_ex(x,t) = ∂u_ex/∂t = ω·cos(ω·t)·sin(2π·x)·sin(3π·y)·sin(4π·z)
   */
  class ExactVelocity : public Function<dim>
  {
  public:
    ExactVelocity()
      : Function<dim>()
      , omega(5.0 * M_PI)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return omega * std::cos(omega * get_time()) * std::sin(2.0 * M_PI * p[0]) *
             std::sin(3.0 * M_PI * p[1]); // 2D: removed z-component
    }

  protected:
    const double omega;
  };

  /**
   * @brief Exact acceleration (second time derivative of exact solution).
   * 
   * a_ex(x,t) = ∂²u_ex/∂t² = -ω²·sin(ω·t)·sin(2π·x)·sin(3π·y)·sin(4π·z)
   */
  class ExactAcceleration : public Function<dim>
  {
  public:
    ExactAcceleration()
      : Function<dim>()
      , omega(5.0 * M_PI)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return -omega * omega * std::sin(omega * get_time()) *
             std::sin(2.0 * M_PI * p[0]) * std::sin(3.0 * M_PI * p[1]); // 2D: removed z-component
    }

  protected:
    const double omega;
  };

  /**
   * @brief Forcing term derived from manufactured solution.
   * 
   * f = ∂²u_ex/∂t² - Δu_ex = (-ω² + 29π²)·sin(ω·t)·sin(2π·x)·sin(3π·y)·sin(4π·z)
   */
  class ForcingTerm : public Function<dim>
  {
  public:
    ForcingTerm()
      : Function<dim>()
      , omega(5.0 * M_PI)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      const double laplacian_coefficient = 4.0 + 9.0; // 2² + 3² (2D)
      return (-omega * omega + laplacian_coefficient * M_PI * M_PI) *
             std::sin(omega * get_time()) * std::sin(2.0 * M_PI * p[0]) *
             std::sin(3.0 * M_PI * p[1]); // 2D: removed z-component
    }

  protected:
    const double omega;
  };

  /**
   * @brief Initial condition for displacement (for non-manufactured tests).
   */
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      // Gaussian wave packet centered at (0.5, 0.5) [2D]
      const double r2 = (p[0] - 0.5) * (p[0] - 0.5) + 
                        (p[1] - 0.5) * (p[1] - 0.5);
      return std::exp(-100.0 * r2);
    }
  };

  /**
   * @brief Initial condition for velocity (for non-manufactured tests).
   */
  class FunctionV0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  /**
   * @brief Constructor.
   * 
   * @param mesh_file_name_ Path to mesh file (empty string for built-in cube mesh)
   * @param N_subdivisions_ Number of subdivisions for built-in cube mesh
   * @param r_ Polynomial degree of finite elements
   * @param T_ Final time
   * @param deltat_ Time step size
   * @param beta_ Newmark beta parameter (default 0.25)
   * @param gamma_ Newmark gamma parameter (default 0.5)
   * @param use_manufactured_ Whether to use manufactured solution
   * @param output_frequency_ Write VTU output every N timesteps (0 = every step)
   */
  Wave(const std::string  &mesh_file_name_,
       const unsigned int &N_subdivisions_,
       const unsigned int &r_,
       const double       &T_,
       const double       &deltat_,
       const double       &beta_    = 0.25,
       const double       &gamma_   = 0.5,
       const bool         &use_manufactured_ = true,
       const unsigned int &output_frequency_ = 0)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , N_subdivisions(N_subdivisions_)
    , r(r_)
    , deltat(deltat_)
    , beta(beta_)
    , gamma(gamma_)
    , use_manufactured_solution(use_manufactured_)
    , output_frequency(output_frequency_)
    , mesh(MPI_COMM_WORLD)
    , time(0.0)
    , initial_energy(0.0)
  {}

  /// Initialize mesh, finite element space, and linear system
  void
  setup();

  /// Solve the time-dependent problem
  void
  solve();

  /// Compute error against exact solution
  double
  compute_error(const VectorTools::NormType &norm_type, const bool for_velocity = false);

  /// Compute total energy (kinetic + potential)
  double
  compute_energy();

protected:
  /// Assemble mass and stiffness matrices (called once)
  void
  assemble_matrices();

  /// Assemble right-hand side for current time step
  void
  assemble_rhs(const double &time);

  /// Solve one time step using Newmark-β predictor-corrector
  void
  solve_time_step();

  /// Project initial acceleration from PDE
  void
  project_initial_acceleration();

  /// Write VTU output
  void
  output(const unsigned int &time_step);

  /// Write time-series data to CSV
  void
  write_time_series(const unsigned int &time_step);

  void
  enforce_boundary_conditions_on_vector(TrilinosWrappers::MPI::Vector &vec);

  // MPI parallelization ////////////////////////////////////////////////////////

  const unsigned int mpi_size; ///< Number of MPI processes
  const unsigned int mpi_rank; ///< Current MPI process rank
  ConditionalOStream pcout;    ///< Parallel output stream

  // Problem definition /////////////////////////////////////////////////////////

  ExactSolution      exact_solution;      ///< Manufactured solution
  ExactVelocity      exact_velocity;      ///< Exact velocity
  ExactAcceleration  exact_acceleration;  ///< Exact acceleration
  ForcingTerm        forcing_term;        ///< Source term
  FunctionU0         u_0;                 ///< Initial displacement
  FunctionV0         v_0;                 ///< Initial velocity

  const double T; ///< Final time

  // Discretization /////////////////////////////////////////////////////////////

  const std::string  mesh_file_name;             ///< Mesh file path
  const unsigned int N_subdivisions;             ///< Subdivisions for built-in mesh
  const unsigned int r;                          ///< Polynomial degree
  const double       deltat;                     ///< Time step
  const double       beta;                       ///< Newmark beta parameter
  const double       gamma;                      ///< Newmark gamma parameter
  const bool         use_manufactured_solution;  ///< Use manufactured solution
  const unsigned int output_frequency;           ///< VTU output frequency

  parallel::fullydistributed::Triangulation<dim> mesh; ///< Computational mesh

  std::unique_ptr<FiniteElement<dim>> fe;         ///< Finite element
  std::unique_ptr<Quadrature<dim>>    quadrature; ///< Quadrature formula

  DoFHandler<dim> dof_handler; ///< Degree of freedom handler

  IndexSet locally_owned_dofs;    ///< DoFs owned by current process
  IndexSet locally_relevant_dofs; ///< DoFs relevant to current process

  AffineConstraints<double> constraints; ///< Constraints for hanging nodes and BCs

  // Linear algebra /////////////////////////////////////////////////////////////

  TrilinosWrappers::SparseMatrix mass_matrix;      ///< Mass matrix M
  TrilinosWrappers::SparseMatrix stiffness_matrix; ///< Stiffness matrix K
  TrilinosWrappers::SparseMatrix lhs_matrix;       ///< LHS: M (for acceleration solve)

  TrilinosWrappers::MPI::Vector system_rhs; ///< Right-hand side vector

  TrilinosWrappers::MPI::Vector solution_owned; ///< Displacement u^n (owned)
  TrilinosWrappers::MPI::Vector solution;       ///< Displacement u^n (ghosted)

  TrilinosWrappers::MPI::Vector velocity_owned; ///< Velocity v^n (owned)
  TrilinosWrappers::MPI::Vector velocity;       ///< Velocity v^n (ghosted)

  TrilinosWrappers::MPI::Vector acceleration_owned; ///< Acceleration a^n (owned)
  TrilinosWrappers::MPI::Vector acceleration;       ///< Acceleration a^n (ghosted)

  // Time stepping //////////////////////////////////////////////////////////////

  double time;           ///< Current simulation time
  double initial_energy; ///< Energy at t=0 for conservation monitoring

  // Timing /////////////////////////////////////////////////////////////////////

  double assembly_time; ///< Time spent in assembly
  double solver_time;   ///< Time spent in linear solver
  double output_time;   ///< Time spent in I/O

  // CSV output /////////////////////////////////////////////////////////////////

  std::ofstream csv_file; ///< Time-series CSV file
};

#endif

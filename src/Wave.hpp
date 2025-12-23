#ifndef WAVE_HPP
#define WAVE_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

using namespace dealii;

/**
 * @brief Risolutore per l'equazione delle onde lineare usando Differenze Centrate.
 * * Schema temporale: M u^{n+1} = M(2u^n - u^{n-1}) - dt^2 K u^n + dt^2 M f^n
 */
class WaveEquation
{
public:
  static constexpr unsigned int dim = 2;

  // --- Classi per la Manufactured Solution (Soluzione Esatta) ---

  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution() : Function<dim>(), omega(5.0 * M_PI) {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
      return std::sin(omega * get_time()) * std::sin(2.0 * M_PI * p[0]) * std::sin(3.0 * M_PI * p[1]);
    }
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override {
      Tensor<1, dim> grad;
      const double t_part = std::sin(omega * get_time());
      grad[0] = t_part * 2.0 * M_PI * std::cos(2.0 * M_PI * p[0]) * std::sin(3.0 * M_PI * p[1]);
      grad[1] = t_part * std::sin(2.0 * M_PI * p[0]) * 3.0 * M_PI * std::cos(3.0 * M_PI * p[1]);
      return grad;
    }
  protected:
    const double omega;
  };

  class ExactVelocity : public Function<dim>
  {
  public:
    ExactVelocity() : Function<dim>(), omega(5.0 * M_PI) {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
      return omega * std::cos(omega * get_time()) * std::sin(2.0 * M_PI * p[0]) * std::sin(3.0 * M_PI * p[1]);
    }
  protected:
    const double omega;
  };

  class ExactAcceleration : public Function<dim>
  {
  public:
    ExactAcceleration() : Function<dim>(), omega(5.0 * M_PI) {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
      return -omega * omega * std::sin(omega * get_time()) * std::sin(2.0 * M_PI * p[0]) * std::sin(3.0 * M_PI * p[1]);
    }
  protected:
    const double omega;
  };

  class ForcingTerm : public Function<dim>
  {
  public:
    ForcingTerm() : Function<dim>(), omega(5.0 * M_PI) {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
      const double laplacian_coeff = 4.0 + 9.0; // (2^2 + 3^2)
      return (-omega * omega + laplacian_coeff * M_PI * M_PI) * std::sin(omega * get_time()) * std::sin(2.0 * M_PI * p[0]) * std::sin(3.0 * M_PI * p[1]);
    }
  protected:
    const double omega;
  };

  // --- Condizioni Iniziali e Contorno ---

  class FunctionU0 : public Function<dim> {
  public:
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
      const double r2 = (p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5);
      return std::exp(-100.0 * r2);
    }
  };

  class FunctionV0 : public Function<dim> {
  public:
    virtual double value(const Point<dim> &, const unsigned int = 0) const override { return 0.0; }
  };

  class FunctionG : public Function<dim> {
  public:
    virtual double value(const Point<dim> &, const unsigned int = 0) const override { return 0.0; }
  };

  // --- Interfaccia Pubblica ---

  WaveEquation(const std::string &mesh_file_name_,
               const unsigned int &r_,
               const double &T_,
               const double &deltat_,
               const bool &use_manufactured_ = true,
               const unsigned int &output_freq_ = 1);

  void setup();
  void solve();
  
  double compute_error(const VectorTools::NormType &norm_type, const bool for_velocity = false);
  double compute_energy();
  double compute_power_input(const double &t);

protected:
  // Metodi Core
  void assemble_matrices();
  void assemble_rhs(const double &t);
  void solve_time_step();
  void project_initial_acceleration(); // Per calcolare u^{-1} iniziale

  // Metodi Supporto
  void output(const unsigned int &time_step) const;
  void write_time_series(const unsigned int &time_step);
  void enforce_boundary_conditions_on_vector(TrilinosWrappers::MPI::Vector &vec);

  // Membri MPI e Parallelismo
  const unsigned int mpi_size;
  const unsigned int mpi_rank;
  ConditionalOStream pcout;

  // Definizioni Problema
  ExactSolution      exact_solution;
  ExactVelocity      exact_velocity;
  ExactAcceleration  exact_acceleration;
  ForcingTerm        forcing_term;
  FunctionU0         u_0;
  FunctionV0         v_0;
  FunctionG          function_g;

  double             time;
  const double       T;
  const std::string  mesh_file_name;
  const unsigned int r;
  const double       deltat;
  const bool         use_manufactured_solution;
  const unsigned int output_frequency;

  // Oggetti deal.II
  parallel::fullydistributed::Triangulation<dim> mesh;
  std::unique_ptr<FiniteElement<dim>>            fe;
  std::unique_ptr<Quadrature<dim>>               quadrature;
  DoFHandler<dim>                                dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  // Matrici e Vettori Trilinos
  TrilinosWrappers::SparseMatrix mass_matrix;
  TrilinosWrappers::SparseMatrix stiffness_matrix;
  TrilinosWrappers::SparseMatrix lhs_matrix;

  TrilinosWrappers::MPI::Vector  system_rhs;
  TrilinosWrappers::MPI::Vector  solution;       // u^n
  TrilinosWrappers::MPI::Vector  solution_old;   // u^{n-1}
  TrilinosWrappers::MPI::Vector  solution_next;  // u^{n+1}
  TrilinosWrappers::MPI::Vector  velocity;       // v^n (per energia)
  TrilinosWrappers::MPI::Vector  acceleration;   // a^n (opzionale)

  // Monitoraggio Fisico e Timing
  double initial_energy;
  double accumulated_work = 0.0;
  std::ofstream csv_file;
  
  double assembly_time = 0.0;
  double solver_time   = 0.0;
  double output_time   = 0.0;
};

#endif
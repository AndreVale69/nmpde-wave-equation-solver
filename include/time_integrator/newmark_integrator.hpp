#ifndef NM4PDE_NEWMARK_INTEGRATOR_HPP
#define NM4PDE_NEWMARK_INTEGRATOR_HPP

#include "time_integrator.hpp"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/matrix_tools.h>

class NewmarkIntegrator : public TimeIntegrator {
public:
    // Default: average acceleration (unconditionally stable)
    explicit NewmarkIntegrator(const double beta_  = 0.25,
                               const double gamma_ = 0.50)
        : beta(beta_), gamma(gamma_) {}

    void initialize(const TrilinosWrappers::SparseMatrix &M,
                    const TrilinosWrappers::SparseMatrix &K,
                    const TrilinosWrappers::MPI::Vector  &U0,
                    const TrilinosWrappers::MPI::Vector  &V0,
                    const double                          dt) override;

    void advance(const double                                     t_n,
                 const double                                     dt,
                 const TrilinosWrappers::SparseMatrix            &M,
                 const TrilinosWrappers::SparseMatrix            &K,
                 const TrilinosWrappers::MPI::Vector             &F_n,
                 const TrilinosWrappers::MPI::Vector             &F_np1,
                 const AffineConstraints<>                       &constraints_u_np1,
                 const std::map<types::global_dof_index, double> &u_boundary_values,
                 const AffineConstraints<>                       &constraints_v_np1,
                 const std::map<types::global_dof_index, double> &v_boundary_values,
                 TrilinosWrappers::MPI::Vector                   &U,
                 TrilinosWrappers::MPI::Vector                   &V) override;

private:
    double beta;
    double gamma;

    bool first_step = true;

    TrilinosWrappers::MPI::Vector A; // acceleration A^n

    // Effective stiffness: K_eff = K + (1/(beta dt^2)) M
    TrilinosWrappers::SparseMatrix K_eff;

    TrilinosWrappers::MPI::Vector tmp_owned;
};

#endif // NM4PDE_NEWMARK_INTEGRATOR_HPP

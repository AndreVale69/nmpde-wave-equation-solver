#ifndef NM4PDE_THETA_INTEGRATOR_HPP
#define NM4PDE_THETA_INTEGRATOR_HPP

#include "time_integrator.hpp"

class ThetaIntegrator : public TimeIntegrator {
public:
    explicit ThetaIntegrator(double theta_) : theta(theta_) {}

    void initialize(const TrilinosWrappers::SparseMatrix &M,
                    const TrilinosWrappers::SparseMatrix &K,
                    const TrilinosWrappers::MPI::Vector  &U0,
                    const TrilinosWrappers::MPI::Vector  &V0,
                    const double                          dt) override;

    void advance(const double                          t_n,
                 const double                          dt,
                 const TrilinosWrappers::SparseMatrix &M,
                 const TrilinosWrappers::SparseMatrix &K,
                 const TrilinosWrappers::MPI::Vector  &F_n,
                 const TrilinosWrappers::MPI::Vector  &F_np1,
                 const std::map<types::global_dof_index, double> &boundary_values_v,
                 TrilinosWrappers::MPI::Vector        &U,
                 TrilinosWrappers::MPI::Vector        &V) override;

private:
    double theta;

    TrilinosWrappers::SparseMatrix lhs_matrix_base;
    // Matrix on the left-hand side (M / deltat + theta A).
    TrilinosWrappers::SparseMatrix lhs_matrix;

    // Matrix on the right-hand side (M / deltat - (1 - theta) A).
    TrilinosWrappers::SparseMatrix rhs_matrix;

    TrilinosWrappers::MPI::Vector system_rhs_owned;
};

#endif // NM4PDE_THETA_INTEGRATOR_HPP

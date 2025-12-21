#include "theta_integrator.hpp"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/numerics/matrix_tools.h>


void ThetaIntegrator::initialize(const TrilinosWrappers::SparseMatrix &M,
                                 const TrilinosWrappers::SparseMatrix &K,
                                 const TrilinosWrappers::MPI::Vector  &U0,
                                 const TrilinosWrappers::MPI::Vector  &V0,
                                 const double                          dt) {
    (void) U0;
    (void) V0;

    // Copy sparsity/layout from M
    lhs_matrix.reinit(M);
    rhs_matrix.reinit(M);

    // Build M/dt
    lhs_matrix.copy_from(M);
    lhs_matrix *= 1.0 / dt;

    rhs_matrix.copy_from(M);
    rhs_matrix *= 1.0 / dt;

    // C_LHS = M/dt + theta^2 * dt * K
    lhs_matrix.add(theta * theta * dt, K);

    // C_RHS^(V) = M/dt - theta(1-theta) * dt * K
    rhs_matrix.add(-theta * (1.0 - theta) * dt, K);

    // Save an untouched copy: we'll restore it every time-step before applying BC
    lhs_matrix_base.copy_from(lhs_matrix);
}


void ThetaIntegrator::advance(const double                          t_n,
                              const double                          dt,
                              const TrilinosWrappers::SparseMatrix &M,
                              const TrilinosWrappers::SparseMatrix &K,
                              const TrilinosWrappers::MPI::Vector  &F_n,
                              const TrilinosWrappers::MPI::Vector  &F_np1,
                              const std::map<types::global_dof_index, double> &boundary_values_v,
                              TrilinosWrappers::MPI::Vector        &U,
                              TrilinosWrappers::MPI::Vector        &V) {
    (void) t_n;
    (void) M;
    (void) K;

    // Save V^n before overwriting it (velocity old)
    TrilinosWrappers::MPI::Vector V_old(V);

    // Build forcing combo: b = θF^{n+1} + (1-θ)F^n
    TrilinosWrappers::MPI::Vector rhs(V); // correct partitioning
    rhs = 0.0;
    rhs.add(theta, F_np1);
    rhs.add(1.0 - theta, F_n);

    // Add scheme terms: + C_RHS^(V) * V^n
    rhs_matrix.vmult_add(rhs, V_old);

    // Add: -K * U^n
    TrilinosWrappers::MPI::Vector KU(U);
    K.vmult(KU, U);
    rhs.add(-1.0, KU);

    // Solve: C_LHS * V^{n+1} = rhs
    SolverControl                           solver_control(1000, 1e-6 * rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    // Restore the original LHS before applying boundary conditions (apply_boundary_values modifies A)
    lhs_matrix.copy_from(lhs_matrix_base);

    TrilinosWrappers::MPI::Vector V_new(V);

    // Apply Dirichlet BC on v (for g=0 => v=0 on boundary), exactly like in Heat
    MatrixTools::apply_boundary_values(boundary_values_v, lhs_matrix, V_new, rhs, false);

    AssertThrow(lhs_matrix.frobenius_norm() > 0.0,
                ExcMessage("LHS matrix is zero: did you call initialize() after assembling M,K?"));

    TrilinosWrappers::PreconditionSSOR preconditioner;
    preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    solver.solve(lhs_matrix, V_new, rhs, preconditioner);

    // Update V^{n+1} in-place
    V = V_new;

    // Update displacement U^{n+1} = U^n + dt[(1-theta) V^n + theta V^{n+1}]
    U.add(dt * (1.0 - theta), V_old);
    U.add(dt * theta, V_new);
}

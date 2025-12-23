#include "time_integrator/theta_integrator.hpp"

#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>

#include "Wave.hpp"

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
}


void ThetaIntegrator::advance(const double                                     t_n,
                              const double                                     dt,
                              const TrilinosWrappers::SparseMatrix            &M,
                              const TrilinosWrappers::SparseMatrix            &K,
                              const TrilinosWrappers::MPI::Vector             &F_n,
                              const TrilinosWrappers::MPI::Vector             &F_np1,
                              const AffineConstraints<>                       &constraints_v_np1,
                              const std::map<types::global_dof_index, double> &v_boundary_values,
                              TrilinosWrappers::MPI::Vector                   &U,
                              TrilinosWrappers::MPI::Vector                   &V) {
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
    SolverControl solver_control(5000, std::max(1e-12, 1e-8 * rhs.l2_norm()));
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
    // Set maximum number of temporary vectors for GMRES
    gmres_data.max_n_tmp_vectors = 50;
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control, gmres_data);

    // Solve for V^{n+1}
    AssertThrow(lhs_matrix.frobenius_norm() > 0.0,
                ExcMessage("LHS matrix is zero: did you call initialize() after assembling M,K?"));

    // Copy lhs_matrix to TrilinosWrappers::SparseMatrix A
    TrilinosWrappers::SparseMatrix A;
    A.reinit(lhs_matrix);
    A.copy_from(lhs_matrix);

    // Create vectors for the linear system, where b is the right-hand side
    TrilinosWrappers::MPI::Vector b(rhs);
    // And solution vector V^{n+1}
    TrilinosWrappers::MPI::Vector V_new(V);

    // Initialize preconditioner
    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize(A, TrilinosWrappers::PreconditionAMG::AdditionalData(1.0));

    // Seed Dirichlet values into the initial guess for V^{n+1}
    for (const auto &[dof, val]: v_boundary_values)
        if (V_new.locally_owned_elements().is_element(dof))
            V_new[dof] = val;
    V_new.compress(VectorOperation::insert);

    // Impose Dirichlet on the linear system for V^{n+1} and eliminate boundary rows
    MatrixTools::apply_boundary_values(v_boundary_values,
                                       A,
                                       V_new,
                                       b,
                                       /*eliminate_columns=*/false); // `true` not implemented

    // Solve the linear system
    solver.solve(A, V_new, b, preconditioner);

    // Distribute constraints to V^{n+1}, i.e., set Dirichlet values
    constraints_v_np1.distribute(V_new);

    // Enforce Dirichlet values again (in case some were modified by the solver)
    for (const auto &[dof, val]: v_boundary_values)
        if (V_new.locally_owned_elements().is_element(dof))
            V_new[dof] = val;
    V_new.compress(VectorOperation::insert);

    // Update V^{n+1} in-place
    V = V_new;

    // Update displacement U^{n+1} = U^n + dt[(1-theta) V^n + theta V^{n+1}]
    U.add(dt * (1.0 - theta), V_old);
    U.add(dt * theta, V_new);
}

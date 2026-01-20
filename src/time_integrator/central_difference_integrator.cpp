#include "time_integrator/central_difference_integrator.hpp"

void CentralDifferenceIntegrator::initialize(const TrilinosWrappers::SparseMatrix &M,
                                             const TrilinosWrappers::SparseMatrix &K,
                                             const TrilinosWrappers::MPI::Vector  &U0,
                                             const TrilinosWrappers::MPI::Vector  &V0,
                                             const double                          dt) {
    (void) M;
    (void) K;
    (void) dt;

    // allocate vectors with correct layout
    U_prev.reinit(U0);
    A.reinit(U0);
    tmp_owned.reinit(U0);

    // We cannot compute U^{-1} here because we don't have F(0).
    // We'll compute A0 at the first advance() and then build U_prev.
    first_step = true;

    // silence unused warnings for now
    (void) V0;
}

void CentralDifferenceIntegrator::advance(
        const double                                     t_n,
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
        TrilinosWrappers::MPI::Vector                   &V) {
    (void) t_n;
    (void) F_np1;

    // ---- 1) compute acceleration A^n from: M A^n = F^n - K U^n
    TrilinosWrappers::MPI::Vector rhs(U);
    rhs = 0.0;

    // rhs = F^n
    rhs.add(1.0, F_n);

    // rhs -= K U
    tmp_owned.reinit(U);
    K.vmult(tmp_owned, U);
    rhs.add(-1.0, tmp_owned);

    // Solve M A = rhs
    TrilinosWrappers::SparseMatrix M_sys;
    M_sys.reinit(M);
    M_sys.copy_from(M);

    TrilinosWrappers::MPI::Vector A_new(A);
    A_new = 0.0;

    // Apply Dirichlet (use displacement boundary dofs as "fixed" for A too)
    TrilinosWrappers::MPI::Vector b(rhs);
    MatrixTools::apply_boundary_values(
            u_boundary_values, M_sys, A_new, b, /*eliminate_columns=*/false);

    SolverControl solver_control(5000, std::max(1e-12, 1e-10 * b.l2_norm()));
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    TrilinosWrappers::PreconditionAMG prec;
    prec.initialize(M_sys, TrilinosWrappers::PreconditionAMG::AdditionalData(1.0));

    solver.solve(M_sys, A_new, b, prec);

    // distribute not strictly needed for A, but keep it consistent
    constraints_u_np1.distribute(A_new);

    A = A_new;

    // ---- 2) first step: build U^{-1} using Taylor: U^{-1} = U^0 - dt V^0 + 0.5 dt^2 A^0
    if (first_step) {
        U_prev = U;
        U_prev.add(-dt, V);
        U_prev.add(0.5 * dt * dt, A);

        first_step = false;
    }

    // ---- 3) Central difference update:
    // U^{n+1} = 2U^n - U^{n-1} + dt^2 A^n
    TrilinosWrappers::MPI::Vector U_new(U);
    U_new.sadd(2.0, -1.0, U_prev); // U_new = 2U - U_prev
    U_new.add(dt * dt, A); // + dt^2 A

    // ---- 4) Velocity (centered): V^{n+1} = (U^{n+1} - U^{n-1}) / (2dt)
    TrilinosWrappers::MPI::Vector V_new(V);
    V_new = U_new;
    V_new.add(-1.0, U_prev);
    V_new *= (1.0 / (2.0 * dt));

    // ---- 5) shift history
    U_prev = U; // old U becomes U^{n-1} for next step
    U      = U_new;
    V      = V_new;

    // Enforce Dirichlet on u and v
    constraints_u_np1.distribute(U);
    constraints_v_np1.distribute(V);

    // Force exact boundary values (robustness)
    for (const auto &[dof, val]: u_boundary_values)
        if (U.locally_owned_elements().is_element(dof))
            U[dof] = val;
    U.compress(VectorOperation::insert);

    for (const auto &[dof, val]: v_boundary_values)
        if (V.locally_owned_elements().is_element(dof))
            V[dof] = val;
    V.compress(VectorOperation::insert);
}

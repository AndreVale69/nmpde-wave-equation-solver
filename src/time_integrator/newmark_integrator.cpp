#include "time_integrator/newmark_integrator.hpp"

void NewmarkIntegrator::initialize(const TrilinosWrappers::SparseMatrix &M,
                                   const TrilinosWrappers::SparseMatrix &K,
                                   const TrilinosWrappers::MPI::Vector  &U0,
                                   const TrilinosWrappers::MPI::Vector  &V0,
                                   const double                          dt)
{
    (void)V0;

    A.reinit(U0);
    A = 0.0;

    tmp_owned.reinit(U0);

    // Build K_eff = K + (1/(beta dt^2)) M
    K_eff.reinit(K);
    K_eff.copy_from(K);
    K_eff.add(1.0 / (beta * dt * dt), M);

    first_step = true;
}

void NewmarkIntegrator::advance(const double                                     t_n,
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
                                TrilinosWrappers::MPI::Vector                   &V)
{
    (void)t_n;

    // ---- 0) If first step, compute A0 from: M A0 = F0 - K U0
    if (first_step) {
        TrilinosWrappers::MPI::Vector rhs(U);
        rhs = 0.0;
        rhs.add(1.0, F_n);

        tmp_owned.reinit(U);
        K.vmult(tmp_owned, U);
        rhs.add(-1.0, tmp_owned);

        TrilinosWrappers::SparseMatrix M_sys;
        M_sys.reinit(M);
        M_sys.copy_from(M);

        TrilinosWrappers::MPI::Vector A0(A);
        A0 = 0.0;

        TrilinosWrappers::MPI::Vector b(rhs);
        MatrixTools::apply_boundary_values(u_boundary_values, M_sys, A0, b, /*eliminate_columns=*/false);

        SolverControl solver_control(5000, std::max(1e-12, 1e-10 * b.l2_norm()));
        SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

        TrilinosWrappers::PreconditionAMG prec;
        prec.initialize(M_sys, TrilinosWrappers::PreconditionAMG::AdditionalData(1.0));

        solver.solve(M_sys, A0, b, prec);

        constraints_u_np1.distribute(A0);
        A = A0;

        first_step = false;
    }

    // ---- 1) Solve for U^{n+1}:
    // K_eff U_{n+1} = F_{n+1} + M * [ (1/(beta dt^2)) U_n + (1/(beta dt)) V_n + (1/(2beta)-1) A_n ]
    TrilinosWrappers::MPI::Vector rhs(U);
    rhs = 0.0;
    rhs.add(1.0, F_np1);

    // tmp = (1/(beta dt^2)) U + (1/(beta dt)) V + (1/(2beta)-1) A
    TrilinosWrappers::MPI::Vector tmp(U);
    tmp = U;
    tmp *= (1.0 / (beta * dt * dt));

    tmp_owned.reinit(U);
    tmp_owned = V;
    tmp_owned *= (1.0 / (beta * dt));
    tmp.add(1.0, tmp_owned);

    tmp_owned = A;
    tmp_owned *= ((1.0 / (2.0 * beta)) - 1.0);
    tmp.add(1.0, tmp_owned);

    // rhs += M * tmp
    tmp_owned.reinit(U);
    M.vmult(tmp_owned, tmp);
    rhs.add(1.0, tmp_owned);

    // System solve: K_eff U_new = rhs with Dirichlet on u at n+1
    TrilinosWrappers::SparseMatrix A_sys;
    A_sys.reinit(K_eff);
    A_sys.copy_from(K_eff);

    TrilinosWrappers::MPI::Vector b(rhs);
    TrilinosWrappers::MPI::Vector U_new(U);

    // seed boundary values into initial guess
    for (const auto &[dof, val] : u_boundary_values)
        if (U_new.locally_owned_elements().is_element(dof))
            U_new[dof] = val;
    U_new.compress(VectorOperation::insert);

    MatrixTools::apply_boundary_values(u_boundary_values, A_sys, U_new, b, /*eliminate_columns=*/false);

    SolverControl solver_control(8000, std::max(1e-12, 1e-9 * b.l2_norm()));
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    TrilinosWrappers::PreconditionAMG prec;
    prec.initialize(A_sys, TrilinosWrappers::PreconditionAMG::AdditionalData(1.0));

    solver.solve(A_sys, U_new, b, prec);

    constraints_u_np1.distribute(U_new);

    // ---- 2) Update acceleration A^{n+1}
    // A_{n+1} = (1/(beta dt^2)) (U_{n+1}-U_n - dt V_n) - (1/(2beta)-1) A_n
    TrilinosWrappers::MPI::Vector A_new(A);
    A_new = U_new;
    A_new.add(-1.0, U);
    tmp_owned = V;
    tmp_owned *= dt;
    A_new.add(-1.0, tmp_owned);
    A_new *= (1.0 / (beta * dt * dt));

    tmp_owned = A;
    tmp_owned *= ((1.0 / (2.0 * beta)) - 1.0);
    A_new.add(-1.0, tmp_owned);

    // ---- 3) Update velocity V^{n+1}
    // V_{n+1} = V_n + dt[(1-gamma)A_n + gamma A_{n+1}]
    TrilinosWrappers::MPI::Vector V_new(V);
    tmp_owned = A;
    tmp_owned *= (1.0 - gamma);
    V_new.add(dt, tmp_owned);

    tmp_owned = A_new;
    tmp_owned *= gamma;
    V_new.add(dt, tmp_owned);

    // ---- 4) Commit and enforce v Dirichlet too
    U = U_new;
    V = V_new;
    A = A_new;

    constraints_u_np1.distribute(U);
    constraints_v_np1.distribute(V);

    for (const auto &[dof, val] : u_boundary_values)
        if (U.locally_owned_elements().is_element(dof))
            U[dof] = val;
    U.compress(VectorOperation::insert);

    for (const auto &[dof, val] : v_boundary_values)
        if (V.locally_owned_elements().is_element(dof))
            V[dof] = val;
    V.compress(VectorOperation::insert);
}

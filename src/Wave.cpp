#include "Wave.hpp"

void Wave::setup()
{
  std::cout << "===============================================" << std::endl;
  std::cout << "Initializing Wave Equation Solver" << std::endl;
  std::cout << "===============================================" << std::endl;

  // Create or load the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;

    if (mesh_file_name.empty())
    {
      // Use built-in mesh generator
      std::cout << "  Generating built-in mesh with " << N_subdivisions
                << " subdivisions" << std::endl;

      // Generate a hypercube mesh then convert to a simplex mesh for FE_SimplexP
      Triangulation<dim> hypercube_mesh;
      GridGenerator::subdivided_hyper_cube(hypercube_mesh, N_subdivisions, 0.0, 1.0);
      GridGenerator::convert_hypercube_to_simplex_mesh(hypercube_mesh, mesh);
    }
    else
    {
      // Read mesh from file
      std::cout << "  Reading mesh from file: " << mesh_file_name << std::endl;
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh);

      std::ifstream grid_in_file(mesh_file_name);
      grid_in.read_msh(grid_in_file);
    }
    std::cout << "  Number of elements = " << mesh.n_active_cells() << std::endl;

    // Write mesh to file for visualization
    if (mesh_file_name.empty())
    {
      std::ofstream mesh_out("../mesh/generated_mesh.vtk");
      if (mesh_out)
      {
        GridOut grid_out;
        grid_out.write_vtk(mesh, mesh_out);
        std::cout << "  Mesh saved to ../mesh/generated_mesh.vtk" << std::endl;
      }
      else
      {
        std::cout << "  Warning: Could not write mesh file (directory may not exist)"
                  << std::endl;
      }
    }
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

    // Quadrature for simplex elements: labs commonly use r+1; this avoids under-integration
    // while ensuring the rule is valid for the chosen degree.
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 2);

    std::cout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity.copy_from(dsp);
    // std::cout << "  Sparsity pattern created with " << sparsity.n_rows() << " rows and " << sparsity.n_cols() << " cols" << std::endl;

    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);

    system_rhs.reinit(dof_handler.n_dofs());

    solution.reinit(dof_handler.n_dofs());
    velocity.reinit(dof_handler.n_dofs());
    acceleration.reinit(dof_handler.n_dofs());
    
  }

  // Initialize timing
  assembly_time = 0.0;
  solver_time = 0.0;
  output_time = 0.0;

  std::cout << "===============================================" << std::endl;
}

void Wave::enforce_boundary_conditions_on_vector(Vector<double> &vec)
{
  // Get boundary values once
  std::map<types::global_dof_index, double> boundary_values;

  std::map<types::boundary_id, const Function<dim> *> boundary_functions;
  for (unsigned int b = 0; b < dim * 2; ++b)
    boundary_functions[b] = &function_g;

  VectorTools::interpolate_boundary_values(dof_handler,
                                           boundary_functions,
                                           boundary_values);

  // Zero out boundary values
  for (const auto &pair : boundary_values)
    vec(pair.first) = 0.0;
}

void Wave::assemble_matrices()
{
  // std::cout << "===============================================" << std::endl;
  std::cout << "Assembling the mass and stiffness matrices" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  
  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);


  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {

    fe_values.reinit(cell);

    cell_mass_matrix = 0.0;
    cell_stiffness_matrix = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // Mass matrix: M_ij = ∫ φ_i φ_j dx
          cell_mass_matrix(i, j) +=
              fe_values.shape_value(i, q) * fe_values.shape_value(j, q) *
              fe_values.JxW(q);

          // Stiffness matrix: K_ij = ∫ ∇φ_i · ∇φ_j dx
          cell_stiffness_matrix(i, j) +=
              fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) *
              fe_values.JxW(q);
        }
      }
    }
    cell->get_dof_indices(dof_indices);
    mass_matrix.add(dof_indices, cell_mass_matrix);
    stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
  }

  std::cout << "  Assembling LHS matrix for Newmark method" << std::endl;
  // For Newmark method, we solve: K_eff·a_{n+1} = RHS where K_eff = M + β·Δt²·K
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(beta * deltat * deltat, stiffness_matrix);

  // Apply homogeneous Dirichlet BCs ONCE to the LHS matrix, since BCs are time-independent.
  // Note: we apply boundary values only to the matrix here; per-step RHS will be zeroed on
  // boundary DOFs to maintain u=0 consistently without reapplying matrix BCs each timestep.
  {
    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    for (unsigned int b = 0; b < dim * 2; ++b)
      boundary_functions[b] = &function_g;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    // Use temporary vectors to modify only the matrix structure.
    Vector<double> tmp_unknown(dof_handler.n_dofs());
    Vector<double> tmp_rhs(dof_handler.n_dofs());
    MatrixTools::apply_boundary_values(boundary_values,
                                       lhs_matrix,
                                       tmp_unknown,
                                       tmp_rhs,
                                       false);
  }

  auto end = std::chrono::high_resolution_clock::now();
  assembly_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

  std::cout << "  Mass matrix Frobenius norm:      " << std::scientific << mass_matrix.frobenius_norm() << std::endl;
  std::cout << "  Stiffness matrix Frobenius norm: " << stiffness_matrix.frobenius_norm() << std::endl;
  std::cout << "  LHS matrix Frobenius norm:       " << lhs_matrix.frobenius_norm() << std::endl;
  std::cout << "===============================================" << std::endl;
}

void Wave::assemble_rhs(const double &time)
{
  auto start = std::chrono::high_resolution_clock::now();

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points | update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Assemble forcing term contribution
  system_rhs = 0.0;
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      // Evaluate forcing term at current time
      forcing_term.set_time(time);
      const double f_loc = forcing_term.value(fe_values.quadrature_point(q));

      // Check for NaN/Inf in forcing term
      if (!std::isfinite(f_loc))
      {
        std::cout << "    ERROR: Forcing term is " << (std::isnan(f_loc) ? "NaN" : "Inf")
                  << " at point (" << fe_values.quadrature_point(q) << ")" << std::endl;
        throw std::runtime_error("Non-finite forcing term");
      }

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        cell_rhs(i) += f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }
    cell->get_dof_indices(dof_indices);
    system_rhs.add(dof_indices, cell_rhs);
  }

  // Add contribution from predicted displacement
  // RHS = f - K·u_pred, where u_pred = u^n + Δt·v^n + Δt²·(0.5-β)·a^n
  Vector<double> u_pred(dof_handler.n_dofs());

  u_pred = solution;
  u_pred.add(deltat, velocity);
  u_pred.add(deltat * deltat * (0.5 - beta), acceleration);

  Vector<double> tmp(dof_handler.n_dofs());
  stiffness_matrix.vmult(tmp, u_pred);
  system_rhs.add(-1.0, tmp); // RHS = f - K·u_pred

  // Zero boundary RHS entries to be consistent with LHS elimination (u=0 on boundary).
  enforce_boundary_conditions_on_vector(system_rhs);

  auto end = std::chrono::high_resolution_clock::now();
  assembly_time +=std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

}

void Wave::solve_time_step()
{
  // std::cout << "  Solving for acceleration at time " << time << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  // ===== DIAGNOSTICS =====
  const double lhs_norm = lhs_matrix.frobenius_norm();
  const double rhs_norm = system_rhs.l2_norm();

  // std::cout << "    LHS matrix Frobenius norm: " << std::scientific << lhs_norm << std::endl;
  // std::cout << "    RHS vector L2 norm: " << rhs_norm << std::endl;

  // Check for NaNs or Infs
  if (!std::isfinite(lhs_norm) || !std::isfinite(rhs_norm))
  {
    std::cout << "    ERROR: Matrix or RHS contains NaN or Inf!" << std::endl;
    throw std::runtime_error("Non-finite values in system");
  }

  // Check diagonal of matrix (should be non-zero)
  bool has_zero_diagonal = false;
  for (unsigned int i = 0; i < lhs_matrix.m(); ++i)
  {
    if (std::abs(lhs_matrix.diag_element(i)) < 1e-14)
    {
      has_zero_diagonal = true;
      break;
    }
  }

  if (has_zero_diagonal)
  {
    std::cout << "    WARNING: Matrix has zero or near-zero diagonal elements!" << std::endl;
    std::cout << "    This may indicate boundary conditions were not applied correctly." << std::endl;
  }

  // BCs are time-independent; LHS already has BCs applied at assembly.
  // We only zero boundary entries on RHS per step in assemble_rhs().

  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);

  // Preconditioner on LHS matrix
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(lhs_matrix, 1.0);

  // Solve for acceleration: (M+βΔt²K)·a_{n+1} = RHS
  solver.solve(lhs_matrix, acceleration, system_rhs, preconditioner);

  // Diagnostic: Check acceleration after solving
  // const double acc_norm = acceleration.l2_norm();
  // std::cout << " | a norm: " << acc_norm << " | iter: " << solver_control.last_step();

  if (solver_control.last_step() >= 999)
  {
    std::cout << "  WARNING: Linear solver did not converge! ("
              << solver_control.last_step() << " iterations)" << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  solver_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

}

void Wave::project_initial_acceleration()
{

  if (use_manufactured_solution)
  {
    // Directly use exact acceleration from manufactured solution

    exact_acceleration.set_time(0.0);
    VectorTools::interpolate(dof_handler, exact_acceleration, acceleration);

    enforce_boundary_conditions_on_vector(acceleration);

    // std::cout << "===============================================" << std::endl;
  }
  else
  {
    // Project from PDE: M·a_0 = f(x,0) - K·u_0
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_quadrature_points |
                                update_JxW_values);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_rhs = 0.0;

    // Assemble forcing term
    for (const auto &cell : dof_handler.active_cell_iterators())
    {

      fe_values.reinit(cell);
      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
      {
        forcing_term.set_time(0.0);
        const double f_loc = forcing_term.value(fe_values.quadrature_point(q));

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          cell_rhs(i) +=
              f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
        }
      }

      cell->get_dof_indices(dof_indices);
      system_rhs.add(dof_indices, cell_rhs);
    }

    // Subtract K·u_0
    Vector<double> tmp(dof_handler.n_dofs());
    stiffness_matrix.vmult(tmp, solution);
    system_rhs.add(-1.0, tmp);

    // Apply Dirichlet BCs in-place on mass system and solve M·a_0 = RHS
    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    for (unsigned int b = 0; b < dim * 2; ++b)
      boundary_functions[b] = &function_g;
    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions, boundary_values);

    MatrixTools::apply_boundary_values(boundary_values, mass_matrix, acceleration, system_rhs, false);

    SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

    SolverCG<Vector<double>> solver(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(mass_matrix, 1.0);

    solver.solve(mass_matrix, acceleration, system_rhs, preconditioner);

    std::cout << "    " << solver_control.last_step() << " CG iterations" << std::endl;
    enforce_boundary_conditions_on_vector(acceleration);
  }
}

void Wave::output(const unsigned int &time_step)
{
  auto start = std::chrono::high_resolution_clock::now();

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "displacement");
  data_out.add_data_vector(dof_handler, velocity, "velocity");
  data_out.add_data_vector(dof_handler, acceleration, "acceleration");

  // std::vector<unsigned int> partition_int(mesh.n_active_cells());
  // GridTools::get_subdomain_association(mesh, partition_int);
  // const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  // data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::ofstream output_file("./output-" + std::to_string(time_step) + ".vtu");
  data_out.write_vtu(output_file);

  auto end = std::chrono::high_resolution_clock::now();
  output_time +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
}

void Wave::write_time_series(const unsigned int &time_step)
{
  const double energy = compute_energy();
  const double power  = compute_power_input(time);
  const double dE_dt  = (time_step == 0) ? 0.0 : (energy - previous_energy) / deltat;
  const double residual = dE_dt - power; // For homogeneous Dirichlet and undamped dynamics, ideally ~0
  // Trapezoidal integration for accumulated work: W_n = W_{n-1} + Δt * (P_n + P_{n-1}) / 2
  if (time_step == 0)
    accumulated_work = 0.0;
  else
    accumulated_work += deltat * (power + previous_power) * 0.5;
  const double energy_balance = energy - initial_energy - accumulated_work;
  const double eL2_u = use_manufactured_solution
                           ? compute_error(VectorTools::L2_norm, false)
                           : 0.0;
  // std::cout << "    eL2_u computed " << eL2_u << std::endl;
  const double eH1_u = use_manufactured_solution
                           ? compute_error(VectorTools::H1_norm, false)
                           : 0.0;
  // std::cout << "    eH1_u computed " << eH1_u << std::endl;
  const double eL2_v = use_manufactured_solution
                           ? compute_error(VectorTools::L2_norm, true)
                           : 0.0;

  if (time_step == 0)
  {
    // std::cout << "    Creating new CSV file for time-series data" << std::endl;
    csv_file.open("../results/time_series.csv");
    csv_file << "time_step,time,energy,power,dE_dt,residual,work,energy_balance,eL2_u,eH1_u,eL2_v" << std::endl;
    previous_energy = energy;
    previous_power  = power;
    initial_energy  = energy;
  }

  csv_file << time_step << "," << time << "," << energy << "," << power << "," << dE_dt
           << "," << residual << "," << accumulated_work << "," << energy_balance
           << "," << eL2_u << "," << eH1_u << "," << eL2_v << std::endl;

  previous_energy = energy;
  previous_power  = power;

  // Check energy conservation
  // if (time_step == 0)
  // {
  //   initial_energy = energy;
  // }
  // else if (initial_energy > 1e-14)
  // {
  //   const double energy_drift = std::abs(energy - initial_energy) / initial_energy;
  //   if (energy_drift > 0.01)
  //   {
  //     std::cout << "  WARNING: Energy drift = " << energy_drift * 100.0
  //           << "% (exceeds 1% threshold)" << std::endl;
  //   }
  // }
}

double Wave::compute_error(const VectorTools::NormType &norm_type, const bool for_velocity)
{
  // std::cout << "  Computing error ("
  //       << (for_velocity ? "velocity" : "displacement") << ") in "
  //       << (norm_type == VectorTools::L2_norm ? "L2 norm" : "H1 norm") << std::endl;

  FE_SimplexP<dim> fe_linear(1);
  MappingFE mapping(fe_linear);

  const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(r + 2);

  Vector<double> error_per_cell(mesh.n_active_cells());

  if (for_velocity)
  {
    exact_velocity.set_time(time);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      velocity,
                                      exact_velocity,
                                      error_per_cell,
                                      quadrature_error,
                                      norm_type);
  }
  else
  {
    exact_solution.set_time(time);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      exact_solution,
                                      error_per_cell,
                                      quadrature_error,
                                      norm_type);
  }

  double error = error_per_cell.l2_norm();

  return error;
}

double Wave::compute_energy()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients | update_JxW_values);

  std::vector<double> velocity_values(n_q);
  std::vector<Tensor<1, dim>> displacement_gradients(n_q);

  double local_kinetic_energy = 0.0;
  double local_potential_energy = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {

    fe_values.reinit(cell);

    fe_values.get_function_values(velocity, velocity_values);
    fe_values.get_function_gradients(solution, displacement_gradients);

    for (unsigned int q = 0; q < n_q; ++q)
    {
      local_kinetic_energy += 0.5 * velocity_values[q] * velocity_values[q] * fe_values.JxW(q);

      local_potential_energy += 0.5 * displacement_gradients[q] *
                                displacement_gradients[q] * fe_values.JxW(q);
    }
  }

  // std::cout << " | Kinetic Energy: " << local_kinetic_energy << " | Potential Energy: " << local_potential_energy << std::endl;

  return local_kinetic_energy + local_potential_energy;
}

double Wave::compute_power_input(const double &time)
{
  // Computes P(t) = ∫ f(x,t) v(x,t) dx over the domain
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_quadrature_points | update_JxW_values | update_values);

  std::vector<double> velocity_values(n_q);
  double power = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    // Evaluate velocity field at quadrature points
    fe_values.get_function_values(velocity, velocity_values);

    // Accumulate f(x,t) * v(x,t)
    for (unsigned int q = 0; q < n_q; ++q)
    {
      forcing_term.set_time(time);
      const double f_loc = forcing_term.value(fe_values.quadrature_point(q));
      power += f_loc * velocity_values[q] * fe_values.JxW(q);
    }
  }

  return power;
}

void Wave::solve()
{
  std::cout << "Starting time integration" << std::endl;
  std::cout << "  Newmark parameters: beta = " << beta << ", gamma = " << gamma
            << std::endl;
  std::cout << "  Time step: " << deltat << std::endl;
  std::cout << "  Final time: " << T << std::endl;
  std::cout << "  Number of time steps: " << static_cast<unsigned int>(T / deltat)
            << std::endl;
  std::cout << "===============================================" << std::endl;

  // Assemble constant matrices
  assemble_matrices();

  // Apply initial conditions
  {
    // std::cout << "Applying initial conditions" << std::endl;

    const Function<dim> *u_init;
    const Function<dim> *v_init;

    if (use_manufactured_solution)
    {
      exact_solution.set_time(0.0);
      exact_velocity.set_time(0.0);
      u_init = &exact_solution;
      v_init = &exact_velocity;
    }
    else
    {
      u_init = &u_0;
      v_init = &v_0;
    }

    VectorTools::interpolate(dof_handler, *u_init, solution);
    VectorTools::interpolate(dof_handler, *v_init, velocity);

    enforce_boundary_conditions_on_vector(velocity);
    enforce_boundary_conditions_on_vector(solution);

    // Compute or project initial acceleration
    project_initial_acceleration();

    // Write initial output
    if (output_frequency == 0)
    {
      std::cout << "Writing initial output at time t=0.0" << std::endl;
      output(0);
    }
    write_time_series(0);
  }

  unsigned int time_step = 0;
  time = 0.0;

  SolverControl solver_control_for_output(1000, 1e-6);

  // Time stepping loop
  while (time < T - 1e-10)
  {
    time += deltat;
    ++time_step;

    std::cout << "Time step " << std::setw(4) << time_step << " | t = " << std::scientific << std::setprecision(4) << time << std::endl;

    // Save old state for corrector step
    Vector<double> acceleration_old(dof_handler.n_dofs());
    Vector<double> velocity_old(dof_handler.n_dofs());
    Vector<double> solution_old(dof_handler.n_dofs());
    acceleration_old = acceleration;
    velocity_old = velocity;
    solution_old = solution;

    // Assemble RHS (includes predictor for displacement)
    assemble_rhs(time);

    // Solve for new acceleration
    solve_time_step();

    // Corrector: update velocity
    // v_{n+1} = v^n + Δt·[(1-γ)·a^n + γ·a_{n+1}]
    velocity = velocity_old; // Start from v^n
    velocity.add(deltat * (1.0 - gamma), acceleration_old);
    velocity.add(deltat * gamma, acceleration);

    // Corrector: update displacement
    // u_{n+1} = u^n + Δt·v^n + Δt²·[(0.5-β)·a^n + β·a_{n+1}]
    solution = solution_old; // Start from u^n
    solution.add(deltat, velocity_old);
    solution.add(deltat * deltat * (0.5 - beta), acceleration_old);
    solution.add(deltat * deltat * beta, acceleration);


    enforce_boundary_conditions_on_vector(solution);
    enforce_boundary_conditions_on_vector(velocity);
    enforce_boundary_conditions_on_vector(acceleration);

    // Write output
    if (output_frequency == 0 || time_step % output_frequency == 0)
      output(time_step);

    write_time_series(time_step);

    // std::cout << std::endl;
  }

  // Close CSV file
  csv_file.close();

  // Print timing summary
  const double total_time = assembly_time + solver_time + output_time;
  std::cout << "===============================================" << std::endl;
  std::cout << "Timing summary:" << std::endl;
  std::cout << "  Assembly:     " << std::fixed << std::setprecision(2) << assembly_time
            << " s (" << std::setw(5) << std::setprecision(1)
            << 100.0 * assembly_time / total_time << "%)" << std::endl;
  std::cout << "  Linear solve: " << std::fixed << std::setprecision(2) << solver_time
            << " s (" << std::setw(5) << std::setprecision(1) << 100.0 * solver_time / total_time
            << "%)" << std::endl;
  std::cout << "  Output:       " << std::fixed << std::setprecision(2) << output_time
            << " s (" << std::setw(5) << std::setprecision(1) << 100.0 * output_time / total_time
            << "%)" << std::endl;
  std::cout << "  Total:        " << std::fixed << std::setprecision(2) << total_time << " s"
            << std::endl;
  std::cout << "===============================================" << std::endl;
}
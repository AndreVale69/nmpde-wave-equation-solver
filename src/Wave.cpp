#include "Wave.hpp"

void Wave::setup()
{
  pcout << "===============================================" << std::endl;
  pcout << "Initializing Wave Equation Solver" << std::endl;
  pcout << "===============================================" << std::endl;

  // Create or load the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    if (mesh_file_name.empty())
    {
      // Use built-in mesh generator
      pcout << "  Generating built-in mesh with " << N_subdivisions
            << " subdivisions" << std::endl;
      
      // Option 1: Square mesh with simplices (default)
      GridGenerator::subdivided_hyper_cube(mesh_serial, N_subdivisions, 0.0, 1.0);

      // Option 2: Circular mesh 
      // const Point<dim> center(0.5, 0.5);
      // const double radius = 1.0;
      // GridGenerator::hyper_ball(mesh_serial, center, radius);
      // mesh_serial.set_all_manifold_ids(0);
      // SphericalManifold<dim> boundary_manifold(center);
      // mesh_serial.set_manifold(0, boundary_manifold);
      // mesh_serial.refine_global(N_subdivisions - 5);  // Adjust refinement level
      
      
      // Convert to simplex mesh
      Triangulation<dim> simplex_mesh;
      GridGenerator::convert_hypercube_to_simplex_mesh(mesh_serial, simplex_mesh);
      mesh_serial.clear();
      mesh_serial.copy_triangulation(simplex_mesh);
    }
    else
    {
      // Read mesh from file
      pcout << "  Reading mesh from file: " << mesh_file_name << std::endl;
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);
      std::ifstream grid_in_file(mesh_file_name);
      grid_in.read_msh(grid_in_file);
    }

    // Partition for MPI
    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;

    // Write mesh to file for visualization
    if (mpi_rank == 0 && mesh_file_name.empty())
    {
      std::ofstream mesh_out("../mesh/generated_mesh.vtk");
      if (mesh_out)
      {
        GridOut grid_out;
        grid_out.write_vtk(mesh_serial, mesh_out);
        pcout << "  Mesh saved to ../mesh/generated_mesh.vtk" << std::endl;
      }
      else
      {
        pcout << "  Warning: Could not write mesh file (directory may not exist)"
              << std::endl;
      }
    }
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

    // Use higher quadrature order for accurate integration of high-frequency solutions
    // QGaussSimplex for 2D triangles supports up to order 5
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    pcout << "  Locally owned DoFs = " << locally_owned_dofs.n_elements() << std::endl;

  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize constraints
  {
    pcout << "Initializing constraints" << std::endl;
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    
    // Add homogeneous Dirichlet BCs on all boundaries
    Functions::ZeroFunction<dim> zero;
    for (unsigned int b = 0; b < 4; ++b) // 2D: 4 boundaries
      VectorTools::interpolate_boundary_values(dof_handler, b, zero, constraints);
    
    constraints.close();
    pcout << "  Constrained DoFs = " << constraints.n_constraints() << std::endl;
  }

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity, constraints);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);

    pcout << "  Initializing vectors" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    velocity_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    velocity.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    acceleration_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    acceleration.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }

  // Initialize timing
  assembly_time = 0.0;
  solver_time = 0.0;
  output_time = 0.0;

  pcout << "===============================================" << std::endl;
}

void Wave::enforce_boundary_conditions_on_vector(TrilinosWrappers::MPI::Vector &vec)
{
  for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
  {
    if (constraints.is_constrained(i) && locally_owned_dofs.is_element(i))
      vec(i) = 0.0;
  }
}

void Wave::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the mass and stiffness matrices" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

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

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // Apply homogeneous Dirichlet boundary conditions (for manufactured solution)
  // We apply BCs ONCE here and never modify the matrix again
  if (use_manufactured_solution)
  {
    std::map<types::global_dof_index, double> boundary_values;

    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    {
      if (constraints.is_constrained(i))
        boundary_values[i] = 0.0;
    }

    TrilinosWrappers::MPI::Vector dummy_solution(locally_owned_dofs, MPI_COMM_WORLD);
    TrilinosWrappers::MPI::Vector dummy_rhs(locally_owned_dofs, MPI_COMM_WORLD);

    dummy_solution = 0.0;
    dummy_rhs = 0.0;

    MatrixTools::apply_boundary_values(
        boundary_values, mass_matrix, dummy_solution, dummy_rhs, true);
    MatrixTools::apply_boundary_values(
        boundary_values, stiffness_matrix, dummy_solution, dummy_rhs, true);

    pcout << "  Applied homogeneous Dirichlet BCs to matrices" << std::endl;
  }

  // For Newmark method, we solve: K_eff·a_{n+1} = RHS where K_eff = M + β·Δt²·K
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(beta * deltat * deltat, stiffness_matrix);

  auto end = std::chrono::high_resolution_clock::now();
  assembly_time +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

  pcout << "  Matrix assembly completed" << std::endl;
  pcout << "  Mass matrix Frobenius norm:      " << std::scientific
        << mass_matrix.frobenius_norm() << std::endl;
  pcout << "  Stiffness matrix Frobenius norm: " << stiffness_matrix.frobenius_norm()
        << std::endl;
  pcout << "  LHS matrix Frobenius norm:       " << lhs_matrix.frobenius_norm()
        << std::endl;
  pcout << "===============================================" << std::endl;
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
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      // Evaluate forcing term at current time
      forcing_term.set_time(time);
      const double f_loc = forcing_term.value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        cell_rhs(i) += f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

    cell->get_dof_indices(dof_indices);
    system_rhs.add(dof_indices, cell_rhs);
  }

  system_rhs.compress(VectorOperation::add);

  // Add contribution from predicted displacement
  // RHS = f - K·u_pred, where u_pred = u^n + Δt·v^n + Δt²·(0.5-β)·a^n
  TrilinosWrappers::MPI::Vector u_pred(locally_owned_dofs, MPI_COMM_WORLD);
  
  u_pred = solution_owned;
  u_pred.add(deltat, velocity_owned);
  u_pred.add(deltat * deltat * (0.5 - beta), acceleration_owned);

  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, MPI_COMM_WORLD);
  stiffness_matrix.vmult(tmp, u_pred);
  system_rhs.add(-1.0, tmp); // RHS = f - K·u_pred

  enforce_boundary_conditions_on_vector(u_pred);
  enforce_boundary_conditions_on_vector(system_rhs);

  auto end = std::chrono::high_resolution_clock::now();
  assembly_time +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
}

void Wave::solve_time_step()
{
  auto start = std::chrono::high_resolution_clock::now();

  // Diagnostic: Check RHS before solving
  const double rhs_norm = system_rhs.l2_norm();
  //pcout << " | RHS norm: " << std::scientific << std::setprecision(2) << rhs_norm;

  SolverControl solver_control(1000, 1e-6 * rhs_norm);
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  // Choose preconditioner: SSOR (default) or AMG (commented, uncomment for large problems)
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(lhs_matrix,
                            TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  // Uncomment for AMG preconditioner (better for large problems):
  // TrilinosWrappers::PreconditionAMG preconditioner;
  // TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  // amg_data.smoother_sweeps = 2;
  // amg_data.aggregation_threshold = 0.02;
  // preconditioner.initialize(lhs_matrix, amg_data);

  // Solve for acceleration: M·a_{n+1} = RHS
  solver.solve(lhs_matrix, acceleration_owned, system_rhs, preconditioner);

  // Diagnostic: Check acceleration after solving
  const double acc_norm = acceleration_owned.l2_norm();
  //pcout << " | a norm: " << acc_norm << " | iter: " << solver_control.last_step();

  if (solver_control.last_step() >= 999)
  {
    pcout << "  WARNING: Linear solver did not converge! ("
          << solver_control.last_step() << " iterations)" << std::endl;
  }
  acceleration = acceleration_owned;

  auto end = std::chrono::high_resolution_clock::now();
  solver_time +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
}

void Wave::project_initial_acceleration()
{
  pcout << "  Projecting initial acceleration from PDE" << std::endl;

  if (use_manufactured_solution)
  {
    // Directly use exact acceleration from manufactured solution
    // Use MappingFE for simplex elements
    const FE_SimplexP<dim> fe_map(1);
    const MappingFE<dim> mapping(fe_map);
    
    exact_acceleration.set_time(0.0);
    VectorTools::project(mapping,
                         dof_handler,
                         constraints,
                         QGaussSimplex<dim>(r + 1),
                         exact_acceleration,
                         acceleration_owned);
    acceleration_owned.compress(VectorOperation::insert);
    acceleration = acceleration_owned;
    acceleration.compress(VectorOperation::insert);

    enforce_boundary_conditions_on_vector(acceleration_owned);
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

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

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

    system_rhs.compress(VectorOperation::add);

    // Subtract K·u_0
    TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, MPI_COMM_WORLD);
    stiffness_matrix.vmult(tmp, solution_owned);
    system_rhs.add(-1.0, tmp);

    // Enforce BCs on RHS
    enforce_boundary_conditions_on_vector(system_rhs);

    // Solve M·a_0 = RHS
    SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
    TrilinosWrappers::PreconditionSSOR preconditioner;
    preconditioner.initialize(mass_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    solver.solve(mass_matrix, acceleration_owned, system_rhs, preconditioner);
    
    pcout << "    " << solver_control.last_step() << " CG iterations" << std::endl;

    acceleration_owned.compress(VectorOperation::insert);
    acceleration = acceleration_owned;
    acceleration.compress(VectorOperation::insert);

    enforce_boundary_conditions_on_vector(acceleration_owned);
  }
}

void Wave::output(const unsigned int &time_step)
{
  auto start = std::chrono::high_resolution_clock::now();

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "displacement");
  data_out.add_data_vector(dof_handler, velocity, "velocity");
  data_out.add_data_vector(dof_handler, acceleration, "acceleration");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", "output", time_step, MPI_COMM_WORLD, 3);

  auto end = std::chrono::high_resolution_clock::now();
  output_time +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
}

void Wave::write_time_series(const unsigned int &time_step)
{
  if (mpi_rank == 0)
  {
    if (time_step == 0)
    {
      csv_file.open("../results/time_series.csv");
      csv_file << "time_step,time,energy,eL2_u,eH1_u,eL2_v" << std::endl;
    }

    const double energy = compute_energy();
    const double eL2_u = use_manufactured_solution
                             ? compute_error(VectorTools::L2_norm, false)
                             : 0.0;
    const double eH1_u = use_manufactured_solution
                             ? compute_error(VectorTools::H1_norm, false)
                             : 0.0;
    const double eL2_v = use_manufactured_solution
                             ? compute_error(VectorTools::L2_norm, true)
                             : 0.0;

    csv_file << time_step << "," << time << "," << energy << "," << eL2_u << ","
             << eH1_u << "," << eL2_v << std::endl;

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
    //     pcout << "  WARNING: Energy drift = " << energy_drift * 100.0
    //           << "% (exceeds 1% threshold)" << std::endl;
    //   }
    // }
  }
}

double Wave::compute_error(const VectorTools::NormType &norm_type, const bool for_velocity)
{
  FE_SimplexP<dim> fe_linear(1);
  MappingFE mapping(fe_linear);

  const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(r + 1);

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

  const double error =
      VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

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
    if (!cell->is_locally_owned())
      continue;

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

  const double global_kinetic_energy =
      Utilities::MPI::sum(local_kinetic_energy, MPI_COMM_WORLD);
  const double global_potential_energy =
      Utilities::MPI::sum(local_potential_energy, MPI_COMM_WORLD);

  pcout << " | Kinetic Energy: " << global_kinetic_energy << " | Potential Energy: " << global_potential_energy;

  return global_kinetic_energy + global_potential_energy;
}

void Wave::solve()
{
  pcout << "===============================================" << std::endl;
  pcout << "Starting time integration" << std::endl;
  pcout << "  Newmark parameters: beta = " << beta << ", gamma = " << gamma
        << std::endl;
  pcout << "  Time step: " << deltat << std::endl;
  pcout << "  Final time: " << T << std::endl;
  pcout << "  Number of time steps: " << static_cast<unsigned int>(T / deltat)
        << std::endl;
  pcout << "===============================================" << std::endl;

  // Assemble constant matrices
  assemble_matrices();

  // Apply initial conditions
  {
    pcout << "Applying initial conditions" << std::endl;

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
    
    // Explicit mapping for simplex elements
    const FE_SimplexP<dim> fe_map(1);
    const MappingFE<dim> mapping(fe_map);

    // Project initial displacement
    VectorTools::project(mapping,
                         dof_handler,
                         constraints,
                         QGaussSimplex<dim>(r + 1),
                         *u_init,
                         solution_owned);
    
    solution_owned.compress(VectorOperation::insert);
    solution = solution_owned;
    solution.compress(VectorOperation::insert);

    // Project initial velocity
    VectorTools::project(mapping,
                         dof_handler,
                         constraints,
                         QGaussSimplex<dim>(r + 1),
                         *v_init,
                         velocity_owned);

    velocity_owned.compress(VectorOperation::insert);
    velocity = velocity_owned;
    velocity.compress(VectorOperation::insert);

    enforce_boundary_conditions_on_vector(solution_owned);
    enforce_boundary_conditions_on_vector(velocity_owned);

    // Compute or project initial acceleration
    project_initial_acceleration();

    // Write initial output
    if (output_frequency == 0)
      output(0);
    write_time_series(0);

    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;
  time = 0.0;

  SolverControl solver_control_for_output(1000, 1e-6);

  // Time stepping loop
  while (time < T - 1e-10)
  {
    time += deltat;
    ++time_step;

    pcout << "Time step " << std::setw(4) << time_step << " | t = " << std::scientific
          << std::setprecision(4) << time << std::flush;

    // Save old acceleration for corrector step
    TrilinosWrappers::MPI::Vector acceleration_old(locally_owned_dofs, MPI_COMM_WORLD);
    acceleration_old = acceleration_owned;
    TrilinosWrappers::MPI::Vector velocity_old(locally_owned_dofs, MPI_COMM_WORLD);
    velocity_old = velocity_owned;
    TrilinosWrappers::MPI::Vector solution_old(locally_owned_dofs, MPI_COMM_WORLD);
    solution_old = solution_owned;

    // Assemble RHS (includes predictor for displacement)
    assemble_rhs(time);

    // Solve for new acceleration
    solve_time_step();

    // Corrector: update velocity
    // v_{n+1} = v^n + Δt·[(1-γ)·a^n + γ·a_{n+1}]
    velocity_owned = velocity_old; // Start from v^n
    velocity_owned.add(deltat * (1.0 - gamma), acceleration_old);
    velocity_owned.add(deltat * gamma, acceleration_owned);

    // velocity = velocity_owned;

    // Corrector: update displacement
    // u_{n+1} = u^n + Δt·v^n + Δt²·[(0.5-β)·a^n + β·a_{n+1}]
    solution_owned = solution_old; // Start from u^n
    solution_owned.add(deltat, velocity_old);
    solution_owned.add(deltat * deltat * (0.5 - beta), acceleration_old);
    solution_owned.add(deltat * deltat * beta, acceleration_owned);

    // solution = solution_owned;

    acceleration_owned.compress(VectorOperation::insert);
    acceleration = acceleration_owned;
    acceleration.compress(VectorOperation::insert);

    velocity_owned.compress(VectorOperation::insert);
    velocity = velocity_owned;
    velocity.compress(VectorOperation::insert);

    solution_owned.compress(VectorOperation::insert);
    solution = solution_owned;
    solution.compress(VectorOperation::insert);

    enforce_boundary_conditions_on_vector(solution_owned);
    enforce_boundary_conditions_on_vector(velocity_owned);

    // Write output
    if (output_frequency == 0 || time_step % output_frequency == 0)
      output(time_step);

    write_time_series(time_step);

    pcout << std::endl;
  }

  // Close CSV file
  if (mpi_rank == 0)
    csv_file.close();

  // Print timing summary
  const double total_time = assembly_time + solver_time + output_time;
  pcout << "===============================================" << std::endl;
  pcout << "Timing summary:" << std::endl;
  pcout << "  Assembly:     " << std::fixed << std::setprecision(2) << assembly_time
        << " s (" << std::setw(5) << std::setprecision(1)
        << 100.0 * assembly_time / total_time << "%)" << std::endl;
  pcout << "  Linear solve: " << std::fixed << std::setprecision(2) << solver_time
        << " s (" << std::setw(5) << std::setprecision(1) << 100.0 * solver_time / total_time
        << "%)" << std::endl;
  pcout << "  Output:       " << std::fixed << std::setprecision(2) << output_time
        << " s (" << std::setw(5) << std::setprecision(1) << 100.0 * output_time / total_time
        << "%)" << std::endl;
  pcout << "  Total:        " << std::fixed << std::setprecision(2) << total_time << " s"
        << std::endl;
  pcout << "===============================================" << std::endl;
}
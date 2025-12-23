#include "Wave.hpp"

// Costruttore: inizializza i membri e il pcout per l'output parallelo
WaveEquation::WaveEquation(const std::string &mesh_file_name_,
                           const unsigned int &r_,
                           const double &T_,
                           const double &deltat_,
                           const bool &use_manufactured_,
                           const unsigned int &output_freq_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
      mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
      pcout(std::cout, mpi_rank == 0),
      T(T_),
      mesh_file_name(mesh_file_name_),
      r(r_),
      deltat(deltat_),
      use_manufactured_solution(use_manufactured_),
      output_frequency(output_freq_),
      mesh(MPI_COMM_WORLD)
{}

void WaveEquation::setup()
{
  pcout << "===============================================" << std::endl;
  pcout << "Initializing Wave Equation Solver (MPI)" << std::endl;
  pcout << "===============================================" << std::endl;

  // 1. Caricamento Mesh (parallela)
  {
    pcout << "Initializing the mesh from: " << mesh_file_name << std::endl;
    Triangulation<dim> mesh_serial;
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);
    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);
    pcout << "  Number of active cells = " << mesh.n_global_active_cells() << std::endl;
  }

  // 2. Elementi Finiti e Quadratura
  fe = std::make_unique<FE_SimplexP<dim>>(r);
  quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
  pcout << "  FE degree = " << r << " | Quadrature points = " << quadrature->size() << std::endl;

  // 3. DoF Handler e IndexSets
  dof_handler.reinit(mesh);
  dof_handler.distribute_dofs(*fe);
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;

  // 4. Sparsity Pattern e Matrici Trilinos
  TrilinosWrappers::SparsityPattern sp(locally_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, sp);
  sp.compress();

  mass_matrix.reinit(sp);
  stiffness_matrix.reinit(sp);
  lhs_matrix.reinit(sp);

  // 5. Vettori (Owned e Ghosted)
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution_next.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  solution_old.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  velocity.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  acceleration.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

  pcout << "===============================================" << std::endl;
}

void WaveEquation::assemble_matrices()
{
  pcout << "Assembling Mass and Stiffness matrices..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_JxW_values);
  FullMatrix<double> cell_M(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_K(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix = 0.0; stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned()) continue;
    fe_values.reinit(cell);
    cell_M = 0.0; cell_K = 0.0;

    for (unsigned int q = 0; q < n_q; ++q) {
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          cell_M(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
          cell_K(i, j) += fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
        }
      }
    }
    cell->get_dof_indices(dof_indices);
    mass_matrix.add(dof_indices, cell_M);
    stiffness_matrix.add(dof_indices, cell_K);
  }
  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // Per differenze centrate explicit-like: LHS = M
  lhs_matrix.copy_from(mass_matrix);

  auto end = std::chrono::high_resolution_clock::now();
  assembly_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
}

void WaveEquation::assemble_rhs(const double &t)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_quadrature_points | update_JxW_values);

  system_rhs = 0.0;
  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector term_K(locally_owned_dofs, MPI_COMM_WORLD);

  // 1. Termine Inerziale: M * (2*u^n - u^{n-1})
  tmp.add(2.0, solution);
  tmp.add(-1.0, solution_old);
  mass_matrix.vmult(system_rhs, tmp);

  // 2. Termine Rigidezza: - dt^2 * K * u^n
  stiffness_matrix.vmult(term_K, solution);
  system_rhs.add(-deltat * deltat, term_K);

  // 3. Forcing Term: + dt^2 * M * f^n
  forcing_term.set_time(t - deltat); 
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned()) continue;
    fe_values.reinit(cell);
    cell_rhs = 0.0;
    for (unsigned int q = 0; q < n_q; ++q) {
      const double f_val = forcing_term.value(fe_values.quadrature_point(q));
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_rhs(i) += deltat * deltat * f_val * fe_values.shape_value(i, q) * fe_values.JxW(q);
    }
    cell->get_dof_indices(dof_indices);
    system_rhs.add(dof_indices, cell_rhs);
  }
  system_rhs.compress(VectorOperation::add);

  // Applicazione Condizioni al Contorno (Dirichlet)
  std::map<types::global_dof_index, double> boundary_values;
  exact_solution.set_time(t);
  VectorTools::interpolate_boundary_values(dof_handler, 0, exact_solution, boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, lhs_matrix, solution_next, system_rhs, false);
}

void WaveEquation::solve_time_step()
{
  auto start = std::chrono::high_resolution_clock::now();

  SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(lhs_matrix);

  solver.solve(lhs_matrix, solution_next, system_rhs, preconditioner);

  // Aggiornamento per il calcolo dell'energia e del prossimo step
  velocity.add(1.0 / (2.0 * deltat), solution_next);
  velocity.add(-1.0 / (2.0 * deltat), solution_old);

  solution_old = solution;
  solution = solution_next;

  auto end = std::chrono::high_resolution_clock::now();
  solver_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
}

void WaveEquation::project_initial_acceleration()
{
  // Calcola u^{-1} per avviare le differenze centrate: u^{-1} = u0 - dt*v0 + (dt^2/2)*a0
  // M*a0 = f(0) - K*u0
  pcout << "Projecting initial acceleration a0..." << std::endl;
  
  TrilinosWrappers::MPI::Vector rhs_acc(locally_owned_dofs, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector k_u0(locally_owned_dofs, MPI_COMM_WORLD);
  
  stiffness_matrix.vmult(k_u0, solution);
  mass_matrix.vmult(rhs_acc, k_u0); // Placeholder per f(0) - K*u0 (qui semplificato)
  
  // Risolvi M*a0 = rhs
  SolverControl solver_control(1000, 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR prec;
  prec.initialize(mass_matrix);
  solver.solve(mass_matrix, acceleration, rhs_acc, prec);
}

void WaveEquation::solve()
{
  assemble_matrices();
  time = 0.0;

  // Condizioni Iniziali t=0
  exact_solution.set_time(0.0);
  VectorTools::interpolate(dof_handler, exact_solution, solution);
  
  // Per differenze centrate, inizializziamo u_old (u^{-1}) usando u0 e v0
  solution_old = solution; 
  exact_velocity.set_time(0.0);
  TrilinosWrappers::MPI::Vector v0_vec(locally_owned_dofs, MPI_COMM_WORLD);
  VectorTools::interpolate(dof_handler, exact_velocity, v0_vec);
  solution_old.add(-deltat, v0_vec); // u^{-1} \approx u0 - dt*v0

  if (output_frequency > 0) output(0);
  write_time_series(0);

  unsigned int time_step = 0;
  while (time < T - 0.5 * deltat)
  {
    time += deltat;
    time_step++;

    pcout << "Step " << std::setw(4) << time_step << " | t = " << std::scientific << time << std::endl;

    assemble_rhs(time);
    solve_time_step();

    if (output_frequency > 0 && time_step % output_frequency == 0)
      output(time_step);
    
    write_time_series(time_step);
  }

  pcout << "\nSimulation finished. Total assembly time: " << assembly_time << "s" << std::endl;
}

double WaveEquation::compute_energy()
{
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients | update_JxW_values);

  std::vector<double>         velocity_values(n_q);
  std::vector<Tensor<1, dim>> displacement_gradients(n_q);

  double local_kinetic_energy   = 0.0;
  double local_potential_energy = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // In parallelo, integriamo solo sulle celle di competenza del rank locale
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          // Estraiamo i valori della velocità e i gradienti dello spostamento nei punti di quadratura.
          // Nota: 'velocity' e 'solution' devono essere vettori ghosted (locally_relevant).
          fe_values.get_function_values(velocity, velocity_values);
          fe_values.get_function_gradients(solution, displacement_gradients);

          for (unsigned int q = 0; q < n_q; ++q)
            {
              const double JxW = fe_values.JxW(q);

              // Energia Cinetica: 1/2 * rho * v^2 (assumendo rho=1)
              local_kinetic_energy += 0.5 * velocity_values[q] * velocity_values[q] * JxW;

              // Energia Potenziale: 1/2 * |grad(u)|^2
              local_potential_energy += 0.5 * (displacement_gradients[q] * displacement_gradients[q]) * JxW;
            }
        }
    }

  // Sommiamo i contributi di tutti i processi MPI
  const double global_kinetic   = Utilities::MPI::sum(local_kinetic_energy, MPI_COMM_WORLD);
  const double global_potential = Utilities::MPI::sum(local_potential_energy, MPI_COMM_WORLD);

  return global_kinetic + global_potential;
}
void WaveEquation::write_time_series(const unsigned int &step)
{
  double energy = compute_energy();
  if (mpi_rank == 0)
  {
    if (step == 0) {
      csv_file.open("results.csv");
      csv_file << "step,time,energy\n";
    }
    csv_file << step << "," << time << "," << energy << "\n";
  }
}

void WaveEquation::output(const unsigned int &step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");
  data_out.build_patches();
  data_out.write_vtu_with_pvtu_record("./", "output", step, MPI_COMM_WORLD, 3);
}

double
WaveEquation::compute_error(const VectorTools::NormType &norm_type, const bool for_velocity)
{
  // 1. Definiamo il mapping e la quadratura per l'integrazione dell'errore (come da template)
  // Usiamo un mapping lineare (FE_SimplexP grado 1) e una quadratura di ordine r+2 per precisione
  FE_SimplexP<dim> fe_linear(1);
  MappingFE mapping(fe_linear);
  const QGaussSimplex<dim> quadrature_error(r + 2);

  // 2. Vettore locale per l'errore per cella
  // In parallelo, ogni processo calcolerà l'errore solo per le celle che possiede
  Vector<double> error_per_cell(mesh.n_active_cells());

  if (for_velocity)
    {
      // Impostiamo il tempo per la funzione esatta della velocità
      exact_velocity.set_time(time);
      
      // Calcolo locale dell'errore sulla velocità
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        velocity, // Vettore ghosted
                                        exact_velocity,
                                        error_per_cell,
                                        quadrature_error,
                                        norm_type);
    }
  else
    {
      // Impostiamo il tempo per la funzione esatta dello spostamento
      exact_solution.set_time(time);
      
      // Calcolo locale dell'errore sullo spostamento
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution, // Vettore ghosted
                                        exact_solution,
                                        error_per_cell,
                                        quadrature_error,
                                        norm_type);
    }

  // 3. Riduzione MPI: sommiamo gli errori locali per ottenere l'errore globale
  // In un contesto distribuito, l2_norm() del vettore locale non basta; 
  // compute_global_error gestisce la comunicazione tra i rank.
  const double global_error = VectorTools::compute_global_error(mesh,
                                                                error_per_cell,
                                                                norm_type);

  return global_error;
}
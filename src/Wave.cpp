#include "Wave.hpp"

#include "theta_integrator.hpp"
#include "time_integrator.hpp"

void Wave::setup() {
    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;
        Triangulation<dim> mesh_serial;

        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);

        std::ifstream grid_in_file(mesh_file_name);
        grid_in.read_msh(grid_in_file);

        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const TriangulationDescription::Description<dim, dim> construction_data =
                TriangulationDescription::Utilities::create_description_from_triangulation(
                        mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);

        pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;
        fe = std::make_unique<FE_SimplexP<dim>>(r);

        pcout << "  Degree                     = " << fe->degree << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;
        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        locally_owned_dofs    = dof_handler.locally_owned_dofs();
        locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

        pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;

        pcout << "  Initializing the sparsity pattern" << std::endl;
        TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, sparsity);
        sparsity.compress();

        pcout << "  Initializing the matrices" << std::endl;
        mass_matrix.reinit(sparsity);
        stiffness_matrix.reinit(sparsity);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        forcing_n.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        forcing_np1.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

        pcout << "  Initializing the velocity vector" << std::endl;
        velocity_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        velocity.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    }

    pcout << "-----------------------------------------------" << std::endl;

    switch (time_scheme) {
        case TimeScheme::Theta:
            pcout << "Initializing the Theta time integrator" << std::endl;
            time_integrator = std::make_unique<ThetaIntegrator>(theta);
            break;
        default:
            AssertThrow(false, ExcMessage("Unknown time scheme"));
    }
}

void Wave::assemble_matrices() {
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the system matrices" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q           = quadrature->size();

    FEValues fe_values(*fe,
                       *quadrature,
                       update_values | update_gradients | update_quadrature_points |
                               update_JxW_values);

    FullMatrix<double>                   cell_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double>                   cell_stiffness(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    mass_matrix      = 0.0;
    stiffness_matrix = 0.0;

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_mass      = 0.0;
        cell_stiffness = 0.0;

        for (unsigned int q = 0; q < n_q; ++q) {
            // Evaluate coefficients on this quadrature node.
            const double mu_loc = mu.value(fe_values.quadrature_point(q));

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_mass(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) *
                                       fe_values.JxW(q);

                    cell_stiffness(i, j) += mu_loc * fe_values.shape_grad(i, q) *
                                            fe_values.shape_grad(j, q) * fe_values.JxW(q);
                }
            }
        }

        cell->get_dof_indices(dof_indices);
        mass_matrix.add(dof_indices, cell_mass);
        stiffness_matrix.add(dof_indices, cell_stiffness);
    }

    mass_matrix.compress(VectorOperation::add);
    stiffness_matrix.compress(VectorOperation::add);
}

void Wave::assemble_rhs(const double &time, TrilinosWrappers::MPI::Vector &F_out) {
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q           = quadrature->size();

    FEValues fe_values(
            *fe, *quadrature, update_values | update_quadrature_points | update_JxW_values);

    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    F_out = 0.0;

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_rhs = 0.0;

        // Compute f at this time
        forcing_term.set_time(time);

        for (unsigned int q = 0; q < n_q; ++q) {
            const double f_loc = forcing_term.value(fe_values.quadrature_point(q));

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
        }

        cell->get_dof_indices(dof_indices);
        F_out.add(dof_indices, cell_rhs);
    }

    F_out.compress(VectorOperation::add);
}

void Wave::output(const unsigned int &time_step) const {
    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "u");
    data_out.add_data_vector(dof_handler, velocity, "v");

    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./", "output", time_step, MPI_COMM_WORLD, 3);
}

void Wave::solve() {
    assemble_matrices();

    pcout << "===============================================" << std::endl;

    // Initialize the time integrator.
    {
        pcout << "Initializing the time integrator" << std::endl;
        time_integrator->initialize(
                mass_matrix, stiffness_matrix, solution_owned, velocity_owned, deltat);
    }

    // Apply the initial conditions.
    {
        pcout << "Applying the initial conditions" << std::endl;

        // U^0
        VectorTools::interpolate(dof_handler, u_0, solution_owned);
        solution = solution_owned;

        // V^0
        VectorTools::interpolate(dof_handler, v_0, velocity_owned);
        velocity = velocity_owned;

        // Output the initial solution (time step 0)
        output(0);
        pcout << "-----------------------------------------------" << std::endl;
    }

    unsigned int time_step = 0;
    double       time      = 0;

    while (time < T) {
        const double t_n   = time;
        const double t_np1 = time + deltat;

        pcout << "n = " << std::setw(3) << time_step + 1 << ", t = " << std::setw(5) << t_np1 << ":"
              << std::flush << std::endl;

        // 1. Assemble the right-hand side at time step n
        assemble_rhs(t_n, forcing_n); // F^n
        assemble_rhs(t_np1, forcing_np1); // F^{n+1}

        time_integrator->advance(t_n,
                                 deltat,
                                 mass_matrix,
                                 stiffness_matrix,
                                 forcing_n,
                                 forcing_np1,
                                 solution_owned,
                                 velocity_owned);

        // 3. Update ghosted vectors for output
        solution = solution_owned;
        solution.update_ghost_values();

        velocity = velocity_owned;
        velocity.update_ghost_values();

        time = t_np1;
        ++time_step;

        output(time_step);
    }
}

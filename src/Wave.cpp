#include "Wave.hpp"

#include "progress_bar.hpp"
#include "theta_integrator.hpp"
#include "time_integrator.hpp"

#include <iomanip>
#include <sstream>

void Wave::process_mesh_input() {
    try {
        const std::filesystem::path p(mesh_file_name);
        if (!p.has_extension()) {
            AssertThrow(false,
                        ExcMessage("Mesh file name must have an extension (.msh or .geo): " +
                                   mesh_file_name));
        }
        if (p.extension() == ".geo") {
            pcout << "-----------------------------------------------" << std::endl;
            pcout << "Generating mesh from .geo file using gmsh..." << std::endl;
            // Output mesh file: same name but .msh extension
            std::filesystem::path out = p;
            out.replace_extension(".msh");

            // Build gmsh command. Use -2 (2D mesh) and explicit output format
            const std::string cmd =
                    "gmsh -2 -format msh2 -o \"" + out.string() + "\" \"" + p.string() + "\"";

            const int ret = std::system(cmd.c_str());
            AssertThrow(ret == 0,
                        ExcMessage("Failed to run gmsh to generate mesh from .geo file: " +
                                   mesh_file_name + ". Command executed: " + cmd +
                                   ". If gmsh is not installed, please install it or provide a "
                                   "mesh file in .msh format."));

            // Replace the mesh file name with the generated mesh file
            mesh_file_name = out.string();
            pcout << "  Mesh generated: " << mesh_file_name << std::endl;
            pcout << "-----------------------------------------------" << std::endl;
        } else if (p.extension() == ".msh") {
            // Nothing to do
        } else {
            AssertThrow(false,
                        ExcMessage("Unsupported mesh file extension (use .msh or .geo): " +
                                   mesh_file_name));
        }
    } catch (const std::exception &e) {
        AssertThrow(false,
                    ExcMessage(std::string("Exception while processing mesh input: ") + e.what()));
    }
}

void Wave::setup() {
    pcout << "===============================================" << std::endl;

    // Set up the problem.
    {
        pcout << "Setting up the problem" << std::endl;
        parameters.initialize_problem<dim>(mu, boundary_g, forcing_term, u_0, v_0);
    }

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

        // Identify boundary IDs.
        boundary_ids.clear();
        // Iterate over all active cells.
        for (const auto &cell: dof_handler.active_cell_iterators())
            // Only consider locally owned cells.
            if (cell->is_locally_owned())
                // Check each face of the cell.
                for (unsigned int f = 0; f < cell->n_faces(); ++f)
                    // If the face is at the boundary, store its boundary ID.
                    if (cell->face(f)->at_boundary())
                        boundary_ids.insert(cell->face(f)->boundary_id());

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

    // Set the time for the forcing term function.
    forcing_term.set_time(time);

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_rhs = 0.0;

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

void Wave::make_dirichlet_constraints(const double &time, AffineConstraints<> &constraints) const {
    // Clear previous constraints
    constraints.clear();

    // Initialize constraints with locally relevant DoFs
    constraints.reinit(locally_relevant_dofs);

    // Handle hanging nodes (i.e., non-conforming meshes)
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Set the time for the boundary condition function
    boundary_g->set_time(time);

    // Apply Dirichlet boundary conditions for all boundary faces
    for (const auto id: boundary_ids)
        VectorTools::interpolate_boundary_values(dof_handler, id, *boundary_g, constraints);

    constraints.close();
}

void Wave::make_velocity_dirichlet_constraints(const double               time,
                                               AffineConstraints<double> &constraints) const {
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    boundary_v->set_time(time);
    for (const auto id: boundary_ids)
        VectorTools::interpolate_boundary_values(dof_handler, id, *boundary_v, constraints);

    constraints.close();
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

std::pair<double, double> Wave::compute_errors(const double time) {
    const auto exact_u = &u_0;
    const auto exact_v = &v_0;
    exact_u->set_time(time);
    exact_v->set_time(time);

    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(
            *fe, *quadrature, update_values | update_quadrature_points | update_JxW_values);

    std::vector<double> uh_values(n_q);
    std::vector<double> vh_values(n_q);

    double local_u_sq = 0.0;
    double local_v_sq = 0.0;

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        fe_values.get_function_values(solution, uh_values);
        fe_values.get_function_values(velocity, vh_values);

        for (unsigned int q = 0; q < n_q; ++q) {
            const Point<dim> &xq = fe_values.quadrature_point(q);

            const double uex = exact_u->value(xq);
            const double vex = exact_v->value(xq);

            const double eu = uh_values[q] - uex;
            const double ev = vh_values[q] - vex;

            local_u_sq += eu * eu * fe_values.JxW(q);
            local_v_sq += ev * ev * fe_values.JxW(q);
        }
    }

    // MPI reduction across ranks
    const double global_u_sq = Utilities::MPI::sum(local_u_sq, MPI_COMM_WORLD);
    const double global_v_sq = Utilities::MPI::sum(local_v_sq, MPI_COMM_WORLD);

    return {std::sqrt(global_u_sq), std::sqrt(global_v_sq)};
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

        // Set time to 0 for initial conditions
        u_0.set_time(0.0);
        v_0.set_time(0.0);

        // U^0
        VectorTools::interpolate(dof_handler, u_0, solution_owned);
        solution = solution_owned;

        // Apply Dirichlet constraints to initial displacement
        AffineConstraints<> constraints_u0;
        make_dirichlet_constraints(0.0, constraints_u0);
        constraints_u0.distribute(solution_owned);

        // V^0
        VectorTools::interpolate(dof_handler, v_0, velocity_owned);
        velocity = velocity_owned;

        // Output the initial solution (time step 0)
        output(0);
        pcout << "-----------------------------------------------" << std::endl;
    }

    ProgressBar progress(static_cast<unsigned int>(std::ceil(T / deltat)), MPI_COMM_WORLD, &pcout);
    progress.for_while(
            [&](const unsigned int step) -> bool {
                // step is 1-based: t_n = (step-1)*deltat, t_np1 = step*deltat
                const double t_n   = (static_cast<double>(step) - 1.0) * deltat;
                const double t_np1 = static_cast<double>(step) * deltat;

                // 1. Assemble the right-hand side at time step n
                assemble_rhs(t_n, forcing_n); // F^n
                assemble_rhs(t_np1, forcing_np1); // F^{n+1}

                // 2. Create Dirichlet constraints at time step n+1
                AffineConstraints<> constraints_u_np1;
                make_dirichlet_constraints(t_np1, constraints_u_np1);

                // 3. Advance the solution to time step n+1
                time_integrator->advance(t_n,
                                         deltat,
                                         mass_matrix,
                                         stiffness_matrix,
                                         forcing_n,
                                         forcing_np1,
                                         solution_owned,
                                         velocity_owned);

                // 4. Apply Dirichlet constraints to the new solution
                constraints_u_np1.distribute(solution_owned);


                // 5. Update ghosted vectors for output
                solution = solution_owned;
                solution.update_ghost_values();

                velocity = velocity_owned;
                velocity.update_ghost_values();

                if (step % output_every == 0) {
                    output(step);
                }

                return t_np1 < T;
            },
            "Time-stepping progress",
            [&](const unsigned int step) -> std::string {
                std::ostringstream oss;
                const double       t_show = static_cast<double>(step) * deltat; // end of this step
                oss << "t=" << std::fixed << std::setprecision(5) << t_show;
                return oss.str();
            });
}

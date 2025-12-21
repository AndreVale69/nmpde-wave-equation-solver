#include "Wave.hpp"

#include "progress_bar.hpp"
#include "theta_integrator.hpp"
#include "time_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>

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
        parameters.initialize_problem<dim>(mu, boundary_g, boundary_v, forcing_term, u_0, v_0);
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

Wave::ErrorStatistics Wave::compute_error_statistics(const std::vector<double> &errors) {
    ErrorStatistics stats{};
    // If no errors, return zeros (already zero-initialized)
    if (errors.empty())
        return stats;

    const size_t        m    = errors.size();
    const double        sum  = std::accumulate(errors.begin(), errors.end(), 0.0);
    const double        mean = sum / static_cast<double>(m);
    std::vector<double> tmp  = errors;
    std::sort(tmp.begin(), tmp.end());
    const double median = (m % 2 == 1) ? tmp[m / 2] : 0.5 * (tmp[m / 2 - 1] + tmp[m / 2]);
    const double sq_sum =
            std::accumulate(errors.begin(), errors.end(), 0.0, [&](double a, double b) {
                return a + (b - mean) * (b - mean);
            });
    const double stddev = (m > 0) ? std::sqrt(sq_sum / static_cast<double>(m)) : 0.0;
    const double rms    = std::sqrt(std::accumulate(errors.begin(),
                                                 errors.end(),
                                                 0.0,
                                                 [](double a, double b) { return a + b * b; }) /
                                 static_cast<double>(m));

    const auto it_min = std::min_element(errors.begin(), errors.end());
    const auto it_max = std::max_element(errors.begin(), errors.end());

    stats.mean    = mean;
    stats.median  = median;
    stats.std     = stddev;
    stats.rms     = rms;
    stats.sum     = sum;
    stats.min     = (it_min != errors.end()) ? *it_min : 0.0;
    stats.max     = (it_max != errors.end()) ? *it_max : 0.0;
    stats.idx_min = (it_min != errors.end())
                            ? static_cast<size_t>(std::distance(errors.begin(), it_min))
                            : 0;
    stats.idx_max = (it_max != errors.end())
                            ? static_cast<size_t>(std::distance(errors.begin(), it_max))
                            : 0;
    return stats;
}

void Wave::print_error_summary() const {
    pcout << "===============================================" << std::endl;
    pcout << "Error summary over time:" << std::endl;
    pcout << "-----------------------------------------------" << std::endl;

    // If no errors were recorded, exit early
    if (time_history.empty()) {
        pcout << "No error history recorded (time_history is empty)." << std::endl;
        return;
    }

    // Number of time steps recorded
    const size_t n = error_u_history.size();

    // u stats
    const ErrorStatistics u_stats       = compute_error_statistics(error_u_history);
    const double          u_mean        = u_stats.mean;
    const double          u_median      = u_stats.median;
    const double          u_stddev      = u_stats.std;
    const double          u_rms         = u_stats.rms;
    const double          u_min         = u_stats.min;
    const double          u_max         = u_stats.max;
    const size_t          u_idx_min     = u_stats.idx_min;
    const size_t          u_idx_max     = u_stats.idx_max;
    const double          u_time_of_max = time_history[u_idx_max];
    const double          u_time_of_min = time_history[u_idx_min];
    const double          u_final       = error_u_history.back();

    // v stats
    const ErrorStatistics v_stats       = compute_error_statistics(error_v_history);
    const double          v_mean        = v_stats.mean;
    const double          v_median      = v_stats.median;
    const double          v_stddev      = v_stats.std;
    const double          v_rms         = v_stats.rms;
    const double          v_min         = v_stats.min;
    const double          v_max         = v_stats.max;
    const size_t          v_idx_min     = v_stats.idx_min;
    const size_t          v_idx_max     = v_stats.idx_max;
    const double          v_time_of_max = time_history[v_idx_max];
    const double          v_time_of_min = time_history[v_idx_min];
    const double          v_final       = error_v_history.back();

    // max relative change between consecutive steps (for u and v)
    double u_max_rel_change = 0.0;
    double v_max_rel_change = 0.0;
    for (size_t i = 1; i < n; ++i) {
        const double du    = std::abs(error_u_history[i] - error_u_history[i - 1]);
        const double dv    = std::abs(error_v_history[i] - error_v_history[i - 1]);
        const double u_rel = (error_u_history[i - 1] != 0.0) ? du / error_u_history[i - 1] : du;
        const double v_rel = (error_v_history[i - 1] != 0.0) ? dv / error_v_history[i - 1] : dv;
        u_max_rel_change   = std::max(u_max_rel_change, u_rel);
        v_max_rel_change   = std::max(v_max_rel_change, v_rel);
    }

    pcout << "-----------------------------------------------" << std::endl;
    pcout << "Error summary (L2 norms) over time-steps (n=" << n << "):" << std::endl;
    pcout << std::fixed;
    pcout << " u: min                  = " << std::scientific << std::setprecision(5) << u_min
          << std::endl
          << "    max                  = " << std::scientific << std::setprecision(5) << u_max
          << std::endl
          << "    mean                 = " << std::scientific << std::setprecision(5) << u_mean
          << std::endl
          << "    median               = " << std::scientific << std::setprecision(5) << u_median
          << std::endl
          << "    stddev               = " << std::scientific << std::setprecision(5) << u_stddev
          << std::endl
          << "    rms                  = " << std::scientific << std::setprecision(5) << u_rms
          << std::endl
          << "    final                = " << std::scientific << std::setprecision(5) << u_final
          << std::endl
          << "    time_of_max          = " << std::fixed << std::setprecision(5) << u_time_of_max
          << std::endl
          << "    time_of_min          = " << std::fixed << std::setprecision(5) << u_time_of_min
          << std::endl
          << "    max_rel_step_change = " << std::scientific << std::setprecision(5)
          << u_max_rel_change << std::endl;

    pcout << std::endl;

    pcout << " v: min                 = " << std::scientific << std::setprecision(5) << v_min
          << std::endl
          << "    max                 = " << std::scientific << std::setprecision(5) << v_max
          << std::endl
          << "    mean                = " << std::scientific << std::setprecision(5) << v_mean
          << std::endl
          << "    median              = " << std::scientific << std::setprecision(5) << v_median
          << std::endl
          << "    stddev              = " << std::scientific << std::setprecision(5) << v_stddev
          << std::endl
          << "    rms                 = " << std::scientific << std::setprecision(5) << v_rms
          << std::endl
          << "    final               = " << std::scientific << std::setprecision(5) << v_final
          << std::endl
          << "    time_of_max         = " << std::fixed << std::setprecision(5) << v_time_of_max
          << std::endl
          << "    time_of_min         = " << std::fixed << std::setprecision(5) << v_time_of_min
          << std::endl
          << "    max_rel_step_change = " << std::scientific << std::setprecision(5)
          << v_max_rel_change << std::endl;
    pcout << "-----------------------------------------------" << std::endl;

    // Ask user whether to save the history to a CSV file
    const std::string default_fname = "./error_history.csv";
    std::cout << "Save error history to CSV file '" << default_fname << "'? [y/N]: " << std::flush;
    std::string answer;
    std::getline(std::cin, answer);
    if (!answer.empty() && (answer[0] == 'y' || answer[0] == 'Y')) {
        if (std::ofstream ofs(default_fname); ofs) {
            // Write CSV header
            ofs << "step,time,error_u,error_v,delta_u,delta_v,rel_delta_u,rel_delta_v,cum_mean_u,"
                   "cum_mean_v,cum_rms_u,cum_rms_v\n";

            // Write data rows
            double prev_u = 0.0, prev_v = 0.0;
            double cum_sum_u = 0.0, cum_sum_v = 0.0;
            double cum_sq_u = 0.0, cum_sq_v = 0.0;

            // Loop over all time steps
            for (size_t i = 0; i < n; ++i) {
                // Current time and errors
                const double t  = time_history[i];
                const double eu = error_u_history[i];
                const double ev = error_v_history[i];

                // Changes from previous step
                const double delta_u     = (i > 0) ? (eu - prev_u) : 0.0;
                const double delta_v     = (i > 0) ? (ev - prev_v) : 0.0;
                const double rel_delta_u = (i > 0 && prev_u != 0.0) ? (delta_u / prev_u) : 0.0;
                const double rel_delta_v = (i > 0 && prev_v != 0.0) ? (delta_v / prev_v) : 0.0;

                // Cumulative statistics
                cum_sum_u += eu;
                cum_sum_v += ev;
                cum_sq_u += eu * eu;
                cum_sq_v += ev * ev;

                // Cumulative mean and RMS
                const double cum_mean_u = cum_sum_u / static_cast<double>(i + 1);
                const double cum_mean_v = cum_sum_v / static_cast<double>(i + 1);
                const double cum_rms_u  = std::sqrt(cum_sq_u / static_cast<double>(i + 1));
                const double cum_rms_v  = std::sqrt(cum_sq_v / static_cast<double>(i + 1));

                // Write row: step (1-based), time, errors and diagnostics
                ofs << (i + 1) << "," << std::fixed << std::setprecision(10) << t << ","
                    << std::scientific << std::setprecision(10) << eu << "," << std::scientific
                    << std::setprecision(10) << ev << "," << std::scientific
                    << std::setprecision(10) << delta_u << "," << std::scientific
                    << std::setprecision(10) << delta_v << "," << std::scientific
                    << std::setprecision(10) << rel_delta_u << "," << std::scientific
                    << std::setprecision(10) << rel_delta_v << "," << std::scientific
                    << std::setprecision(10) << cum_mean_u << "," << std::scientific
                    << std::setprecision(10) << cum_mean_v << "," << std::scientific
                    << std::setprecision(10) << cum_rms_u << "," << std::scientific
                    << std::setprecision(10) << cum_rms_v << std::endl;

                prev_u = eu;
                prev_v = ev;
            }

            ofs.close();
            pcout << "Wrote extended error history to '" << default_fname << "'" << std::endl;
        } else {
            pcout << "Failed to open '" << default_fname << "' for writing." << std::endl;
        }
    } else {
        pcout << "Did not save error history." << std::endl;
    }

    pcout << "===============================================" << std::endl;
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

        // Apply Dirichlet constraints to initial velocity
        AffineConstraints<> constraints_v0;
        make_velocity_dirichlet_constraints(0.0, constraints_v0);
        constraints_v0.distribute(velocity_owned);

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
                AffineConstraints<> constraints_v_np1;
                make_dirichlet_constraints(t_np1, constraints_u_np1);
                make_velocity_dirichlet_constraints(t_np1, constraints_v_np1);

                // 3. Build Dirichlet values map for velocity at time step n+1
                std::map<types::global_dof_index, double> v_boundary_values;
                boundary_v->set_time(t_np1);
                for (const auto id: boundary_ids)
                    VectorTools::interpolate_boundary_values(
                            dof_handler, id, *boundary_v, v_boundary_values);

                // 4. Advance the solution to time step n+1
                time_integrator->advance(t_n,
                                         deltat,
                                         mass_matrix,
                                         stiffness_matrix,
                                         forcing_n,
                                         forcing_np1,
                                         constraints_v_np1,
                                         v_boundary_values,
                                         solution_owned,
                                         velocity_owned);

                // 5. Apply Dirichlet constraints to the solution and velocity at time step n+1
                constraints_u_np1.distribute(solution_owned);
                constraints_v_np1.distribute(velocity_owned);


                // 6. Update the solution and velocity vectors with ghost values
                solution = solution_owned;
                velocity = velocity_owned;
                solution.update_ghost_values();
                velocity.update_ghost_values();

                // 6. Compute errors if exact solution is available: store them for later summary
                if (parameters.problem.type == ProblemType::MMS) {
                    const auto [error_u, error_v] = compute_errors(t_np1);
                    // Only rank 0 stores the time/error history to reduce memory on worker ranks
                    if (mpi_rank == 0) {
                        time_history.push_back(t_np1);
                        error_u_history.push_back(error_u);
                        error_v_history.push_back(error_v);
                    }
                }

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

    // After time-stepping, print an extended error summary if using MMS (only on rank 0)
    if (parameters.problem.type == ProblemType::MMS && mpi_rank == 0) {
        print_error_summary();
    }
}

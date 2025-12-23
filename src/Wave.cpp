#include "Wave.hpp"

#include "time_integrator/central_difference_integrator.hpp"
#include "time_integrator/newmark_integrator.hpp"
#include "time_integrator/theta_integrator.hpp"
#include "time_integrator/time_integrator.hpp"
#include "utils/mesh_generator.hpp"
#include "utils/progress_bar.hpp"

#include <algorithm>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <cmath>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>

class SineMode2D : public Function<2> {
public:
    SineMode2D(const unsigned int k, const unsigned int l) : Function<2>(1), k(k), l(l) {}

    double value(const Point<2> &p, const unsigned int /*component*/ = 0) const override {
        return std::sin(k * numbers::PI * p[0]) * std::sin(l * numbers::PI * p[1]);
    }

private:
    const unsigned int k, l;
};

double Wave::compute_energy(const TrilinosWrappers::MPI::Vector &u_owned,
                            const TrilinosWrappers::MPI::Vector &v_owned) const {
    // tmp = M v
    TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, MPI_COMM_WORLD);
    mass_matrix.vmult(tmp, v_owned);
    const double vMv = v_owned * tmp; // dot product

    // tmp = K u
    stiffness_matrix.vmult(tmp, u_owned);
    const double uKu = u_owned * tmp;

    return 0.5 * (vMv + uKu);
}

void Wave::setup() {
    pcout << "===============================================" << std::endl;

    // Set up the problem.
    {
        pcout << "Setting up the problem" << std::endl;
        parameters->initialize_problem(mu, boundary_g, boundary_v, forcing_term, u_0, v_0);
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

        case TimeScheme::CentralDifference:
            pcout << "Initializing the Central Difference time integrator" << std::endl;
            time_integrator = std::make_unique<CentralDifferenceIntegrator>();
            break;

        case TimeScheme::Newmark:
            pcout << "Initializing the Newmark time integrator" << std::endl;
            time_integrator = std::make_unique<NewmarkIntegrator>(); // beta=1/4, gamma=1/2
            break;

        default:
            AssertThrow(false, ExcMessage("Unknown time scheme"));
    }

}

void Wave::solve() {
    // If convergence study is requested, run it instead of normal solve.
    if (parameters->output.convergence_study) {
        return convergence();
    }
    do_solve();
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

    // Use the vtk output directory from parameters (ensure trailing slash)
    std::string vtk_dir = parameters->output.vtk_output_directory;
    if (!vtk_dir.empty() && vtk_dir.back() != '/')
        vtk_dir.push_back('/');

    data_out.write_vtu_with_pvtu_record(vtk_dir, "output", time_step, MPI_COMM_WORLD, 3);
}

void Wave::do_solve() {
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

        // Quick diagnostic: if both are (almost) zero, E0 will be (almost) zero.
        {
            const double u0_l2 = solution_owned.l2_norm();
            const double v0_l2 = velocity_owned.l2_norm();
            pcout << "IC norms after constraints: ||u0||_2=" << u0_l2 << ", ||v0||_2=" << v0_l2
                  << std::endl;
            if (u0_l2 == 0.0 && v0_l2 == 0.0) {
                pcout << "[Hint] Both u0 and v0 are exactly zero after applying Dirichlet BCs, "
                         "so the discrete energy E0 will be 0 and E/E0 is undefined. "
                         "If you want a nonzero energy, use Problem type 'expr' (u0_expr/v0_expr) "
                         "or provide physical initial conditions."
                      << std::endl;
            }
        }

        // Output the initial solution (time step 0)
        output(0);
        pcout << "-----------------------------------------------" << std::endl;
    }

    // -------------------- Dissipation study (optional) --------------------
    double        E0 = 1.0;
    std::ofstream dissipation_out;

    if (parameters->study.enable_dissipation_study) {
        E0 = compute_energy(solution_owned, velocity_owned);

        if (mpi_rank == 0) {
            dissipation_out.open(parameters->study.dissipation_csv);
            dissipation_out << "n,t,E,E_over_E0\n";

            // Guard against E0=0 (e.g. zero initial conditions) to avoid NaNs in output.
            const double E_over_E0_0 = (std::isfinite(E0) && std::abs(E0) > 0.0) ? 1.0 : 0.0;
            dissipation_out << 0 << "," << 0.0 << "," << E0 << "," << E_over_E0_0 << "\n";
            dissipation_out.flush();
        }

        if (!(std::isfinite(E0) && std::abs(E0) > 0.0)) {
            pcout << "[Study] Dissipation enabled, but E0 is not positive/finite (E0=" << E0
                  << "). E/E0 will be reported as 0 to avoid NaNs.\n";
        } else {
            pcout << "[Study] Dissipation enabled. E0 = " << E0 << ", writing to "
                  << parameters->study.dissipation_csv << std::endl;
        }
        pcout << "-----------------------------------------------" << std::endl;
    }

    // -------------------- Modal study (optional) --------------------
    TrilinosWrappers::MPI::Vector phi_owned(locally_owned_dofs, MPI_COMM_WORLD);
    TrilinosWrappers::MPI::Vector Mphi_owned(locally_owned_dofs, MPI_COMM_WORLD);
    double                        phi_M_phi = 1.0;

    std::ofstream modal_out;

    if (parameters->study.enable_modal_study) {
        // Build phi = sin(k*pi*x) sin(l*pi*y) interpolated on FE space
        SineMode2D mode_fun(parameters->study.modal_k, parameters->study.modal_l);
        VectorTools::interpolate(dof_handler, mode_fun, phi_owned);

        // Enforce Dirichlet zero on phi as well
        const auto boundary_values_zero = build_zero_dirichlet_map();
        for (const auto &[dof, val]: boundary_values_zero)
            if (phi_owned.locally_owned_elements().is_element(dof))
                phi_owned[dof] = val;
        phi_owned.compress(VectorOperation::insert);

        // Precompute Mphi and denom
        mass_matrix.vmult(Mphi_owned, phi_owned);
        phi_M_phi = phi_owned * Mphi_owned;

        if (mpi_rank == 0) {
            modal_out.open(parameters->study.modal_csv);
            modal_out << "n,t,a,adot\n";
        }

        // Log at t=0
        const double a0    = (solution_owned * Mphi_owned) / phi_M_phi;
        const double adot0 = (velocity_owned * Mphi_owned) / phi_M_phi;

        if (mpi_rank == 0) {
            modal_out << 0 << "," << 0.0 << "," << a0 << "," << adot0 << "\n";
            modal_out.flush();
        }

        pcout << "[Study] Modal enabled. Mode (k,l)=(" << parameters->study.modal_k << ","
              << parameters->study.modal_l << "), writing to " << parameters->study.modal_csv
              << std::endl;
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

                // Build Dirichlet values map for displacement at time step n+1
                std::map<types::global_dof_index, double> u_boundary_values;
                boundary_g->set_time(t_np1);
                for (const auto id: boundary_ids)
                    VectorTools::interpolate_boundary_values(
                        dof_handler, id, *boundary_g, u_boundary_values);



                // 4. Advance the solution to time step n+1
                time_integrator->advance(t_n,
                         deltat,
                         mass_matrix,
                         stiffness_matrix,
                         forcing_n,
                         forcing_np1,
                         constraints_u_np1,
                         u_boundary_values,
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
                if (parameters->problem.type == ProblemType::MMS) {
                    const auto [u_L2, u_H1, v_L2] = compute_error_norms(t_np1);
                    if (mpi_rank == 0) {
                        time_history.push_back(t_np1);
                        error_u_L2_history.push_back(u_L2);
                        error_u_H1_history.push_back(u_H1);
                        error_v_L2_history.push_back(v_L2);
                    }
                }

                // -------------------- Dissipation study (optional) --------------------
                if (parameters->study.enable_dissipation_study &&
                    (step % parameters->study.dissipation_every == 0)) {
                    const double E = compute_energy(solution_owned, velocity_owned);
                    if (mpi_rank == 0) {
                        const double E_over_E0 =
                                (std::isfinite(E0) && std::abs(E0) > 0.0) ? (E / E0) : 0.0;
                        dissipation_out << step << "," << t_np1 << "," << E << "," << E_over_E0
                                        << "\n";
                        dissipation_out.flush();
                    }
                }

                // -------------------- Modal study (optional) --------------------
                if (parameters->study.enable_modal_study &&
                    (step % parameters->study.modal_every == 0)) {
                    const double a    = (solution_owned * Mphi_owned) / phi_M_phi;
                    const double adot = (velocity_owned * Mphi_owned) / phi_M_phi;

                    if (mpi_rank == 0)
                        modal_out << step << "," << t_np1 << "," << a << "," << adot << "\n";
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
    if (parameters->problem.type == ProblemType::MMS && mpi_rank == 0) {
        print_error_summary();
    }
}


void Wave::convergence() {
    using dealii::ConvergenceTable;

    AssertThrow(parameters->problem.type == ProblemType::MMS,
                ExcMessage("Convergence study requires MMS (exact solution)."));

    if (parameters->output.convergence_type == ConvergenceType::Time) {
        pcout << "Running time convergence study..." << std::endl;
        const auto rows = run_time_convergence(parameters_file,
                                               {0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005});

        ConvergenceTable table;

        for (const auto &row: rows) {
            table.add_value("dt", row.dt);
            table.add_value("u_L2", row.err.u_L2);
            table.add_value("u_H1", row.err.u_H1);
            table.add_value("v_L2", row.err.v_L2);

            // Also store the observed orders computed in run_time_convergence().
            // First row has NaN (no previous row), which will be printed as-is.
            table.add_value("q_uL2", row.q_uL2);
            table.add_value("q_uH1", row.q_uH1);
            table.add_value("q_vL2", row.q_vL2);
        }

        // dt halves -> use reduction_rate_log2 (same as prof)
        table.evaluate_convergence_rates("u_L2", ConvergenceTable::reduction_rate_log2);
        table.evaluate_convergence_rates("u_H1", ConvergenceTable::reduction_rate_log2);
        table.evaluate_convergence_rates("v_L2", ConvergenceTable::reduction_rate_log2);

        // Formatting
        table.set_precision("dt", 6);
        table.set_scientific("u_L2", true);
        table.set_scientific("u_H1", true);
        table.set_scientific("v_L2", true);

        table.set_precision("q_uL2", 6);
        table.set_precision("q_uH1", 6);
        table.set_precision("q_vL2", 6);

        if (mpi_rank == 0) {
            table.write_text(std::cout);
            if (!parameters->output.convergence_csv.empty()) {
                write_time_convergence_csv(parameters->output.convergence_csv, rows);
            }
        }

        return;
    }
    if (parameters->output.convergence_type == ConvergenceType::Space) {
        pcout << "Running space convergence study..." << std::endl;

        const auto rows = run_space_convergence(parameters_file,
                                                {{"mesh/square_structured.geo", 1.0 / 10.0},
                                                 {"mesh/square_structured.geo", 1.0 / 20.0},
                                                 {"mesh/square_structured.geo", 1.0 / 30.0},
                                                 {"mesh/square_structured.geo", 1.0 / 40.0},
                                                 {"mesh/square_structured.geo", 1.0 / 50.0},
                                                 {"mesh/square_structured.geo", 1.0 / 60.0},
                                                 {"mesh/square_structured.geo", 1.0 / 70.0},
                                                 {"mesh/square_structured.geo", 1.0 / 80.0},
                                                 {"mesh/square_structured.geo", 1.0 / 90.0},
                                                 {"mesh/square_structured.geo", 1.0 / 100.0}},
                                                /*dt_small=*/1e-3);

        ConvergenceTable table;

        for (const auto &row: rows) {
            table.add_value("h", row.h);
            table.add_value("u_L2", row.err.u_L2);
            table.add_value("u_H1", row.err.u_H1);
            table.add_value("v_L2", row.err.v_L2);

            table.add_value("p_uL2", row.p_uL2);
            table.add_value("p_uH1", row.p_uH1);
            table.add_value("p_vL2", row.p_vL2);
        }

        table.evaluate_convergence_rates("u_L2", ConvergenceTable::reduction_rate_log2);
        table.evaluate_convergence_rates("u_H1", ConvergenceTable::reduction_rate_log2);
        table.evaluate_convergence_rates("v_L2", ConvergenceTable::reduction_rate_log2);

        table.set_precision("h", 6);
        table.set_scientific("u_L2", true);
        table.set_scientific("u_H1", true);
        table.set_scientific("v_L2", true);

        table.set_precision("p_uL2", 6);
        table.set_precision("p_uH1", 6);
        table.set_precision("p_vL2", 6);

        if (mpi_rank == 0) {
            table.write_text(std::cout);
            if (!parameters->output.convergence_csv.empty()) {
                write_space_convergence_csv(parameters->output.convergence_csv, rows);
            }
        }

        return;
    }

    AssertThrow(false, ExcMessage("Unknown convergence type"));
}

Wave::ErrorNorms Wave::compute_error_norms(const double time) {
    AssertThrow(parameters->problem.type == ProblemType::MMS,
                ExcMessage("compute_error_norms() is intended for MMS verification."));
    // Exact solutions
    const auto exact_solution_u = &u_0;
    const auto exact_solution_v = &v_0;

    // Set the time for the exact solutions
    exact_solution_u->set_time(time);
    exact_solution_v->set_time(time);

    // Quadrature for error evaluation:
    // for norms, use a bit higher order than assembly because we want accurate errors
    const QGaussSimplex<dim> q_err(fe->degree + 2);

    // Mapping and FE for error evaluation
    FE_SimplexP<dim> fe_linear(1);
    MappingFE<dim>   mapping(fe_linear);

    // Per-cell error vectors
    Vector<double> diff_u(mesh.n_active_cells());
    Vector<double> diff_v(mesh.n_active_cells());

    // ---- u: L2 ----
    VectorTools::integrate_difference(
            mapping, dof_handler, solution, *exact_solution_u, diff_u, q_err, VectorTools::L2_norm);
    const double u_L2 = VectorTools::compute_global_error(mesh, diff_u, VectorTools::L2_norm);

    // ---- u: H1 ----
    VectorTools::integrate_difference(
            mapping, dof_handler, solution, *exact_solution_u, diff_u, q_err, VectorTools::H1_norm);
    const double u_H1 = VectorTools::compute_global_error(mesh, diff_u, VectorTools::H1_norm);

    // ---- v: L2 ----
    VectorTools::integrate_difference(
            mapping, dof_handler, velocity, *exact_solution_v, diff_v, q_err, VectorTools::L2_norm);
    const double v_L2 = VectorTools::compute_global_error(mesh, diff_v, VectorTools::L2_norm);

    // Return the computed norms
    return {u_L2, u_H1, v_L2};
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
        pcout << "===============================================" << std::endl;
        return;
    }

    // Ensure histories are consistent. If not, truncate to the minimum common size.
    const size_t n_time = time_history.size();
    const size_t n_uL2  = error_u_L2_history.size();
    const size_t n_uH1  = error_u_H1_history.size();
    const size_t n_vL2  = error_v_L2_history.size();

    const size_t n = std::min({n_time, n_uL2, n_uH1, n_vL2});
    if (n == 0) {
        pcout << "No error history recorded (one or more error vectors are empty)." << std::endl;
        pcout << "===============================================" << std::endl;
        return;
    }

    if (n != n_time || n != n_uL2 || n != n_uH1 || n != n_vL2) {
        pcout << "Warning: time/error history sizes mismatch. Truncating to n=" << n << "."
              << " (time=" << n_time << ", u_L2=" << n_uL2 << ", u_H1=" << n_uH1
              << ", v_L2=" << n_vL2 << ")" << std::endl;
    }

    // Create truncated views for statistics computations.
    const std::vector<double> time(time_history.begin(), time_history.begin() + n);
    const std::vector<double> u_L2(error_u_L2_history.begin(), error_u_L2_history.begin() + n);
    const std::vector<double> u_H1(error_u_H1_history.begin(), error_u_H1_history.begin() + n);
    const std::vector<double> v_L2(error_v_L2_history.begin(), error_v_L2_history.begin() + n);

    // Stats
    const ErrorStatistics uL2_stats = compute_error_statistics(u_L2);
    const ErrorStatistics uH1_stats = compute_error_statistics(u_H1);
    const ErrorStatistics vL2_stats = compute_error_statistics(v_L2);

    const auto safe_time_at = [&](const size_t idx) -> double {
        return (idx < time.size()) ? time[idx] : time.back();
    };

    const double uL2_time_of_max = safe_time_at(uL2_stats.idx_max);
    const double uL2_time_of_min = safe_time_at(uL2_stats.idx_min);
    const double uH1_time_of_max = safe_time_at(uH1_stats.idx_max);
    const double uH1_time_of_min = safe_time_at(uH1_stats.idx_min);
    const double vL2_time_of_max = safe_time_at(vL2_stats.idx_max);
    const double vL2_time_of_min = safe_time_at(vL2_stats.idx_min);

    const double uL2_final = u_L2.back();
    const double uH1_final = u_H1.back();
    const double vL2_final = v_L2.back();

    // max relative change between consecutive steps
    auto max_rel_step_change = [](const std::vector<double> &e) -> double {
        if (e.size() < 2)
            return 0.0;
        double max_rel = 0.0;
        for (size_t i = 1; i < e.size(); ++i) {
            const double de  = std::abs(e[i] - e[i - 1]);
            const double rel = (e[i - 1] != 0.0) ? de / e[i - 1] : de;
            max_rel          = std::max(max_rel, rel);
        }
        return max_rel;
    };

    const double uL2_max_rel_change = max_rel_step_change(u_L2);
    const double uH1_max_rel_change = max_rel_step_change(u_H1);
    const double vL2_max_rel_change = max_rel_step_change(v_L2);

    pcout << "-----------------------------------------------" << std::endl;
    pcout << "Error summary over time-steps (n=" << n << "):" << std::endl;
    pcout << std::fixed;

    auto print_block = [&](const std::string     &label,
                           const std::string     &norm_name,
                           const ErrorStatistics &stats,
                           const double           final_value,
                           const double           time_of_max,
                           const double           time_of_min,
                           const double           max_rel_change) {
        pcout << " " << label << " (" << norm_name
              << "):     min                  = " << std::scientific << std::setprecision(5)
              << stats.min << std::endl
              << "             max                  = " << std::scientific << std::setprecision(5)
              << stats.max << std::endl
              << "             mean                 = " << std::scientific << std::setprecision(5)
              << stats.mean << std::endl
              << "             median               = " << std::scientific << std::setprecision(5)
              << stats.median << std::endl
              << "             stddev               = " << std::scientific << std::setprecision(5)
              << stats.std << std::endl
              << "             rms                  = " << std::scientific << std::setprecision(5)
              << stats.rms << std::endl
              << "             final                = " << std::scientific << std::setprecision(5)
              << final_value << std::endl
              << "             time_of_max          = " << std::fixed << std::setprecision(5)
              << time_of_max << std::endl
              << "             time_of_min          = " << std::fixed << std::setprecision(5)
              << time_of_min << std::endl
              << "             max_rel_step_change  = " << std::scientific << std::setprecision(5)
              << max_rel_change << std::endl;
    };

    print_block(
            "u", "L2", uL2_stats, uL2_final, uL2_time_of_max, uL2_time_of_min, uL2_max_rel_change);
    pcout << std::endl;
    print_block(
            "u", "H1", uH1_stats, uH1_final, uH1_time_of_max, uH1_time_of_min, uH1_max_rel_change);
    pcout << std::endl;
    print_block(
            "v", "L2", vL2_stats, vL2_final, vL2_time_of_max, vL2_time_of_min, vL2_max_rel_change);

    pcout << "-----------------------------------------------" << std::endl;

    // Save the extended error history to CSV if requested in parameters (only rank 0 writes)
    if (parameters->output.compute_error) {
        if (mpi_rank == 0) {
            const std::filesystem::path outpath(parameters->output.error_history_file);
            try {
                if (outpath.has_parent_path()) {
                    std::filesystem::create_directories(outpath.parent_path());
                }

                if (std::ofstream ofs(outpath); ofs) {
                    // Write CSV header
                    ofs << "step,time,error_u_L2,error_u_H1,error_v_L2,"
                           "delta_u_L2,delta_u_H1,delta_v_L2,"
                           "rel_delta_u_L2,rel_delta_u_H1,rel_delta_v_L2,"
                           "cum_mean_u_L2,cum_mean_u_H1,cum_mean_v_L2,"
                           "cum_rms_u_L2,cum_rms_u_H1,cum_rms_v_L2\n";

                    // Write data rows
                    double prev_uL2 = 0.0, prev_uH1 = 0.0, prev_vL2 = 0.0;
                    double cum_sum_uL2 = 0.0, cum_sum_uH1 = 0.0, cum_sum_vL2 = 0.0;
                    double cum_sq_uL2 = 0.0, cum_sq_uH1 = 0.0, cum_sq_vL2 = 0.0;

                    for (size_t i = 0; i < n; ++i) {
                        const double t    = time[i];
                        const double euL2 = u_L2[i];
                        const double euH1 = u_H1[i];
                        const double evL2 = v_L2[i];

                        const double delta_uL2 = (i > 0) ? (euL2 - prev_uL2) : 0.0;
                        const double delta_uH1 = (i > 0) ? (euH1 - prev_uH1) : 0.0;
                        const double delta_vL2 = (i > 0) ? (evL2 - prev_vL2) : 0.0;

                        const double rel_delta_uL2 =
                                (i > 0 && prev_uL2 != 0.0) ? (delta_uL2 / prev_uL2) : 0.0;
                        const double rel_delta_uH1 =
                                (i > 0 && prev_uH1 != 0.0) ? (delta_uH1 / prev_uH1) : 0.0;
                        const double rel_delta_vL2 =
                                (i > 0 && prev_vL2 != 0.0) ? (delta_vL2 / prev_vL2) : 0.0;

                        cum_sum_uL2 += euL2;
                        cum_sum_uH1 += euH1;
                        cum_sum_vL2 += evL2;
                        cum_sq_uL2 += euL2 * euL2;
                        cum_sq_uH1 += euH1 * euH1;
                        cum_sq_vL2 += evL2 * evL2;

                        const double denom        = static_cast<double>(i + 1);
                        const double cum_mean_uL2 = cum_sum_uL2 / denom;
                        const double cum_mean_uH1 = cum_sum_uH1 / denom;
                        const double cum_mean_vL2 = cum_sum_vL2 / denom;
                        const double cum_rms_uL2  = std::sqrt(cum_sq_uL2 / denom);
                        const double cum_rms_uH1  = std::sqrt(cum_sq_uH1 / denom);
                        const double cum_rms_vL2  = std::sqrt(cum_sq_vL2 / denom);

                        ofs << (i + 1) << "," << std::fixed << std::setprecision(10) << t << ","
                            << std::scientific << std::setprecision(10) << euL2 << ","
                            << std::scientific << std::setprecision(10) << euH1 << ","
                            << std::scientific << std::setprecision(10) << evL2 << ","
                            << std::scientific << std::setprecision(10) << delta_uL2 << ","
                            << std::scientific << std::setprecision(10) << delta_uH1 << ","
                            << std::scientific << std::setprecision(10) << delta_vL2 << ","
                            << std::scientific << std::setprecision(10) << rel_delta_uL2 << ","
                            << std::scientific << std::setprecision(10) << rel_delta_uH1 << ","
                            << std::scientific << std::setprecision(10) << rel_delta_vL2 << ","
                            << std::scientific << std::setprecision(10) << cum_mean_uL2 << ","
                            << std::scientific << std::setprecision(10) << cum_mean_uH1 << ","
                            << std::scientific << std::setprecision(10) << cum_mean_vL2 << ","
                            << std::scientific << std::setprecision(10) << cum_rms_uL2 << ","
                            << std::scientific << std::setprecision(10) << cum_rms_uH1 << ","
                            << std::scientific << std::setprecision(10) << cum_rms_vL2 << std::endl;

                        prev_uL2 = euL2;
                        prev_uH1 = euH1;
                        prev_vL2 = evL2;
                    }

                    ofs.close();
                    pcout << "Wrote extended error history to '" << outpath.string() << "'"
                          << std::endl;
                } else {
                    pcout << "Failed to open '" << outpath.string() << "' for writing."
                          << std::endl;
                }
            } catch (const std::exception &e) {
                pcout << "Exception while trying to write error history to '" << outpath.string()
                      << "': " << e.what() << std::endl;
            }
        } else {
            pcout << "Error history saving disabled by parameter 'compute_error'." << std::endl;
        }

        pcout << "===============================================" << std::endl;
    }
}

double Wave::estimate_order(const double e1, const double e2, const double h1, const double h2) {
    return std::log(e1 / e2) / std::log(h1 / h2);
}

std::vector<Wave::TimeConvRow> Wave::run_time_convergence(const std::string         &prm_base,
                                                          const std::vector<double> &dts) {
    std::vector<TimeConvRow> rows;
    rows.reserve(dts.size());

    for (const double dt: dts) {
        auto prm                      = std::make_shared<Parameters<dim>>(prm_base);
        prm->time.dt                  = dt;
        prm->output.output_every      = 999999; // disable output during convergence tests
        prm->output.compute_error     = true; // enable error computation
        prm->output.convergence_study = false; // disable extra convergence output

        Wave w(prm);
        w.setup();
        const ErrorNorms e = w.solve_and_get_final_errors();

        rows.push_back({dt, e});
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 1; i < rows.size(); ++i) {
        rows[i].q_uL2 =
                estimate_order(rows[i - 1].err.u_L2, rows[i].err.u_L2, rows[i - 1].dt, rows[i].dt);
        rows[i].q_uH1 =
                estimate_order(rows[i - 1].err.u_H1, rows[i].err.u_H1, rows[i - 1].dt, rows[i].dt);
        rows[i].q_vL2 =
                estimate_order(rows[i - 1].err.v_L2, rows[i].err.v_L2, rows[i - 1].dt, rows[i].dt);
    }
    return rows;
}

void Wave::write_time_convergence_csv(const std::string              &filename,
                                      const std::vector<TimeConvRow> &rows) const {
    if (mpi_rank != 0)
        return;

    std::ofstream csv(filename);
    AssertThrow(csv, ExcMessage("Could not open CSV file: " + filename));

    // Header
    csv << "dt,u_L2,u_H1,v_L2,q_uL2,q_uH1,q_vL2\n";

    auto write_double = [&](double v) {
        if (std::isnan(v))
            csv << "nan";
        else
            csv << std::scientific << std::setprecision(10) << v;
    };

    for (const auto &[dt, err, q_uL2, q_uH1, q_vL2]: rows) {
        // dt
        csv << std::fixed << std::setprecision(6) << dt << ",";

        // errors
        write_double(err.u_L2);
        csv << ",";
        write_double(err.u_H1);
        csv << ",";
        write_double(err.v_L2);
        csv << ",";

        // observed orders (computed in run_time_convergence)
        write_double(q_uL2);
        csv << ",";
        write_double(q_uH1);
        csv << ",";
        write_double(q_vL2);

        csv << "\n";
    }

    csv.close();
}

std::vector<Wave::SpaceConvRow>
Wave::run_space_convergence(const std::string                                 &prm_base,
                            const std::vector<std::pair<std::string, double>> &meshes,
                            const double                                       dt_small) const {
    std::vector<SpaceConvRow> rows;
    rows.reserve(meshes.size());

    if (meshes.empty())
        return rows;

    // Pre-generate all meshes (rank 0) so the solve loop stays clean.
    struct GeneratedMesh {
        double       h;
        unsigned int Nx;
        std::string  msh_path;
    };
    std::vector<GeneratedMesh> generated;
    generated.reserve(meshes.size());

    std::error_code       ec;
    std::filesystem::path out_dir;


    if (mpi_rank == 0) {
        // We ignore the provided mesh template file path and instead use a hardcoded
        // structured square template, injecting Nx.
        // This avoids creating per-run temporary .geo files/directories.
        // Note: Nx controls the number of subdivisions per side (Transfinite uses Nx+1 points).
        const std::string square_structured_geo_template =
                "// Structured unit square mesh (generated)\n"
                "// Nx is injected by run_space_convergence\n"
                "L = 1.0;\n"
                "Point(1) = {0, 0, 0, 1.0};\n"
                "Point(2) = {L, 0, 0, 1.0};\n"
                "Point(3) = {L, L, 0, 1.0};\n"
                "Point(4) = {0, L, 0, 1.0};\n\n"
                "Line(1) = {1,2};\n"
                "Line(2) = {2,3};\n"
                "Line(3) = {3,4};\n"
                "Line(4) = {4,1};\n\n"
                "Curve Loop(1) = {1,2,3,4};\n"
                "Plane Surface(1) = {1};\n\n"
                "// Transfinite (structured) discretization\n"
                "Transfinite Curve {1,2,3,4} = Nx+1 Using Progression 1;\n"
                "Transfinite Surface {1};\n\n"
                "Physical Surface(\"domain\") = {1};\n"
                "Physical Curve(\"boundary\") = {1,2,3,4};\n";

        // Stable run id for generated meshes.
        const auto        uuid     = boost::uuids::random_generator()();
        const std::string uuid_str = to_string(uuid);

        out_dir = std::filesystem::temp_directory_path() / ("nmpde_wave_space_conv_" + uuid_str);

        std::filesystem::create_directories(out_dir, ec);
        AssertThrow(!ec, ExcMessage("Failed to create mesh output directory: " + out_dir.string()));

        // Build the list of target meshes (Nx, h) deterministically.
        for (const auto &_mesh: meshes) {
            const double h = _mesh.second;
            AssertThrow(h > 0.0, ExcMessage("Space convergence: h must be > 0"));
            const auto Nx = static_cast<unsigned int>(std::lround(1.0 / h));
            AssertThrow(Nx >= 1u, ExcMessage("Space convergence: computed Nx must be >= 1"));

            const std::string msh_path = (out_dir / ("Nx" + std::to_string(Nx) + ".msh")).string();
            generated.push_back({h, Nx, msh_path});
        }

        for (const auto &gm: generated) {
            // Explicit, unambiguous Nx injection.
            const std::string geo_with_Nx =
                    "Nx = " + std::to_string(gm.Nx) + ";\n" + square_structured_geo_template;
            const std::string inline_geo =
                    std::string(mesh_generator::inline_geo_prefix) + "\n" + geo_with_Nx;

            try {
                mesh_generator::gmsh_generate_msh(inline_geo, gm.msh_path);
            } catch (const std::exception &e) {
                AssertThrow(false,
                            ExcMessage(std::string("Failed generating mesh for Nx=") +
                                       std::to_string(gm.Nx) + ": " + e.what()));
            }
        }
    }
    // Broadcast generated mesh info to all ranks.
    {
        auto n_meshes = generated.size();
        MPI_Bcast(&n_meshes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        if (mpi_rank != 0) {
            generated.resize(n_meshes);
        }

        for (unsigned long i = 0; i < n_meshes; ++i) {
            MPI_Bcast(&generated[i].h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&generated[i].Nx, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

            unsigned long path_size = 0;
            if (mpi_rank == 0)
                path_size = generated[i].msh_path.size();

            MPI_Bcast(&path_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

            if (mpi_rank != 0)
                generated[i].msh_path.resize(path_size);

            if (path_size > 0) {
                MPI_Bcast(generated[i].msh_path.data(),
                          static_cast<int>(path_size),
                          MPI_CHAR,
                          0,
                          MPI_COMM_WORLD);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);


    // Solve loop
    for (const auto &gm: generated) {
        auto prm                      = std::make_shared<Parameters<dim>>(prm_base);
        prm->time.dt                  = dt_small;
        prm->mesh.mesh_file           = gm.msh_path;
        prm->output.output_every      = 999999;
        prm->output.compute_error     = true;
        prm->output.convergence_study = false;

        Wave w(prm);
        w.setup();
        const auto e = w.solve_and_get_final_errors();
        rows.push_back({gm.h, gm.msh_path, e});
    }

    // Compute observed orders.
    for (size_t i = 1; i < rows.size(); ++i) {
        rows[i].p_uL2 =
                estimate_order(rows[i - 1].err.u_L2, rows[i].err.u_L2, rows[i - 1].h, rows[i].h);
        rows[i].p_uH1 =
                estimate_order(rows[i - 1].err.u_H1, rows[i].err.u_H1, rows[i - 1].h, rows[i].h);
        rows[i].p_vL2 =
                estimate_order(rows[i - 1].err.v_L2, rows[i].err.v_L2, rows[i - 1].h, rows[i].h);
    }

    // Cleanup generated meshes (rank 0).
    if (mpi_rank == 0) {
        std::filesystem::remove_all(out_dir, ec);
    }

    return rows;
}

void Wave::write_space_convergence_csv(const std::string               &filename,
                                       const std::vector<SpaceConvRow> &rows) const {
    if (mpi_rank != 0)
        return;

    std::ofstream csv(filename);
    AssertThrow(csv, ExcMessage("Could not open CSV file: " + filename));

    // Header
    csv << "h,u_L2,u_H1,v_L2,p_uL2,p_uH1,p_vL2,mesh\n";

    auto write_double = [&](double v) {
        if (std::isnan(v))
            csv << "nan";
        else
            csv << std::scientific << std::setprecision(10) << v;
    };

    for (const auto &[h, _mesh, err, p_uL2, p_uH1, p_vL2]: rows) {
        // h
        csv << std::fixed << std::setprecision(6) << h << ",";

        // errors
        write_double(err.u_L2);
        csv << ",";
        write_double(err.u_H1);
        csv << ",";
        write_double(err.v_L2);
        csv << ",";

        // orders
        write_double(p_uL2);
        csv << ",";
        write_double(p_uH1);
        csv << ",";
        write_double(p_vL2);
        csv << ",";

        // mesh path (quoted to be CSV-safe)
        csv << '"' << _mesh << '"' << "\n";
    }

    csv.close();
}

Wave::ErrorNorms Wave::solve_and_get_final_errors() {
    do_solve();
    ErrorNorms out = {};
    if (parameters->problem.type == ProblemType::MMS) {
        const double t_final       = T;
        const auto [uL2, uH1, vL2] = compute_error_norms(t_final);
        out                        = {uL2, uH1, vL2};
    }
    return out;
}

void Wave::process_mesh_input() {
    try {
        const std::filesystem::path p(mesh_file_name);

        if (mesh_generator::is_inline_geo(mesh_file_name)) {
            if (mpi_rank == 0) {
                pcout << "-----------------------------------------------" << std::endl;
                pcout << "Generating mesh from inline .geo string using gmsh..." << std::endl;
            }

            std::string uuid_str;
            if (mpi_rank == 0) {
                const auto uuid = boost::uuids::random_generator()();
                uuid_str        = to_string(uuid);
            }
            Utilities::MPI::broadcast(MPI_COMM_WORLD, uuid_str, 0);

            const std::filesystem::path out = std::filesystem::temp_directory_path() /
                                              ("nmpde_wave_inline_" + uuid_str + ".msh");

            if (mpi_rank == 0) {
                try {
                    mesh_generator::gmsh_generate_msh(mesh_file_name, out.string());
                } catch (const std::exception &e) {
                    AssertThrow(false, ExcMessage(e.what()));
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);

            mesh_file_name = out.string();

            if (mpi_rank == 0) {
                pcout << "  Mesh generated: " << mesh_file_name << std::endl;
                pcout << "-----------------------------------------------" << std::endl;
            }
            return;
        }

        if (!p.has_extension()) {
            AssertThrow(false,
                        ExcMessage("Mesh file name must have an extension (.msh or .geo): " +
                                   mesh_file_name));
        }
        if (p.extension() == ".geo") {
            pcout << "-----------------------------------------------" << std::endl;
            pcout << "Generating mesh from .geo file using gmsh..." << std::endl;
            std::filesystem::path out = p;
            out.replace_extension(".msh");

            if (mpi_rank == 0) {
                pcout << "  Running gmsh to generate: " << out.string() << std::endl;
                try {
                    mesh_generator::gmsh_generate_msh(p.string(), out.string());
                } catch (const std::exception &e) {
                    AssertThrow(false, ExcMessage(e.what()));
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);

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

std::map<types::global_dof_index, double> Wave::build_zero_dirichlet_map() const {
    std::map<types::global_dof_index, double> bv;
    Functions::ZeroFunction<dim>              zero;

    for (const auto id: boundary_ids)
        VectorTools::interpolate_boundary_values(dof_handler, id, zero, bv);

    return bv;
}

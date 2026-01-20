#ifndef NM4PDE_TIME_INTEGRATOR_H
#define NM4PDE_TIME_INTEGRATOR_H

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

using namespace dealii;

class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;

    /**
     * @brief Get a string describing the parameters of the time integrator.
     * @return String with time integrator parameters.
     */
    [[nodiscard]] virtual std::string get_parameters_info() const = 0;

    /**
     * @brief Initialize the time integrator with the system matrices and initial conditions.
     * @param M Mass matrix.
     * @param K Stiffness matrix.
     * @param U0 Initial displacement vector.
     * @param V0 Initial velocity vector.
     * @param dt Time step size.
     * @details This method is called once before the time-stepping loop starts.
     */
    virtual void initialize(const TrilinosWrappers::SparseMatrix &M,
                            const TrilinosWrappers::SparseMatrix &K,
                            const TrilinosWrappers::MPI::Vector  &U0,
                            const TrilinosWrappers::MPI::Vector  &V0,
                            const double                          dt) = 0;

    /**
     * @brief Advance the solution from time step n to n+1.
     * @param t_n Current time at step n.
     * @param dt Time step size.
     * @param M Mass matrix.
     * @param K Stiffness matrix.
     * @param F_n Forcing vector at time step n.
     * @param F_np1 Forcing vector at time step n+1.
     * @param constraints_u_np1 Constraints for displacement at time step n+1.
     * @param u_boundary_values Boundary values for displacement at time step n+1.
     * @param constraints_v_np1 Constraints for velocity at time step n+1.
     * @param v_boundary_values Boundary values for velocity at time step n+1.
     * @param U Displacement vector to be updated to time step n+1.
     * @param V Velocity vector to be updated to time step n+1.
     * @details This method is called at each time step to update the solution.
     */
    virtual void advance(const double                                     t_n,
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
                         TrilinosWrappers::MPI::Vector                   &V) = 0;
};

#endif // NM4PDE_TIME_INTEGRATOR_H

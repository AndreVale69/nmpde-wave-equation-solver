#ifndef NM4PDE_TIME_INTEGRATOR_H
#define NM4PDE_TIME_INTEGRATOR_H

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <map>
#include <deal.II/base/types.h>

using namespace dealii;

class TimeIntegrator {
public:
    virtual ~TimeIntegrator() = default;

    virtual void initialize(const TrilinosWrappers::SparseMatrix &M,
                            const TrilinosWrappers::SparseMatrix &K,
                            const TrilinosWrappers::MPI::Vector  &U0,
                            const TrilinosWrappers::MPI::Vector  &V0,
                            const double                          dt) = 0;

    virtual void advance(const double                          t_n,
                         const double                          dt,
                         const TrilinosWrappers::SparseMatrix &M,
                         const TrilinosWrappers::SparseMatrix &K,
                         const TrilinosWrappers::MPI::Vector  &F_n,
                         const TrilinosWrappers::MPI::Vector  &F_np1,
                         const std::map<types::global_dof_index, double> &boundary_values_v,
                         TrilinosWrappers::MPI::Vector        &U,
                         TrilinosWrappers::MPI::Vector        &V) = 0; // V^n -> V^{n+1}
};

#endif // NM4PDE_TIME_INTEGRATOR_H

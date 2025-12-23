#ifndef NM4PDE_PROBLEM_FUNCTIONS_HPP
#define NM4PDE_PROBLEM_FUNCTIONS_HPP

#include <deal.II/base/function.h>

using namespace dealii;

/**
 * @brief Homogeneous Dirichlet boundary condition g = 0.
 * @tparam dim Spatial dimension.
 */
template<int dim>
class BoundaryGZero : public Function<dim> {
public:
    BoundaryGZero() : Function<dim>(1) {}

    double value(const Point<dim> &, const unsigned int = 0) const override { return 0.0; }
};

/**
 * @brief Homogeneous Dirichlet boundary condition for the velocity v = 0.
 * @tparam dim Spatial dimension.
 */
template<int dim>
class BoundaryVZero : public Function<dim> {
public:
    BoundaryVZero() : Function<dim>(1) {}

    double value(const Point<dim> &, const unsigned int = 0) const override { return 0.0; }
};

#endif // NM4PDE_PROBLEM_FUNCTIONS_HPP

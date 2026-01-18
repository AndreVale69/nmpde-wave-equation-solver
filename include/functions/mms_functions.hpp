/**
 * @file mms_functions.hpp
 * @brief Manufactured solutions, velocities, and forcing terms for the wave equation.
 *
 * This file contains classes that define manufactured solutions, velocities,
 * and forcing terms for the wave equation in a 2D domain. These classes inherit
 * from the dealii::Function class and provide methods to evaluate the functions
 * at given points in space and time.
 */
#ifndef NM4PDE_MMS_FUNCTIONS_HPP
#define NM4PDE_MMS_FUNCTIONS_HPP

using namespace dealii;

/**
 * @brief Manufactured solution for the wave equation.
 * @tparam dim Dimension of the problem.
 */
template<int dim>
class ManufacturedSolution : public Function<dim> {
public:
    explicit ManufacturedSolution(const double omega = 1.0) : Function<dim>(), omega(omega) {}

    double value(const Point<dim> &p, const unsigned int = 0) const override {
        const double x = p[0], y = p[1];
        const double t = this->get_time();
        return std::sin(numbers::PI * x) * std::sin(numbers::PI * y) * std::cos(omega * t);
    }

    /**
     * @brief Get the expression string for the solution.
     * @return Expression string.
     * @details
     * The expression is of the form:
     * sin(pi*x)*sin(pi*y)*cos(omega*t)
     */
    [[nodiscard]] std::string get_expression() const {
        return "sin(pi*x)*sin(pi*y)*cos(" + std::to_string(omega) + "*t)";
    }

private:
    const double omega;
};

/**
 * @brief Manufactured velocity for the wave equation.
 * @tparam dim Dimension of the problem.
 */
template<int dim>
class ManufacturedVelocity : public Function<dim> {
public:
    explicit ManufacturedVelocity(const double omega = 1.0) : Function<dim>(), omega(omega) {}

    double value(const Point<dim> &p, const unsigned int = 0) const override {
        const double x = p[0], y = p[1];
        const double t = this->get_time();
        return -omega * std::sin(numbers::PI * x) * std::sin(numbers::PI * y) * std::sin(omega * t);
    }

    /**
     * @brief Get the expression string for the velocity.
     * @return Expression string.
     * @details
     * The expression is of the form:
     * -omega*sin(pi*x)*sin(pi*y)*sin(omega*t)
     */
    [[nodiscard]] std::string get_expression() const {
        return "-" + std::to_string(omega) + "*sin(pi*x)*sin(pi*y)*sin(" + std::to_string(omega) +
               "*t)";
    }

private:
    const double omega;
};

/**
 * @brief Manufactured forcing term for the wave equation.
 * @tparam dim Dimension of the problem.
 */
template<int dim>
class ManufacturedForcing : public Function<dim> {
public:
    explicit ManufacturedForcing(const double omega = 1.0) : Function<dim>(), omega(omega) {}

    double value(const Point<dim> &p, const unsigned int = 0) const override {
        const double x = p[0], y = p[1];
        const double t = this->get_time();
        const double s = std::sin(numbers::PI * x) * std::sin(numbers::PI * y);
        return (2.0 * numbers::PI * numbers::PI - omega * omega) * s * std::cos(omega * t);
    }

    /**
     * @brief Get the expression string for the forcing term.
     * @return Expression string.
     * @details
     * The expression is of the form:
     * (2*pi^2 - omega^2)*sin(pi*x)*sin(pi*y)*cos(omega*t)
     */
    [[nodiscard]] std::string get_expression() const {
        return "(" + std::to_string(2.0 * numbers::PI * numbers::PI - omega * omega) +
               ")*sin(pi*x)*sin(pi*y)*cos(" + std::to_string(omega) + "*t)";
    }

private:
    const double omega;
};

#endif // NM4PDE_MMS_FUNCTIONS_HPP

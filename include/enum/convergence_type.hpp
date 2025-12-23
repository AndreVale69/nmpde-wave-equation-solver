/**
 * @file boundary_type.hpp
 * @brief Definition and utilities for ConvergenceType enumeration.
 * @details
 * This file contains the definition of the ConvergenceType enumeration,
 * along with utility functions for converting between the enum and its
 * string representation, as well as an overloaded output stream operator.
 */
#ifndef NM4PDE_CONVERGENCE_TYPE_HPP
#define NM4PDE_CONVERGENCE_TYPE_HPP

#include <ostream>
#include <stdexcept>
#include <string>

/**
 * @brief Enumeration for different types of convergence studies.
 */
enum class ConvergenceType : char { Time, Space };

/**
 * @brief Convert ConvergenceType enum to string for output.
 * @param conv_type ConvergenceType enum value.
 * @return Corresponding string representation.
 */
inline std::string to_string(const ConvergenceType conv_type) {
    switch (conv_type) {
        case ConvergenceType::Time:
            return "time";
        case ConvergenceType::Space:
            return "space";
        default:
            return "unknown";
    }
}

/**
 * @brief Convert string from parameter file to ConvergenceType enum.
 * @param s Input string.
 * @return Corresponding ConvergenceType enum value.
 * @throws std::invalid_argument if the string does not match any known type.
 */
inline ConvergenceType convergence_type_from_string(const std::string &s) {
    if (s == "time")
        return ConvergenceType::Time;
    if (s == "space")
        return ConvergenceType::Space;
    throw std::invalid_argument("Invalid ConvergenceType string: " + s);
}

/**
 * @brief Overload the output stream operator for ConvergenceType enum.
 * @param os Output stream.
 * @param boundary_type ConvergenceType enum value.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &os, ConvergenceType boundary_type) {
    os << to_string(boundary_type);
    return os;
}

#endif // NM4PDE_CONVERGENCE_TYPE_HPP

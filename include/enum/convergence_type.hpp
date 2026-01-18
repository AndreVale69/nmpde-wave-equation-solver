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

#include <magic_enum/magic_enum.hpp>
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
    const std::string_view name = magic_enum::enum_name(conv_type);
    AssertThrow(!name.empty(),
                dealii::ExcMessage("Invalid ConvergenceType value (out of enum domain)."));
    return std::string(name);
}

/**
 * @brief Convert string from parameter file to ConvergenceType enum.
 * @param s Input string.
 * @return Corresponding ConvergenceType enum value.
 * @throws std::invalid_argument if the string does not match any known type.
 */
inline ConvergenceType convergence_type_from_string(const std::string &s) {
    const auto conv_type = magic_enum::enum_cast<ConvergenceType>(s);
    AssertThrow(conv_type.has_value(), dealii::ExcMessage("Invalid ConvergenceType string: " + s));
    return conv_type.value();
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

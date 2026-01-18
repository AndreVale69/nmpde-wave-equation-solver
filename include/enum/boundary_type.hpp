/**
 * @file boundary_type.hpp
 * @brief Definition and utilities for BoundaryType enumeration.
 * @details
 * This file contains the definition of the BoundaryType enumeration,
 * along with utility functions for converting between the enum and its
 * string representation, as well as an overloaded output stream operator.
 */
#ifndef NM4PDE_BOUNDARY_TYPE_HPP
#define NM4PDE_BOUNDARY_TYPE_HPP

#include <magic_enum/magic_enum.hpp>
#include <string>

/**
 * @brief Enumeration for different types of problems.
 */
enum class BoundaryType : char { Zero, MMS, Expr };

/**
 * @brief Convert BoundaryType enum to string for output.
 * @param boundary_type BoundaryType enum value.
 * @return Corresponding string representation.
 */
inline std::string to_string(const BoundaryType boundary_type) {
    const std::string_view name = magic_enum::enum_name(boundary_type);
    AssertThrow(!name.empty(),
                dealii::ExcMessage("Invalid BoundaryType value (out of enum domain)."));
    return std::string(name);
}

/**
 * @brief Convert string from parameter file to BoundaryType enum.
 * @param s Input string.
 * @return Corresponding BoundaryType enum value.
 * @throws std::invalid_argument if the string does not match any known type.
 */
inline BoundaryType boundary_type_from_string(const std::string &s) {
    const auto boundary_type = magic_enum::enum_cast<BoundaryType>(s);
    AssertThrow(boundary_type.has_value(), dealii::ExcMessage("Invalid BoundaryType string: " + s));
    return boundary_type.value();
}

/**
 * @brief Overload the output stream operator for BoundaryType enum.
 * @param os Output stream.
 * @param boundary_type BoundaryType enum value.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &os, const BoundaryType boundary_type) {
    os << to_string(boundary_type);
    return os;
}

#endif // NM4PDE_BOUNDARY_TYPE_HPP

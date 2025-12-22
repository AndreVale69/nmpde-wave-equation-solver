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
#include <stdexcept>

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
    switch (boundary_type) {
        case BoundaryType::Zero:
            return "zero";
        case BoundaryType::MMS:
            return "mms";
        case BoundaryType::Expr:
            return "expr";
        default:
            return "unknown";
    }
}

/**
 * @brief Convert string from parameter file to BoundaryType enum.
 * @param s Input string.
 * @return Corresponding BoundaryType enum value.
 * @throws std::invalid_argument if the string does not match any known type.
 */
inline BoundaryType boundary_type_from_string(const std::string &s) {
    if (s == "zero")
        return BoundaryType::Zero;
    if (s == "mms")
        return BoundaryType::MMS;
    if (s == "expr")
        return BoundaryType::Expr;
    throw std::invalid_argument("Invalid BoundaryType string: " + s);
}

/**
 * @brief Overload the output stream operator for BoundaryType enum.
 * @param os Output stream.
 * @param boundary_type BoundaryType enum value.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &os, BoundaryType boundary_type) {
    os << to_string(boundary_type);
    return os;
}

#endif // NM4PDE_BOUNDARY_TYPE_HPP

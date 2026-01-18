/**
 * @file problem_type.hpp
 * @brief Definition and utilities for ProblemType enumeration.
 * @details
 * This file defines the ProblemType enumeration used to specify different types of problems
 * in the NM4PDE framework. It includes functions to convert between the enumeration and
 * string representations, as well as an overload for the output stream operator.
 */
#ifndef NM4PDE_PROBLEM_TYPE_HPP
#define NM4PDE_PROBLEM_TYPE_HPP

#include <magic_enum/magic_enum.hpp>
#include <string>

/**
 * @brief Enumeration of available problem types.
 */
enum class ProblemType : char { Physical, MMS, Expr };

/**
 * @brief Convert ProblemType enum to string for output.
 * @param problem_type ProblemType enum value.
 * @return Corresponding string representation.
 */
inline std::string to_string(const ProblemType problem_type) {
    const std::string_view name = magic_enum::enum_name(problem_type);
    AssertThrow(!name.empty(),
                dealii::ExcMessage("Invalid ProblemType value (out of enum domain)."));
    return std::string(name);
}

/**
 * @brief Convert string from parameter file to ProblemType enum.
 * @param s Input string.
 * @return Corresponding ProblemType enum value.
 * @throws std::invalid_argument if the string does not match any known type.
 */
inline ProblemType problem_type_from_string(const std::string &s) {
    const auto problem_type = magic_enum::enum_cast<ProblemType>(s);
    AssertThrow(problem_type.has_value(), dealii::ExcMessage("Invalid ProblemType string: " + s));
    return problem_type.value();
}

/**
 * @brief Overload the output stream operator for ProblemType enum.
 * @param os Output stream.
 * @param problem_type ProblemType enum value.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &os, const ProblemType problem_type) {
    os << to_string(problem_type);
    return os;
}

#endif // NM4PDE_PROBLEM_TYPE_HPP

#ifndef NM4PDE_TIME_SCHEME_HPP
#define NM4PDE_TIME_SCHEME_HPP

#include <magic_enum/magic_enum.hpp>
#include <string>

/**
 * @brief Enumeration of available time integration schemes.
 */
enum class TimeScheme : char { Theta, CentralDifference, Newmark };

/**
 * @brief Convert TimeScheme enum to string for output.
 * @param scheme TimeScheme enum value.
 * @return Corresponding string representation.
 */
inline std::string to_string(const TimeScheme scheme) {
    const std::string_view name = magic_enum::enum_name(scheme);
    AssertThrow(!name.empty(),
                dealii::ExcMessage("Invalid TimeScheme value (out of enum domain)."));
    return std::string(name);
}

/**
 * @brief Convert string from parameter file to TimeScheme enum.
 * @param s Input string.
 * @return Corresponding TimeScheme enum value.
 * @throws std::invalid_argument if the string does not match any known scheme.
 */
inline TimeScheme time_scheme_from_string(const std::string &s) {
    const auto scheme = magic_enum::enum_cast<TimeScheme>(s);
    AssertThrow(scheme.has_value(), dealii::ExcMessage("Invalid TimeScheme string: " + s));
    return scheme.value();
}

/**
 * @brief Overload the output stream operator for TimeScheme enum.
 * @param os Output stream.
 * @param scheme TimeScheme enum value.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &os, const TimeScheme scheme) {
    os << to_string(scheme);
    return os;
}

#endif // NM4PDE_TIME_SCHEME_HPP

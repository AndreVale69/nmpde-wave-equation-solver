#ifndef NM4PDE_TIME_SCHEME_HPP
#define NM4PDE_TIME_SCHEME_HPP

#include <ostream>
#include <stdexcept>
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
    switch (scheme) {
        case TimeScheme::Theta:
            return "theta";
        case TimeScheme::CentralDifference:
            return "central";
        case TimeScheme::Newmark:
            return "newmark";
        default:
            return "unknown";
    }
}

/**
 * @brief Convert string from parameter file to TimeScheme enum.
 * @param s Input string.
 * @return Corresponding TimeScheme enum value.
 * @throws std::invalid_argument if the string does not match any known scheme.
 */
inline TimeScheme time_scheme_from_string(const std::string &s) {
    if (s == "theta")
        return TimeScheme::Theta;
    if (s == "central")
        return TimeScheme::CentralDifference;
    if (s == "newmark")
        return TimeScheme::Newmark;
    throw std::invalid_argument("Invalid TimeScheme string: " + s);
}

/**
 * @brief Overload the output stream operator for TimeScheme enum.
 * @param os Output stream.
 * @param scheme TimeScheme enum value.
 * @return Reference to the output stream.
 */
inline std::ostream &operator<<(std::ostream &os, TimeScheme scheme) {
    os << to_string(scheme);
    return os;
}

#endif // NM4PDE_TIME_SCHEME_HPP

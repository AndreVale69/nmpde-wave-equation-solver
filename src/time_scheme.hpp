#ifndef NM4PDE_TIME_SCHEME_HPP
#define NM4PDE_TIME_SCHEME_HPP

/**
 * @brief Enumeration of available time integration schemes.
 *
 * Currently, only the Theta method is implemented.
 */
enum class TimeScheme : char {
    /**
     * @brief The Theta method for time integration.
     */
    Theta
};

#endif // NM4PDE_TIME_SCHEME_HPP

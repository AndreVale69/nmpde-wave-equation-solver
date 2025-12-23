#ifndef NM4PDE_CONDITIONAL_OS_STREAM_WRAPPER_HPP
#define NM4PDE_CONDITIONAL_OS_STREAM_WRAPPER_HPP

#include <deal.II/base/conditional_ostream.h>

/**
 * @brief A wrapper around dealii::ConditionalOStream
 * that deletes warnings about the return value not being used.
 *
 * @details
 * In CLion, using dealii::ConditionalOStream's operator<<
 * generates warnings because the operator returns a reference
 * to the stream, but this return value is not used.
 * This wrapper class overrides the operator<< to
 * return a reference to itself, effectively suppressing
 * these warnings.
 */
class ConditionalOStreamWrapper : public dealii::ConditionalOStream {
public:
    using ConditionalOStream::ConditionalOStream;

    /**
     * @brief Templated operator<< that forwards to the base class
     * and returns a reference to the wrapper itself.
     */
    template<typename T>
    ConditionalOStreamWrapper &operator<<(const T &value) {
        (void) ConditionalOStream::operator<<(value);
        return *this;
    }

    /**
     * @brief Templated operator<< for const-qualified wrappers.
     */
    template<typename T>
    ConditionalOStreamWrapper &operator<<(const T &value) const {
        (void) const_cast<ConditionalOStream &>(static_cast<const ConditionalOStream &>(*this))
                .operator<<(value);
        return const_cast<ConditionalOStreamWrapper &>(*this);
    }

    /**
     * @brief Overloaded operator<< for std::ostream manipulators.
     */
    ConditionalOStreamWrapper &operator<<(std::ostream &(*manip)(std::ostream &) ) {
        (void) ConditionalOStream::operator<<(manip);
        return *this;
    }

    /**
     * @brief Overloaded operator<< for const-qualified wrappers
     * and std::ostream manipulators.
     */
    ConditionalOStreamWrapper &operator<<(std::ostream &(*manip)(std::ostream &) ) const {
        (void) static_cast<const ConditionalOStream &>(*this).operator<<(manip);
        return const_cast<ConditionalOStreamWrapper &>(*this);
    }

    /**
     * @brief Overloaded operator<< for std::ios manipulators.
     */
    ConditionalOStreamWrapper &operator<<(std::ios &(*manip)(std::ios &) ) {
        (void) ConditionalOStream::operator<<(manip);
        return *this;
    }

    /**
     * @brief Overloaded operator<< for const-qualified wrappers
     * and std::ios manipulators.
     */
    ConditionalOStreamWrapper &operator<<(std::ios &(*manip)(std::ios &) ) const {
        (void) static_cast<const ConditionalOStream &>(*this).operator<<(manip);
        return const_cast<ConditionalOStreamWrapper &>(*this);
    }

    /**
     * @brief Overloaded operator<< for std::ios_base manipulators.
     */
    ConditionalOStreamWrapper &operator<<(std::ios_base &(*manip)(std::ios_base &) ) {
        (void) ConditionalOStream::operator<<(manip);
        return *this;
    }

    /**
     * @brief Overloaded operator<< for const-qualified wrappers
     * and std::ios_base manipulators.
     */
    ConditionalOStreamWrapper &operator<<(std::ios_base &(*manip)(std::ios_base &) ) const {
        (void) static_cast<const ConditionalOStream &>(*this).operator<<(manip);
        return const_cast<ConditionalOStreamWrapper &>(*this);
    }

    /**
     * @brief Overloaded operator<< for std::ios_base::fmtflags.
     */
    ConditionalOStreamWrapper &operator<<(std::ios_base::fmtflags flags) {
        (void) ConditionalOStream::operator<<(flags);
        return *this;
    }

    /**
     * @brief Overloaded operator<< for const-qualified wrappers
     * and std::ios_base::fmtflags.
     */
    ConditionalOStreamWrapper &operator<<(std::ios_base::fmtflags flags) const {
        (void) static_cast<const ConditionalOStream &>(*this).operator<<(flags);
        return const_cast<ConditionalOStreamWrapper &>(*this);
    }
};

#endif // NM4PDE_CONDITIONAL_OS_STREAM_WRAPPER_HPP

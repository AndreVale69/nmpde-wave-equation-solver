/**
 * @file progress_bar.hpp
 * @brief A simple ASCII progress bar for long-running loops. Inspired by Python's tqdm.
 * @author Andrea Valentini
 * @date October 2025
 * @license MIT
 * @details
 * This class provides a simple ASCII progress bar for tracking the progress of
 * long-running loops in C++. It supports both known and unknown total step counts,
 * smoothed ETA estimates, and can be integrated with deal.II's ConditionalOStream
 * for MPI-aware output. The progress bar can be used in traditional for-loops,
 * while-loops, or range-based for-loops.
 *
 * Usage example:
 * @code{cpp}
 * ProgressBar progress(100); // known total of 100 steps
 * const unsigned int start = 1;
 * const unsigned int end   = 100;
 * const std::string description = "Description";
 * progress.for_each(start, end, [&](unsigned int step) -> void {
 *     std::this_thread::sleep_for(std::chrono::milliseconds(50)); // simulate work
 * }, description);
 * @endcode
 *
 * Output (e.g., after 40 steps):
 * @code
 * [##############--------------------]  40.0% (40/100)  Description  elapsed=00:00:02  ETA=00:00:03
 * @endcode
 *
 * Or for unknown total steps:
 * @code{cpp}
 * ProgressBar progress; // unknown total
 * const auto extra_info = [](unsigned int step) {
 *     return "This function will run at the end of each step: step " + std::to_string(step);
 * };
 * progress.for_while([&](unsigned int step) -> bool {
 *     std::this_thread::sleep_for(std::chrono::milliseconds(100)); // simulate work
 *     return step < 50; // stop after 50 steps
 * }, "Amazing work");
 * @endcode
 *
 * Output (e.g., after 20 steps):
 * @code
 * [/] (step 20)  Amazing work  elapsed=00:00:02 This function will run at the end of each step:
 * step 20
 * @endcode
 *
 * Note: The progress bar uses carriage returns to overwrite the same line in the console.
 */
#ifndef NM4PDE_PROGRESS_BAR_HPP
#define NM4PDE_PROGRESS_BAR_HPP

#include <deal.II/base/conditional_ostream.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <io.h> // _isatty, _fileno
#define isatty _isatty
#define fileno _fileno
#else // #ifdef _WIN32
#include <unistd.h> // isatty
#endif // #ifdef _WIN32 #else

/**
 * @brief A simple ASCII progress bar for long-running loops using deal.II's ConditionalOStream and
 * MPI.
 * @details
 * This class provides a progress bar that can be used to track the progress of loops
 * with known or unknown total step counts. It supports smoothed ETA estimates and can
 * be integrated with deal.II's ConditionalOStream for MPI-aware output.
 */
class ProgressBar {
public:
    /**
     * @brief Construct a new Progress Bar object.
     * @param total_steps Total number of steps in the loop (0 for unknown total).
     * @param comm MPI communicator for ConditionalOStream.
     * @param external_out Optional external ConditionalOStream for output.
     * @param bar_width Width of the progress bar in characters.
     * @param smoothing Number of recent steps to average for ETA smoothing. For
     *                  unknown total mode, this controls the responsiveness of the spinner.
     *                  For example, a value of 8 averages the last 8 steps.
     * @param smooth Whether to use smooth Unicode characters for the progress bar.
     *               If false, ASCII characters are used.
     * @param enabled Whether the progress bar is enabled at construction. When disabled,
     *                no output is produced.
     * @details
     * This constructor creates a ProgressBar with the specified total steps. If total_steps
     * is 0, the progress bar operates in "unknown total" mode, similar to Python's tqdm,
     * displaying a spinner instead of a percentage bar. The output is directed to the
     * provided ConditionalOStream or a new one created for the given MPI communicator.
     * The bar width, smoothing window size, and character style can be customized.
     *
     * Note: If stdout is not a TTY (TeleTYpewriter, i.e., a terminal), carriage return
     * updates are disabled to avoid garbled output in non-interactive environments.
     * Additionally, if the progress bar is disabled, no output will be produced (no overhead).
     */
    explicit ProgressBar(const unsigned int  total_steps  = 0,
                         const MPI_Comm      comm         = MPI_COMM_WORLD,
                         ConditionalOStream *external_out = nullptr,
                         const unsigned int  bar_width    = 30,
                         const unsigned int  smoothing    = 8,
                         const bool          smooth       = true,
                         const bool          enabled      = true)
        : enabled(enabled)
        , total_steps(total_steps)
        , bar_width(bar_width)
        , smoothing(smoothing)
        , smooth_unicode(smooth)
        , start(std::chrono::steady_clock::now())
        , last_time(start)
        , last_step_index(0) {
        // if an external ConditionalOStream is provided, use it; otherwise create one
        if (external_out != nullptr) {
            out_ptr  = external_out;
            owns_out = false;
        } else {
            owned_out = std::make_unique<ConditionalOStream>(
                    std::cout, Utilities::MPI::this_mpi_process(comm) == 0);
            out_ptr  = owned_out.get();
            owns_out = true;
        }
        // If stdout is not a TTY (TeleTYpewriter, i.e., a terminal), disable CR updates
        interactive_output = isatty(fileno(stdout)) != 0;
        // Set the appropriate dispatch functions based on total_steps and enabled state
        update_dispatch();
    }

    /**
     * @brief Construct a ProgressBar for unknown total steps using an external ConditionalOStream.
     * @param external_out External ConditionalOStream for output.
     * @details
     * This constructor creates a ProgressBar in "unknown total" mode, similar to Python's
     * tqdm, displaying a spinner instead of a percentage bar. It uses the provided
     * ConditionalOStream for output.
     */
    explicit ProgressBar(ConditionalOStream *external_out)
        : ProgressBar(0, MPI_COMM_WORLD, external_out, 30, 8, true) {}

    /**
     * @brief Destroy the Progress Bar object.
     * @details
     * Cleans up resources if the ProgressBar owns its ConditionalOStream.
     */
    ~ProgressBar() {
        if (owns_out) {
            owned_out.reset();
            out_ptr = nullptr;
        }
    }

    /**
     * @brief Run a callable repeatedly until it returns false or indefinitely if no maximum is set.
     * @param f Callable that receives the 1-based step index and returns true to continue or false
     * to stop.
     * @param desc Optional description shown before the extra text.
     * @param extra_fn Optional function called after each step with the step index, returning a
     * string to append to the status.
     * @details
     * This method runs the provided callable repeatedly until it returns false or indefinitely
     * if no maximum number of steps is set. The callable receives the 1-based step index
     * and should return true to continue or false to stop. This matches a while-loop
     * semantics (e.g., stop when some runtime condition is met). The callable
     * is typed as std::function<bool(unsigned int)> for clearer diagnostics. Example usage:
     * @code{cpp}
     * ProgressBar progress;
     * progress.for_while([&](unsigned int step) -> bool {
     *     // simulate work
     *     std::this_thread::sleep_for(std::chrono::milliseconds(100));
     *     // stop after 50 steps
     *     return step < 50;
     * }, "Amazing work");
     * @endcode
     * <br>
     * Output (e.g., after 20 steps):
     * @code
     * [/] (step 20)  Amazing work  elapsed=00:00:02
     * @endcode
     */
    void for_while(const std::function<bool(unsigned int)>        &f,
                   const std::string                              &desc     = "",
                   const std::function<std::string(unsigned int)> &extra_fn = nullptr) {
        for_while(f, std::numeric_limits<unsigned int>::max(), desc, extra_fn);
    }
    /**
     * @brief Run a callable for up to max_steps or until it returns false.
     * @param f Callable that receives the 1-based step index and returns true to continue or false
     * to stop.
     * @param max_steps Maximum number of steps to run.
     * @param desc Optional description shown before the extra text.
     * @param extra_fn Optional function called after each step with the step index, returning a
     * string to append to the status.
     * @details
     * This method runs the provided callable for up to max_steps or until it returns false.
     * The callable receives the 1-based step index and should return true to continue or false to
     * stop. This matches a while-loop semantics (e.g., stop when some runtime condition is met).
     * The callable is typed as std::function<bool(unsigned int)> for clearer diagnostics.
     * Example usage:
     * @code{cpp}
     * ProgressBar progress;
     * progress.for_while([&](unsigned int step) -> bool {
     *    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // simulate work
     *    return step < 50; // stop after 50 steps
     * }, 100, "Amazing work");
     * @endcode
     * <br>
     * Output (e.g., after 20 steps):
     * @code
     * [/] (step 20)  Amazing work  elapsed=00:00:02
     * @endcode
     */
    void for_while(const std::function<bool(unsigned int)> &f,
                   const unsigned int max_steps = std::numeric_limits<unsigned int>::max(),
                   const std::string &desc      = "",
                   const std::function<std::string(unsigned int)> &extra_fn = nullptr) {
        (this->*for_while_dispatch)(f, max_steps, desc, extra_fn);
    }

    /**
     * @brief Run a callable for each step in a specified range.
     * @param start_index Starting index (1-based). Where 1-based means the first step is 1.
     * @param end_index Ending index (1-based).
     * @param f Callable that receives the 1-based step index.
     * @param desc Optional description shown before the extra text.
     * @param extra_fn Optional function called after each step with the step index, returning a
     * string to append to the status.
     * @details
     * This method runs the provided callable for each step in the specified range
     * [from start_index to end_index]. The callable receives the 1-based step index.
     * This matches a traditional for-loop semantics. The callable is typed as
     * std::function<void(unsigned int)> for clearer diagnostics. Example usage:
     * @code{cpp}
     * ProgressBar progress(100); // known total of 100 steps
     * const unsigned int start = 1;
     * const unsigned int end   = 100;
     * const std::string description = "Description";
     * progress.for_each(start, end, [&](unsigned int step) -> void {
     *    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // simulate work
     * }, description);
     * @endcode
     * <br>
     * Output (e.g., after 40 steps):
     * @code
     * [##############--------------------]  40.0% (40/100)  Description  elapsed=00:00:02
     * ETA=00:00:03
     * @endcode
     */
    void for_each(const unsigned int                              start_index,
                  const unsigned int                              end_index,
                  const std::function<void(unsigned int)>        &f,
                  const std::string                              &desc     = "",
                  const std::function<std::string(unsigned int)> &extra_fn = nullptr) {
        (this->*for_each_dispatch)(start_index, end_index, f, desc, extra_fn);
    }

    /**
     * @brief Represent a range of steps for use in range-based for-loops.
     * @details
     * This struct represents a range of steps that can be used in range-based for-loops.
     * It provides an iterator that yields step indices and performs progress bar updates
     * after each step. Example usage:
     * @code{cpp}
     * ProgressBar progress(100); // known total of 100 steps
     * for (auto step : progress.range(1, 100, "Description")) {
     *    // simulate work
     *    std::this_thread::sleep_for(std::chrono::milliseconds(50));
     *    // use 'step' as an unsigned int
     * }
     * @endcode
     * <br>
     * Output (e.g., after 40 steps):
     * @code
     * [##############--------------------]  40.0% (40/100)  Description  elapsed=00:00:02
     * ETA=00:00:03
     * @endcode
     */
    struct Range {
        /**
         * @brief Pointer to the parent ProgressBar.
         * @details
         * This pointer allows the Range to call back into the ProgressBar
         * to perform timing updates and printing.
         */
        ProgressBar *bar;
        /**
         * @brief Starting index of the range (1-based).
         */
        unsigned int start;
        /**
         * @brief Ending index of the range (1-based).
         */
        unsigned int end_idx;
        /**
         * @brief Optional description shown before the extra text.
         */
        std::string desc;
        /**
         * @brief Optional function called after each step with the step index,
         * returning a string to append to the status.
         */
        std::function<std::string(unsigned int)> extra_fn;

        /**
         * @brief Construct a new Range object.
         * @param b Pointer to the parent ProgressBar.
         * @param s Starting index (1-based).
         * @param e Ending index (1-based).
         * @param d Optional description shown before the extra text.
         * @param ef Optional function called after each step with the step index,
         * returning a string to append to the status.
         */
        Range(ProgressBar                             *b,
              const unsigned int                       s,
              const unsigned int                       e,
              std::string                              d  = "",
              std::function<std::string(unsigned int)> ef = nullptr)
            : bar(b), start(s), end_idx(e), desc(std::move(d)), extra_fn(std::move(ef)) {}

        /**
         * @brief Proxy object representing a single step in the iteration.
         * @details
         * This proxy object allows implicit conversion to unsigned int
         * so that user code can use the step index directly. The destructor
         * performs the post-step timing update and printing.
         */
        struct StepProxy {
            /**
             * @brief Current step index (1-based).
             */
            unsigned int step;
            /**
             * @brief Pointer to the parent Range.
             */
            Range *parent;

            /**
             * @brief Construct a new StepProxy object.
             * @param s Current step index (1-based).
             * @param p Pointer to the parent Range.
             */
            StepProxy(const unsigned int s, Range *p) : step(s), parent(p) {}

            /**
             * @brief Implicit conversion to unsigned int.
             * @return Current step index as unsigned int.
             * @details
             * This operator allows user code to use the StepProxy directly as an
             * unsigned int representing the step index.
             * It is necessary to allow range-based for-loops to work seamlessly.
             */
            explicit operator unsigned int() const { return step; }

            /**
             * @brief Destroy the StepProxy object.
             * @details
             * The destructor performs the post-step timing update and printing
             * by calling back into the parent ProgressBar.
             */
            ~StepProxy() {
                // If parent ProgressBar exists, perform timing update and print
                if (parent && parent->bar) {
                    std::string extra;
                    // If extra_fn is provided, call it to get extra string
                    if (parent->extra_fn)
                        extra = parent->extra_fn(step);
                    parent->bar->step_completed(step, extra, step >= parent->end_idx);
                }
            }
        };

        /**
         * @brief Iterator for the Range.
         * @details
         * This iterator yields StepProxy objects representing each step
         * in the range. It supports standard iterator operations.
         */
        struct iterator {
            /**
             * @brief Current step index (1-based).
             */
            unsigned int cur;
            /**
             * @brief Pointer to the parent Range.
             */
            Range *parent;

            /**
             * @brief Construct a new iterator object.
             * @param c Current step index (1-based).
             * @param p Pointer to the parent Range.
             */
            explicit iterator(const unsigned int c = 0, Range *p = nullptr) : cur(c), parent(p) {}

            /**
             * @brief Dereference operator to get the current StepProxy.
             * @return StepProxy object for the current step.
             */
            StepProxy operator*() const { return StepProxy(cur, parent); }

            /**
             * @brief Pre-increment operator to advance the iterator.
             * @return Reference to the incremented iterator.
             */
            iterator &operator++() {
                ++cur;
                return *this;
            }

            /**
             * @brief Inequality operator to compare iterators.
             * @param o Other iterator to compare with.
             * @return true if the iterators are not equal, false otherwise.
             */
            bool operator!=(const iterator &o) const { return cur != o.cur; }
        };

        /**
         * @brief Get the 'begin' iterator for the range.
         * @return Iterator pointing to the start of the range.
         * @details
         * This method also performs the initial print of the progress bar
         * before starting the iteration.
         */
        iterator begin() {
            // Print an initial status line (0% or just before start)
            if (bar)
                bar->initial_print((start > 0) ? (start - 1) : 0, desc);
            return iterator(start, this);
        }

        /**
         * @brief Get the 'end' iterator for the range.
         * @return Iterator pointing just past the end of the range.
         * @details
         * This method returns an iterator pointing just past the end of the range.
         */
        iterator end_iter() { return iterator(end_idx + 1, this); }

        /**
         * @brief Get the 'end' iterator for the range.
         * @return Iterator pointing just past the end of the range.
         */
        iterator end() { return end_iter(); }
    };

    /**
     * @brief Create a Range object for use in range-based for-loops.
     * @param s Starting index (1-based).
     * @param e Ending index (1-based).
     * @param desc Optional description shown before the extra text.
     * @param extra_fn Optional function called after each step with the step index,
     * returning a string to append to the status.
     * @return Range object representing the specified range of steps.
     * @details
     * This method creates a Range object that can be used in range-based for-loops.
     * The Range provides an iterator that yields step indices and performs progress
     * bar updates after each step. Example usage:
     * @code{cpp}
     * ProgressBar progress(100); // known total of 100 steps
     * for (auto step : progress.range(1, 100, "Description")) {
     *    // simulate work
     *    std::this_thread::sleep_for(std::chrono::milliseconds(50));
     *    // use 'step' as an unsigned int
     * }
     * @endcode
     * <br>
     * Output (e.g., after 40 steps):
     * @code
     * [##############--------------------]  40.0% (40/100)  Description  elapsed=00:00:02
     * ETA=00:00:03
     * @endcode
     */
    Range range(const unsigned int                       s,
                const unsigned int                       e,
                const std::string                       &desc     = "",
                std::function<std::string(unsigned int)> extra_fn = nullptr) {
        return (this->*range_dispatch)(s, e, desc, std::move(extra_fn));
    }

    /**
     * @brief Enable/disable the progress bar at runtime.
     * @details
     * When disabled, the public loop helpers (`for_each`, `for_while`, `range`) become no-ops
     * with respect to timing/printing: they still execute the user-provided loop body, but
     * they do not update or print the progress bar.
     */
    void set_enabled(const bool on) {
        enabled = on;
        update_dispatch();
    }

    /**
     * @brief Return whether the progress bar is enabled at runtime.
     */
    [[nodiscard]] bool is_enabled() const { return enabled; }

private:
    /**
     * @brief Whether the output is interactive (i.e., a TTY). Automatically detected.
     * @details
     * If the output is not a TTY (e.g., redirected to a file), the progress bar
     * uses coarser updates to avoid cluttering the output with carriage returns.
     */
    bool interactive_output = true;
    /**
     * @brief Whether the progress bar is enabled at runtime.
     * @details
     * When disabled, the public loop helpers (`for_each`, `for_while`, `range`) become no-ops
     * with respect to timing/printing: they still execute the user-provided loop body, but
     * they do not update or print the progress bar.
     */
    bool enabled = true;

    /**
     * @brief Total number of steps in the loop (0 for unknown total).
     */
    unsigned int total_steps;
    /**
     * @brief Width of the progress bar in characters.
     */
    unsigned int bar_width;
    /**
     * @brief Number of recent steps to average for ETA smoothing.
     */
    unsigned int smoothing;
    /**
     * @brief Whether to use smooth Unicode characters for the progress bar.
     */
    bool smooth_unicode = true;

    /**
     * @brief Last printed percentage for coarse logging in non-TTY output.
     * @details
     * In non-interactive output (non-TTY), the progress bar uses coarser updates
     * to avoid cluttering the output with carriage returns. This variable tracks
     * the last printed percentage to determine when to print the next update.
     * Initially set to -1 to ensure the first update is printed; then updated in
     * increments of coarse_percent_step.
     */
    int last_percent_print = -1; // for coarse logs
    /**
     * @brief Coarse percentage step for printing in non-TTY output.
     * @details
     * In non-interactive output (non-TTY), the progress bar prints updates
     * every coarse_percent_step percentage points to avoid cluttering the output.
     * For example, with a value of 5, the progress bar prints updates at 0%, 5%, 10%, etc.
     */
    unsigned int coarse_percent_step = 5;
    /**
     * @brief Coarse step interval for printing in unknown total mode.
     * @details
     * In unknown total mode, the progress bar prints updates every
     * coarse_unknown_every step to avoid cluttering the output.
     * For example, with a value of 25, the progress bar prints updates
     * every 25 steps.
     */
    unsigned int coarse_unknown_every = 25;

    /**
     * @brief Start time point of the progress bar.
     */
    std::chrono::steady_clock::time_point start;
    /**
     * @brief Last time point when a step was completed.
     */
    std::chrono::steady_clock::time_point last_time;
    /**
     * @brief Last completed step index. Used to compute step durations for smoothing.
     */
    unsigned int last_step_index;

    /**
     * @brief Deque of recent step durations (in seconds) for smoothing ETA estimates.
     */
    std::deque<double> recent_steps;

    /**
     * @brief ConditionalOStream for output.
     * @details
     * This pointer may point to an external ConditionalOStream provided by the user
     * or to an owned instance created internally. The owns_out flag indicates ownership.
     */
    ConditionalOStream *out_ptr = nullptr;
    /**
     * @brief Owned ConditionalOStream if created internally. Ownership is tracked by owns_out.
     */
    std::unique_ptr<ConditionalOStream> owned_out;
    /**
     * @brief Whether this ProgressBar owns the ConditionalOStream.
     */
    bool owns_out = false;

    /**
     * @brief Type alias for member function pointer used in dispatching for for-loops.
     * @details
     * This alias defines the signature of the member function used for dispatching
     * between enabled and disabled states of the progress bar for for-loops.
     */
    using ForEachMemFn = void (ProgressBar::*)(unsigned int,
                                               unsigned int,
                                               const std::function<void(unsigned int)> &,
                                               const std::string &,
                                               const std::function<std::string(unsigned int)> &);
    /**
     * @brief Type alias for member function pointer used in dispatching for while-loops.
     * @details
     * This alias defines the signature of the member function used for dispatching
     * between enabled and disabled states of the progress bar for while-loops.
     */
    using ForWhileMemFn = void (ProgressBar::*)(const std::function<bool(unsigned int)> &,
                                                unsigned int,
                                                const std::string &,
                                                const std::function<std::string(unsigned int)> &);
    /**
     * @brief Type alias for member function pointer used in dispatching for range-based loops.
     * @details
     * This alias defines the signature of the member function used for dispatching
     * between enabled and disabled states of the progress bar for range-based loops.
     */
    using RangeMemFn = Range (ProgressBar::*)(unsigned int,
                                              unsigned int,
                                              const std::string &,
                                              std::function<std::string(unsigned int)>);

    /**
     * @brief Member function pointer for dispatching for-loops based on enabled state.
     * @details
     * This member function pointer is set to point to either the enabled or disabled
     * implementation of the for-loop helper based on the current enabled state of the
     * progress bar.
     */
    ForEachMemFn for_each_dispatch = &ProgressBar::for_each_enabled;
    /**
     * @brief Member function pointer for dispatching while-loops based on enabled state.
     * @details
     * This member function pointer is set to point to either the enabled or disabled
     * implementation of the while-loop helper based on the current enabled state of the
     * progress bar.
     */
    ForWhileMemFn for_while_dispatch = &ProgressBar::for_while_enabled;
    /**
     * @brief Member function pointer for dispatching range-based loops based on enabled state.
     * @details
     * This member function pointer is set to point to either the enabled or disabled
     * implementation of the range-based loop helper based on the current enabled state of the
     * progress bar.
     */
    RangeMemFn range_dispatch = &ProgressBar::range_enabled;

    /**
     * @brief Update the dispatch function pointers based on the enabled state.
     * @details
     * This method sets the member function pointers for dispatching between
     * enabled and disabled implementations of the loop helpers based on the
     * current enabled state of the progress bar.
     */
    void update_dispatch() {
        if (enabled) {
            for_each_dispatch  = &ProgressBar::for_each_enabled;
            for_while_dispatch = &ProgressBar::for_while_enabled;
            range_dispatch     = &ProgressBar::range_enabled;
        } else {
            for_each_dispatch  = &ProgressBar::for_each_disabled;
            for_while_dispatch = &ProgressBar::for_while_disabled;
            range_dispatch     = &ProgressBar::range_disabled;
        }
    }

    /**
     * @brief Enabled implementation of for_while loop helper.
     * @details
     * This method implements the for_while loop helper when the progress bar
     * is enabled. It tracks progress, updates timing, and prints the progress bar
     * after each step. See `ProgressBar::for_while` for full documentation.
     */
    void for_while_enabled(const std::function<bool(unsigned int)>        &f,
                           const unsigned int                              max_steps,
                           const std::string                              &desc,
                           const std::function<std::string(unsigned int)> &extra_fn) {
        // Validate inputs
        if (!f)
            throw std::invalid_argument("ProgressBar::for_while requires a valid callable");
        if (max_steps == 0)
            return;

        // Reset timing history
        recent_steps.clear();
        last_time       = std::chrono::steady_clock::now();
        last_step_index = 0;

        // Initial print (before starting) (0% or just before start)
        do_print(0, desc.empty() ? "" : desc, /*end_line=*/false);

        // Main loop up to max_steps
        for (unsigned int step = 1; step <= max_steps; ++step) {
            // Run user function; it returns true to continue, false to stop
            const bool cont = f(step);

            // Measure step duration and update smoothing window
            const auto   now = std::chrono::steady_clock::now();
            const double step_seconds =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - last_time)
                            .count();
            // Only record positive durations
            if (step_seconds > 0.0) {
                // Update recent steps window
                recent_steps.push_back(step_seconds);
                // Maintain window size
                if (recent_steps.size() > smoothing)
                    recent_steps.pop_front();
            }
            // Update last time/index
            last_time       = now;
            last_step_index = step;

            // Build extra string from desc + extra_fn
            std::string extra;
            if (!desc.empty())
                extra = desc;
            if (extra_fn) {
                if (const std::string s = extra_fn(step); !s.empty()) {
                    if (!extra.empty())
                        extra += "  ";
                    extra += s;
                }
            }

            // Print status line
            do_print(step, extra, /*end_line=*/(!cont || step >= max_steps));

            // Stop if user function returned false
            if (!cont)
                break;
        }
    }

    /**
     * @brief Enabled implementation of for_each loop helper.
     * @details
     * This method implements the for_each loop helper when the progress bar
     * is enabled. It tracks progress, updates timing, and prints the progress bar
     * after each step. See `ProgressBar::for_each` for full documentation.
     */
    void for_each_enabled(const unsigned int                              start_index,
                          const unsigned int                              end_index,
                          const std::function<void(unsigned int)>        &f,
                          const std::string                              &desc,
                          const std::function<std::string(unsigned int)> &extra_fn) {
        // Validate inputs
        if (!f)
            throw std::invalid_argument("ProgressBar::for_each requires a valid callable");
        if (end_index < start_index)
            return;

        // Determine current step for initial print
        const unsigned int current_step = std::max(start_index, 1u) - 1u;

        // Reset timing history for a fresh run
        recent_steps.clear();
        last_time       = std::chrono::steady_clock::now();
        last_step_index = current_step;

        // Initial print (0% or before starting)
        do_print(current_step, !desc.empty() ? desc : "", /*end_line=*/false);

        // Main loop over the specified range
        for (unsigned int step = start_index; step <= end_index; ++step) {
            // Run user function
            f(step);

            // Measure step duration and update smoothing window
            const auto   now = std::chrono::steady_clock::now();
            const double step_seconds =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - last_time)
                            .count();
            // Only record positive durations
            if (step_seconds > 0.0) {
                // Update recent steps window
                recent_steps.push_back(step_seconds);
                // Maintain window size
                if (recent_steps.size() > smoothing)
                    recent_steps.pop_front();
            }
            // Update last time/index
            last_time       = now;
            last_step_index = step;

            // Build extra string from desc + extra_fn
            std::string extra;
            if (!desc.empty())
                extra = desc;
            if (extra_fn) {
                if (const std::string s = extra_fn(step); !s.empty()) {
                    if (!extra.empty())
                        extra += "  ";
                    extra += s;
                }
            }

            // Print status line
            do_print(step, extra, /*end_line=*/step >= end_index);
        }
    }

    /**
     * @brief Enabled implementation of range-based loop helper.
     * @details
     * This method implements the range-based loop helper when the progress bar
     * is enabled. It returns a Range object that can be used in range-based for-loops.
     * See `ProgressBar::range` for full documentation.
     */
    Range range_enabled(const unsigned int                       s,
                        const unsigned int                       e,
                        const std::string                       &desc,
                        std::function<std::string(unsigned int)> extra_fn) {
        return Range(this, s, e, desc, std::move(extra_fn));
    }

    /**
     * @brief Disabled implementation of for_while loop helper.
     * @details
     * This method implements the for_while loop helper when the progress bar
     * is disabled. It simply runs the user-provided callable without any
     * timing or printing.
     */
    void for_while_disabled(const std::function<bool(unsigned int)> &f,
                            const unsigned int                       max_steps,
                            const std::string & /*desc*/,
                            const std::function<std::string(unsigned int)> & /*extra_fn*/) {
        if (!f)
            throw std::invalid_argument("ProgressBar::for_while requires a valid callable");
        if (max_steps == 0)
            return;

        for (unsigned int step = 1; step <= max_steps; ++step) {
            if (!f(step))
                break;
        }
    }

    void for_each_disabled(const unsigned int                       start_index,
                           const unsigned int                       end_index,
                           const std::function<void(unsigned int)> &f,
                           const std::string & /*desc*/,
                           const std::function<std::string(unsigned int)> & /*extra_fn*/) {
        if (!f)
            throw std::invalid_argument("ProgressBar::for_each requires a valid callable");
        if (end_index < start_index)
            return;

        for (unsigned int step = start_index; step <= end_index; ++step)
            f(step);
    }

    Range range_disabled(const unsigned int s,
                         const unsigned int e,
                         const std::string & /*desc*/,
                         std::function<std::string(unsigned int)> /*extra_fn*/) {
        // Return a Range with a null bar: it will iterate but won't print/update.
        return Range(nullptr, s, e, "", nullptr);
    }

    /**
     * @brief Do a print of the progress bar status line.
     * @param current_step Current step index (1-based).
     * @param extra Extra string to append to the status.
     * @param end_line Whether to end the line after printing.
     * @details
     * This method performs the actual printing of the progress bar status line.
     * It handles both known and unknown total step modes, computes percentages,
     * builds the progress bar, and formats elapsed time and ETA estimates.
     */
    void do_print(const unsigned int current_step, const std::string &extra, const bool end_line) {
        // If total_steps == 0 we are in "unknown total" mode
        // If total_steps == 0 we are in "unknown total" mode
        if (total_steps == 0) {
            // The spinner array cycles through these characters
            static constexpr char  spinner[]   = {'|', '/', '-', '\\'};
            constexpr unsigned int spinner_len = std::size(spinner);
            // The frame is based on current_step modulo spinner length
            const char frame = spinner[current_step % spinner_len];

            const long elapsed_local = std::chrono::duration_cast<std::chrono::seconds>(
                                               std::chrono::steady_clock::now() - start)
                                               .count();

            // ----------------------------
            // NON-INTERACTIVE (logs)
            // ----------------------------
            if (!interactive_output || !enabled) {
                // print only every N steps, plus the final line
                if (!end_line && (current_step % coarse_unknown_every != 0))
                    return;

                std::ostringstream ss;
                ss << '[' << frame << "] "
                   << "(step " << current_step << ")";
                if (!extra.empty())
                    ss << "  " << extra;
                ss << "  elapsed=" << format_hms(elapsed_local) << '\n';

                if (out_ptr)
                    (void) (*out_ptr << ss.str() << std::flush);
                return;
            }

            // ----------------------------
            // INTERACTIVE (TTY)
            // ----------------------------
            std::ostringstream ss;
            ss << '\r';
            ss << '[' << frame << "] "
               << "(step " << current_step << ")";
            if (!extra.empty())
                ss << "  " << extra;
            ss << "  elapsed=" << format_hms(elapsed_local);

            if (end_line)
                ss << '\n';

            if (out_ptr)
                (void) (*out_ptr << ss.str() << std::flush);
            return;
        }

        // Compute percentage complete
        double percent = 0.0;
        // To avoid division by zero
        if (total_steps > 0)
            percent = 100.0 * static_cast<double>(current_step) / static_cast<double>(total_steps);

        // Smoothed ETA based on average of recent steps (seconds per step)
        double eta_seconds = -1.0;
        // Only compute ETA if we have recent steps and not yet complete
        if (!recent_steps.empty() && current_step < total_steps) {
            const double avg = std::accumulate(recent_steps.begin(), recent_steps.end(), 0.0) /
                               static_cast<double>(recent_steps.size());
            eta_seconds = avg * static_cast<double>(total_steps - current_step);
        }

        // Build ASCII or smooth Unicode bar
        std::string bar;
        if (smooth_unicode) {
            // Compute fractional fill across bar_width characters
            const double frac =
                    (total_steps > 0) ? (percent / 100.0) * static_cast<double>(bar_width) : 0.0;
            const auto   full_blocks = static_cast<unsigned int>(std::floor(frac));
            const double rem         = frac - static_cast<double>(full_blocks);

            // Unicode partial blocks from smallest to largest (1/8 .. 7/8)
            static const std::vector<std::string> partial = {
                    "\u258F", "\u258E", "\u258D", "\u258C", "\u258B", "\u258A", "\u2589"};

            // Full block: U+2588
            for (unsigned int i = 0; i < full_blocks && i < bar_width; ++i)
                bar += "\u2588";

            if (full_blocks < bar_width) {
                // Compute an index into partial[] from the fractional remainder;
                // Scale rem by the number of partial levels, round to nearest,
                // then subtract 1 to obtain a 0-based index (or -1 if below 1/2)
                const double scaled  = rem * static_cast<double>(partial.size());
                const int    rounded = static_cast<int>(std::floor(scaled + 0.5));
                const int    idx     = rounded - 1; // idx can be -1 .. (partial.size()-1)

                // Add partial block if idx >= 0
                if (idx >= 0 && static_cast<unsigned int>(idx) < partial.size())
                    bar += partial[static_cast<unsigned int>(idx)];

                // Fill the rest with dashes
                const unsigned int used      = full_blocks + (idx >= 0 ? 1u : 0u);
                const unsigned int remaining = (used < bar_width) ? (bar_width - used) : 0u;
                for (unsigned int i = 0; i < remaining; ++i)
                    bar += '-';
            }
        } else {
            // ASCII bar with '#' and '-'
            const unsigned int filled =
                    (total_steps > 0)
                            ? static_cast<unsigned int>(std::round((percent / 100.0) * bar_width))
                            : 0u;
            bar.reserve(bar_width);
            for (unsigned int i = 0; i < bar_width; ++i)
                bar.push_back(i < filled ? '#' : '-');
        }

        // Elapsed time since start
        const long elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();

        // ----------------------------
        // NON-INTERACTIVE (logs)
        // ----------------------------
        if (!interactive_output || !enabled) {
            const int p = static_cast<int>(std::floor(percent));

            const bool should_print =
                    end_line || (p >= last_percent_print + static_cast<int>(coarse_percent_step)) ||
                    (current_step == 1) || (current_step == total_steps);

            if (!should_print)
                return;

            last_percent_print = p;

            std::ostringstream ss;
            ss << "[" << bar << "] ";
            ss << std::fixed << std::setprecision(1) << std::setw(5) << percent << "% ";
            ss << "(" << current_step << "/" << total_steps << ")";

            if (!extra.empty())
                ss << "  " << extra;

            ss << "  elapsed=" << format_hms(elapsed);

            if (eta_seconds >= 0.0)
                ss << "  ETA=" << format_hms(std::llround(eta_seconds));
            else
                ss << "  ETA=--:--:--";

            ss << '\n';

            if (out_ptr)
                (void) (*out_ptr << ss.str() << std::flush);

            return;
        }

        std::ostringstream ss;
        ss << '\r'; // carriage-return to overwrite the same line
        ss << "[" << bar << "] ";
        ss << std::fixed << std::setprecision(1) << std::setw(5) << percent << "% ";
        ss << "(" << current_step;
        if (total_steps > 0)
            ss << "/" << total_steps;
        ss << ")";

        // Append extra string if provided
        if (!extra.empty()) {
            ss << "  " << extra;
        }

        ss << "  elapsed=" << format_hms(elapsed);

        // Append ETA if available
        if (eta_seconds >= 0.0) {
            ss << "  ETA=" << format_hms(std::llround(eta_seconds));
        } else {
            ss << "  ETA=--:--:--";
        }

        // End line if requested
        if (end_line && total_steps > 0)
            ss << '\n';

        // If out_ptr is valid, print to it
        if (out_ptr) {
            (void) (*out_ptr << ss.str() << std::flush);
        }
    }

    /**
     * @brief Update the progress bar after a step is completed.
     * @param step Completed step index (1-based).
     * @param extra Extra string to append to the status.
     * @param end_line Whether to end the line after printing.
     * @details
     * This method updates the internal timing windows based on the completed step
     * and then calls do_print() to display the updated progress bar status line.
     */
    void step_completed(const unsigned int step, const std::string &extra, const bool end_line) {
        // Measure duration of this step (time since last completed step)
        using namespace std::chrono;
        const auto   now          = steady_clock::now();
        const double step_seconds = duration_cast<duration<double>>(now - last_time).count();

        // Only record positive durations
        if (step > last_step_index && step_seconds > 0.0) {
            recent_steps.push_back(step_seconds);
            if (recent_steps.size() > smoothing)
                recent_steps.pop_front();
        }

        // Update last time/index
        last_time       = now;
        last_step_index = step;

        // Print updated status line
        do_print(step, extra, end_line);
    }

    /**
     * @brief Initial print of the progress bar before starting.
     * @param prev_step Previous step index (0-based).
     * @param desc Description string to show.
     */
    void initial_print(const unsigned int prev_step, const std::string &desc) {
        do_print(prev_step, desc, /*end_line=*/false);
    }

    /**
     * @brief Format seconds as HH:MM:SS string.
     * @param seconds Number of seconds.
     * @return Formatted string in HH:MM:SS format.
     */
    static std::string format_hms(long seconds) {
        if (seconds < 0)
            seconds = 0;
        const long         h = seconds / 3600;
        const long         m = (seconds % 3600) / 60;
        const long         s = seconds % 60;
        std::ostringstream ss;
        ss << std::setfill('0') << std::setw(2) << h << ":" << std::setw(2) << m << ":"
           << std::setw(2) << s;
        return ss.str();
    }
};

#endif // NM4PDE_PROGRESS_BAR_HPP

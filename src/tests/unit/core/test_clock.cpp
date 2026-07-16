// Test for luisa::core::Clock class
// This test covers:
// - Clock construction and initialization
// - tic() functionality (reset timing)
// - toc() functionality (measure elapsed time)
// - Multiple tic/toc cycles

#include <thread>
#include <cmath>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include "ut/ut.hpp"

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

// Helper to sleep for a specified duration in milliseconds
void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

static auto test_clock_registration = [] {
    "test_clock"_test = [] {
        log_level_verbose();

        LUISA_INFO("=== Clock Test Started ===");

        // Test 1: Basic construction and toc() after construction
        {
            LUISA_INFO("Test 1: Construction timing...");
            Clock clock;
            sleep_ms(10);
            double elapsed = clock.toc();
            // Should be at least 10ms, with some tolerance
            expect(static_cast<bool>(elapsed >= 9.0));
            expect(static_cast<bool>(elapsed < 100.0));
            LUISA_INFO("  Elapsed after construction: {} ms", elapsed);
        }

        // Test 2: tic() resets the clock
        {
            LUISA_INFO("Test 2: tic() reset functionality...");
            Clock clock;
            sleep_ms(20);
            clock.tic();// Reset
            sleep_ms(5);
            double elapsed = clock.toc();
            // Should be ~5ms, not ~25ms
            expect(static_cast<bool>(elapsed >= 4.0 && elapsed < 50.0));
            LUISA_INFO("  Elapsed after tic(): {} ms", elapsed);
        }

        // Test 3: Multiple tic/toc cycles
        {
            LUISA_INFO("Test 3: Multiple tic/toc cycles...");
            Clock clock;

            for (int i = 0; i < 3; ++i) {
                clock.tic();
                sleep_ms(5 * (i + 1));
                double elapsed = clock.toc();
                double expected = 5.0 * (i + 1);
                expect(static_cast<bool>(elapsed >= expected - 2.0 && elapsed < expected + 50.0));
                LUISA_INFO("  Cycle {}: elapsed = {} ms (expected ~{} ms)", i, elapsed, expected);
            }
        }

        // Test 4: toc() does not reset the clock (consecutive calls should return increasing values)
        {
            LUISA_INFO("Test 4: toc() does not reset clock...");
            Clock clock;
            sleep_ms(5);
            double t1 = clock.toc();
            sleep_ms(5);
            double t2 = clock.toc();
            expect(static_cast<bool>(t2 > t1));
            LUISA_INFO("  First toc(): {} ms, Second toc(): {} ms", t1, t2);
        }

        // Test 5: Zero or near-zero elapsed time
        {
            LUISA_INFO("Test 5: Near-zero elapsed time...");
            Clock clock;
            clock.tic();
            // Minimal delay (just function call overhead)
            double elapsed = clock.toc();
            expect(static_cast<bool>(elapsed >= 0.0));
            expect(static_cast<bool>(elapsed < 10.0));
            LUISA_INFO("  Near-zero elapsed: {} ms", elapsed);
        }

        // Test 6: High precision test (shorter intervals)
        {
            LUISA_INFO("Test 6: High precision test...");
            Clock clock;
            clock.tic();
            sleep_ms(1);
            double elapsed = clock.toc();
            // 1ms sleep should result in measurable time
            expect(static_cast<bool>(elapsed > 0.0));
            LUISA_INFO("  1ms sleep measured: {} ms", elapsed);
        }

        LUISA_INFO("=== All Clock Tests Passed ===");
    };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}

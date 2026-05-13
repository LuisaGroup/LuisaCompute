#include "tests.h"
#include "cli_entry.h"
#include <cstring>

int main(int argc, char *argv[]) {
    bool test_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test") == 0) {
            test_mode = true;
            break;
        }
    }

    if (test_mode) {
        // run test suite when --test is passed
        return run_tests(argc, argv);
    }
    // otherwise run interactive CLI
    return run_cli(argc, argv);
}

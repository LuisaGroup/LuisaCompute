#ifndef NDEBUG
#define NDEBUG
#endif

#include "ut/ut.hpp"

#include "log.h"

#if defined(__unix__) || defined(__APPLE__)
#include <csignal>
#include <cstdlib>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_release_result_contract_evaluates_success_once"_test = [] {
        auto evaluations = 0;
        VK_CHECK_RESULT(
            (++evaluations, static_cast<VkResult>(VK_SUCCESS)));
        expect(evaluations == 1_i);
    };

#if defined(__unix__) || defined(__APPLE__)
    "vk_release_result_contract_rejects_device_loss"_test = [] {
        const auto child = fork();
        expect(child >= 0_i);
        if (child == 0) {
            const rlimit no_core{0u, 0u};
            static_cast<void>(setrlimit(RLIMIT_CORE, &no_core));
            VK_CHECK_RESULT(
                static_cast<VkResult>(VK_ERROR_DEVICE_LOST));
            _exit(EXIT_SUCCESS);
        }

        int status = 0;
        expect(waitpid(child, &status, 0) == child);
        expect(WIFSIGNALED(status));
        expect(WTERMSIG(status) == SIGABRT);
    };
#endif

    return 0;
}

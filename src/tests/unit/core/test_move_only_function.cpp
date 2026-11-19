#include <memory>
#include <utility>
#include <luisa/core/stl/functional.h>
#include "ut/ut.hpp"

using namespace boost::ut;

namespace {
struct Report {
    int value{};
    Report() = default;
    Report(const Report &) = delete;
    Report &operator=(const Report &) = delete;
};
}

int main() {
    "move_only_function_reference_arguments"_test = [] {
        int module = 7;
        Report report;
        luisa::move_only_function<bool(int *, Report &)> callback = [](int *m, Report &r) {
            r.value = *m;
            return true;
        };
        expect(callback(&module, report));
        expect(report.value == 7);
        auto moved = std::move(callback);
        module = 11;
        expect(moved(&module, report));
        expect(report.value == 11);
    };
    "move_only_function_move_and_const_arguments"_test = [] {
        auto offset = std::make_unique<int>(3);
        luisa::move_only_function<int(std::unique_ptr<int>, const int &) noexcept> callback =
            [offset = std::move(offset)](std::unique_ptr<int> value, const int &extra) noexcept {
                return *offset + *value + extra;
            };
        const int extra = 5;
        expect(callback(std::make_unique<int>(7), extra) == 15);
    };
}

#include "llvm_value_layout.h"

#include <iostream>
#include <string_view>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <luisa/ast/type_registry.h>

using namespace luisa::compute;
using namespace luisa::compute::simd;
using namespace luisa::compute::simd::schedule;

namespace {

[[nodiscard]] bool check(bool condition, const char *expression,
                         const char *file, int line) noexcept {
    if (!condition) {
        std::cerr << file << ':' << line << ": check failed: "
                  << expression << '\n';
    }
    return condition;
}

#define CHECK(EXPR)                                                           \
    do {                                                                      \
        if (!check(static_cast<bool>(EXPR), #EXPR, __FILE__, __LINE__)) {     \
            return false;                                                     \
        }                                                                     \
    } while (false)

template<size_t Width>
[[nodiscard]] bool test_layout() noexcept {
    ::llvm::LLVMContext context;
    LLVMValueLayout layout{context, Width};
    Value warp_uniform{
        .value_class = ValueClass::warp_uniform,
        .type = Type::of<uint32_t>(),
    };
    Value cohort_uniform{
        .value_class = ValueClass::cohort_uniform,
        .type = Type::of<uint32_t>(),
    };
    Value varying{
        .value_class = ValueClass::varying,
        .type = Type::of<float>(),
    };
    Value varying_vector{
        .value_class = ValueClass::varying,
        .type = Type::of<luisa::float3>(),
    };

    CHECK(layout.expression_type(warp_uniform)->isIntegerTy(32u));
    CHECK(layout.state_type(warp_uniform)->isIntegerTy(32u));
    CHECK(layout.expression_type(cohort_uniform)->isIntegerTy(32u));
    auto *cohort_state = ::llvm::dyn_cast<::llvm::FixedVectorType>(
        layout.state_type(cohort_uniform));
    CHECK(cohort_state != nullptr);
    CHECK(cohort_state->getNumElements() == Width);
    auto *varying_type = ::llvm::dyn_cast<::llvm::FixedVectorType>(
        layout.expression_type(varying));
    CHECK(varying_type != nullptr);
    CHECK(varying_type->getNumElements() == Width);

    auto *soa = ::llvm::dyn_cast<::llvm::ArrayType>(
        layout.expression_type(varying_vector));
    CHECK(soa != nullptr);
    CHECK(soa->getNumElements() == 3u);
    auto *component = ::llvm::dyn_cast<::llvm::FixedVectorType>(
        soa->getElementType());
    CHECK(component != nullptr);
    CHECK(component->getNumElements() == Width);
    CHECK(layout.mask_type()->getElementCount().getKnownMinValue() == Width);
    CHECK(layout.succeeded());
    return true;
}

[[nodiscard]] bool test_invalid_width() noexcept {
    ::llvm::LLVMContext context;
    LLVMValueLayout layout{context, 0u};
    CHECK(!layout.succeeded());
    CHECK(layout.mask_type() == nullptr);
    return true;
}

}// namespace

int main() {
    struct Test {
        std::string_view name;
        bool (*run)();
    };
    constexpr Test tests[]{
        {"layout warp1", &test_layout<1u>},
        {"layout warp4", &test_layout<4u>},
        {"layout warp8", &test_layout<8u>},
        {"layout warp16", &test_layout<16u>},
        {"invalid width", &test_invalid_width},
    };
    auto failures = 0u;
    for (auto test : tests) {
        if (test.run()) {
            std::cout << "[pass] " << test.name << '\n';
        } else {
            std::cerr << "[fail] " << test.name << '\n';
            ++failures;
        }
    }
    return failures == 0u ? 0 : 1;
}

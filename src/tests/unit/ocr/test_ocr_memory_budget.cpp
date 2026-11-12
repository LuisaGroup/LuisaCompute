// Test for ocr/runtime/memory_budget.h (Phase 2, host-only).
// Covers: checked category/component accounting, KV capacity formula,
// overflow refusal, limit preflight with useful required-vs-available
// messages, component release, additional-device preflight.
#include "ut/ut.hpp"

#include "ocr/runtime/memory_budget.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_accounting() {
    MemoryBudget budget;
    expect(budget.require("vision", MemoryCategory::weights, 1000).ok());
    expect(budget.require("vision", MemoryCategory::weights, 500).ok());
    expect(budget.require("decoder", MemoryCategory::weights, 2000).ok());
    expect(budget.require("decoder", MemoryCategory::workspace, 4096).ok());
    expect(budget.require("decoder", MemoryCategory::kv_cache, 8192).ok());
    expect(budget.require("decoder", MemoryCategory::shader_cache, 128).ok());
    expect(budget.require("staging", MemoryCategory::host_staging, 777).ok());
    expect(budget.component_bytes("vision") == 1500u);
    expect(budget.category_bytes(MemoryCategory::weights) == 3500u);
    expect(budget.device_bytes() == 1500u + 2000u + 4096u + 8192u + 128u);
    expect(budget.host_bytes() == 777u);
    expect(budget.total_bytes() == budget.device_bytes() + 777u);
    // set_requirement replaces (resized arena).
    expect(budget.set_requirement("decoder", MemoryCategory::workspace, 1024).ok());
    expect(budget.component_bytes("decoder") == 2000u + 1024u + 8192u + 128u);
    // release_component drops every category of that component.
    budget.release_component("vision");
    expect(budget.component_bytes("vision") == 0u);
    expect(budget.category_bytes(MemoryCategory::weights) == 2000u);
}

void test_overflow() {
    MemoryBudget budget;
    expect(budget.require("c", MemoryCategory::weights, UINT64_MAX - 10u).ok());
    auto s = budget.require("c", MemoryCategory::weights, 100u);
    expect(!s.ok());
    expect(s.code == StatusCode::invalid_argument);
    // KV plan overflow.
    auto kv = plan_kv_cache_bytes(UINT64_MAX / 2u, 4, 1024, 8, 128, 4);
    expect(!kv.ok());
}

void test_kv_formula() {
    // 28 layers x 1 request x 2048 tokens x 2 kv heads x 128 dim x 2 x 4 bytes.
    auto kv = plan_kv_cache_bytes(28, 1, 2048, 2, 128, 4);
    expect(kv.ok());
    expect(*kv.value == 28ull * 1 * 2048 * 2 * 128 * 2 * 4);
    auto zero = plan_kv_cache_bytes(0, 1, 2048, 2, 128, 4);
    expect(zero.ok());
    expect(*zero.value == 0u);
}

void test_preflight() {
    MemoryLimits limits;
    limits.max_device_bytes = 10000;
    limits.max_host_bytes = 1000;
    MemoryBudget budget(limits);
    expect(budget.require("m", MemoryCategory::weights, 6000).ok());
    expect(budget.require("m", MemoryCategory::workspace, 3000).ok());
    expect(budget.require("m", MemoryCategory::host_staging, 500).ok());
    expect(budget.preflight().ok());
    expect(budget.require("m", MemoryCategory::kv_cache, 2000).ok());// device 11000 > 10000
    auto s = budget.preflight();
    expect(!s.ok());
    expect(s.code == StatusCode::out_of_memory);
    expect(s.message.find("11000") != luisa::string::npos);
    expect(s.message.find("10000") != luisa::string::npos);
    // Host limit refusal.
    MemoryBudget b2(limits);
    expect(b2.require("m", MemoryCategory::host_staging, 1001).ok());
    expect(!b2.preflight().ok());
    // Additional-device preflight without recording.
    MemoryBudget b3(limits);
    expect(b3.require("m", MemoryCategory::weights, 9000).ok());
    expect(b3.preflight_additional_device(1000).ok());
    expect(!b3.preflight_additional_device(1001).ok());
    expect(b3.device_bytes() == 9000u);// unchanged
    // 0 limits = no explicit limit.
    MemoryBudget b4;
    expect(b4.require("m", MemoryCategory::weights, UINT64_MAX / 2u).ok());
    expect(b4.preflight().ok());
}

}// namespace

static auto test_ocr_memory_budget_registration = [] {
    "memory_budget_accounting"_test = [] { test_accounting(); };
    "memory_budget_overflow"_test = [] { test_overflow(); };
    "memory_budget_kv_formula"_test = [] { test_kv_formula(); };
    "memory_budget_preflight"_test = [] { test_preflight(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
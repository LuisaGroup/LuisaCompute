// Test for Vulkan SPIR-V motion blur patching.
// This test covers:
// - Field-store payload time extraction
// - Whole-payload composite time extraction

#include "ut/ut.hpp"

#include <cstdint>
#include <vector>

#include <luisa/vstl/common.h>

#include "spirv_motion_patch.h"

using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto spv_magic_number = 0x07230203u;
constexpr auto spv_op_capability = 17u;
constexpr auto spv_op_extension = 10u;
constexpr auto spv_op_type_void = 19u;
constexpr auto spv_op_constant = 43u;
constexpr auto spv_op_variable = 59u;
constexpr auto spv_op_store = 62u;
constexpr auto spv_op_access_chain = 65u;
constexpr auto spv_op_composite_construct = 80u;
constexpr auto spv_op_trace_ray_khr = 4445u;
constexpr auto spv_op_trace_ray_motion_nv = 5339u;
constexpr auto spv_capability_shader = 1u;
constexpr auto spv_capability_ray_tracing_motion_blur_nv = 5341u;

[[nodiscard]] constexpr uint32_t spv_word(uint32_t op, uint32_t wc) noexcept {
    return (wc << 16u) | op;
}

[[nodiscard]] uint32_t spv_opcode(uint32_t word) noexcept {
    return word & 0xffffu;
}

void append_extension(std::vector<uint32_t> &spirv) noexcept {
    spirv.emplace_back(spv_word(spv_op_extension, 3u));
    spirv.emplace_back(0x565053u);
    spirv.emplace_back(0u);
}

[[nodiscard]] std::vector<uint32_t> make_header() noexcept {
    std::vector<uint32_t> spirv{
        spv_magic_number,
        0x00010500u,
        0u,
        128u,
        0u,
        spv_word(spv_op_capability, 2u),
        spv_capability_shader};
    append_extension(spirv);
    spirv.emplace_back(spv_word(spv_op_type_void, 2u));
    spirv.emplace_back(1u);
    return spirv;
}

void append_trace_ray(std::vector<uint32_t> &spirv, uint32_t payload_id) noexcept {
    spirv.emplace_back(spv_word(spv_op_trace_ray_khr, 12u));
    spirv.emplace_back(20u);
    spirv.emplace_back(21u);
    spirv.emplace_back(22u);
    spirv.emplace_back(23u);
    spirv.emplace_back(24u);
    spirv.emplace_back(25u);
    spirv.emplace_back(26u);
    spirv.emplace_back(27u);
    spirv.emplace_back(28u);
    spirv.emplace_back(29u);
    spirv.emplace_back(payload_id);
}

[[nodiscard]] bool has_opcode(luisa::span<uint32_t const> spirv, uint32_t opcode) noexcept {
    for (auto i = 5u; i < spirv.size();) {
        auto wc = spirv[i] >> 16u;
        if (wc == 0u) { return false; }
        if (spv_opcode(spirv[i]) == opcode) { return true; }
        i += wc;
    }
    return false;
}

[[nodiscard]] bool trace_ray_motion_has_time(luisa::span<uint32_t const> spirv, uint32_t time_id) noexcept {
    for (auto i = 5u; i < spirv.size();) {
        auto wc = spirv[i] >> 16u;
        if (wc == 0u) { return false; }
        if (spv_opcode(spirv[i]) == spv_op_trace_ray_motion_nv) {
            return wc == 13u && spirv[i + 11u] == time_id;
        }
        i += wc;
    }
    return false;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    "nested_field_store_time"_test = [] {
        constexpr auto payload_id = 10u;
        constexpr auto time_ptr_id = 11u;
        constexpr auto time_id = 12u;
        auto spirv = make_header();
        spirv.emplace_back(spv_word(spv_op_constant, 4u));
        spirv.emplace_back(2u);
        spirv.emplace_back(3u);
        spirv.emplace_back(4u);
        spirv.emplace_back(spv_word(spv_op_variable, 4u));
        spirv.emplace_back(4u);
        spirv.emplace_back(payload_id);
        spirv.emplace_back(0u);
        spirv.emplace_back(spv_word(spv_op_access_chain, 6u));
        spirv.emplace_back(5u);
        spirv.emplace_back(time_ptr_id);
        spirv.emplace_back(payload_id);
        spirv.emplace_back(6u);
        spirv.emplace_back(3u);
        spirv.emplace_back(spv_word(spv_op_store, 3u));
        spirv.emplace_back(time_ptr_id);
        spirv.emplace_back(time_id);
        append_trace_ray(spirv, payload_id);

        auto patched = lc::vk::patch_spirv_for_motion_blur(spirv);
        expect(has_opcode(patched, spv_op_capability));
        expect(has_opcode(patched, spv_op_trace_ray_motion_nv));
        expect(!has_opcode(patched, spv_op_trace_ray_khr));
        expect(trace_ray_motion_has_time(patched, time_id));
    };

    "direct_field_store_time"_test = [] {
        constexpr auto payload_id = 10u;
        constexpr auto time_ptr_id = 11u;
        constexpr auto time_id = 12u;
        auto spirv = make_header();
        spirv.emplace_back(spv_word(spv_op_constant, 4u));
        spirv.emplace_back(2u);
        spirv.emplace_back(3u);
        spirv.emplace_back(4u);
        spirv.emplace_back(spv_word(spv_op_variable, 4u));
        spirv.emplace_back(4u);
        spirv.emplace_back(payload_id);
        spirv.emplace_back(0u);
        spirv.emplace_back(spv_word(spv_op_access_chain, 5u));
        spirv.emplace_back(5u);
        spirv.emplace_back(time_ptr_id);
        spirv.emplace_back(payload_id);
        spirv.emplace_back(3u);
        spirv.emplace_back(spv_word(spv_op_store, 3u));
        spirv.emplace_back(time_ptr_id);
        spirv.emplace_back(time_id);
        append_trace_ray(spirv, payload_id);

        auto patched = lc::vk::patch_spirv_for_motion_blur(spirv);
        expect(has_opcode(patched, spv_op_trace_ray_motion_nv));
        expect(!has_opcode(patched, spv_op_trace_ray_khr));
        expect(trace_ray_motion_has_time(patched, time_id));
    };

    "composite_store_time"_test = [] {
        constexpr auto payload_id = 10u;
        constexpr auto payload_value_id = 11u;
        constexpr auto time_id = 12u;
        auto spirv = make_header();
        spirv.emplace_back(spv_word(spv_op_composite_construct, 8u));
        spirv.emplace_back(4u);
        spirv.emplace_back(payload_value_id);
        spirv.emplace_back(30u);
        spirv.emplace_back(31u);
        spirv.emplace_back(32u);
        spirv.emplace_back(33u);
        spirv.emplace_back(time_id);
        spirv.emplace_back(spv_word(spv_op_store, 3u));
        spirv.emplace_back(payload_id);
        spirv.emplace_back(payload_value_id);
        append_trace_ray(spirv, payload_id);

        auto patched = lc::vk::patch_spirv_for_motion_blur(spirv);
        expect(has_opcode(patched, spv_op_trace_ray_motion_nv));
        expect(!has_opcode(patched, spv_op_trace_ray_khr));
        expect(trace_ray_motion_has_time(patched, time_id));
    };

    "no_trace_ray_unchanged"_test = [] {
        auto spirv = make_header();
        auto patched = lc::vk::patch_spirv_for_motion_blur(spirv);
        expect(patched.size() == spirv.size());
    };
}

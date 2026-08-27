// Test for XIR canonical text and bitcode interchange.
// This test covers:
// - Deterministic empty-module text and binary round-trips
// - Structured diagnostics for malformed text and bitcode envelopes
// - Ordered metadata and escaped metadata payload round-trips

#include "ut/ut.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#include <luisa/ast/type_registry.h>
#include <luisa/core/stl/format.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/metadata/signature_constraint.h>
#include <luisa/xir/module.h>
#include <luisa/xir/translators/xir_interchange.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] luisa::string metadata_name(luisa::string_view prefix) {
    return luisa::format("{}_name", prefix);
}

[[nodiscard]] luisa::string metadata_comment(luisa::string_view prefix, uint32_t index) {
    return luisa::format("{} comment {}: quoted=\"yes\", slash=\\\\, line=\nnext", prefix, index);
}

[[nodiscard]] luisa::filesystem::path metadata_path(luisa::string_view prefix) {
    return luisa::filesystem::path{luisa::format("root/{}/quoted\"path\\segment\nfile.xir", prefix)};
}

void attach_all_metadata(MetadataListMixin &owner, luisa::string_view prefix) {
    owner.metadata_list().push_front(luisa::make_managed<SignatureConstraintMD>());
    owner.metadata_list().push_front(luisa::make_managed<CurveBasisMD>(
        CurveBasisSet::make(CurveBasis::PIECEWISE_LINEAR,
                            CurveBasis::CATMULL_ROM,
                            CurveBasis::BEZIER)));
    owner.metadata_list().push_front(luisa::make_managed<CommentMD>(metadata_comment(prefix, 2u)));
    owner.metadata_list().push_front(luisa::make_managed<LocationMD>(metadata_path(prefix), 137));
    owner.metadata_list().push_front(luisa::make_managed<CommentMD>(metadata_comment(prefix, 1u)));
    owner.metadata_list().push_front(luisa::make_managed<NameMD>(metadata_name(prefix)));
}

void expect_all_metadata(const MetadataListMixin &owner, luisa::string_view prefix) {
    constexpr std::array expected_tags{
        DerivedMetadataTag::NAME,
        DerivedMetadataTag::COMMENT,
        DerivedMetadataTag::LOCATION,
        DerivedMetadataTag::COMMENT,
        DerivedMetadataTag::CURVE_BASIS,
        DerivedMetadataTag::SIGNATURE_CONSTRAINT};
    size_t index = 0u;
    for (auto metadata : owner.metadata_list()) {
        expect(index < expected_tags.size());
        if (index >= expected_tags.size()) { continue; }
        expect(metadata->derived_metadata_tag() == expected_tags[index]);
        switch (index) {
            case 0u: {
                auto is_name = metadata->isa<NameMD>();
                expect(is_name);
                if (is_name) {
                    expect(static_cast<const NameMD *>(metadata)->name() == metadata_name(prefix));
                }
                break;
            }
            case 1u: {
                auto is_comment = metadata->isa<CommentMD>();
                expect(is_comment);
                if (is_comment) {
                    expect(static_cast<const CommentMD *>(metadata)->comment() == metadata_comment(prefix, 1u));
                }
                break;
            }
            case 2u: {
                auto is_location = metadata->isa<LocationMD>();
                expect(is_location);
                if (is_location) {
                    auto location = static_cast<const LocationMD *>(metadata);
                    expect(location->file().string() == metadata_path(prefix).string());
                    expect(location->line() == 137);
                }
                break;
            }
            case 3u: {
                auto is_comment = metadata->isa<CommentMD>();
                expect(is_comment);
                if (is_comment) {
                    expect(static_cast<const CommentMD *>(metadata)->comment() == metadata_comment(prefix, 2u));
                }
                break;
            }
            case 4u: {
                auto is_curve_basis = metadata->isa<CurveBasisMD>();
                expect(is_curve_basis);
                if (is_curve_basis) {
                    auto expected = CurveBasisSet::make(
                        CurveBasis::PIECEWISE_LINEAR,
                        CurveBasis::CATMULL_ROM,
                        CurveBasis::BEZIER);
                    expect(static_cast<const CurveBasisMD *>(metadata)->curve_basis_set() == expected);
                }
                break;
            }
            case 5u:
                expect(metadata->isa<SignatureConstraintMD>());
                break;
            default: break;
        }
        index++;
    }
    expect(index == expected_tags.size());
}

void test_debug_break_callback(void *, DebugBreakInst::Evaluate) {}

void expect_interchange_rejected(luisa::string_view text) {
    auto decoded = xir_from_interchange_text(text);
    expect(!decoded.succeeded());
    expect(decoded.module == nullptr);
    expect(!decoded.diagnostics.empty());
}

void expect_interchange_rejected_with_diagnostic(
    luisa::string_view text,
    luisa::string_view expected_diagnostic) {
    auto decoded = xir_from_interchange_text(text);
    expect(!decoded.succeeded());
    expect(decoded.module == nullptr);
    expect(!decoded.diagnostics.empty());
    auto found = std::any_of(
        decoded.diagnostics.begin(), decoded.diagnostics.end(),
        [&](auto &&diagnostic) noexcept {
            return diagnostic.message == expected_diagnostic;
        });
    auto first_diagnostic = decoded.diagnostics.empty() ?
                                luisa::string_view{} :
                                luisa::string_view{decoded.diagnostics.front().message};
    expect(found)
        << "malformed operation failed outside the intended validation category: "
        << text << " first diagnostic: " << first_diagnostic;
}

[[nodiscard]] uint64_t test_bitcode_checksum(luisa::span<const std::byte> bytes) {
    auto hash = uint64_t{14695981039346656037ull};
    for (auto byte : bytes) {
        hash ^= std::to_integer<uint8_t>(byte);
        hash *= 1099511628211ull;
    }
    return hash;
}

void test_append_u32(luisa::vector<std::byte> &bytes, uint32_t value) {
    for (auto i = 0u; i < 4u; i++) {
        bytes.emplace_back(static_cast<std::byte>((value >> (i * 8u)) & 0xffu));
    }
}

void test_append_u64(luisa::vector<std::byte> &bytes, uint64_t value) {
    for (auto i = 0u; i < 8u; i++) {
        bytes.emplace_back(static_cast<std::byte>((value >> (i * 8u)) & 0xffu));
    }
}

void test_append_uleb(luisa::vector<std::byte> &bytes, uint64_t value) {
    do {
        auto byte = static_cast<uint8_t>(value & 0x7fu);
        value >>= 7u;
        if (value != 0u) { byte |= 0x80u; }
        bytes.emplace_back(static_cast<std::byte>(byte));
    } while (value != 0u);
}

[[nodiscard]] luisa::vector<std::byte>
make_test_bitcode(luisa::span<const std::byte> payload, uint32_t version = 2u) {
    constexpr std::array magic{
        std::byte{'L'}, std::byte{'X'}, std::byte{'I'}, std::byte{'R'},
        std::byte{'B'}, std::byte{'C'}, std::byte{0u}, std::byte{1u}};
    luisa::vector<std::byte> bitcode;
    bitcode.insert(bitcode.end(), magic.begin(), magic.end());
    test_append_u32(bitcode, version);
    test_append_u32(bitcode, 0u);
    test_append_u64(bitcode, payload.size());
    test_append_u64(bitcode, test_bitcode_checksum(payload));
    bitcode.insert(bitcode.end(), payload.begin(), payload.end());
    return bitcode;
}

}// namespace

void reg_empty_module_round_trip() {
    "xir_interchange_empty_module_round_trip"_test = [] {
        Module module;
        auto first = xir_to_interchange_text(&module);
        auto second = xir_to_interchange_text(&module);
        expect(first.succeeded());
        expect(second.succeeded());
        expect(first.text == second.text) << "canonical text must be deterministic";

        auto parsed_text = xir_from_interchange_text(first.text);
        expect(parsed_text.succeeded());
        expect(parsed_text.module != nullptr);

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        expect(bitcode.bitcode.size() < first.text.size());
        auto parsed_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(parsed_bitcode.succeeded());
        expect(parsed_bitcode.module != nullptr);
    };
}

void reg_malformed_text() {
    "xir_interchange_malformed_text"_test = [] {
        auto missing_version = xir_from_interchange_text("xir.text module {}\n");
        expect(!missing_version.succeeded());
        expect(!missing_version.diagnostics.empty());
        expect(missing_version.diagnostics.front().line == 1u);
        expect(missing_version.diagnostics.front().column > 0u);

        constexpr auto trailing_text = luisa::string_view{
            "xir.text 1 module { globals 0 functions 0 } trailing\n"};
        auto trailing = xir_from_interchange_text(trailing_text);
        expect(!trailing.succeeded());
        expect(trailing.module == nullptr);
        expect(!trailing.diagnostics.empty());
        if (!trailing.diagnostics.empty()) {
            auto &&diagnostic = trailing.diagnostics.front();
            expect(diagnostic.message == "Malformed XIR interchange text.");
            expect(diagnostic.offset == trailing_text.find("trailing"));
            expect(diagnostic.line == 1u);
            expect(diagnostic.column == diagnostic.offset + 1u);
        }
    };
}

void reg_malformed_bitcode() {
    "xir_interchange_malformed_bitcode"_test = [] {
        Module module;
        auto encoded = xir_to_bitcode(&module);
        expect(encoded.succeeded());

        auto truncated = encoded.bitcode;
        truncated.pop_back();
        auto truncated_result = xir_from_bitcode(truncated);
        expect(!truncated_result.succeeded());

        auto wrong_version = encoded.bitcode;
        wrong_version[8u] = std::byte{3u};
        auto wrong_version_result = xir_from_bitcode(wrong_version);
        expect(!wrong_version_result.succeeded());

        auto wrong_checksum = encoded.bitcode;
        wrong_checksum[24u] ^= std::byte{1u};
        auto wrong_checksum_result = xir_from_bitcode(wrong_checksum);
        expect(!wrong_checksum_result.succeeded());

        auto trailing = encoded.bitcode;
        trailing.emplace_back(std::byte{0u});
        auto trailing_result = xir_from_bitcode(trailing);
        expect(!trailing_result.succeeded());

        const std::array invalid_spill_kind_payload{
            std::byte{0x00}, std::byte{0x01},
            std::byte{0x05}, std::byte{0x02}};
        auto invalid_spill_kind = xir_from_bitcode(
            make_test_bitcode(invalid_spill_kind_payload));
        expect(!invalid_spill_kind.succeeded());
        expect(!invalid_spill_kind.diagnostics.empty());
        if (!invalid_spill_kind.diagnostics.empty()) {
            expect(invalid_spill_kind.diagnostics.front().message ==
                   "Unknown XIR binary reg2mem-spill metadata kind.");
        }
    };
}

void reg_semantic_module_round_trip() {
    "xir_interchange_semantic_module_round_trip"_test = [] {
        Module module;
        auto int_type = luisa::compute::Type::of<int32_t>();
        auto uint_type = luisa::compute::Type::of<uint32_t>();
        auto one = module.create_constant_one(int_type);

        auto callable = module.create_callable(int_type);
        auto callable_argument = callable->create_value_argument(int_type);
        auto callable_body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callable_body);
        auto sum = builder.call(int_type, ArithmeticOp::BINARY_ADD, {callable_argument, one});
        builder.return_(sum);

        auto kernel = module.create_kernel();
        kernel->set_block_size(luisa::make_uint3(32u, 2u, 1u));
        auto kernel_argument = kernel->create_value_argument(int_type);
        auto kernel_body = kernel->create_body_block();
        builder.set_insertion_point(kernel_body);
        auto call = builder.call(int_type, callable, {kernel_argument});
        auto local = builder.alloca_local(int_type);
        builder.store(local, call);
        auto loaded = builder.load(int_type, local);
        builder.cast_(uint_type, CastOp::STATIC_CAST, loaded);
        builder.return_void();

        auto encoded = xir_to_interchange_text(&module);
        expect(encoded.succeeded());
        if (!encoded.succeeded()) { return; }
        expect(encoded.text.find("function") != luisa::string::npos);
        expect(encoded.text.find("arithmetic") != luisa::string::npos);
        expect(encoded.text.find("call") != luisa::string::npos);

        auto decoded = xir_from_interchange_text(encoded.text);
        expect(decoded.succeeded());
        expect(decoded.module != nullptr);
        if (!decoded.succeeded()) { return; }
        expect(decoded.module->function_list().count_size() == 2u);
        expect(decoded.module->constant_list().count_size() == 1u);

        auto canonical_again = xir_to_interchange_text(decoded.module.get());
        expect(canonical_again.succeeded());
        expect(canonical_again.text == encoded.text) << "text round-trip must preserve canonical form";

        size_t callable_count = 0u;
        size_t kernel_count = 0u;
        size_t call_count = 0u;
        for (auto function : decoded.module->function_list()) {
            if (function->isa<CallableFunction>()) {
                callable_count++;
                expect(function->type() == int_type);
            } else if (function->isa<KernelFunction>()) {
                kernel_count++;
                auto block_size = static_cast<KernelFunction *>(function)->block_size();
                expect(static_cast<bool>(block_size.x == 32u && block_size.y == 2u && block_size.z == 1u));
            }
            for (auto block : function->basic_blocks()) {
                for (auto instruction : block->instructions()) {
                    if (instruction->isa<CallInst>()) {
                        call_count++;
                        auto call_instruction = static_cast<CallInst *>(instruction);
                        expect(call_instruction->callee()->isa<CallableFunction>());
                        expect(call_instruction->argument_count() == 1u);
                    }
                }
            }
        }
        expect(callable_count == 1u);
        expect(kernel_count == 1u);
        expect(call_count == 1u);

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(decoded_bitcode.succeeded());
        if (!decoded_bitcode.succeeded()) { return; }
        auto bitcode_text = xir_to_interchange_text(decoded_bitcode.module.get());
        expect(bitcode_text.succeeded());
        expect(bitcode_text.text == encoded.text);
        auto bitcode_again = xir_to_bitcode(decoded_bitcode.module.get());
        expect(bitcode_again.succeeded());
        expect(static_cast<bool>(bitcode_again.bitcode == bitcode.bitcode))
            << "binary string-table ordering must be deterministic";
    };
}

void reg_bindless_access_round_trip() {
    "xir_interchange_bindless_access_axes_round_trip"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *bindless = kernel->create_resource_argument(
            Type::from("bindless_array"));
        auto *body = kernel->create_body_block();
        auto *zero = module.create_constant_zero(Type::of<uint32_t>());
        auto *one = module.create_constant_one(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.call(
            Type::of<uint32_t>(), ResourceReadOp::BINDLESS_BUFFER_READ,
            {bindless, zero, zero},
            {.typed = true, .uniform = true});
        builder.call(
            Type::of<uint32_t>(), ResourceQueryOp::BINDLESS_BUFFER_SIZE,
            {bindless, zero, one},
            {.typed = false, .uniform = true});
        builder.call(
            ResourceWriteOp::BINDLESS_BUFFER_WRITE,
            {bindless, zero, zero, one},
            {.typed = true, .uniform = false});
        builder.return_void();
        expect(xir_verify_module(&module).succeeded());

        auto text = xir_to_interchange_text(&module);
        expect(text.succeeded());
        if (!text.succeeded()) { return; }
        auto decoded = xir_from_interchange_text(text.text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }

        std::array<bool, 3u> found{};
        for (auto *function : decoded.module->function_list()) {
            for (auto *block : function->basic_blocks()) {
                for (auto *instruction : block->instructions()) {
                    if (instruction->isa<ResourceReadInst>()) {
                        auto access = static_cast<ResourceReadInst *>(
                                          instruction)
                                          ->bindless_access();
                        found[0] |= access == BindlessResourceAccess{
                                                  .typed = true,
                                                  .uniform = true};
                    } else if (instruction->isa<ResourceQueryInst>()) {
                        auto access = static_cast<ResourceQueryInst *>(
                                          instruction)
                                          ->bindless_access();
                        found[1] |= access == BindlessResourceAccess{
                                                  .typed = false,
                                                  .uniform = true};
                    } else if (instruction->isa<ResourceWriteInst>()) {
                        auto access = static_cast<ResourceWriteInst *>(
                                          instruction)
                                          ->bindless_access();
                        found[2] |= access == BindlessResourceAccess{
                                                  .typed = true,
                                                  .uniform = false};
                    }
                }
            }
        }
        expect(found[0] && found[1] && found[2]);
        auto canonical = xir_to_interchange_text(decoded.module.get());
        expect(canonical.succeeded());
        expect(canonical.text == text.text);

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(decoded_bitcode.succeeded());
        if (decoded_bitcode.succeeded()) {
            auto bitcode_text = xir_to_interchange_text(
                decoded_bitcode.module.get());
            expect(bitcode_text.succeeded());
            expect(bitcode_text.text == text.text);
        }
    };

    "xir_interchange_rejects_bindless_access_on_direct_resource"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.call(
            Type::of<uint32_t>(), ResourceQueryOp::BUFFER_SIZE,
            {buffer}, {.typed = true});
        builder.return_void();
        expect(!xir_verify_module(&module).succeeded());
        expect(!xir_to_interchange_text(&module).succeeded());
    };
}

void reg_unsupported_instruction_fails_closed() {
    "xir_interchange_debug_break_null_callback_round_trip"_test = [] {
        Module module;
        auto callable = module.create_callable(nullptr);
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.debug_break();
        builder.return_void();
        auto encoded = xir_to_interchange_text(&module);
        expect(encoded.succeeded());
        if (!encoded.succeeded()) { return; }
        expect(encoded.text.find("debug_break \"void\" null_callback") != luisa::string::npos);
        auto decoded = xir_from_interchange_text(encoded.text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        auto decoded_body = decoded.module->function_list().front()->definition()->body_block();
        auto instruction = decoded_body->instructions().front();
        expect(instruction->isa<DebugBreakInst>());
        if (instruction->isa<DebugBreakInst>()) {
            expect(static_cast<DebugBreakInst *>(instruction)->callback() == nullptr);
        }
        auto canonical = xir_to_interchange_text(decoded.module.get());
        expect(canonical.succeeded());
        expect(canonical.text == encoded.text);
    };

    "xir_interchange_debug_break_nonnull_callback_fails_closed"_test = [] {
        Module module;
        auto callable = module.create_callable(nullptr);
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.debug_break(test_debug_break_callback);
        builder.return_void();
        auto encoded = xir_to_interchange_text(&module);
        expect(!encoded.succeeded());
        expect(encoded.text.empty());
        expect(!encoded.diagnostics.empty());
    };
}

void reg_symbolic_op_tokens_and_compatibility() {
    "xir_interchange_legacy_numeric_ops_canonicalize_to_symbols"_test = [] {
        constexpr auto text = R"(
xir.text 1
module {
  globals 0
  functions 1
  function 0 callable "int" 0 0 0 {
    arguments 1
    argument 1 value "int"
    blocks 1
    block 2
    body 2
    instructions 6
    instruction 3 2 alloca "int" 0 0 0
    instruction 4 2 store "void" -1 2 3 1 0
    instruction 5 2 load "int" -1 1 3 0
    instruction 6 2 cast "uint" 0 1 5 0
    instruction 7 2 arithmetic "int" 2 2 1 1 0
    instruction 8 2 return "void" -1 1 7 0
  }
}
)";
        auto decoded = xir_from_interchange_text(text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        auto canonical = xir_to_interchange_text(decoded.module.get());
        expect(canonical.succeeded());
        if (!canonical.succeeded()) { return; }
        expect(canonical.text.find("alloca \"int\" local") != luisa::string::npos);
        expect(canonical.text.find("cast \"uint\" static_cast") != luisa::string::npos);
        expect(canonical.text.find("arithmetic \"int\" binary_add") != luisa::string::npos);
        expect(canonical.text.find("arithmetic \"int\" 2 ") == luisa::string::npos);

        auto decoded_again = xir_from_interchange_text(canonical.text);
        expect(decoded_again.succeeded());
        if (!decoded_again.succeeded()) { return; }
        auto canonical_again = xir_to_interchange_text(decoded_again.module.get());
        expect(canonical_again.succeeded());
        expect(canonical_again.text == canonical.text);
    };

    "xir_interchange_invalid_symbolic_and_legacy_ops_rejected"_test = [] {
        constexpr auto unknown_symbol = R"(
xir.text 1 module { globals 0 functions 1
function 0 callable "int" 0 0 0 {
arguments 1 argument 1 value "int"
blocks 1 block 2 body 2 instructions 2
instruction 3 2 arithmetic "int" future_add 2 1 1 0
instruction 4 2 return "void" -1 1 3 0 }
})";
        auto unknown = xir_from_interchange_text(unknown_symbol);
        expect(!unknown.succeeded());
        expect(!unknown.diagnostics.empty());
        expect(unknown.diagnostics.front().line > 0u);

        constexpr auto invalid_legacy = R"(
xir.text 1 module { globals 0 functions 1
function 0 callable "int" 0 0 0 {
arguments 1 argument 1 value "int"
blocks 1 block 2 body 2 instructions 2
instruction 3 2 arithmetic "int" 93 2 1 1 0
instruction 4 2 return "void" -1 1 3 0 }
})";
        auto legacy = xir_from_interchange_text(invalid_legacy);
        expect(!legacy.succeeded());
        expect(!legacy.diagnostics.empty());
        expect(legacy.diagnostics.front().line > 0u);
    };
}

void reg_vulkan_priority_instruction_round_trip() {
    "xir_interchange_vulkan_priority_instructions_round_trip"_test = [] {
        Module module;
        auto int_type = Type::of<int32_t>();
        auto uint_type = Type::of<uint32_t>();
        auto ulong_type = Type::of<uint64_t>();
        auto buffer_type = Type::buffer(int_type);

        auto resource_callable = module.create_callable(nullptr);
        auto buffer = resource_callable->create_resource_argument(buffer_type);
        auto index = resource_callable->create_value_argument(uint_type);
        auto value = resource_callable->create_value_argument(int_type);
        auto resource_body = resource_callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(resource_body);
        builder.call(ulong_type, ResourceQueryOp::BUFFER_SIZE, {buffer});
        builder.call(int_type, ResourceReadOp::BUFFER_READ, {buffer, index});
        std::array<Value *, 1u> atomic_indices{index};
        builder.atomic_fetch_add(int_type, buffer, luisa::span<Value *const>{atomic_indices}, value);
        auto warp_sum = builder.call(int_type, ThreadGroupOp::WARP_ACTIVE_SUM, {value});
        builder.call(ResourceWriteOp::BUFFER_WRITE, {buffer, index, warp_sum});
        builder.return_void();

        auto break_callable = module.create_callable(nullptr);
        auto break_entry = break_callable->create_body_block();
        builder.set_insertion_point(break_entry);
        auto break_loop = builder.simple_loop();
        auto break_body = break_loop->create_body_block();
        auto break_merge = break_loop->create_merge_block();
        builder.set_insertion_point(break_body);
        builder.break_(break_merge);
        builder.set_insertion_point(break_merge);
        builder.return_void();

        auto continue_callable = module.create_callable(nullptr);
        auto continue_entry = continue_callable->create_body_block();
        builder.set_insertion_point(continue_entry);
        auto continue_loop = builder.simple_loop();
        auto continue_body = continue_loop->create_body_block();
        auto continue_merge = continue_loop->create_merge_block();
        builder.set_insertion_point(continue_body);
        builder.continue_(continue_body);
        builder.set_insertion_point(continue_merge);
        builder.return_void();

        auto encoded = xir_to_interchange_text(&module);
        expect(encoded.succeeded());
        if (!encoded.succeeded()) { return; }
        expect(encoded.text.find("resource_query \"ulong\" buffer_size") != luisa::string::npos);
        expect(encoded.text.find("resource_read \"int\" buffer_read") != luisa::string::npos);
        expect(encoded.text.find("atomic \"int\" fetch_add") != luisa::string::npos);
        expect(encoded.text.find("thread_group \"int\" warp_active_sum") != luisa::string::npos);
        expect(encoded.text.find("resource_write \"void\" buffer_write") != luisa::string::npos);
        expect(encoded.text.find(" break \"void\" -1") != luisa::string::npos);
        expect(encoded.text.find(" continue \"void\" -1") != luisa::string::npos);

        auto decoded = xir_from_interchange_text(encoded.text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        std::array seen{false, false, false, false, false, false, false};
        for (auto function : decoded.module->function_list()) {
            for (auto block : function->basic_blocks()) {
                for (auto instruction : block->instructions()) {
                    switch (instruction->derived_instruction_tag()) {
                        case DerivedInstructionTag::RESOURCE_QUERY:
                            seen[0] = static_cast<ResourceQueryInst *>(instruction)->op() == ResourceQueryOp::BUFFER_SIZE;
                            break;
                        case DerivedInstructionTag::RESOURCE_READ:
                            seen[1] = static_cast<ResourceReadInst *>(instruction)->op() == ResourceReadOp::BUFFER_READ;
                            break;
                        case DerivedInstructionTag::ATOMIC:
                            seen[2] = static_cast<AtomicInst *>(instruction)->op() == AtomicOp::FETCH_ADD;
                            break;
                        case DerivedInstructionTag::THREAD_GROUP:
                            seen[3] = static_cast<ThreadGroupInst *>(instruction)->op() == ThreadGroupOp::WARP_ACTIVE_SUM;
                            break;
                        case DerivedInstructionTag::RESOURCE_WRITE:
                            seen[4] = static_cast<ResourceWriteInst *>(instruction)->op() == ResourceWriteOp::BUFFER_WRITE;
                            break;
                        case DerivedInstructionTag::BREAK: seen[5] = true; break;
                        case DerivedInstructionTag::CONTINUE: seen[6] = true; break;
                        default: break;
                    }
                }
            }
        }
        for (auto value_seen : seen) { expect(value_seen); }

        auto canonical_again = xir_to_interchange_text(decoded.module.get());
        expect(canonical_again.succeeded());
        expect(canonical_again.text == encoded.text);

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(decoded_bitcode.succeeded());
        if (!decoded_bitcode.succeeded()) { return; }
        auto bitcode_text = xir_to_interchange_text(decoded_bitcode.module.get());
        expect(bitcode_text.succeeded());
        expect(bitcode_text.text == encoded.text);
    };
}

void reg_vulkan_priority_instruction_validation() {
    "xir_interchange_vulkan_priority_malformed_ops_rejected"_test = [] {
        struct MalformedCase {
            luisa::string_view text;
            luisa::string_view diagnostic;
        };
        constexpr std::array malformed{
            MalformedCase{R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 resource "buffer<int>" blocks 1 block 2 body 2 instructions 2 instruction 3 2 atomic "int" fetch_add 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
                          "XIR instruction has an invalid operand, auxiliary, or opcode layout."},
            MalformedCase{R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 resource "buffer<int>" blocks 1 block 2 body 2 instructions 2 instruction 3 2 resource_query "ulong" buffer_size 2 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
                          "XIR instruction has an invalid operand, auxiliary, or opcode layout."},
            MalformedCase{R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 resource "buffer<int>" argument 2 value "uint" blocks 1 block 3 body 3 instructions 2 instruction 4 3 resource_read "void" buffer_read 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
                          "XIR instruction has an invalid result type."},
            MalformedCase{R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<int>" argument 2 value "uint" argument 3 value "int" blocks 1 block 4 body 4 instructions 2 instruction 5 4 resource_write "int" buffer_write 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
                          "XIR instruction has an invalid result type."},
            MalformedCase{R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "int" blocks 1 block 2 body 2 instructions 2 instruction 3 2 thread_group "bool" warp_active_sum 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
                          "XIR instruction operands or result type do not match its operation."},
            MalformedCase{R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 2 block 1 block 2 body 1 instructions 3 instruction 3 1 thread_group "void" shader_execution_reorder 2 1 2 0 instruction 4 1 return "void" -1 1 -1 0 instruction 5 2 return "void" -1 1 -1 0 } })",
                          "XIR instruction operands or result type do not match its operation."},
            MalformedCase{R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "int" blocks 1 block 2 body 2 instructions 1 instruction 3 2 break "void" -1 1 1 0 } })",
                          "XIR instruction operands or result type do not match its operation."}};
        for (auto &&test_case : malformed) {
            auto decoded = xir_from_interchange_text(test_case.text);
            expect(!decoded.succeeded());
            expect(decoded.module == nullptr);
            expect(!decoded.diagnostics.empty());
            auto found = std::any_of(
                decoded.diagnostics.begin(), decoded.diagnostics.end(),
                [&](auto &&diagnostic) noexcept { return diagnostic.message == test_case.diagnostic; });
            expect(found) << "malformed operation must fail for its operation-specific validation";
        }
    };

    "xir_interchange_writer_rejects_invalid_thread_group_result"_test = [] {
        Module module;
        auto callable = module.create_callable(nullptr);
        auto argument = callable->create_value_argument(Type::of<int32_t>());
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.call(Type::of<bool>(), ThreadGroupOp::WARP_ACTIVE_SUM, {argument});
        builder.return_void();
        auto encoded = xir_to_interchange_text(&module);
        expect(!encoded.succeeded());
        expect(encoded.text.empty());
        expect(!encoded.diagnostics.empty());
    };
}

void reg_strict_atomic_validation() {
    "xir_interchange_atomic_aggregate_paths_round_trip"_test = [] {
        constexpr auto structured = R"(
xir.text 1 module {
globals 1 constant 0 "uint" "01000000"
functions 1 function 1 callable "void" 0 0 0 {
arguments 3
argument 2 resource "buffer<struct<4,int,uint>>"
argument 3 value "uint"
argument 4 value "uint"
blocks 1 block 5 body 5 instructions 2
instruction 6 5 atomic "uint" fetch_add 4 2 3 0 4 0
instruction 7 5 return "void" -1 1 -1 0
} })";
        auto decoded_structured = xir_from_interchange_text(structured);
        expect(decoded_structured.succeeded());
        if (decoded_structured.succeeded()) {
            auto canonical = xir_to_interchange_text(decoded_structured.module.get());
            expect(canonical.succeeded());
            auto decoded_again = xir_from_interchange_text(canonical.text);
            expect(decoded_again.succeeded());
        }

        constexpr auto shared = R"(
xir.text 1 module { globals 0 functions 1
function 0 callable "void" 0 0 0 {
arguments 2 argument 1 value "uint" argument 2 value "int"
blocks 1 block 3 body 3 instructions 3
instruction 4 3 alloca "array<int,4>" shared 0 0
instruction 5 3 atomic "int" exchange 3 4 1 2 0
instruction 6 3 return "void" -1 1 -1 0
} })";
        auto decoded_shared = xir_from_interchange_text(shared);
        expect(decoded_shared.succeeded());
        if (decoded_shared.succeeded()) {
            auto canonical = xir_to_interchange_text(decoded_shared.module.get());
            expect(canonical.succeeded());
            expect(canonical.text.find("atomic \"int\" exchange") != luisa::string::npos);
        }

        // Atomic address paths use the same integer-index contract as GEP and
        // aggregate extract/insert. A 64-bit buffer index is therefore valid
        // and must survive interchange round-tripping.
        constexpr auto wide_buffer_index = R"(
xir.text 1 module { globals 0 functions 1
function 0 callable "void" 0 0 0 {
arguments 3 argument 1 resource "buffer<int>" argument 2 value "ulong" argument 3 value "int"
blocks 1 block 4 body 4 instructions 2
instruction 5 4 atomic "int" fetch_add 3 1 2 3 0
instruction 6 4 return "void" -1 1 -1 0
} })";
        auto decoded_wide = xir_from_interchange_text(wide_buffer_index);
        expect(decoded_wide.succeeded());
        if (decoded_wide.succeeded()) {
            auto canonical = xir_to_interchange_text(decoded_wide.module.get());
            expect(canonical.succeeded());
            expect(xir_from_interchange_text(canonical.text).succeeded());
        }
    };

    "xir_interchange_integer_64_bit_atomics_round_trip"_test = [] {
        Module module;
        for (auto type : {Type::of<luisa::slong>(),
                          Type::of<luisa::ulong>()}) {
            auto *index = module.create_constant_zero(Type::of<uint>());
            auto *value = module.create_constant_one(type);
            auto *callable = module.create_callable(nullptr);
            auto *buffer = callable->create_resource_argument(
                Type::buffer(type));
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *shared = builder.alloca_shared(Type::array(type, 1u));
            std::array<Value *, 1u> indices{index};
            builder.atomic_fetch_add(
                type, buffer, luisa::span<Value *const>{indices}, value);
            builder.atomic_fetch_add(
                type, shared, luisa::span<Value *const>{indices}, value);
            builder.return_void();
        }

        expect(xir_verify_module(&module).succeeded());
        auto text = xir_to_interchange_text(&module);
        expect(text.succeeded());
        if (text.succeeded()) {
            expect(text.text.find("atomic \"long\" fetch_add") !=
                   luisa::string::npos);
            expect(text.text.find("atomic \"ulong\" fetch_add") !=
                   luisa::string::npos);
            expect(xir_from_interchange_text(text.text).succeeded());
        }

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        if (bitcode.succeeded()) {
            expect(xir_from_bitcode(bitcode.bitcode).succeeded());
        }
    };

    "xir_interchange_atomic_invalid_address_paths_rejected"_test = [] {
        constexpr std::array malformed{
            // A buffer atomic must include at least one address index.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 resource "buffer<int>" argument 2 value "int" blocks 1 block 3 body 3 instructions 2 instruction 4 3 atomic "int" fetch_add 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // The declared/result/value type must equal the addressed leaf.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<int>" argument 2 value "uint" argument 3 value "float" blocks 1 block 4 body 4 instructions 2 instruction 5 4 atomic "float" fetch_add 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            // Indexing beyond the aggregate leaf is invalid.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<array<int,2>>" argument 2 value "uint" argument 3 value "int" blocks 1 block 4 body 4 instructions 2 instruction 5 4 atomic "int" fetch_add 5 1 2 2 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            // Structure member indices must be constants.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<struct<4,int,uint>>" argument 2 value "uint" argument 3 value "uint" blocks 1 block 4 body 4 instructions 2 instruction 5 4 atomic "uint" fetch_add 4 1 2 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            // Structure member indices must be in bounds.
            R"(xir.text 1 module { globals 1 constant 0 "uint" "02000000" functions 1 function 1 callable "void" 0 0 0 { arguments 3 argument 2 resource "buffer<struct<4,int,uint>>" argument 3 value "uint" argument 4 value "uint" blocks 1 block 5 body 5 instructions 2 instruction 6 5 atomic "uint" fetch_add 4 2 3 0 4 0 instruction 7 5 return "void" -1 1 -1 0 } })",
            // Local allocas are not shared-memory atomic roots.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 value "uint" argument 2 value "int" blocks 1 block 3 body 3 instructions 3 instruction 4 3 alloca "array<int,4>" local 0 0 instruction 5 3 atomic "int" exchange 3 4 1 2 0 instruction 6 3 return "void" -1 1 -1 0 } })",
            // XIR atomics do not admit 8-bit, 16-bit, or float64 leaves.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<byte>" argument 2 value "uint" argument 3 value "byte" blocks 1 block 4 body 4 instructions 2 instruction 5 4 atomic "byte" fetch_add 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<short>" argument 2 value "uint" argument 3 value "short" blocks 1 block 4 body 4 instructions 2 instruction 5 4 atomic "short" fetch_add 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<double>" argument 2 value "uint" argument 3 value "double" blocks 1 block 4 body 4 instructions 2 instruction 5 4 atomic "double" fetch_add 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            // Bitwise atomics cannot target floating-point leaves.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<float>" argument 2 value "uint" argument 3 value "float" blocks 1 block 4 body 4 instructions 2 instruction 5 4 atomic "float" fetch_and 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })"};
        for (auto text : malformed) {
            expect_interchange_rejected_with_diagnostic(
                text,
                "XIR instruction operands or result type do not match its operation.");
        }
    };
}

void reg_strict_operand_and_resource_validation() {
    "xir_interchange_signed_ray_instance_id_is_valid"_test = [] {
        constexpr auto text = R"(
xir.text 1 module { globals 0 functions 1
function 0 callable "void" 0 0 0 {
arguments 2 argument 1 resource "accel" argument 2 value "int"
blocks 1 block 3 body 3 instructions 2
instruction 4 3 resource_query "matrix<4>" ray_tracing_instance_transform 2 1 2 0
instruction 5 3 return "void" -1 1 -1 0
} })";
        auto decoded = xir_from_interchange_text(text);
        expect(decoded.succeeded());
        if (decoded.succeeded()) {
            auto canonical = xir_to_interchange_text(decoded.module.get());
            expect(canonical.succeeded());
            expect(canonical.text.find("ray_tracing_instance_transform") != luisa::string::npos);
        }
    };

    "xir_interchange_data_operand_categories_rejected"_test = [] {
        constexpr std::array malformed{
            // Blocks are not thread-group data operands (this used to crash).
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 1 block 1 body 1 instructions 2 instruction 2 1 thread_group "void" shader_execution_reorder 2 1 1 0 instruction 3 1 return "void" -1 1 -1 0 } })",
            // Blocks are not arithmetic values.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 1 block 1 body 1 instructions 2 instruction 2 1 arithmetic "int" binary_add 2 1 1 0 instruction 3 1 return "void" -1 1 -1 0 } })",
            // Function symbols are only valid in callee positions.
            R"(xir.text 1 module { globals 0 functions 2 function 0 external "int" 0 0 0 { arguments 0 blocks 0 body -1 instructions 0 } function 1 callable "void" 0 0 0 { arguments 0 blocks 1 block 2 body 2 instructions 2 instruction 3 2 cast "uint" static_cast 1 0 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            // Call arguments must match the formal value category.
            R"(xir.text 1 module { globals 0 functions 2 function 0 external "void" 0 0 0 { arguments 1 argument 1 value "int" blocks 0 body -1 instructions 0 } function 2 callable "void" 0 0 0 { arguments 0 blocks 1 block 3 body 3 instructions 2 instruction 4 3 call "void" -1 2 0 3 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Resource indices must be data, not blocks.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 resource "buffer<int>" blocks 1 block 2 body 2 instructions 2 instruction 3 2 resource_read "int" buffer_read 2 1 2 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            // Warp lane indices are exactly uint32.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 value "int" argument 2 value "int" blocks 1 block 3 body 3 instructions 2 instruction 4 3 thread_group "int" warp_read_lane 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Void instructions cannot be reused as data values.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "int" blocks 1 block 2 body 2 instructions 3 instruction 3 2 debug_break "void" null_callback 0 0 instruction 4 2 arithmetic "int" binary_add 2 3 1 0 instruction 5 2 return "void" -1 1 -1 0 } })"};
        for (auto text : malformed) {
            expect_interchange_rejected_with_diagnostic(
                text,
                "XIR instruction operands or result type do not match its operation.");
        }
    };

    "xir_interchange_typed_and_byte_buffers_are_distinct"_test = [] {
        constexpr std::array malformed{
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 resource "buffer<void>" argument 2 value "uint" blocks 1 block 3 body 3 instructions 2 instruction 4 3 resource_read "int" buffer_read 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 resource "buffer<int>" argument 2 value "uint" blocks 1 block 3 body 3 instructions 2 instruction 4 3 resource_read "int" byte_buffer_read 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 resource "buffer<void>" blocks 1 block 2 body 2 instructions 2 instruction 3 2 resource_query "ulong" buffer_size 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 resource "buffer<int>" blocks 1 block 2 body 2 instructions 2 instruction 3 2 resource_query "ulong" byte_buffer_size 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<void>" argument 2 value "uint" argument 3 value "int" blocks 1 block 4 body 4 instructions 2 instruction 5 4 resource_write "void" buffer_write 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "buffer<int>" argument 2 value "uint" argument 3 value "int" blocks 1 block 4 body 4 instructions 2 instruction 5 4 resource_write "void" byte_buffer_write 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })"};
        for (auto text : malformed) {
            expect_interchange_rejected_with_diagnostic(
                text,
                "XIR instruction operands or result type do not match its operation.");
        }
    };

    "xir_interchange_ray_resource_signatures_are_exact"_test = [] {
        constexpr std::array malformed{
            // Transform result must be float4x4.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 resource "accel" argument 2 value "uint" blocks 1 block 3 body 3 instructions 2 instruction 4 3 resource_query "int" ray_tracing_instance_transform 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Trace ray operand must be the canonical Ray structure.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "accel" argument 2 value "int" argument 3 value "uint" blocks 1 block 4 body 4 instructions 2 instruction 5 4 resource_query "struct<8,uint,uint,vector<float,2>,float>" ray_tracing_trace_closest 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            // Trace result must be the canonical SurfaceHit structure.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "accel" argument 2 value "struct<16,array<float,3>,float,array<float,3>,float>" argument 3 value "uint" blocks 1 block 4 body 4 instructions 2 instruction 5 4 resource_query "int" ray_tracing_trace_closest 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            // Motion time is exactly float32.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 4 argument 1 resource "accel" argument 2 value "struct<16,array<float,3>,float,array<float,3>,float>" argument 3 value "uint" argument 4 value "uint" blocks 1 block 5 body 5 instructions 2 instruction 6 5 resource_query "bool" ray_tracing_trace_any_motion_blur 4 1 2 3 4 0 instruction 7 5 return "void" -1 1 -1 0 } })",
            // Transform setter payload is exactly float4x4.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 3 argument 1 resource "accel" argument 2 value "int" argument 3 value "int" blocks 1 block 4 body 4 instructions 2 instruction 5 4 resource_write "void" ray_tracing_set_instance_transform 3 1 2 3 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            // SRT setter payload is the canonical SRT structure.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 4 argument 1 resource "accel" argument 2 value "uint" argument 3 value "int" argument 4 value "matrix<4>" blocks 1 block 5 body 5 instructions 2 instruction 6 5 resource_write "void" ray_tracing_set_instance_motion_srt 4 1 2 3 4 0 instruction 7 5 return "void" -1 1 -1 0 } })"};
        for (auto text : malformed) {
            expect_interchange_rejected_with_diagnostic(
                text,
                "XIR instruction operands or result type do not match its operation.");
        }
    };
}

void reg_interchange_safety_validation() {
    "xir_interchange_ids_auxiliary_and_payload_shapes_rejected"_test = [] {
        constexpr std::array malformed{
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 1 block 1 body -2 instructions 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 1 block 1 body 1 instructions 1 instruction 2 1 return "void" -1 1 -2 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 1 block 1 body 1 instructions 1 instruction 2 1 coro_suspend "void" -1 1 -1 1 -1 payloads 1 "bad-token" } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 1 block 1 body 1 instructions 1 instruction 2 1 coro_suspend "void" -1 1 -1 1 4294967296 payloads 1 "bad-token" } })",
            // Invalid forward state must be rejected before the asserting constructor runs.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 3 block 1 block 2 block 3 body 1 instructions 3 instruction 4 1 autodiff_scope "void" -1 1 2 3 3 1 0 instruction 5 2 branch "void" -1 1 3 0 instruction 6 3 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 2 block 1 block 2 body 1 instructions 2 instruction 3 1 outline "void" -1 1 2 1 -1 instruction 4 2 return "void" -1 1 -1 0 } })",
            // Message-bearing instructions require exactly one payload record.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 0 blocks 1 block 1 body 1 instructions 1 instruction 2 1 unreachable "void" -1 0 0 } })",
            // Cooperative-vector constants are deliberately fail-closed.
            R"(xir.text 1 module { globals 1 constant 0 "coopvec<float,4>" "00000000000000000000000000000000" functions 0 })",
            // External functions cannot smuggle in blocks or instructions.
            R"(xir.text 1 module { globals 0 functions 1 function 0 external "void" 0 0 0 { arguments 0 blocks 1 block 1 body -1 instructions 0 } })"};
        for (auto text : malformed) { expect_interchange_rejected(text); }

        auto oversized_tiny_input = xir_from_interchange_text(
            "xir.text 1 module { globals 1048577");
        expect(!oversized_tiny_input.succeeded());
        expect(!oversized_tiny_input.diagnostics.empty());
        if (!oversized_tiny_input.diagnostics.empty()) {
            expect(oversized_tiny_input.diagnostics.front().message ==
                   "Record count exceeds the supported limit.");
        }
    };

    "xir_interchange_writer_rejects_external_definitions"_test = [] {
        Module module;
        auto external = module.create_external_function(nullptr);
        auto block = external->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(block);
        builder.return_void();
        auto encoded = xir_to_interchange_text(&module);
        expect(!encoded.succeeded());
        expect(encoded.text.empty());
        expect(!encoded.diagnostics.empty());
    };
}

void reg_structured_control_flow_round_trip() {
    "xir_interchange_structured_control_flow_round_trip"_test = [] {
        Module module;
        auto bool_type = luisa::compute::Type::of<bool>();
        auto int_type = luisa::compute::Type::of<int32_t>();
        auto true_value = module.create_constant_one(bool_type);
        auto false_value = module.create_constant_zero(bool_type);
        auto zero = module.create_constant_zero(int_type);
        auto one = module.create_constant_one(int_type);
        auto kernel = module.create_kernel();
        auto entry = kernel->create_body_block();
        XIRBuilder builder;

        builder.set_insertion_point(entry);
        auto if_instruction = builder.if_(true_value);
        auto if_true = if_instruction->create_true_block();
        auto if_false = if_instruction->create_false_block();
        auto if_merge = if_instruction->create_merge_block();
        builder.set_insertion_point(if_true);
        builder.br(if_merge);
        builder.set_insertion_point(if_false);
        builder.br(if_merge);
        builder.set_insertion_point(if_merge);
        auto selector = builder.phi(int_type, {{one, if_true}, {zero, if_false}});

        auto switch_instruction = builder.switch_(selector);
        auto switch_default = switch_instruction->create_default_block();
        auto switch_case = switch_instruction->create_case_block(1);
        auto switch_merge = switch_instruction->create_merge_block();
        builder.set_insertion_point(switch_default);
        builder.br(switch_merge);
        builder.set_insertion_point(switch_case);
        builder.br(switch_merge);

        builder.set_insertion_point(switch_merge);
        auto loop = builder.loop();
        auto prepare = loop->create_prepare_block();
        auto loop_body = loop->create_body_block();
        auto update = loop->create_update_block();
        auto loop_merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(false_value, loop_body, loop_merge);
        builder.set_insertion_point(loop_body);
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);

        builder.set_insertion_point(loop_merge);
        auto simple_loop = builder.simple_loop();
        auto simple_body = simple_loop->create_body_block();
        auto simple_merge = simple_loop->create_merge_block();
        builder.set_insertion_point(simple_body);
        builder.cond_br(false_value, simple_body, simple_merge);
        builder.set_insertion_point(simple_merge);
        builder.return_void();

        auto encoded = xir_to_interchange_text(&module);
        expect(encoded.succeeded());
        auto decoded = xir_from_interchange_text(encoded.text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) {
            return;
        }
        auto canonical_again = xir_to_interchange_text(decoded.module.get());
        expect(canonical_again.succeeded());
        expect(canonical_again.text == encoded.text);

        size_t if_count = 0u;
        size_t switch_count = 0u;
        size_t loop_count = 0u;
        size_t simple_loop_count = 0u;
        size_t phi_count = 0u;
        for (auto function : decoded.module->function_list()) {
            for (auto block : function->basic_blocks()) {
                for (auto instruction : block->instructions()) {
                    if_count += instruction->isa<IfInst>() ? 1u : 0u;
                    switch_count += instruction->isa<SwitchInst>() ? 1u : 0u;
                    loop_count += instruction->isa<LoopInst>() ? 1u : 0u;
                    simple_loop_count += instruction->isa<SimpleLoopInst>() ? 1u : 0u;
                    phi_count += instruction->isa<PhiInst>() ? 1u : 0u;
                    if (instruction->isa<SwitchInst>()) {
                        auto value = static_cast<SwitchInst *>(instruction);
                        expect(value->case_count() == 1u);
                        expect(value->case_value(0u) == 1);
                    }
                }
            }
        }
        expect(if_count == 1u);
        expect(switch_count == 1u);
        expect(loop_count == 1u);
        expect(simple_loop_count == 1u);
        expect(phi_count == 1u);
    };
}

void reg_quoted_escape_and_reference_validation() {
    "xir_interchange_hex_escape_decodes_once"_test = [] {
        constexpr auto text = R"(
xir.text 1
module {
  globals 0
  functions 1
  function 0 external "void" 0 0 0 {
    arguments 1
    argument 1 value "i\x6et"
    blocks 0
    body -1
    instructions 0
  }
}
)";
        auto decoded = xir_from_interchange_text(text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        auto function = *decoded.module->function_list().begin();
        auto argument = *function->arguments().begin();
        expect(argument->type() == luisa::compute::Type::of<int32_t>());

        auto malformed = xir_from_interchange_text(R"(
xir.text 1 module { globals 0 functions 1
function 0 external "void" 0 0 0 {
arguments 1 argument 1 value "i\x6t"
blocks 0 body -1 instructions 0 }
})");
        expect(!malformed.succeeded());
        expect(!malformed.diagnostics.empty());
    };
}

void reg_remaining_misc_instruction_round_trip() {
    "xir_interchange_remaining_misc_instructions_round_trip"_test = [] {
        Module module;
        auto bool_type = Type::of<bool>();
        auto int_type = Type::of<int32_t>();
        auto condition = module.create_constant_one(bool_type);
        auto value = module.create_constant_one(int_type);
        XIRBuilder builder;

        auto unreachable_function = module.create_callable(nullptr);
        builder.set_insertion_point(unreachable_function->create_body_block());
        builder.unreachable_("unreachable: quoted=\"yes\"\nnext");

        auto raster_function = module.create_callable(nullptr);
        builder.set_insertion_point(raster_function->create_body_block());
        builder.raster_discard();

        auto suspend_function = module.create_callable(nullptr);
        builder.set_insertion_point(suspend_function->create_body_block());
        builder.coro_suspend(17u, "suspend-name", nullptr);

        auto resume_function = module.create_callable(nullptr);
        builder.set_insertion_point(resume_function->create_body_block());
        builder.coro_resume(17u, nullptr);
        builder.coro_terminate();

        auto misc_function = module.create_callable(nullptr);
        auto misc_body = misc_function->create_body_block();
        builder.set_insertion_point(misc_body);
        builder.print("value={}\n", {value});
        builder.clock();
        builder.debug_break();
        builder.assert_(condition, "assert-message");
        builder.assume_(condition, "assume-message");
        builder.return_void();

        auto encoded = xir_to_interchange_text(&module);
        expect(encoded.succeeded());
        if (!encoded.succeeded()) { return; }
        for (auto token : {"unreachable", "raster_discard", "coro_suspend", "coro_resume",
                           "coro_terminate", "print", "clock", "debug_break", "assert", "assume"}) {
            expect(encoded.text.find(token) != luisa::string::npos);
        }
        expect(encoded.text.find("payloads 1 \"suspend-name\"") != luisa::string::npos);
        expect(encoded.text.find("payloads 1 \"assert-message\"") != luisa::string::npos);

        auto decoded = xir_from_interchange_text(encoded.text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        size_t suspend_count = 0u;
        size_t print_count = 0u;
        size_t assert_count = 0u;
        for (auto function : decoded.module->function_list()) {
            for (auto block : function->basic_blocks()) {
                for (auto instruction : block->instructions()) {
                    if (instruction->isa<CoroSuspendInst>()) {
                        auto suspend = static_cast<CoroSuspendInst *>(instruction);
                        expect(suspend->token() == 17u);
                        expect(suspend->name() == "suspend-name");
                        suspend_count++;
                    } else if (instruction->isa<PrintInst>()) {
                        expect(static_cast<PrintInst *>(instruction)->format() == "value={}\n");
                        print_count++;
                    } else if (instruction->isa<AssertInst>()) {
                        expect(static_cast<AssertInst *>(instruction)->message() == "assert-message");
                        assert_count++;
                    }
                }
            }
        }
        expect(suspend_count == 1u);
        expect(print_count == 1u);
        expect(assert_count == 1u);
        auto canonical = xir_to_interchange_text(decoded.module.get());
        expect(canonical.succeeded());
        expect(canonical.text == encoded.text);
    };
}

void reg_autodiff_and_outline_round_trip() {
    "xir_interchange_autodiff_and_outline_round_trip"_test = [] {
        Module module;
        auto float_type = Type::of<float>();
        auto uint_type = Type::of<uint32_t>();
        auto index = module.create_constant_zero(uint_type);
        XIRBuilder builder;

        auto intrinsic_function = module.create_callable(nullptr);
        auto value = intrinsic_function->create_value_argument(float_type);
        auto intrinsic_body = intrinsic_function->create_body_block();
        builder.set_insertion_point(intrinsic_body);
        builder.call(nullptr, AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, {value});
        auto gradient = builder.call(float_type, AutodiffIntrinsicOp::AUTODIFF_GRADIENT, {value});
        builder.call(nullptr, AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {value, gradient});
        builder.call(nullptr, AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT, {value, gradient});
        builder.call(nullptr, AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});
        builder.call(float_type, AutodiffIntrinsicOp::AUTODIFF_DETACH, {value});
        builder.call(nullptr, AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, {value, gradient});
        builder.call(float_type, AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, {value, index});
        builder.return_void();

        auto scope_function = module.create_callable(nullptr);
        auto scope_entry = scope_function->create_body_block();
        builder.set_insertion_point(scope_entry);
        auto scope = builder.forward_autodiff_scope(2u);
        auto differentiated = scope->create_entry_block();
        auto scope_merge = scope->create_merge_block();
        builder.set_insertion_point(differentiated);
        builder.br(scope_merge);
        builder.set_insertion_point(scope_merge);
        builder.return_void();

        auto outline_function = module.create_callable(nullptr);
        auto outline_entry = outline_function->create_body_block();
        builder.set_insertion_point(outline_entry);
        auto outline = builder.outline();
        auto outlined = outline->create_target_block();
        auto outline_merge = outline->create_merge_block();
        builder.set_insertion_point(outlined);
        builder.br(outline_merge);
        builder.set_insertion_point(outline_merge);
        builder.return_void();

        auto encoded = xir_to_interchange_text(&module);
        expect(encoded.succeeded());
        if (!encoded.succeeded()) { return; }
        expect(encoded.text.find("autodiff_scope") != luisa::string::npos);
        expect(encoded.text.find("autodiff_gradient") != luisa::string::npos);
        expect(encoded.text.find("outline") != luisa::string::npos);
        auto decoded = xir_from_interchange_text(encoded.text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        size_t intrinsic_count = 0u;
        size_t scope_count = 0u;
        size_t outline_count = 0u;
        for (auto function : decoded.module->function_list()) {
            for (auto block : function->basic_blocks()) {
                for (auto instruction : block->instructions()) {
                    intrinsic_count += instruction->isa<AutodiffIntrinsicInst>() ? 1u : 0u;
                    if (instruction->isa<AutodiffScopeInst>()) {
                        auto decoded_scope = static_cast<AutodiffScopeInst *>(instruction);
                        expect(decoded_scope->is_forward());
                        expect(decoded_scope->n_forward_grads() == 2u);
                        expect(decoded_scope->entry_block() != nullptr);
                        expect(decoded_scope->merge_block() != nullptr);
                        scope_count++;
                    }
                    outline_count += instruction->isa<OutlineInst>() ? 1u : 0u;
                }
            }
        }
        expect(intrinsic_count == 8u);
        expect(scope_count == 1u);
        expect(outline_count == 1u);
        auto canonical = xir_to_interchange_text(decoded.module.get());
        expect(canonical.succeeded());
        expect(canonical.text == encoded.text);
    };
}

void reg_ray_query_instruction_round_trip() {
    "xir_interchange_ray_query_instructions_round_trip"_test = [] {
        Module module;
        auto float_type = Type::of<float>();
        auto bool_type = Type::of<bool>();
        auto uint_type = Type::of<uint32_t>();
        auto float2_type = Type::vector(float_type, 2u);
        auto ray = Type::structure(
            16u, {Type::array(float_type, 3u), float_type,
                  Type::array(float_type, 3u), float_type});
        auto surface_hit = Type::structure(
            8u, {uint_type, uint_type, float2_type, float_type});
        auto procedural_hit = Type::structure(8u, {uint_type, uint_type});
        auto committed_hit = Type::structure(
            8u, {uint_type, uint_type, float2_type, uint_type, float_type});
        auto query_type = Type::custom("LC_RayQueryAll");
        XIRBuilder builder;

        auto object_function = module.create_callable(nullptr);
        auto query = object_function->create_reference_argument(query_type);
        auto distance = object_function->create_value_argument(float_type);
        builder.set_insertion_point(object_function->create_body_block());
        builder.call(ray, RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY, {query});
        builder.call(ray, RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY, {query});
        builder.call(procedural_hit, RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT, {query});
        builder.call(surface_hit, RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT, {query});
        builder.call(committed_hit, RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT, {query});
        builder.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE, {query});
        builder.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE, {query});
        builder.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, {query});
        builder.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE, {query});
        builder.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL, {query, distance});
        builder.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE, {query});
        builder.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, {query});
        builder.return_void();

        auto surface_handler = module.create_callable(nullptr);
        surface_handler->create_reference_argument(query_type);
        surface_handler->create_value_argument(float_type);
        builder.set_insertion_point(surface_handler->create_body_block());
        builder.return_void();
        auto procedural_handler = module.create_callable(nullptr);
        procedural_handler->create_reference_argument(query_type);
        procedural_handler->create_value_argument(float_type);
        builder.set_insertion_point(procedural_handler->create_body_block());
        builder.return_void();

        auto pipeline_function = module.create_callable(nullptr);
        auto pipeline_query = pipeline_function->create_reference_argument(query_type);
        auto capture = pipeline_function->create_value_argument(float_type);
        builder.set_insertion_point(pipeline_function->create_body_block());
        std::array<Value *, 1u> captures{capture};
        builder.ray_query_pipeline(pipeline_query, surface_handler, procedural_handler,
                                   luisa::span<Value *const>{captures});
        builder.return_void();

        auto loop_function = module.create_callable(nullptr);
        auto loop_query = loop_function->create_reference_argument(query_type);
        auto entry = loop_function->create_body_block();
        builder.set_insertion_point(entry);
        auto loop = builder.ray_query_loop();
        auto dispatch_block = loop->create_dispatch_block();
        auto merge_block = loop->create_merge_block();
        builder.set_insertion_point(dispatch_block);
        auto dispatch = builder.ray_query_dispatch(loop_query);
        dispatch->set_exit_block(merge_block);
        auto surface_block = dispatch->create_on_surface_candidate_block();
        auto procedural_block = dispatch->create_on_procedural_candidate_block();
        builder.set_insertion_point(surface_block);
        builder.br(dispatch_block);
        builder.set_insertion_point(procedural_block);
        builder.br(dispatch_block);
        builder.set_insertion_point(merge_block);
        builder.return_void();

        auto encoded = xir_to_interchange_text(&module);
        expect(encoded.succeeded());
        if (!encoded.succeeded()) { return; }
        for (auto token : {"ray_query_loop", "ray_query_dispatch", "ray_query_object_read",
                           "ray_query_object_write", "ray_query_pipeline"}) {
            expect(encoded.text.find(token) != luisa::string::npos);
        }
        auto decoded = xir_from_interchange_text(encoded.text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        std::array<size_t, 5u> counts{};
        for (auto function : decoded.module->function_list()) {
            for (auto block : function->basic_blocks()) {
                for (auto instruction : block->instructions()) {
                    switch (instruction->derived_instruction_tag()) {
                        case DerivedInstructionTag::RAY_QUERY_LOOP: counts[0u]++; break;
                        case DerivedInstructionTag::RAY_QUERY_DISPATCH: counts[1u]++; break;
                        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: counts[2u]++; break;
                        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: counts[3u]++; break;
                        case DerivedInstructionTag::RAY_QUERY_PIPELINE: counts[4u]++; break;
                        default: break;
                    }
                }
            }
        }
        expect(counts[0u] == 1u);
        expect(counts[1u] == 1u);
        expect(counts[2u] == 8u);
        expect(counts[3u] == 4u);
        expect(counts[4u] == 1u);
        auto canonical = xir_to_interchange_text(decoded.module.get());
        expect(canonical.succeeded());
        expect(canonical.text == encoded.text);
    };
}

void reg_reference_validation() {

    "xir_interchange_cross_function_block_reference_rejected"_test = [] {
        constexpr auto text = R"(
xir.text 1
module {
  globals 0
  functions 2
  function 0 callable "void" 0 0 0 {
    arguments 0
    blocks 1
    block 1
    body 1
    instructions 1
    instruction 2 1 branch "void" -1 1 4 0
  }
  function 3 callable "void" 0 0 0 {
    arguments 0
    blocks 1
    block 4
    body 4
    instructions 1
    instruction 5 4 return "void" -1 1 -1 0
  }
}
)";
        auto decoded = xir_from_interchange_text(text);
        expect(!decoded.succeeded());
        expect(decoded.module == nullptr);
        expect(!decoded.diagnostics.empty());
    };

    "xir_interchange_arithmetic_arity_rejected"_test = [] {
        constexpr auto text = R"(
xir.text 1
module {
  globals 0
  functions 1
  function 0 callable "int" 0 0 0 {
    arguments 1
    argument 1 value "int"
    blocks 1
    block 2
    body 2
    instructions 2
    instruction 3 2 arithmetic "int" 2 1 1 0
    instruction 4 2 return "void" -1 1 3 0
  }
}
)";
        auto decoded = xir_from_interchange_text(text);
        expect(!decoded.succeeded());
        expect(decoded.module == nullptr);
        expect(!decoded.diagnostics.empty());
    };
}

void reg_instruction_type_validation() {
    "xir_interchange_rejects_invalid_kernel_block_size"_test = [] {
        constexpr luisa::string_view malformed =
            R"(xir.text 1 module { globals 0 functions 1 function 0 kernel "void" 1 1 1 { arguments 0 blocks 1 block 1 body 1 instructions 1 instruction 2 1 return "void" -1 1 -1 0 } })";
        auto decoded_text = xir_from_interchange_text(malformed);
        expect(!decoded_text.succeeded());
        expect(decoded_text.module == nullptr);
        expect(!decoded_text.diagnostics.empty());

        auto payload = luisa::span{
            reinterpret_cast<const std::byte *>(malformed.data()),
            malformed.size()};
        auto decoded_bitcode = xir_from_bitcode(make_test_bitcode(payload, 1u));
        expect(!decoded_bitcode.succeeded());
        expect(decoded_bitcode.module == nullptr);
        expect(!decoded_bitcode.diagnostics.empty());
    };

    "xir_interchange_integer_and_bool_switch_selectors_round_trip"_test = [] {
        for (auto selector_type : {
                 Type::of<bool>(),
                 Type::of<int8_t>(), Type::of<uint8_t>(),
                 Type::of<int16_t>(), Type::of<uint16_t>(),
                 Type::of<int32_t>(), Type::of<uint32_t>(),
                 Type::of<int64_t>(), Type::of<uint64_t>()}) {
            Module module;
            auto callable = module.create_callable(nullptr);
            auto selector = callable->create_value_argument(selector_type);
            auto body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto switch_inst = builder.switch_(selector);
            auto case_block = switch_inst->create_case_block(
                std::numeric_limits<uint64_t>::max());
            auto default_block = switch_inst->create_default_block();
            auto merge_block = switch_inst->create_merge_block();
            builder.set_insertion_point(case_block);
            builder.br(merge_block);
            builder.set_insertion_point(default_block);
            builder.br(merge_block);
            builder.set_insertion_point(merge_block);
            builder.return_void();

            expect(xir_verify_module(&module).succeeded());
            auto text = xir_to_interchange_text(&module);
            expect(text.succeeded());
            if (!text.succeeded()) { continue; }
            auto decoded_text = xir_from_interchange_text(text.text);
            expect(decoded_text.succeeded());
            if (decoded_text.succeeded()) {
                expect(xir_verify_module(decoded_text.module.get()).succeeded());
                expect(xir_to_interchange_text(
                           decoded_text.module.get())
                           .succeeded());
            }

            auto bitcode = xir_to_bitcode(&module);
            expect(bitcode.succeeded());
            if (!bitcode.succeeded()) { continue; }
            auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
            expect(decoded_bitcode.succeeded());
        }
    };

    "xir_interchange_zero_sized_data_types_round_trip"_test = [] {
        std::array<const Type *, 0u> no_members{};
        for (auto *type : {
                 Type::array(Type::of<uint32_t>(), 0u),
                 Type::structure(4u, luisa::span{no_members})}) {
            Module module;
            auto *constant = module.create_constant_zero(type);
            auto *callable = module.create_callable(type);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            builder.return_(constant);
            expect(xir_verify_module(&module).succeeded());

            auto verify = [type](const XIRInterchangeParseResult &decoded) noexcept {
                expect(decoded.succeeded());
                if (!decoded.succeeded()) { return; }
                auto found_constant = false;
                for (auto *value : decoded.module->constant_list()) {
                    if (value->type() == type) {
                        expect(eq(value->type()->size(), size_t{0u}));
                        found_constant = true;
                    }
                }
                expect(found_constant);
                expect(xir_verify_module(decoded.module.get()).succeeded());
            };

            auto text = xir_to_interchange_text(&module);
            expect(text.succeeded());
            if (text.succeeded()) {
                verify(xir_from_interchange_text(text.text));
            }
            auto bitcode = xir_to_bitcode(&module);
            expect(bitcode.succeeded());
            if (bitcode.succeeded()) {
                verify(xir_from_bitcode(bitcode.bitcode));
            }
        }
    };

    "xir_interchange_rejects_noncanonical_narrow_switch_cases"_test = [] {
        constexpr std::array malformed_cases{
            std::pair{"ubyte", "511"},
            std::pair{"ubyte", "-1"},
            std::pair{"byte", "-129"},
            std::pair{"byte", "255"},
            std::pair{"bool", "2"}};
        constexpr luisa::string_view expected_diagnostic =
            "XIR switch case value is outside the selector type range.";
        for (auto [selector_type, case_value] : malformed_cases) {
            auto text = luisa::format(
                R"(xir.text 1 module {{ globals 0 functions 1 function 0 callable "void" 0 0 0 {{ arguments 1 argument 1 value "{}" blocks 4 block 2 block 3 block 4 block 5 body 2 instructions 4 instruction 6 2 switch "void" -1 3 1 3 4 2 5 {} instruction 7 3 branch "void" -1 1 5 0 instruction 8 4 branch "void" -1 1 5 0 instruction 9 5 return "void" -1 1 -1 0 }} }})",
                selector_type, case_value);
            expect_interchange_rejected_with_diagnostic(text, expected_diagnostic);

            auto payload = luisa::span{
                reinterpret_cast<const std::byte *>(text.data()), text.size()};
            auto decoded_bitcode = xir_from_bitcode(make_test_bitcode(payload, 1u));
            expect(!decoded_bitcode.succeeded());
            expect(decoded_bitcode.module == nullptr);
            auto found = std::any_of(
                decoded_bitcode.diagnostics.begin(), decoded_bitcode.diagnostics.end(),
                [&](auto &&diagnostic) noexcept {
                    return diagnostic.message == expected_diagnostic;
                });
            expect(found);
        }
    };

    "xir_interchange_preserves_distinct_u64_switch_case_bits"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *selector = callable->create_value_argument(Type::of<uint64_t>());
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *switch_inst = builder.switch_(selector);
        constexpr auto low_word_ones = uint64_t{0x00000000ffffffffull};
        constexpr auto all_ones = uint64_t{0xffffffffffffffffull};
        auto *low_word_block = switch_inst->create_case_block(low_word_ones);
        auto *all_ones_block = switch_inst->create_case_block(all_ones);
        auto *default_block = switch_inst->create_default_block();
        auto *merge_block = switch_inst->create_merge_block();
        for (auto *block : {low_word_block, all_ones_block, default_block}) {
            builder.set_insertion_point(block);
            builder.br(merge_block);
        }
        builder.set_insertion_point(merge_block);
        builder.return_void();
        expect(xir_verify_module(&module).succeeded());

        auto verify = [&](Module *decoded) noexcept {
            auto found = false;
            for (auto *function : decoded->function_list()) {
                for (auto *block : function->basic_blocks()) {
                    for (auto *instruction : block->instructions()) {
                        if (!instruction->isa<SwitchInst>()) { continue; }
                        auto *value = static_cast<const SwitchInst *>(instruction);
                        expect(value->value()->type() == Type::of<uint64_t>());
                        expect(value->case_count() == 2u);
                        expect(value->case_value(0u) == low_word_ones);
                        expect(value->case_value(1u) == all_ones);
                        found = true;
                    }
                }
            }
            expect(found);
        };

        auto text = xir_to_interchange_text(&module);
        expect(text.succeeded());
        if (text.succeeded()) {
            auto decoded = xir_from_interchange_text(text.text);
            expect(decoded.succeeded());
            if (decoded.succeeded()) { verify(decoded.module.get()); }
        }

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        if (bitcode.succeeded()) {
            auto decoded = xir_from_bitcode(bitcode.bitcode);
            expect(decoded.succeeded());
            if (decoded.succeeded()) { verify(decoded.module.get()); }
        }
    };

    "xir_interchange_terminal_if_and_indexed_branch_round_trip"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *selector = kernel->create_value_argument(Type::of<int>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *if_inst = builder.if_(condition);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        auto *indexed_branch = builder.indexed_branch(selector);
        auto *case_block = indexed_branch->create_case_block(1);
        auto *default_block =
            indexed_branch->create_default_block();
        builder.set_insertion_point(case_block);
        builder.return_void();
        builder.set_insertion_point(default_block);
        builder.return_void();

        auto check_decoded = [](const Module *decoded) noexcept {
            size_t null_if_count = 0u;
            size_t indexed_branch_count = 0u;
            for (auto *function : decoded->function_list()) {
                for (auto *block : function->basic_blocks()) {
                    for (auto *instruction : block->instructions()) {
                        if (instruction->isa<IfInst>()) {
                            null_if_count += static_cast<const IfInst *>(instruction)->merge_block() == nullptr ? 1u : 0u;
                        }
                        indexed_branch_count +=
                            instruction->isa<IndexedBranchInst>() ?
                                1u :
                                0u;
                    }
                }
            }
            expect(null_if_count == 1u);
            expect(indexed_branch_count == 1u);
        };

        auto text = xir_to_interchange_text(&module);
        expect(text.succeeded());
        if (!text.succeeded()) { return; }
        auto decoded_text = xir_from_interchange_text(text.text);
        expect(decoded_text.succeeded());
        if (decoded_text.succeeded()) { check_decoded(decoded_text.module.get()); }

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        if (!bitcode.succeeded()) { return; }
        auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(decoded_bitcode.succeeded());
        if (decoded_bitcode.succeeded()) { check_decoded(decoded_bitcode.module.get()); }
    };

    "xir_interchange_instruction_type_paths_round_trip"_test = [] {
        Module module;
        auto int_type = Type::of<int32_t>();
        auto uint_type = Type::of<uint32_t>();
        auto float_type = Type::of<float>();
        auto float2_type = Type::vector(float_type, 2u);
        auto float_array_type = Type::array(float_type, 2u);
        auto uint2_type = Type::vector(uint_type, 2u);
        auto ushort4_type = Type::vector(Type::of<uint16_t>(), 4u);
        auto aggregate_type = Type::structure(
            {int_type, float_array_type});
        auto one = module.create_constant_one(int_type);
        auto wide_one = module.create_constant_one(Type::of<uint64_t>());
        auto narrow_zero = module.create_constant_zero(Type::of<int16_t>());
        auto zero_float = module.create_constant_zero(float_type);
        auto float2_zero = module.create_constant_zero(float2_type);
        auto matrix2_zero = module.create_constant_zero(Type::matrix(2u));
        int32_t negative_index_value = -1;
        uint32_t upper_bound_index_value = 2u;
        auto negative_index = module.create_constant(
            int_type, &negative_index_value);
        auto upper_bound_index = module.create_constant(
            uint_type, &upper_bound_index_value);
        auto field_index = module.create_constant_one(uint_type);
        auto element_index = module.create_constant_zero(uint_type);
        auto callable = module.create_callable(nullptr);
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto storage = builder.alloca_local(aggregate_type);
        builder.gep(float_type, storage, {field_index, element_index});
        auto homogeneous_storage = builder.alloca_local(float_array_type);
        builder.gep(float_type, homogeneous_storage, {negative_index});
        builder.gep(float_type, homogeneous_storage, {upper_bound_index});
        builder.static_cast_(uint_type, one);
        builder.bit_cast_(float_type, one);
        builder.bit_cast_(ushort4_type, wide_one);
        builder.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
                     {zero_float, zero_float});
        builder.call(uint_type, ArithmeticOp::CLZ, {field_index});
        builder.call(
            uint2_type, ArithmeticOp::REVERSE,
            {module.create_constant_zero(uint2_type)});
        builder.call(
            float2_type, ArithmeticOp::SHUFFLE,
            {float2_zero, narrow_zero, wide_one});
        builder.call(
            float2_type, ArithmeticOp::SHUFFLE,
            {float2_zero, negative_index, upper_bound_index});
        auto float_array = builder.call(
            float_array_type, ArithmeticOp::AGGREGATE,
            {zero_float, zero_float});
        builder.call(
            float_type, ArithmeticOp::EXTRACT,
            {float_array, narrow_zero});
        builder.call(
            float_type, ArithmeticOp::EXTRACT,
            {float_array, upper_bound_index});
        builder.call(
            float_array_type, ArithmeticOp::INSERT,
            {float_array, zero_float, wide_one});
        builder.call(
            float_array_type, ArithmeticOp::INSERT,
            {float_array, zero_float, negative_index});
        builder.call(float_type, ArithmeticOp::SATURATE, {zero_float});
        builder.call(float_type, ArithmeticOp::ACOS, {zero_float});
        builder.call(
            float_type, ArithmeticOp::DOT, {float2_zero, float2_zero});
        builder.call(
            float_type, ArithmeticOp::REDUCE_SUM, {float2_zero});
        builder.call(
            float_type, ArithmeticOp::MATRIX_DETERMINANT, {matrix2_zero});
        builder.return_void();

        auto text = xir_to_interchange_text(&module);
        expect(text.succeeded());
        if (!text.succeeded()) { return; }
        auto decoded_text = xir_from_interchange_text(text.text);
        expect(decoded_text.succeeded());

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        if (!bitcode.succeeded()) { return; }
        auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(decoded_bitcode.succeeded());
    };

    "xir_interchange_malformed_instruction_types_rejected"_test = [] {
        constexpr std::array<luisa::string_view, 15u> malformed{
            // Arithmetic result and operand types must agree.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 value "int" argument 2 value "int" blocks 1 block 3 body 3 instructions 2 instruction 4 3 arithmetic "float" binary_add 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Comparison results must have the matching boolean shape.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 value "int" argument 2 value "int" blocks 1 block 3 body 3 instructions 2 instruction 4 3 arithmetic "int" binary_equal 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Bit-count operations use uint32 scalar/vector registers.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "int" blocks 1 block 2 body 2 instructions 2 instruction 3 2 arithmetic "int" clz 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "ushort" blocks 1 block 2 body 2 instructions 2 instruction 3 2 arithmetic "ushort" popcount 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            // Static casts preserve scalar/vector shape.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "int" blocks 1 block 2 body 2 instructions 2 instruction 3 2 cast "vector<uint,2>" static_cast 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            // Bitwise casts preserve logical register width.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "int" blocks 1 block 2 body 2 instructions 2 instruction 3 2 cast "ulong" bitwise_cast 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            // ABI padding must not make float3 and uint4 appear bit-compatible.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "vector<float,3>" blocks 1 block 2 body 2 instructions 2 instruction 3 2 cast "vector<uint,4>" bitwise_cast 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            // Portable bitwise casts do not admit boolean registers.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "bool" blocks 1 block 2 body 2 instructions 2 instruction 3 2 cast "bool" bitwise_cast 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "vector<bool,2>" blocks 1 block 2 body 2 instructions 2 instruction 3 2 cast "vector<bool,2>" bitwise_cast 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            // A GEP result must equal the addressed leaf type.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 reference "array<int,2>" argument 2 value "uint" blocks 1 block 3 body 3 instructions 2 instruction 4 3 gep "float" -1 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // GEP indices must be integer rvalues.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 reference "array<int,2>" argument 2 value "float" blocks 1 block 3 body 3 instructions 2 instruction 4 3 gep "int" -1 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Structure indices must be compile-time constants.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 reference "struct<4,int,uint>" argument 2 value "uint" blocks 1 block 3 body 3 instructions 2 instruction 4 3 gep "int" -1 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Constant structure indices must be in range.
            R"(xir.text 1 module { globals 1 constant 0 "uint" "02000000" functions 1 function 1 callable "void" 0 0 0 { arguments 1 argument 2 reference "struct<4,int,uint>" blocks 1 block 3 body 3 instructions 2 instruction 4 3 gep "int" -1 2 2 0 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Switch selectors must be integer scalars.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "float" blocks 3 block 2 block 3 block 4 body 2 instructions 3 instruction 5 2 switch "void" -1 2 1 3 1 4 instruction 6 3 branch "void" -1 1 4 0 instruction 7 4 return "void" -1 1 -1 0 } })",
            // Integer vectors are not scalar switch selectors.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "vector<int,2>" blocks 3 block 2 block 3 block 4 body 2 instructions 3 instruction 5 2 switch "void" -1 2 1 3 1 4 instruction 6 3 branch "void" -1 1 4 0 instruction 7 4 return "void" -1 1 -1 0 } })"};
        for (auto text : malformed) {
            auto decoded = xir_from_interchange_text(text);
            expect(!decoded.succeeded());
            expect(decoded.module == nullptr);
            expect(!decoded.diagnostics.empty());
            auto semantic_failure = std::any_of(
                decoded.diagnostics.begin(), decoded.diagnostics.end(),
                [](auto &&diagnostic) noexcept {
                    return diagnostic.message ==
                           "XIR instruction operands or result type do not match its operation.";
                });
            auto first_diagnostic = decoded.diagnostics.empty() ?
                                        luisa::string_view{} :
                                        luisa::string_view{decoded.diagnostics.front().message};
            expect(semantic_failure)
                << "otherwise well-formed text must fail operation-specific type validation: "
                << text << " first diagnostic: "
                << first_diagnostic;
        }

        auto payload = luisa::span{
            reinterpret_cast<const std::byte *>(malformed.front().data()),
            malformed.front().size()};
        auto decoded_bitcode = xir_from_bitcode(make_test_bitcode(payload, 1u));
        expect(!decoded_bitcode.succeeded());
        expect(decoded_bitcode.module == nullptr);
        expect(!decoded_bitcode.diagnostics.empty());
    };

    "xir_interchange_rejects_invalid_value_categories_and_targets"_test = [] {
        constexpr std::array<luisa::string_view, 9u> malformed{
            // Structured and unstructured conditions must be boolean rvalues.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 reference "bool" blocks 3 block 2 block 3 block 4 body 2 instructions 3 instruction 5 2 if "void" -1 3 1 3 3 1 4 instruction 6 3 branch "void" -1 1 4 0 instruction 7 4 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 reference "bool" blocks 2 block 2 block 3 body 2 instructions 2 instruction 4 2 conditional_branch "void" -1 3 1 3 3 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Branch targets must be basic blocks, never arbitrary values.
            R"(xir.text 1 module { globals 1 constant 0 "int" "00000000" functions 1 function 1 callable "void" 0 0 0 { arguments 0 blocks 1 block 2 body 2 instructions 1 instruction 3 2 branch "void" -1 1 0 0 } })",
            R"(xir.text 1 module { globals 1 constant 0 "int" "00000000" functions 1 function 1 callable "void" 0 0 0 { arguments 1 argument 2 value "bool" blocks 2 block 3 block 4 body 3 instructions 2 instruction 5 3 conditional_branch "void" -1 3 2 0 4 0 instruction 6 4 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 1 constant 0 "int" "00000000" functions 1 function 1 callable "void" 0 0 0 { arguments 1 argument 2 value "bool" blocks 3 block 3 block 4 block 5 body 3 instructions 3 instruction 6 3 if "void" -1 3 2 0 4 1 5 instruction 7 4 branch "void" -1 1 5 0 instruction 8 5 return "void" -1 1 -1 0 } })",
            // Loads/stores distinguish lvalues from rvalues.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 value "int" blocks 1 block 2 body 2 instructions 2 instruction 3 2 load "int" -1 1 1 0 instruction 4 2 return "void" -1 1 -1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 2 argument 1 reference "int" argument 2 reference "int" blocks 1 block 3 body 3 instructions 2 instruction 4 3 store "void" -1 2 1 2 0 instruction 5 3 return "void" -1 1 -1 0 } })",
            // Return and PHI values must be rvalues.
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "int" 0 0 0 { arguments 1 argument 1 reference "int" blocks 1 block 2 body 2 instructions 1 instruction 3 2 return "void" -1 1 1 0 } })",
            R"(xir.text 1 module { globals 0 functions 1 function 0 callable "void" 0 0 0 { arguments 1 argument 1 reference "int" blocks 2 block 2 block 3 body 2 instructions 3 instruction 4 2 branch "void" -1 1 3 0 instruction 5 3 phi "int" -1 1 1 1 2 instruction 6 3 return "void" -1 1 -1 0 } })"};
        for (auto text : malformed) {
            auto decoded = xir_from_interchange_text(text);
            expect(!decoded.succeeded());
            expect(decoded.module == nullptr);
            expect(!decoded.diagnostics.empty());
            expect(std::any_of(
                decoded.diagnostics.begin(), decoded.diagnostics.end(),
                [](auto &&diagnostic) noexcept {
                    return diagnostic.message ==
                           "XIR instruction operands or result type do not match its operation.";
                }));
        }

        auto payload = luisa::span{
            reinterpret_cast<const std::byte *>(malformed[2u].data()),
            malformed[2u].size()};
        auto decoded_bitcode = xir_from_bitcode(make_test_bitcode(payload, 1u));
        expect(!decoded_bitcode.succeeded());
        expect(decoded_bitcode.module == nullptr);
        expect(!decoded_bitcode.diagnostics.empty());
    };

    "xir_interchange_custom_rvalue_load_store_round_trip"_test = [] {
        Module module;
        auto query_type = Type::custom("LC_RayQueryAll");
        auto callable = module.create_callable(nullptr);
        auto query = callable->create_reference_argument(query_type);
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto source = builder.load(query_type, query);
        auto local = builder.alloca_local(query_type);
        builder.store(local, source);
        builder.load(query_type, local);
        builder.return_void();

        auto text = xir_to_interchange_text(&module);
        expect(text.succeeded());
        if (!text.succeeded()) { return; }
        auto decoded_text = xir_from_interchange_text(text.text);
        expect(decoded_text.succeeded());

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        if (!bitcode.succeeded()) { return; }
        auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(decoded_bitcode.succeeded());
    };

    "xir_interchange_writer_rejects_malformed_instruction_types"_test = [] {
        {
            Module module;
            auto callable = module.create_callable(nullptr);
            auto lhs = callable->create_value_argument(Type::of<int32_t>());
            auto rhs = callable->create_value_argument(Type::of<int32_t>());
            auto body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            builder.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {lhs, rhs});
            builder.return_void();
            auto encoded = xir_to_interchange_text(&module);
            expect(!encoded.succeeded());
            expect(encoded.text.empty());
            expect(!encoded.diagnostics.empty());
        }
        {
            Module module;
            auto callable = module.create_callable(nullptr);
            auto value = callable->create_value_argument(Type::of<int32_t>());
            auto body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            builder.bit_cast_(Type::of<uint64_t>(), value);
            builder.return_void();
            auto encoded = xir_to_interchange_text(&module);
            expect(!encoded.succeeded());
            expect(encoded.text.empty());
            expect(!encoded.diagnostics.empty());
        }
        {
            Module module;
            auto callable = module.create_callable(nullptr);
            auto index = callable->create_value_argument(Type::of<uint32_t>());
            auto body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto storage = builder.alloca_local(
                Type::array(Type::of<int32_t>(), 2u));
            builder.gep(Type::of<float>(), storage, {index});
            builder.return_void();
            auto encoded = xir_to_interchange_text(&module);
            expect(!encoded.succeeded());
            expect(encoded.text.empty());
            expect(!encoded.diagnostics.empty());
        }
    };

    "xir_interchange_writer_rejects_invalid_instruction_opcode"_test = [] {
        Module module;
        auto callable = module.create_callable(nullptr);
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto alloca = builder.alloca_local(Type::of<int32_t>());
        alloca->set_op(static_cast<AllocaOp>(999));
        builder.return_void();

        auto text = xir_to_interchange_text(&module);
        expect(!text.succeeded());
        expect(text.text.empty());
        expect(!text.diagnostics.empty());
        auto bitcode = xir_to_bitcode(&module);
        expect(!bitcode.succeeded());
        expect(bitcode.bitcode.empty());
        expect(!bitcode.diagnostics.empty());
    };

    "xir_interchange_writer_rejects_argument_kind_type_mismatches"_test = [] {
        Module module;
        auto buffer_type = Type::buffer(Type::of<int32_t>());
        auto custom_type = Type::custom("InterchangeOpaque");
        auto int_type = Type::of<int32_t>();

        auto value_resource = module.create_external_function(nullptr);
        value_resource->arguments().push_back(
            luisa::make_managed<ValueArgument>(value_resource, buffer_type));
        auto value_custom = module.create_external_function(nullptr);
        value_custom->arguments().push_back(
            luisa::make_managed<ValueArgument>(value_custom, custom_type));
        auto reference_resource = module.create_external_function(nullptr);
        reference_resource->arguments().push_back(
            luisa::make_managed<ReferenceArgument>(
                reference_resource, buffer_type));
        auto resource_data = module.create_external_function(nullptr);
        resource_data->arguments().push_back(
            luisa::make_managed<ResourceArgument>(resource_data, int_type));

        auto text = xir_to_interchange_text(&module);
        expect(!text.succeeded());
        expect(text.text.empty());
        expect(!text.diagnostics.empty());
        auto bitcode = xir_to_bitcode(&module);
        expect(!bitcode.succeeded());
        expect(bitcode.bitcode.empty());
        expect(!bitcode.diagnostics.empty());
    };
}

void reg_metadata_round_trip() {
    "xir_interchange_all_metadata_round_trip"_test = [] {
        Module module;
        auto int_type = luisa::compute::Type::of<int32_t>();
        auto one = module.create_constant_one(int_type);
        auto undefined = module.create_undefined(int_type);
        auto thread_id = module.create_thread_id();
        auto callable = module.create_callable(int_type);
        auto argument = callable->create_value_argument(int_type);
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto sum = builder.call(int_type, ArithmeticOp::BINARY_ADD, {argument, one});
        auto return_instruction = builder.return_(sum);

        attach_all_metadata(module, "module");
        attach_all_metadata(*one, "constant");
        attach_all_metadata(*undefined, "undefined");
        attach_all_metadata(*thread_id, "special");
        attach_all_metadata(*callable, "function");
        attach_all_metadata(*argument, "argument");
        attach_all_metadata(*body, "block");
        attach_all_metadata(*sum, "arithmetic");
        attach_all_metadata(*return_instruction, "return_inst");

        auto encoded = xir_to_interchange_text(&module);
        expect(encoded.succeeded());
        if (!encoded.succeeded()) { return; }
        expect(encoded.text.find("md name \"function_name\"") != luisa::string::npos);
        expect(encoded.text.find("md signature_constraint") != luisa::string::npos);
        expect(encoded.text.find("quoted\\\"path\\\\segment\\nfile.xir") != luisa::string::npos);

        auto decoded = xir_from_interchange_text(encoded.text);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        expect_all_metadata(*decoded.module, "module");

        auto decoded_constant = decoded.module->constant_list().front();
        auto decoded_undefined = decoded.module->undefined_list().front();
        auto decoded_special = decoded.module->special_register_list().front();
        auto decoded_function = decoded.module->function_list().front();
        auto decoded_argument = decoded_function->arguments().front();
        auto decoded_body = decoded_function->definition()->body_block();
        expect_all_metadata(*decoded_constant, "constant");
        expect_all_metadata(*decoded_undefined, "undefined");
        expect_all_metadata(*decoded_special, "special");
        expect_all_metadata(*decoded_function, "function");
        expect_all_metadata(*decoded_argument, "argument");
        expect_all_metadata(*decoded_body, "block");
        expect(decoded_function->name().has_value());
        expect(decoded_function->name().value() == "function_name");

        size_t instruction_index = 0u;
        for (auto instruction : decoded_body->instructions()) {
            if (instruction_index == 0u) {
                expect(instruction->isa<ArithmeticInst>());
                expect_all_metadata(*instruction, "arithmetic");
            } else if (instruction_index == 1u) {
                expect(instruction->isa<ReturnInst>());
                expect_all_metadata(*instruction, "return_inst");
            }
            instruction_index++;
        }
        expect(instruction_index == 2u);

        auto canonical_again = xir_to_interchange_text(decoded.module.get());
        expect(canonical_again.succeeded());
        expect(canonical_again.text == encoded.text);

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(decoded_bitcode.succeeded());
        if (!decoded_bitcode.succeeded()) { return; }
        auto bitcode_text = xir_to_interchange_text(decoded_bitcode.module.get());
        expect(bitcode_text.succeeded());
        expect(bitcode_text.text == encoded.text);
    };

    "xir_interchange_reg2mem_spill_metadata_round_trip"_test = [] {
        Module module;
        auto callable = module.create_callable(nullptr);
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto phi_spill = builder.alloca_local(Type::of<int32_t>());
        phi_spill->create_metadata<Reg2MemSpillMD>()->set_kind(
            Reg2MemSpillKind::PHI);
        auto cross_block_spill = builder.alloca_local(Type::of<float>());
        cross_block_spill->create_metadata<Reg2MemSpillMD>()->set_kind(
            Reg2MemSpillKind::CROSS_BLOCK);
        builder.return_void();

        auto expect_spills = [](const Module *decoded) noexcept {
            luisa::vector<Reg2MemSpillKind> kinds;
            auto function = decoded->function_list().front();
            for (auto block : function->basic_blocks()) {
                for (auto instruction : block->instructions()) {
                    if (auto metadata =
                            instruction->find_metadata<Reg2MemSpillMD>()) {
                        kinds.emplace_back(metadata->kind());
                    }
                }
            }
            expect(kinds.size() == 2u);
            if (kinds.size() == 2u) {
                expect(kinds[0] == Reg2MemSpillKind::PHI);
                expect(kinds[1] == Reg2MemSpillKind::CROSS_BLOCK);
            }
        };

        auto encoded = xir_to_interchange_text(&module);
        expect(encoded.succeeded());
        if (!encoded.succeeded()) { return; }
        expect(encoded.text.find("md reg2mem_spill phi") !=
               luisa::string::npos);
        expect(encoded.text.find("md reg2mem_spill cross_block") !=
               luisa::string::npos);

        auto decoded_text = xir_from_interchange_text(encoded.text);
        expect(decoded_text.succeeded());
        if (!decoded_text.succeeded()) { return; }
        expect_spills(decoded_text.module.get());
        auto canonical_text =
            xir_to_interchange_text(decoded_text.module.get());
        expect(canonical_text.succeeded());
        expect(canonical_text.text == encoded.text);

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        if (!bitcode.succeeded()) { return; }
        auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(decoded_bitcode.succeeded());
        if (!decoded_bitcode.succeeded()) { return; }
        expect_spills(decoded_bitcode.module.get());
        auto bitcode_text =
            xir_to_interchange_text(decoded_bitcode.module.get());
        expect(bitcode_text.succeeded());
        expect(bitcode_text.text == encoded.text);
    };

    "xir_interchange_malformed_metadata_rejected"_test = [] {
        auto unknown_bits = xir_from_interchange_text(R"(
xir.text 1 module {
metadata 1 md curve_basis 16
globals 0 functions 0
})");
        expect(!unknown_bits.succeeded());
        expect(!unknown_bits.diagnostics.empty());

        auto invalid_name = xir_from_interchange_text(R"(
xir.text 1 module {
metadata 1 md name "9invalid"
globals 0 functions 0
})");
        expect(!invalid_name.succeeded());
        expect(!invalid_name.diagnostics.empty());

        auto unknown_tag = xir_from_interchange_text(R"(
xir.text 1 module {
metadata 1 md mystery "payload"
globals 0 functions 0
})");
        expect(!unknown_tag.succeeded());
        expect(!unknown_tag.diagnostics.empty());

        auto unknown_spill_kind = xir_from_interchange_text(R"(
xir.text 1 module {
metadata 1 md reg2mem_spill mystery
globals 0 functions 0
})");
        expect(!unknown_spill_kind.succeeded());
        expect(!unknown_spill_kind.diagnostics.empty());
    };
}

void reg_canonical_constant_payloads() {
    "xir_interchange_canonical_packed_little_endian_constants"_test = [] {
        Module module;
        auto struct_type = Type::from("struct<16,ubyte,uint,float>");
        auto fp8_e4m3_type = Type::from("float8e4m3");
        auto fp8_e5m2_type = Type::from("float8e5m2");
        expect(struct_type != nullptr);
        expect(fp8_e4m3_type != nullptr);
        expect(fp8_e5m2_type != nullptr);
        if (struct_type == nullptr || fp8_e4m3_type == nullptr || fp8_e5m2_type == nullptr) { return; }

        luisa::vector<std::byte> native(struct_type->size(), std::byte{0xcc});
        auto byte_value = uint8_t{0x7au};
        auto uint_value = uint32_t{0x12345678u};
        auto float_bits = uint32_t{0x7fc12345u};
        std::memcpy(native.data(), &byte_value, sizeof(byte_value));
        std::memcpy(native.data() + 4u, &uint_value, sizeof(uint_value));
        std::memcpy(native.data() + 8u, &float_bits, sizeof(float_bits));
        auto struct_constant = module.create_constant(struct_type, native.data());
        auto fp8_e4m3_bits = uint8_t{0xa5u};
        auto fp8_e5m2_bits = uint8_t{0x5au};
        auto fp8_e4m3_constant = module.create_constant(fp8_e4m3_type, &fp8_e4m3_bits);
        auto fp8_e5m2_constant = module.create_constant(fp8_e5m2_type, &fp8_e5m2_bits);
        expect(struct_constant != nullptr);
        expect(fp8_e4m3_constant != nullptr);
        expect(fp8_e5m2_constant != nullptr);

        auto text = xir_to_interchange_text(&module);
        expect(text.succeeded());
        if (!text.succeeded()) { return; }
        expect(text.text.find("\"7a785634124523c17f\"") != luisa::string::npos)
            << "struct padding must not appear on the wire";
        expect(text.text.find("\"a5\"") != luisa::string::npos);
        expect(text.text.find("\"5a\"") != luisa::string::npos);

        auto decoded_text = xir_from_interchange_text(text.text);
        expect(decoded_text.succeeded());
        if (!decoded_text.succeeded()) { return; }
        auto text_again = xir_to_interchange_text(decoded_text.module.get());
        expect(text_again.succeeded());
        expect(text_again.text == text.text);

        auto bitcode = xir_to_bitcode(&module);
        expect(bitcode.succeeded());
        if (!bitcode.succeeded()) { return; }
        auto decoded_bitcode = xir_from_bitcode(bitcode.bitcode);
        expect(decoded_bitcode.succeeded());
        if (!decoded_bitcode.succeeded()) { return; }
        auto bitcode_text = xir_to_interchange_text(decoded_bitcode.module.get());
        expect(bitcode_text.succeeded());
        expect(bitcode_text.text == text.text);

        auto malformed_bool = xir_from_interchange_text(R"(
xir.text 1 module {
globals 1
constant 0 "bool" "02"
functions 0
})");
        expect(!malformed_bool.succeeded());
        expect(!malformed_bool.diagnostics.empty());
    };
}

void reg_compact_binary_codec() {
    "xir_interchange_compact_binary_golden_and_determinism"_test = [] {
        Module module;
        auto first = xir_to_bitcode(&module);
        auto second = xir_to_bitcode(&module);
        expect(first.succeeded());
        expect(second.succeeded());
        expect(static_cast<bool>(first.bitcode == second.bitcode));
        constexpr std::array golden{
            std::byte{0x4c}, std::byte{0x58}, std::byte{0x49}, std::byte{0x52},
            std::byte{0x42}, std::byte{0x43}, std::byte{0x00}, std::byte{0x01},
            std::byte{0x02}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x04}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
            std::byte{0xf5}, std::byte{0x13}, std::byte{0xce}, std::byte{0x9d},
            std::byte{0x7f}, std::byte{0x76}, std::byte{0x25}, std::byte{0x4d},
            std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}};
        expect(first.bitcode.size() == golden.size());
        if (first.bitcode.size() == golden.size()) {
            expect(std::equal(first.bitcode.begin(), first.bitcode.end(), golden.begin()));
        }
        auto decoded = xir_from_bitcode(first.bitcode);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        auto encoded_again = xir_to_bitcode(decoded.module.get());
        expect(encoded_again.succeeded());
        expect(static_cast<bool>(encoded_again.bitcode == first.bitcode));

        Module legacy_module;
        auto int_type = Type::of<int32_t>();
        auto one = legacy_module.create_constant_one(int_type);
        auto callable = legacy_module.create_callable(int_type);
        auto argument = callable->create_value_argument(int_type);
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto sum = builder.call(
            int_type, ArithmeticOp::BINARY_ADD, {argument, one});
        builder.return_(sum);
        attach_all_metadata(legacy_module, "legacy_module");
        attach_all_metadata(*callable, "legacy_function");
        attach_all_metadata(*sum, "legacy_sum");

        auto legacy_text = xir_to_interchange_text(&legacy_module);
        expect(legacy_text.succeeded());
        if (!legacy_text.succeeded()) { return; }
        auto legacy_payload = luisa::span{
            reinterpret_cast<const std::byte *>(legacy_text.text.data()),
            legacy_text.text.size()};
        auto decoded_legacy = xir_from_bitcode(make_test_bitcode(legacy_payload, 1u));
        expect(decoded_legacy.succeeded());
        expect(decoded_legacy.module != nullptr);
        if (!decoded_legacy.succeeded()) { return; }
        expect(decoded_legacy.module->constant_list().count_size() == 1u);
        expect(decoded_legacy.module->function_list().count_size() == 1u);
        auto decoded_legacy_text = xir_to_interchange_text(decoded_legacy.module.get());
        expect(decoded_legacy_text.succeeded());
        expect(decoded_legacy_text.text == legacy_text.text);
        expect_all_metadata(*decoded_legacy.module, "legacy_module");
        auto *decoded_legacy_function = decoded_legacy.module->function_list().front();
        expect_all_metadata(*decoded_legacy_function, "legacy_function");
        auto *decoded_legacy_sum =
            decoded_legacy_function->definition()->body_block()->instructions().front();
        expect(decoded_legacy_sum->isa<ArithmeticInst>());
        expect_all_metadata(*decoded_legacy_sum, "legacy_sum");
    };

    "xir_interchange_compact_binary_nonempty_golden"_test = [] {
        Module module;
        auto value = uint32_t{0x01020304u};
        auto constant = module.create_constant(Type::of<uint32_t>(), &value);
        auto callable = module.create_callable(Type::of<uint32_t>());
        auto body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_(constant);
        auto encoded = xir_to_bitcode(&module);
        expect(encoded.succeeded());
        constexpr std::array golden{
            std::byte{0x4c}, std::byte{0x58}, std::byte{0x49}, std::byte{0x52},
            std::byte{0x42}, std::byte{0x43}, std::byte{0x00}, std::byte{0x01},
            std::byte{0x02}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x3a}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x6b}, std::byte{0xf6}, std::byte{0x9c}, std::byte{0xd2},
            std::byte{0x4a}, std::byte{0xc3}, std::byte{0xb7}, std::byte{0x0e},
            std::byte{0x04}, std::byte{0x02}, std::byte{0x2d}, std::byte{0x31},
            std::byte{0x08}, std::byte{0x63}, std::byte{0x61}, std::byte{0x6c},
            std::byte{0x6c}, std::byte{0x61}, std::byte{0x62}, std::byte{0x6c},
            std::byte{0x65}, std::byte{0x04}, std::byte{0x75}, std::byte{0x69},
            std::byte{0x6e}, std::byte{0x74}, std::byte{0x04}, std::byte{0x76},
            std::byte{0x6f}, std::byte{0x69}, std::byte{0x64}, std::byte{0x00},
            std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0x02},
            std::byte{0x04}, std::byte{0x04}, std::byte{0x03}, std::byte{0x02},
            std::byte{0x01}, std::byte{0x00}, std::byte{0x01}, std::byte{0x01},
            std::byte{0x01}, std::byte{0x02}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x01},
            std::byte{0x02}, std::byte{0x00}, std::byte{0x04}, std::byte{0x01},
            std::byte{0x03}, std::byte{0x02}, std::byte{0x0b}, std::byte{0x03},
            std::byte{0x00}, std::byte{0x01}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x00}, std::byte{0x00}};
        expect(encoded.bitcode.size() == golden.size());
        if (encoded.bitcode.size() == golden.size()) {
            expect(std::equal(encoded.bitcode.begin(), encoded.bitcode.end(), golden.begin()));
        }
        auto decoded = xir_from_bitcode(encoded.bitcode);
        expect(decoded.succeeded());
        if (!decoded.succeeded()) { return; }
        expect(decoded.module->constant_list().count_size() == 1u);
        expect(decoded.module->function_list().count_size() == 1u);
        auto text = xir_to_interchange_text(decoded.module.get());
        expect(text.succeeded());
        expect(text.text.find(R"(constant 0 "uint" "04030201")") != luisa::string::npos);
        expect(text.text.find("return \"void\" -1 1 0") != luisa::string::npos);
        auto encoded_again = xir_to_bitcode(decoded.module.get());
        expect(encoded_again.succeeded());
        expect(static_cast<bool>(encoded_again.bitcode == encoded.bitcode));
    };

    "xir_interchange_compact_binary_truncation_and_corruption"_test = [] {
        Module module;
        auto encoded = xir_to_bitcode(&module);
        expect(encoded.succeeded());
        if (!encoded.succeeded()) { return; }
        for (auto size = 0u; size < encoded.bitcode.size(); size++) {
            auto truncated = xir_from_bitcode(luisa::span{encoded.bitcode}.first(size));
            expect(!truncated.succeeded());
            expect(!truncated.diagnostics.empty());
        }

        const std::array nonminimal_payload{
            std::byte{0x80}, std::byte{0x00},
            std::byte{0x00}, std::byte{0x00}, std::byte{0x00}};
        auto nonminimal = xir_from_bitcode(make_test_bitcode(nonminimal_payload));
        expect(!nonminimal.succeeded());
        expect(!nonminimal.diagnostics.empty());
        if (!nonminimal.diagnostics.empty()) { expect(nonminimal.diagnostics.front().offset == 34u); }

        const std::array unsorted_strings_payload{
            std::byte{0x02},
            std::byte{0x01}, std::byte{'b'},
            std::byte{0x01}, std::byte{'a'},
            std::byte{0x00}, std::byte{0x00}, std::byte{0x00}};
        auto unsorted = xir_from_bitcode(make_test_bitcode(unsorted_strings_payload));
        expect(!unsorted.succeeded());
        expect(!unsorted.diagnostics.empty());
        if (!unsorted.diagnostics.empty()) { expect(unsorted.diagnostics.front().offset == 37u); }

        const std::array trailing_payload{
            std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
            std::byte{0x00}, std::byte{0x00}};
        auto trailing = xir_from_bitcode(make_test_bitcode(trailing_payload));
        expect(!trailing.succeeded());
        expect(!trailing.diagnostics.empty());
        if (!trailing.diagnostics.empty()) { expect(trailing.diagnostics.front().offset == 36u); }

        luisa::vector<std::byte> invalid_string_index_payload;
        test_append_uleb(invalid_string_index_payload, 1u);
        test_append_uleb(invalid_string_index_payload, 9u);
        constexpr auto thread_id = luisa::string_view{"thread_id"};
        auto thread_id_bytes = reinterpret_cast<const std::byte *>(thread_id.data());
        invalid_string_index_payload.insert(
            invalid_string_index_payload.end(), thread_id_bytes,
            thread_id_bytes + thread_id.size());
        test_append_uleb(invalid_string_index_payload, 0u);// module metadata
        test_append_uleb(invalid_string_index_payload, 1u);// globals
        test_append_uleb(invalid_string_index_payload, 2u);// special
        test_append_uleb(invalid_string_index_payload, 0u);// id
        test_append_uleb(invalid_string_index_payload, 1u);// invalid string index
        auto invalid_string_index = xir_from_bitcode(
            make_test_bitcode(invalid_string_index_payload));
        expect(!invalid_string_index.succeeded());
        expect(!invalid_string_index.diagnostics.empty());

        luisa::vector<std::byte> oversized_count_payload;
        test_append_uleb(oversized_count_payload, (1u << 20u) + 1u);
        auto oversized_count = xir_from_bitcode(make_test_bitcode(oversized_count_payload));
        expect(!oversized_count.succeeded());
        expect(!oversized_count.diagnostics.empty());
        if (!oversized_count.diagnostics.empty()) {
            expect(oversized_count.diagnostics.front().message ==
                   "Record count exceeds the supported limit.");
        }
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_empty_module_round_trip();
    reg_malformed_text();
    reg_malformed_bitcode();
    reg_semantic_module_round_trip();
    reg_bindless_access_round_trip();
    reg_unsupported_instruction_fails_closed();
    reg_symbolic_op_tokens_and_compatibility();
    reg_vulkan_priority_instruction_round_trip();
    reg_vulkan_priority_instruction_validation();
    reg_strict_atomic_validation();
    reg_strict_operand_and_resource_validation();
    reg_interchange_safety_validation();
    reg_structured_control_flow_round_trip();
    reg_quoted_escape_and_reference_validation();
    reg_reference_validation();
    reg_instruction_type_validation();
    reg_remaining_misc_instruction_round_trip();
    reg_autodiff_and_outline_round_trip();
    reg_ray_query_instruction_round_trip();
    reg_metadata_round_trip();
    reg_canonical_constant_payloads();
    reg_compact_binary_codec();
    return 0;
}

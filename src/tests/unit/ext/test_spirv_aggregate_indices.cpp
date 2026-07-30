// Test for XIR aggregate-index lowering at the SPIR-V boundary.
// This test covers:
// - lossless i8/i16/i64 structure-member planning
// - preservation of dynamic array indices
// - validator-backed GEP, INSERT/EXTRACT, and typed atomic lowering
// - explicit negative, dynamic-structure, and out-of-range rejection

#include "ut/ut.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>

#include "spirv_codegen/aggregate_index.h"
#include "spirv_codegen/dialect.h"
#include "spirv_codegen/entry.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

struct SpirvAggregateIndexLeaf {
    std::array<uint32_t, 4u> values;
};

struct SpirvAggregateIndexRoot {
    uint32_t prefix;
    SpirvAggregateIndexLeaf leaf;
};

LUISA_STRUCT(SpirvAggregateIndexLeaf, values) {};
LUISA_STRUCT(SpirvAggregateIndexRoot, prefix, leaf) {};

namespace {

template<typename T>
[[nodiscard]] luisa::compute::xir::Constant *make_constant(
    Module &module, T value) noexcept {
    return module.create_constant(Type::of<T>(), &value);
}

[[nodiscard]] bool has_diagnostic(
    const lc::spirv::SpirvXIRDialectValidationResult &result,
    luisa::string_view needle) noexcept {
    for (auto &&diagnostic : result.diagnostics) {
        if (diagnostic.message.find(needle) !=
            luisa::string_view::npos) {
            return true;
        }
    }
    return false;
}

struct NestedAggregateAccessFacts {
    bool nested_atomic_chain_found{false};
    bool atomic_value_is_one{false};
    bool nested_function_load_uses_same_dynamic_index{false};
    bool dynamic_insert_extract_round_trip_found{false};
    bool dynamic_insert_extract_reaches_storage_output{false};
};

struct IntegerTypeFacts {
    uint32_t width;
    bool is_signed;
};

struct PointerTypeFacts {
    spv::StorageClass storage;
    uint32_t pointee_type;
};

struct ConstantFacts {
    uint32_t type;
    uint32_t value;
    bool is_ordinary;
};

struct AccessChainInstruction {
    uint32_t result_type;
    uint32_t base;
    std::vector<uint32_t> indices;
};

struct LoadInstruction {
    uint32_t result_type;
    uint32_t result;
    uint32_t pointer;
};

struct StoreInstruction {
    uint32_t pointer;
    uint32_t object;
};

struct IAddInstruction {
    uint32_t result_type;
    uint32_t result;
    uint32_t lhs;
    uint32_t rhs;
};

struct AtomicIAddInstruction {
    uint32_t result_type;
    uint32_t pointer;
    uint32_t value;
};

[[nodiscard]] NestedAggregateAccessFacts inspect_nested_aggregate_accesses(
    const std::vector<uint32_t> &words) noexcept {
    std::unordered_map<uint32_t, IntegerTypeFacts> integer_types;
    std::unordered_map<uint32_t, PointerTypeFacts> pointer_types;
    std::unordered_map<uint32_t, ConstantFacts> constants;
    std::unordered_map<uint32_t, AccessChainInstruction> access_chains;
    std::vector<LoadInstruction> loads;
    std::vector<StoreInstruction> stores;
    std::vector<IAddInstruction> iadds;
    std::vector<AtomicIAddInstruction> atomic_iadds;
    for (auto offset = 5u; offset < words.size();) {
        auto word_count = words[offset] >> 16u;
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (word_count == 0u || offset + word_count > words.size()) {
            break;
        }
        if (op == spv::Op::OpTypeInt && word_count >= 4u) {
            integer_types.emplace(
                words[offset + 1u],
                IntegerTypeFacts{
                    .width = words[offset + 2u],
                    .is_signed = words[offset + 3u] != 0u});
        } else if (op == spv::Op::OpTypePointer &&
                   word_count >= 4u) {
            pointer_types.emplace(
                words[offset + 1u],
                PointerTypeFacts{
                    .storage = static_cast<spv::StorageClass>(
                        words[offset + 2u]),
                    .pointee_type = words[offset + 3u]});
        } else if ((op == spv::Op::OpConstant ||
                    op == spv::Op::OpSpecConstant) &&
                   word_count >= 4u) {
            constants.emplace(
                words[offset + 2u],
                ConstantFacts{
                    .type = words[offset + 1u],
                    .value = words[offset + 3u],
                    .is_ordinary = op == spv::Op::OpConstant});
        } else if ((op == spv::Op::OpAccessChain ||
                    op == spv::Op::OpInBoundsAccessChain) &&
                   word_count >= 5u) {
            AccessChainInstruction access_chain{
                .result_type = words[offset + 1u],
                .base = words[offset + 3u]};
            access_chain.indices.reserve(word_count - 4u);
            for (auto operand = 4u; operand < word_count; ++operand) {
                access_chain.indices.emplace_back(words[offset + operand]);
            }
            access_chains.emplace(
                words[offset + 2u], std::move(access_chain));
        } else if (op == spv::Op::OpLoad && word_count >= 4u) {
            loads.emplace_back(LoadInstruction{
                .result_type = words[offset + 1u],
                .result = words[offset + 2u],
                .pointer = words[offset + 3u]});
        } else if (op == spv::Op::OpStore && word_count >= 3u) {
            stores.emplace_back(StoreInstruction{
                .pointer = words[offset + 1u],
                .object = words[offset + 2u]});
        } else if (op == spv::Op::OpIAdd && word_count == 5u) {
            iadds.emplace_back(IAddInstruction{
                .result_type = words[offset + 1u],
                .result = words[offset + 2u],
                .lhs = words[offset + 3u],
                .rhs = words[offset + 4u]});
        } else if (op == spv::Op::OpAtomicIAdd) {
            if (word_count >= 7u) {
                atomic_iadds.emplace_back(AtomicIAddInstruction{
                    .result_type = words[offset + 1u],
                    .pointer = words[offset + 3u],
                    .value = words[offset + 6u]});
            }
        }
        offset += word_count;
    }

    auto is_u32 = [&](uint32_t type) noexcept {
        auto iter = integer_types.find(type);
        return iter != integer_types.end() &&
               iter->second.width == 32u &&
               !iter->second.is_signed;
    };
    auto is_u32_constant = [&](uint32_t id,
                               uint32_t expected) noexcept {
        auto iter = constants.find(id);
        return iter != constants.end() &&
               iter->second.is_ordinary &&
               is_u32(iter->second.type) &&
               iter->second.value == expected;
    };
    auto pointer_has_shape = [&](uint32_t type,
                                 spv::StorageClass storage) noexcept {
        auto iter = pointer_types.find(type);
        return iter != pointer_types.end() &&
               iter->second.storage == storage &&
               is_u32(iter->second.pointee_type);
    };
    auto is_zero_based_u32_index = [&](uint32_t id) noexcept {
        if (is_u32_constant(id, 0u)) { return true; }
        // Direct buffer descriptors may begin at a nonzero byte offset. The
        // typed-buffer lowering therefore adds the runtime element bias to
        // the fixture's logical element index before OpAccessChain. Keep the
        // aggregate-index check independent of that descriptor ABI while
        // still proving that the logical zero reached the address expression.
        for (auto &&add : iadds) {
            auto lhs_is_zero = is_u32_constant(add.lhs, 0u);
            auto rhs_is_zero = is_u32_constant(add.rhs, 0u);
            if (add.result == id && is_u32(add.result_type) &&
                ((lhs_is_zero && !constants.contains(add.rhs)) ||
                 (rhs_is_zero && !constants.contains(add.lhs)))) {
                return true;
            }
        }
        return false;
    };

    NestedAggregateAccessFacts facts;
    uint32_t intended_dynamic_index = 0u;
    for (auto &&atomic : atomic_iadds) {
        auto chain_iter = access_chains.find(atomic.pointer);
        if (chain_iter == access_chains.end()) { continue; }
        auto &&chain = chain_iter->second;
        if (!is_u32(atomic.result_type) ||
            !pointer_has_shape(chain.result_type,
                               spv::StorageClass::StorageBuffer) ||
            chain.indices.size() != 5u ||
            !is_u32_constant(chain.indices[0u], 0u) ||
            !is_zero_based_u32_index(chain.indices[1u]) ||
            !is_u32_constant(chain.indices[2u], 1u) ||
            !is_u32_constant(chain.indices[3u], 0u) ||
            constants.contains(chain.indices[4u])) {
            continue;
        }
        facts.nested_atomic_chain_found = true;
        facts.atomic_value_is_one =
            is_u32_constant(atomic.value, 1u);
        intended_dynamic_index = chain.indices[4u];
        break;
    }

    if (intended_dynamic_index == 0u) { return facts; }
    for (auto &&load : loads) {
        auto chain_iter = access_chains.find(load.pointer);
        if (chain_iter == access_chains.end()) { continue; }
        auto &&chain = chain_iter->second;
        if (is_u32(load.result_type) &&
            pointer_has_shape(chain.result_type,
                              spv::StorageClass::Function) &&
            chain.indices.size() == 3u &&
            is_u32_constant(chain.indices[0u], 1u) &&
            is_u32_constant(chain.indices[1u], 0u) &&
            chain.indices[2u] == intended_dynamic_index) {
            facts.nested_function_load_uses_same_dynamic_index = true;
            break;
        }
    }

    auto is_nested_function_chain = [&](const AccessChainInstruction &chain) noexcept {
        return pointer_has_shape(chain.result_type,
                                 spv::StorageClass::Function) &&
               chain.indices.size() == 3u &&
               is_u32_constant(chain.indices[0u], 1u) &&
               is_u32_constant(chain.indices[1u], 0u) &&
               chain.indices[2u] == intended_dynamic_index;
    };
    auto stored_object = [&](uint32_t pointer,
                             auto predicate) noexcept -> uint32_t {
        for (auto &&store : stores) {
            if (store.pointer == pointer && predicate(store.object)) {
                return store.object;
            }
        }
        return 0u;
    };
    auto loaded_value = [&](uint32_t pointer,
                            auto predicate) noexcept -> uint32_t {
        for (auto &&load : loads) {
            if (load.pointer == pointer && predicate(load)) {
                return load.result;
            }
        }
        return 0u;
    };

    uint32_t round_trip_value = 0u;
    for (auto &&[insert_member_pointer, insert_chain] : access_chains) {
        if (!is_nested_function_chain(insert_chain) ||
            stored_object(insert_member_pointer, [&](uint32_t object) noexcept {
                return is_u32_constant(object, 1u);
            }) == 0u) {
            continue;
        }
        auto base_aggregate = stored_object(
            insert_chain.base, [](uint32_t) noexcept { return true; });
        auto inserted_aggregate = loaded_value(
            insert_chain.base, [&](const LoadInstruction &load) noexcept {
                return !is_u32(load.result_type);
            });
        if (base_aggregate == 0u || inserted_aggregate == 0u) {
            continue;
        }
        for (auto &&[extract_member_pointer, extract_chain] : access_chains) {
            if (extract_chain.base == insert_chain.base ||
                !is_nested_function_chain(extract_chain) ||
                stored_object(
                    extract_chain.base,
                    [&](uint32_t object) noexcept {
                        return object == inserted_aggregate;
                    }) == 0u) {
                continue;
            }
            auto extracted = loaded_value(
                extract_member_pointer,
                [&](const LoadInstruction &load) noexcept {
                    return is_u32(load.result_type);
                });
            if (extracted != 0u) {
                facts.dynamic_insert_extract_round_trip_found = true;
                round_trip_value = extracted;
                break;
            }
        }
        if (round_trip_value != 0u) { break; }
    }

    if (round_trip_value != 0u) {
        std::unordered_map<uint32_t, bool> depends_on_round_trip;
        depends_on_round_trip.emplace(round_trip_value, true);
        auto changed = true;
        while (changed) {
            changed = false;
            for (auto &&add : iadds) {
                if (!is_u32(add.result_type) ||
                    depends_on_round_trip.contains(add.result)) {
                    continue;
                }
                if (depends_on_round_trip.contains(add.lhs) ||
                    depends_on_round_trip.contains(add.rhs)) {
                    depends_on_round_trip.emplace(add.result, true);
                    changed = true;
                }
            }
        }
        for (auto &&store : stores) {
            if (!depends_on_round_trip.contains(store.object)) { continue; }
            auto chain = access_chains.find(store.pointer);
            if (chain != access_chains.end() &&
                pointer_has_shape(chain->second.result_type,
                                  spv::StorageClass::StorageBuffer)) {
                facts.dynamic_insert_extract_reaches_storage_output = true;
                break;
            }
        }
    }
    return facts;
}

void set_environment_variable(const char *name,
                              const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

class ScopedEnvironmentVariable {
private:
    const char *_name;
    std::optional<std::string> _previous;

public:
    ScopedEnvironmentVariable(const char *name,
                              const char *value) noexcept
        : _name{name} {
        if (auto previous = std::getenv(name)) {
            _previous.emplace(previous);
        }
        set_environment_variable(name, value);
    }
    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            _name, _previous ? _previous->c_str() : nullptr);
    }
    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(
        const ScopedEnvironmentVariable &) = delete;
};

[[nodiscard]] lc::spirv::SpirvResult compile_exact_xir(
    luisa::compute::Function kernel, const Module *module) {
    ScopedEnvironmentVariable disable_spirv_optimization{
        "LUISA_SPIRV_OPT_LEVEL", "0"};
    ScopedEnvironmentVariable clear_spirv_pass_override{
        "LUISA_SPIRV_OPT_PASSES", nullptr};
    return lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
        kernel, module, ShaderOption{.enable_cache = false});
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_aggregate_index_plan_is_lossless_and_preserves_dynamic_sequences"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *dynamic_index = kernel->create_value_argument(
            Type::of<uint32_t>());
        auto *field_i8 = make_constant<int8_t>(module, 1);
        auto *field_i16 = make_constant<int16_t>(module, 0);
        auto *field_i64 = make_constant<int64_t>(module, 0);
        auto *root_type = Type::of<SpirvAggregateIndexRoot>();

        for (auto *leaf_field : std::array<const Value *, 2u>{
                 field_i16, field_i64}) {
            std::array<const Value *, 3u> indices{
                field_i8, leaf_field, dynamic_index};
            auto plan = lc::spirv::plan_spirv_aggregate_indices(
                root_type, luisa::span{indices});
            expect(plan.succeeded());
            if (!plan) { continue; }
            expect(eq(plan.steps.size(), 3u));
            expect(plan.steps[0].kind ==
                   lc::spirv::SpirvAggregateIndexKind::STRUCTURE_MEMBER);
            expect(plan.steps[1].kind ==
                   lc::spirv::SpirvAggregateIndexKind::STRUCTURE_MEMBER);
            expect(plan.steps[2].kind ==
                   lc::spirv::SpirvAggregateIndexKind::SEQUENCE_ELEMENT);
            expect(plan.steps[0].constant_index == 1u);
            expect(plan.steps[1].constant_index == 0u);
            expect(!plan.steps[2].is_constant);
            expect(plan.steps[2].index == dynamic_index);
            expect(plan.indexed_type == Type::of<uint32_t>());
        }
    };

    "spirv_aggregate_index_plan_rejects_invalid_structure_boundaries"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *dynamic_index = kernel->create_value_argument(
            Type::of<uint32_t>());
        auto *root_type = Type::of<SpirvAggregateIndexRoot>();

        auto *negative = make_constant<int64_t>(module, -1);
        std::array<const Value *, 1u> negative_indices{negative};
        auto negative_plan = lc::spirv::plan_spirv_aggregate_indices(
            root_type, luisa::span{negative_indices});
        expect(!negative_plan.succeeded());
        expect(negative_plan.diagnostic.find("negative") !=
               luisa::string_view::npos);

        auto *too_wide = make_constant<uint64_t>(
            module, uint64_t{1u} << 40u);
        std::array<const Value *, 1u> too_wide_indices{too_wide};
        auto too_wide_plan = lc::spirv::plan_spirv_aggregate_indices(
            root_type, luisa::span{too_wide_indices});
        expect(!too_wide_plan.succeeded());
        expect(too_wide_plan.diagnostic.find("32-bit") !=
               luisa::string_view::npos);

        auto *out_of_range = make_constant<int16_t>(module, 2);
        std::array<const Value *, 1u> out_of_range_indices{
            out_of_range};
        auto out_of_range_plan =
            lc::spirv::plan_spirv_aggregate_indices(
                root_type, luisa::span{out_of_range_indices});
        expect(!out_of_range_plan.succeeded());
        expect(out_of_range_plan.diagnostic.find("out of bounds") !=
               luisa::string_view::npos);

        std::array<const Value *, 1u> dynamic_structure_indices{
            dynamic_index};
        auto dynamic_structure_plan =
            lc::spirv::plan_spirv_aggregate_indices(
                root_type, luisa::span{dynamic_structure_indices});
        expect(!dynamic_structure_plan.succeeded());
        expect(dynamic_structure_plan.diagnostic.find(
                   "compile-time integer constant") !=
               luisa::string_view::npos);
    };

    "spirv_aggregate_index_rejections_are_dialect_diagnostics"_test = [] {
        auto run = [](auto bad_index) noexcept {
            Module module;
            auto *kernel = module.create_kernel();
            auto *dynamic = kernel->create_value_argument(
                Type::of<uint32_t>());
            auto *root_type = Type::of<SpirvAggregateIndexRoot>();
            luisa::compute::xir::Value *root =
                module.create_constant_zero(root_type);
            auto *body = kernel->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            luisa::compute::xir::Value *index =
                bad_index(module, dynamic);
            std::array<luisa::compute::xir::Value *, 2u> operands{
                root, index};
            builder.call(Type::of<SpirvAggregateIndexLeaf>(),
                         ArithmeticOp::EXTRACT,
                         luisa::span<luisa::compute::xir::Value *const>{
                             operands});
            builder.return_void();
            auto result =
                lc::spirv::validate_spirv_xir_codegen_dialect(
                    &module);
            luisa::vector<luisa::string> messages;
            messages.reserve(result.diagnostics.size());
            for (auto &&diagnostic : result.diagnostics) {
                messages.emplace_back(diagnostic.message);
            }
            return messages;
        };
        auto contains_diagnostic =
            [](const luisa::vector<luisa::string> &messages,
               luisa::string_view needle) noexcept {
                for (auto &&message : messages) {
                    if (message.find(needle) !=
                        luisa::string_view::npos) {
                        return true;
                    }
                }
                return false;
            };

        auto negative = run([](Module &module, Value *) noexcept {
            return make_constant<int8_t>(module, -1);
        });
        expect(!negative.empty());
        expect(contains_diagnostic(negative, "negative"));

        auto out_of_range = run([](Module &module, Value *) noexcept {
            return make_constant<uint64_t>(
                module, uint64_t{1u} << 40u);
        });
        expect(!out_of_range.empty());
        expect(contains_diagnostic(out_of_range, "32-bit"));

        auto dynamic = run([](Module &, Value *value) noexcept {
            return value;
        });
        expect(!dynamic.empty());
        expect(contains_diagnostic(
            dynamic, "compile-time integer constant"));

        Module atomic_module;
        auto *atomic_kernel = atomic_module.create_kernel();
        auto *atomic_buffer = atomic_kernel->create_resource_argument(
            Type::buffer(Type::of<SpirvAggregateIndexRoot>()));
        auto *dynamic_member = atomic_kernel->create_value_argument(
            Type::of<uint32_t>());
        auto *buffer_element = atomic_module.create_constant_zero(
            Type::of<uint32_t>());
        auto *increment = atomic_module.create_constant_one(
            Type::of<uint32_t>());
        auto *atomic_body = atomic_kernel->create_body_block();
        XIRBuilder atomic_builder;
        atomic_builder.set_insertion_point(atomic_body);
        std::array<Value *, 2u> bad_atomic_indices{
            buffer_element, dynamic_member};
        atomic_builder.atomic_fetch_add(
            Type::of<uint32_t>(), atomic_buffer,
            luisa::span<Value *const>{bad_atomic_indices}, increment);
        atomic_builder.return_void();
        auto atomic = lc::spirv::validate_spirv_xir_codegen_dialect(
            &atomic_module);
        expect(!atomic.succeeded());
        expect(has_diagnostic(
            atomic, "rejected atomic aggregate indices"));
        expect(has_diagnostic(
            atomic, "compile-time integer constant"));
    };

    "spirv_aggregate_indices_validate_for_gep_insert_extract_and_atomic"_test = [] {
        auto *root_type = Type::of<SpirvAggregateIndexRoot>();
        auto *uint_type = Type::of<uint32_t>();

        Module module;
        auto *kernel = module.create_kernel();
        luisa::compute::xir::Value *values =
            kernel->create_resource_argument(
                Type::buffer(root_type));
        luisa::compute::xir::Value *output =
            kernel->create_resource_argument(
                Type::buffer(uint_type));
        luisa::compute::xir::Value *dynamic_index =
            kernel->create_value_argument(uint_type);
        auto *body = kernel->create_body_block();

        luisa::compute::xir::Value *zero_root =
            module.create_constant_zero(root_type);
        luisa::compute::xir::Value *one =
            module.create_constant_one(uint_type);
        luisa::compute::xir::Value *zero_u32 =
            module.create_constant_zero(uint_type);
        luisa::compute::xir::Value *field_one_i8 =
            make_constant<int8_t>(module, 1);
        luisa::compute::xir::Value *field_zero_i8 =
            make_constant<int8_t>(module, 0);
        luisa::compute::xir::Value *field_one_i16 =
            make_constant<int16_t>(module, 1);
        luisa::compute::xir::Value *field_zero_i16 =
            make_constant<int16_t>(module, 0);
        luisa::compute::xir::Value *field_one_i64 =
            make_constant<int64_t>(module, 1);
        luisa::compute::xir::Value *field_zero_i64 =
            make_constant<int64_t>(module, 0);

        XIRBuilder builder;
        builder.set_insertion_point(body);

        luisa::compute::xir::Value *local =
            builder.alloca_local(root_type);
        builder.store(local, zero_root);
        std::array<luisa::compute::xir::Value *, 3u> gep_indices{
            field_one_i8, field_zero_i16, dynamic_index};
        luisa::compute::xir::Value *element_ptr = builder.gep(
            uint_type, local,
            luisa::span<luisa::compute::xir::Value *const>{
                gep_indices});
        luisa::compute::xir::Value *gep_value =
            builder.load(uint_type, element_ptr);

        std::array<luisa::compute::xir::Value *, 5u> insert_operands{
            zero_root, one, field_one_i64,
            field_zero_i8, dynamic_index};
        luisa::compute::xir::Value *inserted = builder.call(
            root_type, ArithmeticOp::INSERT,
            luisa::span<luisa::compute::xir::Value *const>{
                insert_operands});
        std::array<luisa::compute::xir::Value *, 4u>
            inserted_extract_operands{
                inserted, field_one_i16, field_zero_i64,
                dynamic_index};
        luisa::compute::xir::Value *inserted_value = builder.call(
            uint_type, ArithmeticOp::EXTRACT,
            luisa::span<luisa::compute::xir::Value *const>{
                inserted_extract_operands});
        std::array<luisa::compute::xir::Value *, 4u>
            extract_operands{
                zero_root, field_one_i8, field_zero_i16,
                dynamic_index};
        luisa::compute::xir::Value *extracted_value = builder.call(
            uint_type, ArithmeticOp::EXTRACT,
            luisa::span<luisa::compute::xir::Value *const>{
                extract_operands});

        std::array<luisa::compute::xir::Value *, 4u> atomic_indices{
            zero_u32, field_one_i64, field_zero_i16,
            dynamic_index};
        luisa::compute::xir::Value *atomic_value =
            builder.atomic_fetch_add(
                uint_type, values,
                luisa::span<luisa::compute::xir::Value *const>{
                    atomic_indices},
                one);

        std::array<luisa::compute::xir::Value *, 2u> add_operands{
            gep_value, inserted_value};
        luisa::compute::xir::Value *sum = builder.call(
            uint_type, ArithmeticOp::BINARY_ADD,
            luisa::span<luisa::compute::xir::Value *const>{
                add_operands});
        std::array<luisa::compute::xir::Value *, 2u>
            extracted_add_operands{sum, extracted_value};
        sum = builder.call(
            uint_type, ArithmeticOp::BINARY_ADD,
            luisa::span<luisa::compute::xir::Value *const>{
                extracted_add_operands});
        std::array<luisa::compute::xir::Value *, 2u>
            atomic_add_operands{sum, atomic_value};
        sum = builder.call(
            uint_type, ArithmeticOp::BINARY_ADD,
            luisa::span<luisa::compute::xir::Value *const>{
                atomic_add_operands});
        std::array<luisa::compute::xir::Value *, 3u> write_operands{
            output, zero_u32, sum};
        builder.call(
            ResourceWriteOp::BUFFER_WRITE,
            luisa::span<luisa::compute::xir::Value *const>{
                write_operands});
        builder.return_void();

        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(dialect.succeeded());

        Kernel1D ast_kernel = [](
                                  BufferVar<SpirvAggregateIndexRoot>,
                                  BufferUInt, UInt) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            ast_kernel.function()->function(), &module);

        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        expect(tools.Validate(compiled.spv_bin.data(),
                              compiled.spv_bin.size()))
            << "aggregate-index SPIR-V fixture must validate";
        expect(eq(compiled.required_target_features,
                  lc::spirv::SpirvTargetFeatureMask{0u}))
            << "canonicalized i8/i16/i64 structure indices must not leak "
               "source-width target-feature requirements";
        auto facts = inspect_nested_aggregate_accesses(compiled.spv_bin);
        expect(facts.nested_atomic_chain_found)
            << "the uint32 atomic pointer must be the typed StorageBuffer "
               "OpAccessChain [wrapper=0, zero-based view-adjusted element, "
               "root=1, leaf=0, dynamic]";
        expect(facts.atomic_value_is_one)
            << "the nested OpAtomicIAdd must consume the fixture's uint32 one";
        expect(facts.nested_function_load_uses_same_dynamic_index)
            << "a typed Function OpAccessChain [root=1, leaf=0, dynamic] "
               "feeding OpLoad must reuse the atomic chain's dynamic index ID";
        expect(facts.dynamic_insert_extract_round_trip_found)
            << "dynamic INSERT must store through [root=1, leaf=0, dynamic], "
               "reload the aggregate, and make that exact aggregate the base "
               "of the matching dynamic EXTRACT";
        expect(facts.dynamic_insert_extract_reaches_storage_output)
            << "the value reloaded by dynamic EXTRACT must participate in the "
               "uint32 dataflow stored to the output StorageBuffer";
    };
}

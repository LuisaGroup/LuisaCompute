//
// Created by Mike Smith on 2023/4/15.
//

#include <core/logging.h>
#include <runtime/rtx/ray.h>
#include <runtime/rtx/hit.h>
#include <dsl/rtx/ray_query.h>
#include <backends/metal/metal_builtin_embedded.h>
#include <backends/metal/metal_codegen_ast.h>

namespace luisa::compute::metal {

MetalCodegenAST::MetalCodegenAST(StringScratch &scratch) noexcept
    : _scratch{scratch},
      _ray_type{Type::of<Ray>()},
      _triangle_hit_type{Type::of<TriangleHit>()},
      _procedural_hit_type{Type::of<ProceduralHit>()},
      _committed_hit_type{Type::of<CommittedHit>()},
      _ray_query_all_type{Type::of<RayQueryAll>()},
      _ray_query_any_type{Type::of<RayQueryAny>()} {}

size_t MetalCodegenAST::type_size_bytes(const Type *type) noexcept {
    if (!type->is_custom()) { return type->size(); }
    LUISA_ERROR_WITH_LOCATION("Cannot get size of custom type.");
}

static void collect_types_in_function(Function f,
                                      luisa::unordered_set<const Type *> &types,
                                      luisa::unordered_set<Function> &visited) noexcept {

    // already visited
    if (!visited.emplace(f).second) { return; }

    // types from variables
    auto add = [&](auto &&self, auto t) noexcept -> void {
        if (t != nullptr && types.emplace(t).second) {
            if (t->is_array() || t->is_buffer()) {
                self(self, t->element());
            } else if (t->is_structure()) {
                for (auto m : t->members()) {
                    self(self, m);
                }
            }
        }
    };
    for (auto &&a : f.arguments()) { add(add, a.type()); }
    for (auto &&l : f.local_variables()) { add(add, l.type()); }
    traverse_expressions<true>(
        f.body(),
        [&add](auto expr) noexcept {
        if (auto type = expr->type()) {
            add(add, type);
        }
        },
        [](auto) noexcept {},
        [](auto) noexcept {});
    add(add, f.return_type());

    // types from called callables
    for (auto &&c : f.custom_callables()) {
        collect_types_in_function(
            Function{c.get()}, types, visited);
    }
}

void MetalCodegenAST::_emit_type_decls(Function kernel) noexcept {

    // collect used types in the kernel
    luisa::unordered_set<const Type *> types;
    luisa::unordered_set<Function> visited;
    collect_types_in_function(kernel, types, visited);

    // sort types by name so the generated
    // source is identical across runs
    luisa::vector<const Type *> sorted;
    sorted.reserve(types.size());
    std::copy(types.cbegin(), types.cend(),
              std::back_inserter(sorted));
    std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) noexcept {
        return a->hash() < b->hash();
    });

    auto do_emit = [this](const Type *type) noexcept {
        if (type->is_structure() &&
            type != _ray_type &&
            type != _triangle_hit_type &&
            type != _procedural_hit_type &&
            type != _committed_hit_type &&
            type != _ray_query_all_type &&
            type != _ray_query_any_type) {
            _scratch << "struct alignas(" << type->alignment() << ") ";
            _emit_type_name(type);
            _scratch << " {\n";
            for (auto i = 0u; i < type->members().size(); i++) {
                _scratch << "  ";
                _emit_type_name(type->members()[i]);
                _scratch << " m" << i << "{};\n";
            }
            _scratch << "};\n\n";
        }
        if (type->is_structure()) {
            // lc_zero and lc_one
            auto lc_make_value = [&](luisa::string_view name) noexcept {
                _scratch << "template<> inline auto " << name << "<";
                _emit_type_name(type);
                _scratch << ">() {\n"
                         << "  return ";
                _emit_type_name(type);
                _scratch << "{\n";
                for (auto i = 0u; i < type->members().size(); i++) {
                    _scratch << "    " << name << "<";
                    _emit_type_name(type->members()[i]);
                    _scratch << ">(),\n";
                }
                _scratch << "  };\n"
                         << "}\n\n";
            };
            lc_make_value("lc_zero");
            lc_make_value("lc_one");
            // lc_accumulate_grad
            _scratch << "inline void lc_accumulate_grad(thread ";
            _emit_type_name(type);
            _scratch << " *dst, ";
            _emit_type_name(type);
            _scratch << " grad) {\n";
            for (auto i = 0u; i < type->members().size(); i++) {
                _scratch << "  lc_accumulate_grad(&dst->m" << i << ", grad.m" << i << ");\n";
            }
            _scratch << "}\n\n";
        }
    };

    // process types in topological order
    types.clear();
    auto emit = [&](auto &&self, auto type) noexcept -> void {
        if (types.emplace(type).second) {
            if (type->is_array() || type->is_buffer()) {
                self(self, type->element());
            } else if (type->is_structure()) {
                for (auto m : type->members()) {
                    self(self, m);
                }
            }
            do_emit(type);
        }
    };
    for (auto t : sorted) { emit(emit, t); }
}

void MetalCodegenAST::_emit_type_name(const Type *type) noexcept {

    switch (type->tag()) {
        case Type::Tag::BOOL: _scratch << "bool"; break;
        case Type::Tag::FLOAT16: _scratch << "half"; break;
        case Type::Tag::FLOAT32: _scratch << "float"; break;
        case Type::Tag::INT16: _scratch << "short"; break;
        case Type::Tag::UINT16: _scratch << "ushort"; break;
        case Type::Tag::INT32: _scratch << "int"; break;
        case Type::Tag::UINT32: _scratch << "uint"; break;
        case Type::Tag::INT64: _scratch << "long"; break;
        case Type::Tag::UINT64: _scratch << "ulong"; break;
        case Type::Tag::VECTOR:
            _emit_type_name(type->element());
            _scratch << type->dimension();
            break;
        case Type::Tag::MATRIX:
            _scratch << "float"
                     << type->dimension()
                     << "x"
                     << type->dimension();
            break;
        case Type::Tag::ARRAY:
            _scratch << "array<";
            _emit_type_name(type->element());
            _scratch << ", ";
            _scratch << type->dimension() << ">";
            break;
        case Type::Tag::STRUCTURE: {
            if (type == _ray_type) {
                _scratch << "LCRay";
            } else if (type == _triangle_hit_type) {
                _scratch << "LCTriangleHit";
            } else if (type == _procedural_hit_type) {
                _scratch << "LCProceduralHit";
            } else if (type == _committed_hit_type) {
                _scratch << "LCCommittedHit";
            } else {
                _scratch << "S" << hash_to_string(type->hash());
            }
            break;
        }
        case Type::Tag::CUSTOM: {
            if (type == _ray_query_all_type) {
                _scratch << "LCRayQueryAll";
            } else if (type == _ray_query_any_type) {
                _scratch << "LCRayQueryAny";
            } else {
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported custom type: {}.",
                    type->description());
            }
            break;
        }
        default: break;
    }
}

void MetalCodegenAST::emit(Function kernel) noexcept {

    _scratch << luisa::string_view{luisa_metal_builtin_metal_device_lib,
                                   sizeof(luisa_metal_builtin_metal_device_lib)}
             << "\n"
             << "// block_size = ("
             << kernel.block_size().x << ", "
             << kernel.block_size().y << ", "
             << kernel.block_size().z << ")\n\n";

    _emit_type_decls(kernel);

    _scratch << "[[kernel]] void kernel_main() {"
                "}\n";
}

}// namespace luisa::compute::metal

#include <yyjson.h>

#include <luisa/xir/translators/xir2json.h>
#include <luisa/xir/translators/xir2text.h>

namespace luisa::compute::xir {

namespace {

[[nodiscard]] yyjson_alc make_allocator() noexcept {
    return yyjson_alc{
        .malloc = [](void *, size_t size) noexcept { return luisa::detail::allocator_allocate(size, 16u); },
        .realloc = [](void *, void *ptr, size_t, size_t size) noexcept { return luisa::detail::allocator_reallocate(ptr, size, 16u); },
        .free = [](void *, void *ptr) noexcept { luisa::detail::allocator_deallocate(ptr, 16u); },
        .ctx = nullptr,
    };
}

[[nodiscard]] luisa::string write_document(yyjson_mut_doc *doc,
                                           const yyjson_alc &allocator) noexcept {
    if (doc == nullptr) { return "{}\n"; }
    auto size = size_t{0u};
    auto *json = yyjson_mut_write_opts(
        doc, YYJSON_WRITE_PRETTY_TWO_SPACES | YYJSON_WRITE_NEWLINE_AT_END,
        &allocator, &size, nullptr);
    if (json == nullptr) { return "{}\n"; }
    auto result = luisa::string{json, size};
    allocator.free(allocator.ctx, json);
    return result;
}

}// namespace

luisa::string xir_to_json_translate(const Module *module) noexcept {
    auto allocator = make_allocator();
    auto *doc = yyjson_mut_doc_new(&allocator);
    if (doc == nullptr) { return "{}\n"; }
    auto *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "schema", "luisa.xir.debug");
    yyjson_mut_obj_add_uint(doc, root, "version", 1u);

    if (module == nullptr) {
        yyjson_mut_obj_add_bool(doc, root, "ok", false);
        yyjson_mut_obj_add_str(doc, root, "error", "null XIR module");
    } else {
        auto function_count = uint64_t{0u};
        auto block_count = uint64_t{0u};
        auto instruction_count = uint64_t{0u};
        for (auto *function : module->function_list()) {
            function_count++;
            if (auto *definition = function->definition()) {
                for (auto *block : definition->basic_blocks()) {
                    block_count++;
                    for ([[maybe_unused]] auto *instruction : block->instructions()) {
                        instruction_count++;
                    }
                }
            }
        }
        auto constant_count = uint64_t{0u};
        for ([[maybe_unused]] auto *constant : module->constant_list()) { constant_count++; }
        auto undefined_count = uint64_t{0u};
        for ([[maybe_unused]] auto *undefined : module->undefined_list()) { undefined_count++; }
        auto special_register_count = uint64_t{0u};
        for ([[maybe_unused]] auto *special : module->special_register_list()) { special_register_count++; }
        auto text = xir_to_flat_text_translate(module, true);
        yyjson_mut_obj_add_bool(doc, root, "ok", true);
        yyjson_mut_obj_add_uint(doc, root, "function_count", function_count);
        yyjson_mut_obj_add_uint(doc, root, "block_count", block_count);
        yyjson_mut_obj_add_uint(doc, root, "instruction_count", instruction_count);
        yyjson_mut_obj_add_uint(doc, root, "constant_count", constant_count);
        yyjson_mut_obj_add_uint(doc, root, "undefined_count", undefined_count);
        yyjson_mut_obj_add_uint(doc, root, "special_register_count", special_register_count);
        if (auto name = module->name()) {
            yyjson_mut_obj_add_strncpy(doc, root, "name", name->data(), name->size());
        }
        yyjson_mut_obj_add_strncpy(doc, root, "text", text.data(), text.size());
    }

    auto result = write_document(doc, allocator);
    yyjson_mut_doc_free(doc);
    return result;
}

}// namespace luisa::compute::xir

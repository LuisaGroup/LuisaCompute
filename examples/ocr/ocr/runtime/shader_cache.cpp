// OCR engine -- per-owner typed shader cache (Phase 2).
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

[[nodiscard]] luisa::string make_shader_key(luisa::string_view family, luisa::string_view variant) {
    luisa::string key;
    key.reserve(family.size() + variant.size() + 5u);
    key.append("ocr_");
    key.append(family);
    key.append("_");
    key.append(variant);
    return key;
}

void ShaderKeyRegistry::record_compile(luisa::string_view key) {
    _counts[luisa::string{key}]++;
    _total++;
}

[[nodiscard]] uint32_t ShaderKeyRegistry::compile_count(luisa::string_view key) const noexcept {
    auto it = _counts.find(luisa::string{key});
    return it == _counts.end() ? 0u : it->second;
}

[[nodiscard]] luisa::vector<luisa::string> ShaderKeyRegistry::keys() const {
    luisa::vector<luisa::string> out;
    out.reserve(_counts.size());
    for (const auto &entry : _counts) { out.emplace_back(entry.first); }
    return out;
}

}// namespace luisa::ocr
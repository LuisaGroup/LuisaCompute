#include <algorithm>

#include <luisa/ast/coro_suspend.h>
#include <luisa/core/logging.h>

namespace luisa::compute {
namespace {

struct CoroSuspendExtensionStorage {
    luisa::string schema;
    uint32_t version{0u};
    CoroSuspendFallback fallback{CoroSuspendFallback::reject};
    luisa::vector<CoroSuspendBinding> bindings;
    luisa::vector<CoroSuspendAttribute> attributes;

    CoroSuspendExtensionStorage(
        luisa::string schema, uint32_t version,
        CoroSuspendFallback fallback,
        luisa::vector<CoroSuspendBinding> bindings,
        luisa::vector<CoroSuspendAttribute> attributes) noexcept
        : schema{std::move(schema)}, version{version}, fallback{fallback},
          bindings{std::move(bindings)}, attributes{std::move(attributes)} {
        LUISA_ASSERT(!this->schema.empty(),
                     "Coroutine suspend extension schema must be non-empty.");
        LUISA_ASSERT(this->version != 0u,
                     "Coroutine suspend extension '{}' has reserved version 0.",
                     this->schema);
        std::sort(this->attributes.begin(), this->attributes.end(),
                  [](auto &&lhs, auto &&rhs) noexcept {
                      return lhs.name < rhs.name;
                  });
        for (size_t i = 0u; i < this->bindings.size(); ++i) {
            LUISA_ASSERT(!this->bindings[i].name.empty(),
                         "Coroutine suspend extension '{}' has an unnamed binding.",
                         this->schema);
            for (size_t j = 0u; j < i; ++j) {
                LUISA_ASSERT(this->bindings[j].name !=
                                 this->bindings[i].name,
                             "Coroutine suspend extension '{}' has duplicate "
                             "binding '{}'.",
                             this->schema, this->bindings[i].name);
            }
        }
        for (size_t i = 0u; i < this->attributes.size(); ++i) {
            LUISA_ASSERT(!this->attributes[i].name.empty(),
                         "Coroutine suspend extension '{}' has an unnamed attribute.",
                         this->schema);
            if (i != 0u) {
                LUISA_ASSERT(this->attributes[i - 1u].name !=
                                 this->attributes[i].name,
                             "Coroutine suspend extension '{}' has duplicate "
                             "attribute '{}'.",
                             this->schema, this->attributes[i].name);
            }
        }
    }
};

template<typename Base>
class DataBackedCoroSuspendExtension final : public Base {
private:
    CoroSuspendExtensionStorage _storage;

public:
    DataBackedCoroSuspendExtension(
        luisa::string schema, uint32_t version,
        CoroSuspendFallback fallback,
        luisa::vector<CoroSuspendBinding> bindings,
        luisa::vector<CoroSuspendAttribute> attributes) noexcept
        : _storage{std::move(schema), version, fallback,
                   std::move(bindings), std::move(attributes)} {}

    [[nodiscard]] luisa::string_view schema() const noexcept override {
        return _storage.schema;
    }
    [[nodiscard]] uint32_t version() const noexcept override {
        return _storage.version;
    }
    [[nodiscard]] CoroSuspendFallback fallback() const noexcept override {
        return _storage.fallback;
    }
    [[nodiscard]] luisa::span<const CoroSuspendBinding>
    bindings() const noexcept override {
        return _storage.bindings;
    }
    [[nodiscard]] luisa::span<const CoroSuspendAttribute>
    attributes() const noexcept override {
        return _storage.attributes;
    }
    [[nodiscard]] CoroSuspendExtensionPtr clone() const noexcept override {
        return luisa::make_unique<DataBackedCoroSuspendExtension>(
            _storage.schema, _storage.version, _storage.fallback,
            _storage.bindings, _storage.attributes);
    }
    [[nodiscard]] CoroSuspendExtensionPtr freeze(
        CoroSuspendExtensionRecorder &) && noexcept override {
        return clone();
    }
};

}// namespace

[[nodiscard]] CoroSuspendExtensionPtr make_coro_suspend_extension_data(
    luisa::string schema, uint32_t version,
    CoroSuspendFallback fallback,
    luisa::vector<CoroSuspendBinding> bindings,
    luisa::vector<CoroSuspendAttribute> attributes) noexcept {
    return luisa::make_unique<
        DataBackedCoroSuspendExtension<CoroSuspendExtension>>(
        std::move(schema), version, fallback,
        std::move(bindings), std::move(attributes));
}

[[nodiscard]] CoroSuspendExtensionPtr make_coro_suspend_annotation_data(
    luisa::string schema, uint32_t version,
    CoroSuspendFallback fallback,
    luisa::vector<CoroSuspendBinding> bindings,
    luisa::vector<CoroSuspendAttribute> attributes) noexcept {
    return luisa::make_unique<
        DataBackedCoroSuspendExtension<CoroSuspendAnnotation>>(
        std::move(schema), version, fallback,
        std::move(bindings), std::move(attributes));
}

}// namespace luisa::compute

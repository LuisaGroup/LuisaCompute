#pragma once

#include <type_traits>
#include <utility>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/tile/ir.h>

namespace luisa::compute::tile {

class AnalysisManager final {

private:
    struct ResultBase {
        virtual ~ResultBase() noexcept = default;
    };

    template<typename T>
    struct ResultHolder final : ResultBase {
        T value;
        explicit ResultHolder(T value) noexcept : value{std::move(value)} {}
    };

    const Function *_function{nullptr};
    luisa::unordered_map<const void *, luisa::unique_ptr<ResultBase>> _results;

    template<typename Analysis>
    [[nodiscard]] static const void *_key() noexcept {
        static const uint8_t key{0u};
        return &key;
    }

public:
    explicit AnalysisManager(const Function *function = nullptr) noexcept
        : _function{function} {}
    AnalysisManager(AnalysisManager &&) noexcept = delete;
    AnalysisManager(const AnalysisManager &) noexcept = delete;
    AnalysisManager &operator=(AnalysisManager &&) noexcept = delete;
    AnalysisManager &operator=(const AnalysisManager &) noexcept = delete;
    ~AnalysisManager() noexcept = default;

    void bind(const Function *function) noexcept {
        if (_function != function) {
            _function = function;
            _results.clear();
        }
    }
    [[nodiscard]] const Function *function() const noexcept { return _function; }

    template<typename Analysis>
    [[nodiscard]] const typename Analysis::Result *get() noexcept {
        using Result = typename Analysis::Result;
        static_assert(std::is_same_v<decltype(Analysis::run(std::declval<const Function &>())), Result>);
        if (_function == nullptr) { return nullptr; }
        auto key = _key<Analysis>();
        if (auto iter = _results.find(key); iter != _results.end()) {
            return &static_cast<ResultHolder<Result> *>(iter->second.get())->value;
        }
        auto holder = luisa::unique_ptr<ResultBase>{new ResultHolder<Result>{Analysis::run(*_function)}};
        auto result = &static_cast<ResultHolder<Result> *>(holder.get())->value;
        _results.emplace(key, std::move(holder));
        return result;
    }

    template<typename Analysis>
    void invalidate() noexcept {
        _results.erase(_key<Analysis>());
    }

    void invalidate_all() noexcept { _results.clear(); }
};

class IRRewriter final {

private:
    AnalysisManager *_analyses{nullptr};

    void _invalidate() noexcept {
        if (_analyses != nullptr) { _analyses->invalidate_all(); }
    }

public:
    explicit IRRewriter(AnalysisManager *analyses = nullptr) noexcept
        : _analyses{analyses} {}

    [[nodiscard]] bool replace_all_uses(Value *value, Value *replacement) noexcept {
        if (value == nullptr || !value->replace_all_uses_with(replacement)) { return false; }
        _invalidate();
        return true;
    }

    [[nodiscard]] bool erase(Operation *operation) noexcept {
        if (operation == nullptr || operation->parent_block() == nullptr || !operation->parent_block()->erase(operation)) { return false; }
        _invalidate();
        return true;
    }

    [[nodiscard]] bool move_before(Operation *operation, Operation *position) noexcept {
        if (operation == nullptr || position == nullptr || operation == position ||
            !operation->is_linked() || !position->is_linked() ||
            operation->is_sentinel() || position->is_sentinel() ||
            operation->parent_function() != position->parent_function()) { return false; }
        auto owned = operation->remove_self();
        if (owned == nullptr) { return false; }
        static_cast<void>(position->insert_before_self(std::move(owned)));
        _invalidate();
        return true;
    }

    [[nodiscard]] bool replace(Operation *operation, luisa::span<Value *const> replacements) noexcept {
        if (operation == nullptr || operation->result_count() != replacements.size()) { return false; }
        for (auto i = 0u; i < replacements.size(); i++) {
            if (replacements[i] == nullptr || replacements[i]->defining_operation() == operation ||
                !(operation->result(i)->type() == replacements[i]->type())) { return false; }
        }
        for (auto i = 0u; i < replacements.size(); i++) {
            if (operation->result(i) != replacements[i]) { static_cast<void>(operation->result(i)->replace_all_uses_with(replacements[i])); }
        }
        if (!operation->parent_block()->erase(operation)) { return false; }
        _invalidate();
        return true;
    }
};

}// namespace luisa::compute::tile

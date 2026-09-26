#pragma once

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/functional.h>
#include <luisa/core/platform.h>
#include <luisa/core/concepts.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>

namespace luisa {

class MemorySanitizer;

/**
 * @brief Dynamic module loader
 *
 */
class LUISA_CORE_API DynamicModule : concepts::Noncopyable {

private:
    friend ::luisa::MemorySanitizer;
    void *_handle{nullptr};

private:
    explicit DynamicModule(void *handle) noexcept : _handle{handle} {}
    static void reset_search_paths() noexcept;

public:
    [[nodiscard]] explicit operator bool() const noexcept { return _handle != nullptr; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    DynamicModule() noexcept = default;
    DynamicModule(DynamicModule &&another) noexcept;
    DynamicModule &operator=(DynamicModule &&rhs) noexcept;
    ~DynamicModule() noexcept;

    // give out the ownership of the module without destroying it
    [[nodiscard]] void *release() noexcept;
    // destroy the module and reset the handle to nullptr
    void reset() noexcept;

    /**
     * @brief Return address of given name
     *
     * @param name
     * @return void*
     */
    [[nodiscard]] void *address(const char *name) const noexcept {
        return dynamic_module_find_symbol(_handle, name);
    }

    /**
     * @brief Return function pointer of given name
     *
     * @tparam F function type
     * @param name name of function
     * @return pointer to function
     */
    template<concepts::function F>
    [[nodiscard]] auto function(const char *name) const noexcept {
        return reinterpret_cast<std::add_pointer_t<F>>(address(name));
    }

    /**
     * @brief Invoke function
     *
     * @tparam F function type
     * @tparam Args function args
     * @param name name of function
     * @param args function args
     * @return function return
     */
    template<concepts::function F, typename... Args>
        requires(std::is_invocable_v<F, Args && ...>)
    decltype(auto) invoke(const char *name, Args &&...args) const noexcept {
        return luisa::invoke(function<F>(name), std::forward<Args>(args)...);
    }

    /**
     * @brief Apply function to each element
     *
     * @tparam F function type
     * @tparam Tuple tuple type
     * @param name name of function
     * @param t tuple to be applied
     * @return decltype(auto)
     */
    template<concepts::function F, typename Tuple>
    decltype(auto) apply(const char *name, Tuple &&t) const noexcept {
        return std::apply(function<F>(name), std::forward<Tuple>(t));
    }

    /**
     * @brief Add dynamic module search path
     *
     * @param path
     */
    static void add_search_path(const luisa::filesystem::path &path) noexcept;

    /**
     * @brief Remove dynamic module search path
     *
     * @param path
     */
    static void remove_search_path(const luisa::filesystem::path &path) noexcept;

    /**
     * @brief Load module with the specified name in search paths and the working directory
     * @param name Name of the module
     * @return The module if successfully loaded, otherwise a nullopt
     */
    [[nodiscard]] static DynamicModule load(luisa::string_view name) noexcept;

    /**
     * @brief Load module with the specified name in a folder
     * @param folder The folder when the module is expected to exist
     * @param name Name of the module
     * @return The module if successfully loaded, otherwise a nullopt
     */
    [[nodiscard]] static DynamicModule load(const luisa::filesystem::path &folder,
                                            luisa::string_view name) noexcept;

    /**
     * @brief Load module with the specified path, without adding prefixes or suffices.
     */
    [[nodiscard]] static DynamicModule load_exact(const luisa::filesystem::path &path) noexcept;

};

}// namespace luisa

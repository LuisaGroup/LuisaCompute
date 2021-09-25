//
// Created by Mike Smith on 2021/9/3.
//

#include <iostream>
#include <functional>
#include <memory>
#include <map>

template<typename T>
struct Pool {

};

template<typename T>
struct Allocator {
    Pool<T> pool;
    using value_type = T;
    Allocator() noexcept {
        std::cout << "Allocator()" << std::endl;
    }
    Allocator(Allocator &&another) noexcept : pool{std::move(another.pool)} {
        std::cout << "Allocator(Allocator &&)" << std::endl;
    }
    Allocator(const Allocator &another) noexcept : pool{another.pool} {
        std::cout << "Allocator(const Allocator &)" << std::endl;
    }
    Allocator &operator=(Allocator &&rhs) noexcept {
        pool = std::move(rhs.pool);
        std::cout << "operator=(Allocator &&)" << std::endl;
    }
    Allocator &operator=(const Allocator &rhs) noexcept {
        pool = rhs.pool;
        std::cout << "operator=(const Allocator &)" << std::endl;
    }
    [[nodiscard]] auto allocate(std::size_t n) const noexcept {
        auto p = malloc(n * sizeof(T));
        std::cout << "allocate(" << n << ") -> " << p << std::endl;
        return p;
    }
    void deallocate(T *p, size_t n) const noexcept {
        std::cout << "deallocate(" << static_cast<void *>(p) << ", " << n << ")" << std::endl;
        free(p);
    }
    template<typename R>
    [[nodiscard]] constexpr auto operator==(const Allocator<R> &rhs) const noexcept -> bool {
        auto same = this == &rhs;
        std::cout << "operator==() -> " << same << std::endl;
        return same;
    }
};

int main() {
    std::map<int, int, std::equal_to<>, Allocator<std::pair<const int, int>>> map;
    std::cout << "after constructor..." << std::endl;
}

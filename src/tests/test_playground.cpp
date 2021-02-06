//
// Created by Mike Smith on 2021/2/2.
//

#include <string>
#include <fstream>
#include <memory>
#include <variant>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>

#include <core/logging.h>
#include <core/arena.h>
#include <core/hash.h>
#include <core/type.h>

struct Test {
    
    std::string s;
    int a;
    
    template<typename Archive>
    void serialize(Archive &&ar) noexcept {
        ar(s, a);
    }
    
};

struct BB;

struct alignas(128) AA {
    float a;
};

struct BB {
    AA a;
    float b;
};

LUISA_MAKE_STRUCTURE_TYPE_SPECIALIZATION(AA, a)
LUISA_MAKE_STRUCTURE_TYPE_SPECIALIZATION(BB, a, b)

struct Interface {
    ~Interface() noexcept = default;
};

struct Impl : public Interface {

};

int main() {
    
    Test test{"Hello", 123};
    Test luisa{"world", 233};
    std::ofstream of{"hello.json", std::ios::binary};
    cereal::JSONOutputArchive ar{of};
    
    spdlog::debug("Debugging...");
    spdlog::info("Hello!!!");
    
    char buffer[128];
    fmt::format_to_n(buffer, 128, FMT_STRING("{:a}f"), 5.5);
    spdlog::critical(buffer);
    
    ar(test, luisa);
    
    std::variant<float, int> x;
    
    LUISA_VERBOSE("verbose...");
    LUISA_VERBOSE_WITH_LOCATION("verbose with {}...", "location");
    LUISA_INFO("info...");
    LUISA_INFO_WITH_LOCATION("info with location...");
    LUISA_WARNING("warning...");
    LUISA_WARNING_WITH_LOCATION("warning with location...");
    
    LUISA_INFO("size = {}, alignment = {}", sizeof(AA), alignof(AA));
    LUISA_INFO("size = {}, alignment = {}", sizeof(BB), alignof(BB));
    
    using namespace luisa;
    Arena arena;
    
    Arena another{std::move(arena)};
    auto p = another.allocate<int, 1024>(1);
    LUISA_INFO("{}", fmt::ptr(p.data()));
    
    auto &&type_aa = typeid(AA);
    auto &&type_bb = typeid(BB);
    LUISA_INFO("{}", type_aa.before(type_bb));
    
    LUISA_INFO("trivially destructible: {}", std::is_trivially_destructible_v<Impl>);
    
    auto hash_aa = luisa::xxh32_hash32(type_aa.name(), std::strlen(type_aa.name()), 0);
    auto hash_bb = luisa::xxh3_hash64(type_bb.name(), std::strlen(type_bb.name()), 0);
    LUISA_INFO("{} {}", hash_aa, hash_bb);
    
    using StructBB = luisa::Type<BB>;
    LUISA_INFO("{} {}", StructBB::alignment, typeid(typename StructBB::Members).name());
}

//
// Created by Mike Smith on 2021/2/2.
//

#include <string>
#include <fstream>
#include <memory>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>

#include <core/logging.h>

struct Test {
    
    std::string s;
    int a;
    
    template<typename Archive>
    void serialize(Archive &&ar) noexcept {
        ar(s, a);
    }
    
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
    
    
    LUISA_VERBOSE("verbose...");
    LUISA_VERBOSE_WITH_LOCATION("verbose with {}...", "location");
    LUISA_INFO("info...");
    LUISA_INFO_WITH_LOCATION("info with location...");
    LUISA_WARNING("warning...");
    LUISA_WARNING_WITH_LOCATION("warning with location...");
}

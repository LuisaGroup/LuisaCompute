//
// Created by Mike Smith on 2021/9/3.
//

#include <iostream>
#include <nlohmann/json.hpp>

int main() {

    auto j = nlohmann::json::parse(R"({
"Hello": "World",
"test": 1234
})");

    std::cout << j.count("Hello")
              << std::endl;
}

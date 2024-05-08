#include <spdlog/fmt/fmt.h>
#include <string>
#include <iostream>
template<typename String, typename Format, typename... Args>
[[nodiscard]] inline auto format(Format &&f, Args &&...args) noexcept {
    using char_type = typename String::value_type;
    using memory_buffer = fmt::basic_memory_buffer<char_type, fmt::inline_buffer_size, std::allocator<char_type>>;
    memory_buffer buffer;
    fmt::format_to(std::back_inserter(buffer), std::forward<Format>(f), std::forward<Args>(args)...);
    return String{buffer.data(), buffer.size()};
}

template<typename Format, typename... Args>
[[nodiscard]] inline auto format(Format &&f, Args &&...args) noexcept {
    return format<std::string>(std::forward<Format>(f), std::forward<Args>(args)...);
}

int main(int argc, char *argv[]) {
    using fmt::format_to;
    std::cout << fmt::format("shit: {}", argv[0]) << std::endl;
}

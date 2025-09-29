//
// Created by mike on 9/22/25.
//

#include <algorithm>
#include <filesystem>
#include <span>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <string_view>

[[noreturn]]
void print_help_and_exit(std::ostream &os, const char *program, int code, bool print_desc) noexcept {
    if (print_desc) {
        os << "Command-line tool to embed device library files into C arrays\n";
    }
    os << "Usage: " << program << " [OPTIONS] input-file-1 input-file-2 ...\n"
       << "Options:\n"
       << "  -o, --output <file>   [REQUIRED] Output C++ source file\n"
       << "  -h, --header <file>   Print header file as well (default: false)\n"
       << "  -u, --unsigned        Use unsigned char for byte (default: false)\n"
       << "  -p, --prefix <prefix> Prefix for variable names (default: empty)\n"
       << "  -s, --suffix <suffix> Suffix for variable names (default: empty)\n"
       << "  -w, --wrap <width>    Wrap lines at <width> elements (default: 16)\n"
       << "  -i, --indent <spaces> Number of spaces for indentation (default: 4)\n"
       << "  -e, --preserve-ext    Preserve file extensions in variable names (default: false)\n"
       << "      --help            Show this help message\n"
       << std::endl;
    exit(code);
}

struct Options {
    std::vector<std::filesystem::path> input_files;
    std::filesystem::path output_file;
    std::filesystem::path header_file;
    std::string prefix;
    std::string suffix;
    int wrap_width = 16;
    int indent = 4;
    bool use_unsigned_char = false;
    bool preserve_extension = false;
};

[[nodiscard]] auto parse(int argc, char *argv[]) noexcept {
    Options o;
    for (auto i = 1; i < argc; i++) {
        auto opt = std::string_view{argv[i]};
        auto require_next = [&]() {
            if (i + 1 >= argc) {
                std::cerr << "Error: option " << opt << " requires an argument.\n";
                print_help_and_exit(std::cerr, argv[0], 1, false);
            }
            return std::string_view{argv[++i]};
        };
        if (opt == "-o" || opt == "--output") {
            o.output_file = require_next();
        } else if (opt == "-h" || opt == "--header") {
            o.header_file = require_next();
        } else if (opt == "-u" || opt == "--unsigned") {
            o.use_unsigned_char = true;
        } else if (opt == "-s" || opt == "--suffix") {
            o.suffix = require_next();
        } else if (opt == "-p" || opt == "--prefix") {
            o.prefix = require_next();
        } else if (opt == "-e" || opt == "--preserve-ext") {
            o.preserve_extension = true;
        } else if (opt == "-w" || opt == "--wrap") {
            auto w = require_next();
            std::istringstream iss{std::string{w}};
            if (!(iss >> o.wrap_width) || o.wrap_width <= 0) {
                std::cerr << "Error: invalid wrap width: " << w << "\n";
                print_help_and_exit(std::cerr, argv[0], 1, false);
            }
        } else if (opt == "-i" || opt == "--indent") {
            auto s = require_next();
            std::istringstream iss{std::string{s}};
            if (!(iss >> o.indent) || o.indent < 0) {
                std::cerr << "Error: invalid indent spaces: " << s << "\n";
                print_help_and_exit(std::cerr, argv[0], 1, false);
            }
        } else if (opt == "--help") {
            print_help_and_exit(std::cout, argv[0], 0, true);
        } else if (opt.starts_with('-')) {
            std::cerr << "Error: invalid option " << opt << "\n";
            print_help_and_exit(std::cerr, argv[0], 1, false);
        } else {
            std::error_code ec;
            auto path = std::filesystem::canonical(opt, ec);
            if (ec) {
                std::cerr << "Error: input file does not exist: " << opt << "\n";
                print_help_and_exit(std::cerr, argv[0], 1, false);
            }
            o.input_files.emplace_back(std::move(path));
        }
    }
    // validate that output file is set
    if (o.output_file.empty()) {
        std::cerr << "Error: output file is required.\n";
        print_help_and_exit(std::cerr, argv[0], 1, false);
    }
    // validate that there is at least one input file
    if (o.input_files.empty()) {
        std::cerr << "Error: at least one input file is required.\n";
        print_help_and_exit(std::cerr, argv[0], 1, false);
    }
    // sort and unique input files
    std::sort(o.input_files.begin(), o.input_files.end());
    return o;
}

[[nodiscard]] std::vector<char> read_file_content(const std::filesystem::path &file_path, bool allow_failure) noexcept {
    std::ifstream ifs{file_path, std::ios::binary | std::ios::ate};
    if (!ifs) {
        if (allow_failure) { return {}; }
        std::cerr << "Error: failed to open input file: " << file_path << std::endl;
        exit(1);
    }
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> content(size);
    if (!ifs.read(content.data(), size)) {
        std::cerr << "Error: failed to read input file: " << file_path << "\n";
        exit(1);
    }
    return content;
}

void append_file_as_c_array(std::ostringstream &oss_source, std::ostringstream &oss_header,
                            const std::filesystem::path &file_path, const Options &options) noexcept {
    auto data = read_file_content(file_path, false);
    auto name = options.prefix + (options.preserve_extension ? file_path.filename() : file_path.stem()).string() + options.suffix;
    // canonicalize name: replace non-alphanumeric characters with '_'
    for (auto &c : name) {
        if (!std::isalnum(c) && c != '_') {
            c = '_';
        }
    }
    // emit header declaration
    if (!options.header_file.empty()) {
        oss_header << "\n"
                   << "extern \"C\" const unsigned long long " << name << "_size;\n"
                   << "extern \"C\" const " << (options.use_unsigned_char ? "unsigned char" : "char") << " " << name << "[];\n";
    }
    // emit C++ code
    oss_source << "\n"
               << "extern \"C\" const unsigned long long " << name << "_size = " << data.size() << ";\n\n"
               << "extern \"C\" const " << (options.use_unsigned_char ? "unsigned char" : "char") << " " << name << "[" << data.size() << "] = {\n";
    auto print_indent = [&oss_source, n = options.indent]() noexcept {
        for (auto i = 0; i < n; i++) { oss_source << ' '; }
    };
    auto print_byte_as_hex = [&oss_source](char byte) noexcept {
        constexpr char hex_chars[] = "0123456789abcdef";
        oss_source << "0x";
        oss_source << hex_chars[(byte >> 4) & 0x0f];
        oss_source << hex_chars[byte & 0x0f];
    };
    for (size_t i = 0; i < data.size(); i++) {
        if (i % options.wrap_width == 0) { print_indent(); }
        print_byte_as_hex(data[i]);
        if ((i + 1) % options.wrap_width == 0 || i + 1 == data.size()) {
            oss_source << ",\n";
        } else {
            oss_source << ", ";
        }
    }
    oss_source << "};\n";
}

[[nodiscard]] auto generate_source(const Options &options) noexcept {
    auto print_notice_and_include = [](std::ostringstream &oss) noexcept {
        oss << "// !!!!!!!!! THIS FILE IS AUTOMATICALLY GENERATED !!!!!!!!!\n"
            << "// !!!!!!!!!    PLEASE DO NOT EDIT IT MANUALLY    !!!!!!!!!\n";
    };
    std::ostringstream oss_source;
    print_notice_and_include(oss_source);
    std::ostringstream oss_header;
    if (!options.header_file.empty()) {
        print_notice_and_include(oss_header);
        oss_header << "\n"
                   << "#pragma once\n";
    }
    for (auto const &file : options.input_files) {
        append_file_as_c_array(oss_source, oss_header, file, options);
    }
    return std::make_pair(oss_source.str(), oss_header.str());
}

void update_file_if_changed(const std::filesystem::path &file_path, std::string_view content) noexcept {
    std::error_code ec;
    auto abs_path = std::filesystem::absolute(file_path, ec);
    if (ec) {
        std::cerr << "Error: failed to get absolute path of output file: " << ec.message() << std::endl;
        exit(1);
    }
    std::filesystem::create_directories(abs_path.parent_path(), ec);
    if (ec) {
        std::cerr << "Error: failed to create directories for output file: " << ec.message() << std::endl;
        exit(1);
    }
    if (auto existing = read_file_content(abs_path, true);
        std::equal(existing.begin(), existing.end(), content.begin(), content.end())) {
        return;
    }
    std::ofstream ofs{abs_path, std::ios::binary | std::ios::trunc};
    if (!ofs) {
        std::cerr << "Error: failed to open output file for writing: " << abs_path << std::endl;
    }
    ofs << content;
}

int main(int argc, char *argv[]) {
    auto options = parse(argc, argv);
    auto [source, header] = generate_source(options);
    update_file_if_changed(options.output_file, source);
    if (!options.header_file.empty()) { update_file_if_changed(options.header_file, header); }
}

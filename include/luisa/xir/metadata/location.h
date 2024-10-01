#pragma once

#include <luisa/core/stl/filesystem.h>
#include <luisa/xir/metadata.h>

namespace luisa::compute::xir {

class LC_XIR_API LocationMD : public Metadata {

private:
    luisa::filesystem::path _file;
    int _line;
    int _column;

public:
    explicit LocationMD(Pool *pool, luisa::filesystem::path file = {},
                        int line = -1, int column = -1) noexcept;
    void set_file(luisa::filesystem::path file) noexcept { _file = std::move(file); }
    void set_line(int line) noexcept { _line = line; }
    void set_column(int column) noexcept { _column = column; }
    [[nodiscard]] auto &file() const noexcept { return _file; }
    [[nodiscard]] auto line() const noexcept { return _line; }
    [[nodiscard]] auto column() const noexcept { return _column; }
};

}// namespace luisa::compute::xir
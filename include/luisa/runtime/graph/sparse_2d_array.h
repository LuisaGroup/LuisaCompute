#pragma once
#include <cstdint>
#include <luisa/vstl/vector.h>

namespace luisa::compute {
template<typename T>
class Sparse2DArray {
public:
    using value_type = T;
    using size_type = size_t;
    Sparse2DArray() noexcept = default;
    Sparse2DArray(const vector<vector<value_type>> &input) noexcept { set(input); }

    Sparse2DArray(const Sparse2DArray &) noexcept = default;
    Sparse2DArray(Sparse2DArray &&) noexcept = default;
    Sparse2DArray &operator=(const Sparse2DArray &) noexcept = default;
    Sparse2DArray &operator=(Sparse2DArray &&) noexcept = default;
    ~Sparse2DArray() noexcept = default;

    void set(const vector<vector<value_type>> &input) {
        _offsets.resize(input.size() + 1);
        _offsets[0] = 0;
        for (size_t i = 0; i < input.size(); i++) {
            _offsets[i + 1] = _offsets[i] + input[i].size();
        }
        _values.resize(_offsets.back());
        for (size_t i = 0; i < input.size(); i++) {
            for (size_t j = 0; j < input[i].size(); j++) {
                _values[_offsets[i] + j] = input[i][j];
            }
        }
    }

    auto operator()(size_type i) const noexcept 
    {
        auto offset = _offsets[i];
        auto size = _offsets[i + 1] - offset;
        return span{_values}.subspan(_offsets[i], size); 
    }
    //auto operator()(size_type i) noexcept { return span{_values}.subspan(_offsets[i], _offsets[i + 1]); }
private:
    vector<size_type> _offsets;
    vector<value_type> _values;
};
}// namespace luisa::compute
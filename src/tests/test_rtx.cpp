#include <core/stl/vector.h>
#include <iostream>
template<typename T>
struct VectorData {
    T *begin;
    T *end;
    T *capacity;
};
int main() {
    using namespace luisa;
    vector<int> a;
    a.resize(5);
    VectorData<int> *ptr = std::launder(reinterpret_cast<VectorData<int> *>(&a));
    std::cout << a.data() << ' ' << ptr->begin << '\n'
              << a.size() << ' ' << (ptr->end - ptr->begin) << '\n'
              << a.capacity()<< ' '  << (ptr->capacity - ptr->begin) << '\n';

    return 0;
}
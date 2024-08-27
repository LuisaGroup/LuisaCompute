#include <zlib/zlib.h>
#include <iostream>
#include <luisa/core/binary_file_stream.h>
#include <luisa/core/stl/vector.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Bad arguments.\n";
        return -1;
    }
    luisa::vector<std::byte> vec;
    luisa::vector<std::byte> result;
    {
        luisa::BinaryFileStream fs{argv[1]};
        if (!fs.valid()) {
            std::cout << "Bad input.\n";
            return -1;
        }
        vec.push_back_uninitialized(fs.length());
        fs.read(vec);
    }
    result.push_back_uninitialized(compressBound(vec.size()));
    uLong size = result.size();
    auto id = compress2((Bytef *)result.data(), &size, (const Bytef *)vec.data(), vec.size(), Z_BEST_COMPRESSION);
    if (id != Z_OK) {
        std::cout << "Compress failed.\n";
        return -1;
    }
    auto f = fopen(argv[2], "wb");
    if (!f) {
        std::cout << "Wrilte file failed.\n";
        return -1;
    }
    result.resize(size);
    fwrite(result.data(), result.size(), 1, f);
    fclose(f);
    return 0;
}
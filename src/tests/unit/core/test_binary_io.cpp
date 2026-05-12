// Test for binary_io.h
// Tests BinaryBlob class for memory management, data access, and move semantics.

#include <cstring>

#include "ut/ut.hpp"

#include <luisa/core/binary_io.h>
#include <luisa/core/logging.h>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;

// Custom test allocator tracking
namespace {
int g_allocate_count = 0;
int g_dispose_count = 0;
}// namespace

void _luisa_reg_binaryblob_default_construction() {

    boost::ut::detail::test{"test", "BinaryBlob default construction"} = [] {
        BinaryBlob blob;
        boost::ut::expect(static_cast<bool>(blob.data() == nullptr));
        boost::ut::expect(static_cast<bool>(blob.size() == 0));
        boost::ut::expect(static_cast<bool>(blob.empty() == true));
    };
}

void _luisa_reg_binaryblob_construction_with_data() {

    boost::ut::detail::test{"test", "BinaryBlob construction with data"} = [] {
        g_allocate_count = 0;
        g_dispose_count = 0;

        size_t size = 1024;
        auto *ptr = static_cast<std::byte *>(::operator new(size));
        g_allocate_count++;

        {
            BinaryBlob blob{
                ptr,
                size,
                [](void *p) {
                    ::operator delete(p);
                    g_dispose_count++;
                }};

            boost::ut::expect(static_cast<bool>(blob.data() == ptr));
            boost::ut::expect(static_cast<bool>(blob.size() == size));
            boost::ut::expect(static_cast<bool>(blob.empty() == false));
        }

        // Destructor should have called disposer
        boost::ut::expect(static_cast<bool>(g_dispose_count == 1));
    };
}

void _luisa_reg_binaryblob_move_construction() {

    boost::ut::detail::test{"test", "BinaryBlob move construction"} = [] {
        size_t size = 256;
        auto *ptr = static_cast<std::byte *>(::operator new(size));

        BinaryBlob blob1{
            ptr,
            size,
            [](void *p) { ::operator delete(p); }};

        BinaryBlob blob2{std::move(blob1)};

        // blob1 should be empty after move
        boost::ut::expect(static_cast<bool>(blob1.data() == nullptr));
        boost::ut::expect(static_cast<bool>(blob1.size() == 0));
        boost::ut::expect(static_cast<bool>(blob1.empty() == true));

        // blob2 should have the data
        boost::ut::expect(static_cast<bool>(blob2.data() == ptr));
        boost::ut::expect(static_cast<bool>(blob2.size() == size));
        boost::ut::expect(static_cast<bool>(blob2.empty() == false));
    };
}

void _luisa_reg_binaryblob_move_assignment() {

    boost::ut::detail::test{"test", "BinaryBlob move assignment"} = [] {
        size_t size1 = 128;
        size_t size2 = 256;
        auto *ptr1 = static_cast<std::byte *>(::operator new(size1));
        auto *ptr2 = static_cast<std::byte *>(::operator new(size2));
        int dispose_count = 0;

        {
            BinaryBlob blob1{
                ptr1,
                size1,
                [&dispose_count](void *p) {
                    ::operator delete(p);
                    dispose_count++;
                }};

            BinaryBlob blob2{
                ptr2,
                size2,
                [&dispose_count](void *p) {
                    ::operator delete(p);
                    dispose_count++;
                }};

            blob1 = std::move(blob2);

            // blob1's old data should be disposed
            boost::ut::expect(static_cast<bool>(dispose_count == 1));

            // blob1 should now have blob2's data
            boost::ut::expect(static_cast<bool>(blob1.data() == ptr2));
            boost::ut::expect(static_cast<bool>(blob1.size() == size2));
        }

        // Both blobs disposed
        boost::ut::expect(static_cast<bool>(dispose_count == 2));
    };
}

void _luisa_reg_binaryblob_const_and_non_const_data_access() {

    boost::ut::detail::test{"test", "BinaryBlob const and non-const data access"} = [] {
        size_t size = 64;
        auto *ptr = static_cast<std::byte *>(::operator new(size));

        // Non-const blob
        {
            BinaryBlob blob{
                ptr,
                size,
                [](void *p) { ::operator delete(p); }};

            std::byte *data = blob.data();
            boost::ut::expect(static_cast<bool>(data == ptr));

            // Modify through non-const pointer
            data[0] = std::byte{0x42};
            boost::ut::expect(static_cast<bool>(static_cast<int>(blob.data()[0]) == 0x42));
        }

        // Const blob
        {
            auto *ptr2 = static_cast<std::byte *>(::operator new(size));
            const BinaryBlob blob{
                ptr2,
                size,
                [](void *p) { ::operator delete(p); }};

            const std::byte *data = blob.data();
            boost::ut::expect(static_cast<bool>(data == ptr2));
        }
    };
}

void _luisa_reg_binaryblob_span_conversion() {

    boost::ut::detail::test{"test", "BinaryBlob span conversion"} = [] {
        size_t size = 32;
        auto data = new std::byte[size];
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<std::byte>(i);
        }

        {
            BinaryBlob blob{
                data,
                size,
                [](void *p) { delete[] static_cast<std::byte *>(p); }};

            // Non-const span
            luisa::span<std::byte> mutable_span = static_cast<luisa::span<std::byte>>(blob);
            boost::ut::expect(static_cast<bool>(mutable_span.size() == size));
            boost::ut::expect(static_cast<bool>(mutable_span.data() == data));

            // Modify through span
            mutable_span[0] = std::byte{0xFF};
            boost::ut::expect(static_cast<bool>(static_cast<int>(blob.data()[0]) == 0xFF));
        }

        // Const span test
        auto data2 = new std::byte[size];
        {
            const BinaryBlob blob{
                data2,
                size,
                [](void *p) { delete[] static_cast<std::byte *>(p); }};

            luisa::span<const std::byte> const_span = static_cast<luisa::span<const std::byte>>(blob);
            boost::ut::expect(static_cast<bool>(const_span.size() == size));
            boost::ut::expect(static_cast<bool>(const_span.data() == data2));
        }
    };
}

void _luisa_reg_binaryblob_empty_check() {

    boost::ut::detail::test{"test", "BinaryBlob empty check"} = [] {
        BinaryBlob empty_blob;
        boost::ut::expect(static_cast<bool>(empty_blob.empty() == true));

        auto *ptr = static_cast<std::byte *>(::operator new(1));
        BinaryBlob non_empty_blob{
            ptr,
            1,
            [](void *p) { ::operator delete(p); }};
        boost::ut::expect(static_cast<bool>(non_empty_blob.empty() == false));
    };
}

void _luisa_reg_binaryblob_self_move_assignment() {

    boost::ut::detail::test{"test", "BinaryBlob self-move assignment"} = [] {
        size_t size = 64;
        auto *ptr = static_cast<std::byte *>(::operator new(size));
        int dispose_count = 0;

        BinaryBlob blob{
            ptr,
            size,
            [&dispose_count](void *p) {
                ::operator delete(p);
                dispose_count++;
            }};

        // Self-move assignment (should be safe)
        blob = std::move(blob);

        // Data should still be valid
        boost::ut::expect(static_cast<bool>(blob.data() == ptr));
        boost::ut::expect(static_cast<bool>(blob.size() == size));
        boost::ut::expect(static_cast<bool>(dispose_count == 0));
    };
}

void _luisa_reg_binaryblob_multiple_moves() {

    boost::ut::detail::test{"test", "BinaryBlob multiple moves"} = [] {
        size_t size = 100;
        auto *ptr = static_cast<std::byte *>(::operator new(size));
        int dispose_count = 0;

        {
            BinaryBlob blob1{
                ptr,
                size,
                [&dispose_count](void *p) {
                    ::operator delete(p);
                    dispose_count++;
                }};

            BinaryBlob blob2{std::move(blob1)};
            BinaryBlob blob3{std::move(blob2)};
            BinaryBlob blob4{std::move(blob3)};

            boost::ut::expect(static_cast<bool>(blob4.data() == ptr));
            boost::ut::expect(static_cast<bool>(blob4.size() == size));
            boost::ut::expect(static_cast<bool>(blob1.data() == nullptr));
            boost::ut::expect(static_cast<bool>(blob2.data() == nullptr));
            boost::ut::expect(static_cast<bool>(blob3.data() == nullptr));
        }

        boost::ut::expect(static_cast<bool>(dispose_count == 1));
    };
}

void _luisa_reg_binaryblob_with_custom_allocator() {

    boost::ut::detail::test{"test", "BinaryBlob with custom allocator"} = [] {
        struct TestAllocator {
            size_t allocated = 0;
            size_t deallocated = 0;

            void *allocate(size_t size) {
                allocated += size;
                return std::malloc(size);
            }

            void deallocate(void *ptr, size_t size) {
                deallocated += size;
                std::free(ptr);
            }
        };

        TestAllocator allocator;
        size_t size = 512;
        {
            auto *ptr = static_cast<std::byte *>(allocator.allocate(size));
            BinaryBlob blob{
                ptr,
                size,
                [&allocator, size](void *p) { allocator.deallocate(p, size); }};

            boost::ut::expect(static_cast<bool>(allocator.allocated == size));
        }

        boost::ut::expect(static_cast<bool>(allocator.deallocated == size));
    };
}

int main(int argc, char *argv[]) {

    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    _luisa_reg_binaryblob_default_construction();
    _luisa_reg_binaryblob_construction_with_data();
    _luisa_reg_binaryblob_move_construction();
    _luisa_reg_binaryblob_move_assignment();
    _luisa_reg_binaryblob_const_and_non_const_data_access();
    _luisa_reg_binaryblob_span_conversion();
    _luisa_reg_binaryblob_empty_check();
    _luisa_reg_binaryblob_self_move_assignment();
    _luisa_reg_binaryblob_multiple_moves();
    _luisa_reg_binaryblob_with_custom_allocator();
    return 0;
}

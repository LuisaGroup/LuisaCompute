// Test for BinaryFileStream class
// This test covers:
// - Construction from file path (valid and invalid files)
// - Move constructor and move assignment
// - valid() and operator bool()
// - length() and pos()
// - read() - reading data into buffers
// - set_pos() - seeking to positions
// - close() - explicit file closing
// - Reading beyond end of file

#include <cstdio>
#include <cstring>

#include <luisa/core/binary_file_stream.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include "ut/ut.hpp"

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

// Helper function to create a test file with known content
static void create_test_file(const char *path, const luisa::vector<std::byte> &content) noexcept {
    FILE *file = std::fopen(path, "wb");
    expect(static_cast<bool>(file != nullptr));
    std::fwrite(content.data(), 1, content.size(), file);
    std::fclose(file);
}

// Helper function to cleanup test file
static void remove_test_file(const char *path) noexcept {
    std::remove(path);
}

static auto test_binary_file_stream_registration = [] {
    "test_binary_file_stream"_test = [] {
        log_level_verbose();

        const char *test_file_path = "test_binary_file_stream.tmp";
        const char *nonexistent_path = "nonexistent_file_xyz.tmp";

        // Prepare test data
        luisa::vector<std::byte> test_data;
        test_data.reserve(256);
        for (int i = 0; i < 256; ++i) {
            test_data.push_back(static_cast<std::byte>(i));
        }

        // Create test file
        create_test_file(test_file_path, test_data);
        LUISA_INFO("Test file created with {} bytes", test_data.size());

        // Test 1: Construction with invalid file path
        {
            LUISA_INFO("Test 1: Construction with invalid file path...");
            BinaryFileStream stream(luisa::string{nonexistent_path});
            expect(static_cast<bool>(!stream.valid()));
            expect(static_cast<bool>(!stream));
            expect(static_cast<bool>(stream.length() == 0));
            expect(static_cast<bool>(stream.pos() == 0));
            LUISA_INFO("Test 1 passed: Invalid file handled correctly");
        }

        // Test 2: Construction with valid file path
        {
            LUISA_INFO("Test 2: Construction with valid file path...");
            BinaryFileStream stream(luisa::string{test_file_path});
            expect(static_cast<bool>(stream.valid()));
            expect(static_cast<bool>(static_cast<bool>(stream)));
            expect(static_cast<bool>(stream.length() == test_data.size()));
            expect(static_cast<bool>(stream.pos() == 0));
            LUISA_INFO("Test 2 passed: Valid file opened correctly, length = {}", stream.length());
        }

        // Test 3: Read full file content
        {
            LUISA_INFO("Test 3: Read full file content...");
            BinaryFileStream stream(luisa::string{test_file_path});
            luisa::vector<std::byte> buffer(test_data.size());
            stream.read(luisa::span<std::byte>(buffer.data(), buffer.size()));

            expect(static_cast<bool>(stream.pos() == test_data.size()));
            expect(static_cast<bool>(std::memcmp(buffer.data(), test_data.data(), test_data.size()) == 0));
            LUISA_INFO("Test 3 passed: Full file read correctly");
        }

        // Test 4: Partial read
        {
            LUISA_INFO("Test 4: Partial read...");
            BinaryFileStream stream(luisa::string{test_file_path});
            const size_t partial_size = 64;
            luisa::vector<std::byte> buffer(partial_size);
            stream.read(luisa::span<std::byte>(buffer.data(), buffer.size()));

            expect(static_cast<bool>(stream.pos() == partial_size));
            expect(static_cast<bool>(std::memcmp(buffer.data(), test_data.data(), partial_size) == 0));
            LUISA_INFO("Test 4 passed: Partial read correctly, pos = {}", stream.pos());
        }

        // Test 5: Multiple sequential reads
        {
            LUISA_INFO("Test 5: Multiple sequential reads...");
            BinaryFileStream stream(luisa::string{test_file_path});
            const size_t chunk_size = 32;
            luisa::vector<std::byte> buffer(chunk_size);

            for (size_t offset = 0; offset < test_data.size(); offset += chunk_size) {
                stream.read(luisa::span<std::byte>(buffer.data(), buffer.size()));
                size_t expected_pos = std::min(offset + chunk_size, test_data.size());
                expect(static_cast<bool>(stream.pos() == expected_pos));
            }
            LUISA_INFO("Test 5 passed: Sequential reads work correctly");
        }

        // Test 6: set_pos - seek to arbitrary position
        {
            LUISA_INFO("Test 6: set_pos - seek to arbitrary position...");
            BinaryFileStream stream(luisa::string{test_file_path});

            // Seek to middle
            stream.set_pos(128);
            expect(static_cast<bool>(stream.pos() == 128));

            std::byte buffer[16];
            stream.read(luisa::span<std::byte>(buffer, 16));
            expect(static_cast<bool>(stream.pos() == 144));
            expect(static_cast<bool>(buffer[0] == static_cast<std::byte>(128)));

            // Seek to beginning
            stream.set_pos(0);
            expect(static_cast<bool>(stream.pos() == 0));

            // Seek to end
            stream.set_pos(test_data.size());
            expect(static_cast<bool>(stream.pos() == test_data.size()));

            LUISA_INFO("Test 6 passed: Seek operations work correctly");
        }

        // Test 7: Read beyond end of file
        {
            LUISA_INFO("Test 7: Read beyond end of file...");
            BinaryFileStream stream(luisa::string{test_file_path});
            stream.set_pos(200);

            // Try to read more than available (only 56 bytes left)
            luisa::vector<std::byte> buffer(100);
            stream.read(luisa::span<std::byte>(buffer.data(), buffer.size()));

            expect(static_cast<bool>(stream.pos() == test_data.size()));
            // The read should only fill available bytes (56 bytes from position 200)
            LUISA_INFO("Test 7 passed: Read beyond EOF handled correctly");
        }

        // Test 8: Move constructor
        {
            LUISA_INFO("Test 8: Move constructor...");
            BinaryFileStream stream1(luisa::string{test_file_path});
            expect(static_cast<bool>(stream1.valid()));

            BinaryFileStream stream2(std::move(stream1));
            expect(static_cast<bool>(!stream1.valid()));
            expect(static_cast<bool>(stream2.valid()));
            expect(static_cast<bool>(stream2.length() == test_data.size()));
            expect(static_cast<bool>(stream2.pos() == 0));

            LUISA_INFO("Test 8 passed: Move constructor works correctly");
        }

        // Test 9: Move assignment
        {
            LUISA_INFO("Test 9: Move assignment...");
            BinaryFileStream stream1(luisa::string{test_file_path});
            BinaryFileStream stream2(luisa::string{test_file_path});
            stream2.set_pos(100);

            stream2 = std::move(stream1);
            expect(static_cast<bool>(!stream1.valid()));
            expect(static_cast<bool>(stream2.valid()));
            expect(static_cast<bool>(stream2.pos() == 0));

            LUISA_INFO("Test 9 passed: Move assignment works correctly");
        }

        // Test 10: Self-move assignment (should not crash)
        {
            LUISA_INFO("Test 10: Self-move assignment...");
            BinaryFileStream stream(luisa::string{test_file_path});
            stream = std::move(stream);
            // Should still be valid (or at least not crash)
            LUISA_INFO("Test 10 passed: Self-move assignment handled");
        }

        // Test 11: close() explicit call
        {
            LUISA_INFO("Test 11: Explicit close()...");
            BinaryFileStream stream(luisa::string{test_file_path});
            expect(static_cast<bool>(stream.valid()));

            stream.close();
            expect(static_cast<bool>(!stream.valid()));
            expect(static_cast<bool>(stream.length() == 0));
            expect(static_cast<bool>(stream.pos() == 0));

            // Read after close should be safe (no-op)
            std::byte buffer[16];
            stream.read(luisa::span<std::byte>(buffer, 16));

            LUISA_INFO("Test 11 passed: close() works correctly");
        }

        // Test 12: Destructor closes file (implicit test - if we get here without crashes/handles leaks, it works)
        {
            LUISA_INFO("Test 12: Destructor closes file...");
            {
                BinaryFileStream stream(luisa::string{test_file_path});
                expect(static_cast<bool>(stream.valid()));
            }// Destructor called here
            LUISA_INFO("Test 12 passed: Destructor works correctly");
        }

        // Test 13: Construction from FILE* and length
        {
            LUISA_INFO("Test 13: Construction from FILE* and length...");
            FILE *file = std::fopen(test_file_path, "rb");
            expect(static_cast<bool>(file != nullptr));

            // Get file length manually
            std::fseek(file, 0, SEEK_END);
            size_t length = std::ftell(file);
            std::fseek(file, 0, SEEK_SET);

            BinaryFileStream stream(file, length);
            expect(static_cast<bool>(stream.valid()));
            expect(static_cast<bool>(stream.length() == length));

            // Stream will close the file in destructor
            LUISA_INFO("Test 13 passed: FILE* constructor works correctly");
        }

        // Test 14: Read with zero-length buffer
        {
            LUISA_INFO("Test 14: Read with zero-length buffer...");
            BinaryFileStream stream(luisa::string{test_file_path});
            stream.read(luisa::span<std::byte>(static_cast<std::byte *>(nullptr), size_t{0}));
            expect(static_cast<bool>(stream.pos() == 0));
            LUISA_INFO("Test 14 passed: Zero-length read handled correctly");
        }

        // Cleanup
        remove_test_file(test_file_path);
        LUISA_INFO("Test file cleaned up");

        LUISA_INFO("All BinaryFileStream tests passed!");
    };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}

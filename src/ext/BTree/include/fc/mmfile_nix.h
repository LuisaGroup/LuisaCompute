#ifndef FC_MMFILE_NIX_H
#define FC_MMFILE_NIX_H

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace frozenca {

class MemoryMappedFileImpl {
 public:
  static inline constexpr std::size_t new_file_size_ = (1UL << 20UL);
  using handle_type = int;
  using path_type = std::filesystem::path::value_type;

 private:
  const std::filesystem::path path_;
  void *data_ = nullptr;
  std::size_t size_ = 0;

  handle_type handle_ = 0;
  int flags_ = 0;

 public:
  MemoryMappedFileImpl(const std::filesystem::path &path,
                       std::size_t init_file_size = new_file_size_,
                       bool trunc = false)
      : path_{path} {
    bool exists = std::filesystem::exists(path);
    if (exists && trunc) {
      std::filesystem::remove(path);
      exists = false;
    }
    open_file(path.c_str(), exists, init_file_size);
    map_file();
  }

  ~MemoryMappedFileImpl() noexcept {
    if (!data_) {
      return;
    }
    bool error = false;
    error = !unmap_file() || error;
    error = !close_file() || error;
  }

 private:
  void open_file(const path_type *path, bool exists,
                 std::size_t init_file_size) {
    flags_ = O_RDWR;
    if (!exists) {
      flags_ |= (O_CREAT | O_TRUNC);
    }
#ifdef _LARGEFILE64_SOURCE
    flags_ |= O_LARGEFILE;
#endif
    errno = 0;
    handle_ = open(path, flags_, S_IRWXU);
    if (errno != 0) {
      throw std::runtime_error("file open failed\n");
    }

    if (!exists) {
      if (ftruncate(handle_, init_file_size) == -1) {
        throw std::runtime_error("failed setting file size\n");
      }
    }

    struct stat info {};
    bool success = (fstat(handle_, &info) != -1);
    size_ = info.st_size;
    if (!success) {
      throw std::runtime_error("failed querying file size\n");
    }
  }

  void map_file() {
    void *data =
        mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, handle_, 0);
    if (data == reinterpret_cast<void *>(-1)) {
      throw std::runtime_error("failed mapping file");
    }
    data_ = data;
  }

  bool close_file() noexcept {
    return close(handle_) == 0;
  }

  bool unmap_file() noexcept {
    return (munmap(data_, size_) == 0);
  }

 public:
  void resize(std::size_t new_size) {
    if (!data_) {
      throw std::runtime_error("file is closed\n");
    }
    if (!unmap_file()) {
      throw std::runtime_error("failed unmappping file\n");
    }
    if (ftruncate(handle_, new_size) == -1) {
      throw std::runtime_error("failed resizing mapped file\n");
    }
    size_ = static_cast<std::size_t>(new_size);
    map_file();
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  [[nodiscard]] void *data() noexcept { return data_; }

  [[nodiscard]] const void *data() const noexcept { return data_; }

  friend bool operator==(const MemoryMappedFileImpl &mmfile1,
                         const MemoryMappedFileImpl &mmfile2) {
    auto res =
        (mmfile1.path_ == mmfile2.path_ && mmfile1.data_ == mmfile2.data_ &&
         mmfile1.size_ == mmfile2.size_ && mmfile1.handle_ == mmfile2.handle_ &&
         mmfile1.flags_ == mmfile2.flags_);
    return res;
  }

  friend bool operator!=(const MemoryMappedFileImpl &mmfile1,
                         const MemoryMappedFileImpl &mmfile2) {
    return !(mmfile1 == mmfile2);
  }
};

}  // namespace frozenca

#endif  // FC_MMFILE_NIX_H

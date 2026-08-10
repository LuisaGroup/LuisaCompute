#ifndef FC_MMFILE_WIN_H
#define FC_MMFILE_WIN_H

#include <windows.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace frozenca {

class MemoryMappedFileImpl {
 public:
  static inline constexpr std::size_t new_file_size_ = (1UL << 20UL);
  using handle_type = HANDLE;
  using path_type = std::filesystem::path::value_type;

 private:
  const std::filesystem::path path_;
  void *data_ = nullptr;
  std::size_t size_ = 0;

  handle_type handle_ = 0;
  int flags_ = 0;
  handle_type mapped_handle_ = 0;

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
    DWORD dwDesiredAccess = GENERIC_READ | GENERIC_WRITE;
    DWORD dwCreationDisposition = exists ? OPEN_EXISTING : CREATE_ALWAYS;
    DWORD dwFlagsandAttributes = FILE_ATTRIBUTE_TEMPORARY;
    handle_ = CreateFileW(path, dwDesiredAccess, FILE_SHARE_READ, 0,
                          dwCreationDisposition, dwFlagsandAttributes, 0);
    if (handle_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("file open failed\n");
    }

    if (!exists) {
      LONG sizehigh = (init_file_size >> (sizeof(LONG) * 8));
      LONG sizelow = (init_file_size & 0xffffffff);
      DWORD result = SetFilePointer(handle_, sizelow, &sizehigh, FILE_BEGIN);
      if ((result == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) ||
          !SetEndOfFile(handle_)) {
        throw std::runtime_error("failed setting file size\n");
      }
    }

    typedef BOOL(WINAPI * func)(HANDLE, PLARGE_INTEGER);
    HMODULE hmod = GetModuleHandleA("kernel32.dll");
    func get_size =
        reinterpret_cast<func>(GetProcAddress(hmod, "GetFileSizeEx"));
    if (get_size) {
      LARGE_INTEGER info;
      if (get_size(handle_, &info)) {
        std::int64_t size =
            ((static_cast<std::int64_t>(info.HighPart) << 32) | info.LowPart);
        size_ = static_cast<std::size_t>(size);
      } else {
        throw std::runtime_error("failed querying file size");
      }
    } else {
      DWORD hi = 0;
      DWORD low = 0;
      if ((low = GetFileSize(handle_, &hi)) != INVALID_FILE_SIZE) {
        std::int64_t size = (static_cast<std::int64_t>(hi) << 32) | low;
        size_ = static_cast<std::size_t>(size);
      } else {
        throw std::runtime_error("failed querying file size");
        return;
      }
    }
  }

  void map_file() {
    DWORD protect = PAGE_READWRITE;
    mapped_handle_ = CreateFileMappingA(handle_, 0, protect, 0, 0, 0);
    if (!mapped_handle_) {
      throw std::runtime_error("failed mapping file");
    }

    DWORD access = FILE_MAP_WRITE;
    void *data = MapViewOfFileEx(mapped_handle_, access, 0, 0, size_, 0);
    if (!data) {
      throw std::runtime_error("failed mapping file");
    }
    data_ = data;
  }

  bool close_file() noexcept { return CloseHandle(handle_); }

  bool unmap_file() noexcept {
    bool error = false;
    error = !UnmapViewOfFile(data_) || error;
    error = !CloseHandle(mapped_handle_) || error;
    mapped_handle_ = NULL;
    return !error;
  }

 public:
  void resize(std::size_t new_size) {
    if (!data_) {
      throw std::runtime_error("file is closed\n");
    }
    if (!unmap_file()) {
      throw std::runtime_error("failed unmappping file\n");
    }

    std::int64_t offset = SetFilePointer(handle_, 0, 0, FILE_CURRENT);
    if (offset == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) {
      throw std::runtime_error("failed querying file pointer");
    }
    LONG sizehigh = (new_size >> (sizeof(LONG) * 8));
    LONG sizelow = (new_size & 0xffffffff);
    DWORD result = SetFilePointer(handle_, sizelow, &sizehigh, FILE_BEGIN);
    if ((result == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) ||
        !SetEndOfFile(handle_)) {
      throw std::runtime_error("failed resizing mapped file");
    }
    sizehigh = (offset >> (sizeof(LONG) * 8));
    sizelow = (offset & 0xffffffff);
    SetFilePointer(handle_, sizelow, &sizehigh, FILE_BEGIN);

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

    res = res && (mmfile1.mapped_handle_ == mmfile2.mapped_handle_);

    return res;
  }

  friend bool operator!=(const MemoryMappedFileImpl &mmfile1,
                         const MemoryMappedFileImpl &mmfile2) {
    return !(mmfile1 == mmfile2);
  }
};

}  // namespace frozenca

#endif  // FC_MMFILE_WIN_H

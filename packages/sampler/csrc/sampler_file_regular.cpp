#define PYSAMPLER_SAMPLER_CPP_IMPL

#include "sampler_file_regular.h"

#include <cerrno>
#include <cstring>  // strerror
#include <stdexcept>
#include <string>

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
#endif

namespace pysampler {

#if defined(_WIN32)

RegularGridFile::RegularGridFile(const std::string& filename) {
  HANDLE h = CreateFileA(filename.c_str(),
                         GENERIC_READ,
                         FILE_SHARE_READ,
                         nullptr,
                         OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                         nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("RegularGridFile: failed to open " + filename);
  }

  LARGE_INTEGER size{};
  if (!GetFileSizeEx(h, &size)) {
    CloseHandle(h);
    throw std::runtime_error("RegularGridFile: failed to query file size for " + filename);
  }

  if (size.QuadPart == 0) {
    CloseHandle(h);
    throw std::runtime_error("RegularGridFile: file is empty: " + filename);
  }

  HANDLE mapping = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping) {
    CloseHandle(h);
    throw std::runtime_error("RegularGridFile: CreateFileMapping failed for " + filename);
  }

  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, (SIZE_T)size.QuadPart);
  if (!view) {
    CloseHandle(mapping);
    CloseHandle(h);
    throw std::runtime_error("RegularGridFile: MapViewOfFile failed for " + filename);
  }

  m_handle  = h;
  m_mapping = mapping;
  m_data    = (const char*)view;
  m_size    = (size_t)size.QuadPart;
}

RegularGridFile::~RegularGridFile() {
  if (m_data) {
    UnmapViewOfFile((LPCVOID)m_data);
    m_data = nullptr;
  }
  if (m_mapping) {
    CloseHandle((HANDLE)m_mapping);
    m_mapping = nullptr;
  }
  if (m_handle) {
    CloseHandle((HANDLE)m_handle);
    m_handle = nullptr;
  }
  m_size = 0;
}

RegularGridFile::RegularGridFile(RegularGridFile&& other) noexcept
  : m_handle(other.m_handle)
  , m_mapping(other.m_mapping)
  , m_data(other.m_data)
  , m_size(other.m_size)
{
  other.m_handle  = nullptr;
  other.m_mapping = nullptr;
  other.m_data    = nullptr;
  other.m_size    = 0;
}

RegularGridFile& RegularGridFile::operator=(RegularGridFile&& other) noexcept {
  if (this != &other) {
    this->~RegularGridFile();
    m_handle  = other.m_handle;
    m_mapping = other.m_mapping;
    m_data    = other.m_data;
    m_size    = other.m_size;
    other.m_handle  = nullptr;
    other.m_mapping = nullptr;
    other.m_data    = nullptr;
    other.m_size    = 0;
  }
  return *this;
}

void RegularGridFile::pread_into(size_t byte_offset, void* dst, size_t nbytes) const {
  if (!m_handle) {
    throw std::runtime_error("RegularGridFile: file is not open");
  }
  if (byte_offset + nbytes > m_size) {
    throw std::runtime_error("RegularGridFile: read past end of file");
  }

  size_t total = 0;
  while (total < nbytes) {
    OVERLAPPED ol{};
    LARGE_INTEGER off;
    off.QuadPart = (LONGLONG)(byte_offset + total);
    ol.Offset     = off.LowPart;
    ol.OffsetHigh = (DWORD)off.HighPart;

    const size_t want = nbytes - total;
    const DWORD  to_read = (DWORD)((want > MAXDWORD) ? MAXDWORD : want);
    DWORD got = 0;
    if (!ReadFile((HANDLE)m_handle, (char*)dst + total, to_read, &got, &ol)) {
      throw std::runtime_error("RegularGridFile: ReadFile failed");
    }
    if (got == 0) {
      throw std::runtime_error("RegularGridFile: unexpected EOF during ReadFile");
    }
    total += got;
  }
}

#else // POSIX

RegularGridFile::RegularGridFile(const std::string& filename) {
  const int fd = ::open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("RegularGridFile: failed to open " + filename);
  }

  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("RegularGridFile: fstat failed for " + filename);
  }

  if (st.st_size <= 0) {
    ::close(fd);
    throw std::runtime_error("RegularGridFile: file is empty: " + filename);
  }

  void* map = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    ::close(fd);
    throw std::runtime_error("RegularGridFile: mmap failed for " + filename);
  }

  // Hint that we will read the volume randomly; the kernel may pre-fetch
  // and reduces SIGBUS exposure for transient I/O hiccups by warming the
  // page cache eagerly.  Best-effort: ignore failures.
  (void)::madvise(map, (size_t)st.st_size, MADV_WILLNEED);

  m_fd   = fd;
  m_data = (const char*)map;
  m_size = (size_t)st.st_size;
}

RegularGridFile::~RegularGridFile() {
  if (m_data) {
    ::munmap((void*)m_data, m_size);
    m_data = nullptr;
  }
  if (m_fd >= 0) {
    ::close(m_fd);
    m_fd = -1;
  }
  m_size = 0;
}

RegularGridFile::RegularGridFile(RegularGridFile&& other) noexcept
  : m_fd(other.m_fd)
  , m_data(other.m_data)
  , m_size(other.m_size)
{
  other.m_fd   = -1;
  other.m_data = nullptr;
  other.m_size = 0;
}

RegularGridFile& RegularGridFile::operator=(RegularGridFile&& other) noexcept {
  if (this != &other) {
    this->~RegularGridFile();
    m_fd   = other.m_fd;
    m_data = other.m_data;
    m_size = other.m_size;
    other.m_fd   = -1;
    other.m_data = nullptr;
    other.m_size = 0;
  }
  return *this;
}

void RegularGridFile::pread_into(size_t byte_offset, void* dst, size_t nbytes) const {
  if (m_fd < 0) {
    throw std::runtime_error("RegularGridFile: file is not open");
  }
  if (byte_offset + nbytes > m_size) {
    throw std::runtime_error("RegularGridFile: read past end of file");
  }

  size_t total = 0;
  while (total < nbytes) {
    const ssize_t n = ::pread(m_fd,
                              (char*)dst + total,
                              nbytes - total,
                              (off_t)(byte_offset + total));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(
        std::string("RegularGridFile: pread failed: ") + std::strerror(errno)
      );
    }
    if (n == 0) {
      throw std::runtime_error("RegularGridFile: unexpected EOF during pread");
    }
    total += (size_t)n;
  }
}

#endif

} // namespace pysampler

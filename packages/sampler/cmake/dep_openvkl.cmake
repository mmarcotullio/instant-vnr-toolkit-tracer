cmake_minimum_required(VERSION 3.24)
include(FetchContent)

option(ENABLE_OPENVKL "Enable OpenVKL CPU sampling backend" OFF)
if(NOT ENABLE_OPENVKL)
  return()
endif()

# OpenVDB is a sub-feature of OpenVKL: it only exists in this build to feed
# OpenVKL's `openvkl_utility_vdb` (the .vdb loader bridge consumed by
# csrc/sampler_openvkl.cpp).  Keeping the option declared here makes the
# `ENABLE_OPENVDB` ⇒ `ENABLE_OPENVKL` dependency a property of CMake's
# control flow rather than a runtime check.
option(ENABLE_OPENVDB "Enable OpenVDB .vdb file loading via openvkl_utility_vdb" OFF)

# All four versions are pinned to OpenVKL v2.0.2's superbuild defaults
# (https://github.com/RenderKit/openvkl/blob/v2.0.2/superbuild/CMakeLists.txt)
# so the rkcommon/Embree/OpenVDB stack matches a single tested upstream release.
set(OPENVKL_VERSION  "2.0.2"  CACHE STRING "OpenVKL version")
set(RKCOMMON_VERSION "1.15.2" CACHE STRING "rkcommon version")
set(EMBREE_VERSION   "4.4.0"  CACHE STRING "Embree version")
set(OPENVDB_VERSION  "12.1.1" CACHE STRING "OpenVDB version")

# TBB is required by rkcommon (tasking system) and OpenVDB (mandatory).
find_package(TBB CONFIG QUIET)
if(NOT TARGET TBB::tbb)
  message(FATAL_ERROR "ENABLE_OPENVKL requires TBB. Install libtbb-dev or set TBB_DIR.")
endif()

# ── rkcommon ──────────────────────────────────────────────────────────────────
set(RKCOMMON_TASKING_SYSTEM "TBB" CACHE STRING "" FORCE)
set(BUILD_TESTING           OFF   CACHE BOOL   "" FORCE)
FetchContent_Declare(rkcommon
  GIT_REPOSITORY  https://github.com/ospray/rkcommon.git
  GIT_TAG         v${RKCOMMON_VERSION}
  GIT_SHALLOW     ON
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(rkcommon)
if(TARGET rkcommon AND NOT TARGET rkcommon::rkcommon)
  add_library(rkcommon::rkcommon ALIAS rkcommon)
endif()

# ── embree ────────────────────────────────────────────────────────────────────
set(EMBREE_ISPC_SUPPORT      OFF CACHE BOOL   "" FORCE)
set(EMBREE_TUTORIALS         OFF CACHE BOOL   "" FORCE)
set(EMBREE_TESTING_INTENSITY 0   CACHE STRING "" FORCE)
set(EMBREE_STATIC_RUNTIME    OFF CACHE BOOL   "" FORCE)
FetchContent_Declare(embree
  GIT_REPOSITORY  https://github.com/RenderKit/embree.git
  GIT_TAG         v${EMBREE_VERSION}
  GIT_SHALLOW     ON
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(embree)

# ── openvdb (optional, must precede openvkl) ──────────────────────────────────
# Declared *before* OpenVKL so OpenVKL's `utility/vdb/CMakeLists.txt` can pick
# it up via its own `find_package(OpenVDB 7.0.0 COMPONENTS openvdb)` call.
# `OVERRIDE_FIND_PACKAGE` makes CMake auto-redirect that find_package to our
# FetchContent build (CMake ≥3.24 generates a minimal OpenVDBConfig and
# OpenVDBConfigVersion in CMAKE_FIND_PACKAGE_REDIRECTS_DIR).
if(ENABLE_OPENVDB)
  # 1. Prefer a system install (matches system TBB/Blosc/Boost ABI; saves
  #    several minutes of subproject build).
  find_package(OpenVDB CONFIG QUIET)
  if(TARGET OpenVDB::openvdb)
    message(STATUS "Found system OpenVDB ${OpenVDB_VERSION}; skipping FetchContent build.")
  else()
    # 2. Otherwise build from source. OpenVDB's hard runtime deps are TBB
    #    (already checked above), zlib (system), and Blosc (optional but
    #    strongly recommended — virtually every real-world .vdb is Blosc
    #    compressed).
    find_package(ZLIB QUIET)
    if(NOT TARGET ZLIB::ZLIB)
      message(FATAL_ERROR "ENABLE_OPENVDB requires zlib. Install zlib1g-dev.")
    endif()

    # Blosc has no Config package on most distros — probe the library directly.
    find_library(_openvdb_blosc_lib NAMES blosc)
    if(_openvdb_blosc_lib)
      set(_openvdb_use_blosc ON)
      message(STATUS "OpenVDB: enabling Blosc support (${_openvdb_blosc_lib})")
    else()
      set(_openvdb_use_blosc OFF)
      message(WARNING
        "libblosc not found; OpenVDB will be built without Blosc compression. "
        "Most real-world .vdb files will fail to load — install libblosc-dev.")
    endif()
    unset(_openvdb_blosc_lib CACHE)

    # Trim the OpenVDB build to just the shared core library.  Order matches
    # OpenVDB's own CMakeLists.txt for easy diffing.
    set(OPENVDB_BUILD_CORE                      ON                    CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_BINARIES                  OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_DOCS                      OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_UNITTESTS                 OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_PYTHON_MODULE             OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_AX                        OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_NANOVDB                   OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_VDB_PRINT                 OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_VDB_LOD                   OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_VDB_RENDER                OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_VDB_VIEW                  OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_BUILD_VDB_TOOL                  OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_CORE_SHARED                     ON                    CACHE BOOL "" FORCE)
    set(OPENVDB_CORE_STATIC                     OFF                   CACHE BOOL "" FORCE)
    # Delayed loading is the only OpenVDB-12 feature that still pulls in Boost.
    # We don't need it (sampler_openvkl.cpp eagerly reads the whole grid).
    set(OPENVDB_USE_DELAYED_LOADING             OFF                   CACHE BOOL "" FORCE)
    set(OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING  ON                    CACHE BOOL "" FORCE)
    # Don't define a top-level `uninstall` custom target — Embree (also
    # FetchContent'd above) defines one with the same name, which CMake refuses
    # under policy CMP0002.
    set(OPENVDB_ENABLE_UNINSTALL                OFF                   CACHE BOOL "" FORCE)
    # CONCURRENT_MALLOC only links into OpenVDB binaries / unit tests, both of
    # which are off above.  Setting it to None skips the find_package(Jemalloc)
    # probe and silences the "Unable to find Jemalloc" warning.
    set(CONCURRENT_MALLOC                       "None"                CACHE STRING "" FORCE)
    # Don't generate explicit instantiations — they slow the OpenVDB build
    # significantly and we only ever instantiate FloatGrid / Vec3SGrid.
    set(USE_EXPLICIT_INSTANTIATION              OFF                   CACHE BOOL "" FORCE)
    set(USE_IMATH_HALF                          OFF                   CACHE BOOL "" FORCE)
    set(USE_PKGCONFIG                           OFF                   CACHE BOOL "" FORCE)
    set(USE_TBB                                 ON                    CACHE BOOL "" FORCE)
    set(USE_ZLIB                                ON                    CACHE BOOL "" FORCE)
    set(USE_BLOSC                               ${_openvdb_use_blosc} CACHE BOOL "" FORCE)

    FetchContent_Declare(openvdb
      GIT_REPOSITORY  https://github.com/AcademySoftwareFoundation/openvdb.git
      GIT_TAG         v${OPENVDB_VERSION}
      GIT_SHALLOW     ON
      OVERRIDE_FIND_PACKAGE
    )
    FetchContent_MakeAvailable(openvdb)

    # OpenVDB FetchContent exports `openvdb_shared` (and/or `openvdb_static`)
    # but does NOT create the `OpenVDB::openvdb` IMPORTED alias that
    # find_package(OpenVDB) would.  Add it here so OpenVKL's
    # `target_link_libraries(openvkl_utility_vdb INTERFACE OpenVDB::openvdb)`
    # resolves correctly when the auto-redirect kicks in.
    if(NOT TARGET OpenVDB::openvdb)
      if(TARGET openvdb_shared)
        add_library(OpenVDB::openvdb ALIAS openvdb_shared)
      elseif(TARGET openvdb)
        add_library(OpenVDB::openvdb ALIAS openvdb)
      else()
        message(FATAL_ERROR
          "OpenVDB FetchContent succeeded but no openvdb / openvdb_shared "
          "target was created — this likely means OPENVDB_BUILD_CORE was overridden.")
      endif()
    endif()
  endif()

  # OpenVKL's `utility/vdb/CMakeLists.txt` only builds the OpenVDB code path
  # when `OpenVDB_ROOT` is DEFINED.  We don't actually use the value (the
  # find_package(OpenVDB) call inside OpenVKL is satisfied by our
  # OVERRIDE_FIND_PACKAGE redirect) — we just need the variable defined.
  set(OpenVDB_ROOT "${openvdb_BINARY_DIR}" CACHE PATH "" FORCE)
endif()

# ── openvkl ───────────────────────────────────────────────────────────────────
set(OPENVKL_DEVICE_GPU OFF CACHE BOOL "" FORCE)
set(OPENVKL_DEVICE_CPU ON  CACHE BOOL "" FORCE)
set(BUILD_TESTING      OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS   OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
set(BUILD_TOOLS        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(openvkl
  GIT_REPOSITORY  https://github.com/openvkl/openvkl.git
  GIT_TAG         v${OPENVKL_VERSION}
  GIT_SHALLOW     ON
)
FetchContent_MakeAvailable(openvkl)
if(TARGET openvkl AND NOT TARGET openvkl::openvkl)
  add_library(openvkl::openvkl ALIAS openvkl)
endif()
if(TARGET openvkl_module_cpu_device AND NOT TARGET openvkl::openvkl_module_cpu_device)
  add_library(openvkl::openvkl_module_cpu_device ALIAS openvkl_module_cpu_device)
endif()

foreach(_w 4 8 16)
  if(TARGET openvkl_module_cpu_device_${_w})
    target_include_directories(openvkl_module_cpu_device_${_w}
      PUBLIC $<BUILD_INTERFACE:${openvkl_SOURCE_DIR}>)
  endif()
endforeach()
unset(_w)

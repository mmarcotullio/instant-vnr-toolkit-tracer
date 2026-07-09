include(FetchContent)

option(ENABLE_WITCHER "Enable ExaBrick/ExaStitch sampling backend" OFF)
if(NOT ENABLE_WITCHER)
  return()
endif()

include(configure_optix)

# ── owlExaStitcher ────────────────────────────────────────────────────────────
set(EXA_STANDALONE_PROJECT OFF CACHE BOOL "" FORCE)
set(OWL_BUILD_SHARED       ON  CACHE BOOL "" FORCE)
set(OWL_BUILD_SAMPLES      OFF CACHE BOOL "" FORCE)
set(TBB_LIBRARIES TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)

# owl's box.h uses uint64_t without including <cstdint> (upstream bug)
string(APPEND CMAKE_CXX_FLAGS " -include cstdint")

FetchContent_Declare(witcher
  GIT_REPOSITORY  https://github.com/wilsonCernWq/owlExaStitcher.git
  GIT_TAG         main
)
FetchContent_MakeAvailable(witcher)

# visionaray (witcher submodule) uses if constexpr; force C++17 on all witcher
# targets so Clang doesn't warn about [-Wc++17-extensions].
foreach(_t witcher_core owl umesh)
  if(TARGET ${_t})
    target_compile_features(${_t} PUBLIC cxx_std_17)
  endif()
endforeach()
unset(_t)

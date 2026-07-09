cmake_minimum_required(VERSION 3.24)  # OVERRIDE_FIND_PACKAGE requires 3.24
include(FetchContent)


# ------------------------------------------------------------------
# Witcher
# ------------------------------------------------------------------

set(EXA_STANDALONE_PROJECT OFF CACHE BOOL "Build owl as standalone project" FORCE)
set(OWL_BUILD_SHARED ON CACHE BOOL "Build owl as shared library" FORCE)
set(OWL_BUILD_SAMPLES OFF CACHE BOOL "Build owl samples" FORCE)
set(TBB_LIBRARIES TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)
FetchContent_Declare(witcher
  GIT_REPOSITORY  https://github.com/wilsonCernWq/owlExaStitcher.git
  GIT_TAG         main
)
FetchContent_MakeAvailable(witcher)

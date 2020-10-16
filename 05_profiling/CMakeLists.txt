cmake_minimum_required(VERSION 3.8) # CUDA language support

project(template LANGUAGES C CXX CUDA)
find_package(CUDA REQUIRED)

set(CUDA_PROGRAM_LIST
  sgmemm
  sgmemm_man
  )

set(GCC_FLAGS "-lgomp")
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_FLAGS}")
# This may vary across different setups
SET(CUDA_PATH "/usr/local/cuda-10.2/")

if(CUDA_FOUND)
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_53;--compiler-options;-std=c++11;-O3;-DVERBOSE;-mcmodel=medium;-lgomp")
endif(CUDA_FOUND)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

############################################################
# Create executables
############################################################

foreach(PROGR ${CUDA_PROGRAM_LIST})
  set(MyProgram ${PROGR})
  add_executable(${MyProgram} ${PROJECT_SOURCE_DIR}/src/${PROGR}.cu)
  set_target_properties(${MyProgram} PROPERTIES COMPILE_FLAGS "${GCC_FLAGS}")
  target_include_directories(${MyProgram} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_PATH}/samples/common/inc)
  target_link_libraries(${MyProgram} PRIVATE "-lgomp")
endforeach()


foreach(PROGR ${CUDA_PROGRAM_LIST})
  target_compile_options(${PROGR}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
    -Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:MSVC>:
    /W4>>)

  set_target_properties(${PROGR}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
endforeach()

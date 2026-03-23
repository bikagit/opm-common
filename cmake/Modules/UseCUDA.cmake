if(CMAKE_BUILD_TYPE)
  set(_flags_suffix "_${CMAKE_BUILD_TYPE}")
endif()
if(NOT DEFINED ENV{CUDAHOSTCXX} AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND
    (NOT CMAKE_CUDA_FLAGS${_flags_suffix} OR NOT CMAKE_CUDA_FLAGS${_flags_suffix} MATCHES ".*-ccbin .*"))
  message(STATUS "Setting CUDA host compiler CMAKE_CUDA_HOST_COMPILER to ${CMAKE_CXX_COMPILER} to "
    "prevent incompatibilities. Note that this might report that there "
    "is not CUDA compiler if your system's CUDA compiler does not support "
    "${CMAKE_CXX_COMPILER}.")
  # check_language does not seem to care about ${CMAKE_CUDA_FLAGS} or $(CUDA_NVCC_FLAGS}.
  # Hence we set CMAKE_CUDA_HOST_COMPILER to our C++ compiler.
  # In check_language(CUDA) we will get an error if we in addition put
  # "-ccbin ${CMAKE_CXX_COMPILER}" into CMAKE_CUDA_FLAGS. It results
  # in "${NVCC} -ccbin=${CMAKE_CXX_COMPILER}   -ccbin ${CMAKE_CXX_COMPILER}"
  # which causes nvcc to abort
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(ENV{CUDAHOSTCXX} ${CMAKE_CUDA_HOST_COMPILER}) # The only thing honored by check_language(CUDA)!
endif()
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # OPTIONAL is ignored. Hence the magic above to check whether enabling CUDA works
  enable_language(CUDA)
  set(CUDA_FOUND ON)
endif()
if(CUDA_FOUND AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
  set(CUDA_FOUND OFF)
  message(WARNING
    "Deactivating CUDA as we require version 9.0 or newer."
    " Found only CUDA version ${CUDA_VERSION}."
  )
endif()

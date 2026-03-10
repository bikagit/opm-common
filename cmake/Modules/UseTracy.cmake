option(USE_TRACY_PROFILER "Enable tracy profiling" OFF)

if(USE_TRACY_PROFILER)
  find_package(Tracy)
endif()

function(use_tracy)
  cmake_parse_arguments(PARAM "" "TARGET" "" ${ARGN})
  if(NOT PARAM_TARGET)
    message(FATAL_ERROR "Function needs a TARGET parameter")
  endif()

  if(TARGET Tracy::TracyClient)
    target_link_libraries(${PARAM_TARGET} PUBLIC Tracy::TracyClient)
    target_compile_definitions(${PARAM_TARGET} PUBLIC USE_TRACY=1)
  endif()
endfunction()

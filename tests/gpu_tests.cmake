add_executable(
  gpu_tests
  gpu_tests.c
)

add_test(
  NAME gpu_tests
  COMMAND gpu_tests
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

target_link_libraries(gpu_tests PRIVATE unity)
target_link_libraries(gpu_tests PRIVATE merge_sort_gpu_headers)
target_link_libraries(gpu_tests PRIVATE wgpu_native)

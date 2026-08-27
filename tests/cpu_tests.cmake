add_executable(
  cpu_tests
  cpu_tests.c
)

add_test(
  NAME cpu_tests
  COMMAND cpu_tests
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

target_link_libraries(cpu_tests PRIVATE unity)
target_link_libraries(cpu_tests PRIVATE merge_sort_cpu_headers)

cmake_minimum_required(VERSION 3.16)
project(PBCE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Threads REQUIRED)

# The bundled Abseil version predates CMake 4's policy compatibility floor.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "Build Abseil tests" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Use the project C++ standard" FORCE)
add_subdirectory(src/open_spiel/abseil-cpp EXCLUDE_FROM_ALL)

# Object libraries preserve the static registration of every bundled game.
add_library(pbce_core OBJECT
  src/hr_edl/samplers.cc
  src/hr_edl/spiel_extra.cc
  src/hr_edl/policy_evaluation.cc
  src/hr_edl/decision_point.cc
  src/hr_edl/best_response.cc
  src/hr_edl/math.cc
  src/open_spiel/action_view.cc
  src/open_spiel/game_parameters.cc
  src/open_spiel/matrix_game.cc
  src/open_spiel/observer.cc
  src/open_spiel/policy.cc
  src/open_spiel/simultaneous_move_game.cc
  src/open_spiel/spiel.cc
  src/open_spiel/spiel_utils.cc
  src/open_spiel/algorithms/minimax.cc
  src/open_spiel/utils/file.cc
  src/open_spiel/game_transforms/turn_based_simultaneous_game.cc
  src/open_spiel/games/kuhn_poker.cc
  src/open_spiel/games/leduc_poker.cc
  src/open_spiel/games/goofspiel.cc
  src/open_spiel/games/sheriff.cc
  src/open_spiel/games/tiny_bridge.cc
  src/open_spiel/games/tic_tac_toe.cc
  src/open_spiel/games/tiny_hanabi.cc
  src/open_spiel/games/battleship.cc
  src/open_spiel/games/battleship_types.cc
  src/open_spiel/games/efg_game.cc
)
target_include_directories(pbce_core PUBLIC src src/open_spiel/abseil-cpp)
target_link_libraries(pbce_core PUBLIC
  absl::flags absl::flags_parse absl::flat_hash_map absl::optional
  absl::random_random absl::str_format absl::strings absl::time Threads::Threads
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(pbce_core PUBLIC stdc++fs)
endif()

add_executable(run_simultaneous_ltbr_pbce src/bin/run_simultaneous_ltbr_pbce.cc)
target_link_libraries(run_simultaneous_ltbr_pbce PRIVATE pbce_core)

include(CTest)
if(BUILD_TESTING)
  foreach(test_name action_transformation decision_point policy_evaluation best_response)
    add_executable(${test_name}_test src/hr_edl/${test_name}_test.cc
      src/open_spiel/games/liars_dice.cc)
    target_link_libraries(${test_name}_test PRIVATE pbce_core)
    add_test(NAME ${test_name} COMMAND ${test_name}_test)
  endforeach()
  target_sources(best_response_test PRIVATE
    src/open_spiel/algorithms/tabular_exploitability.cc
    src/open_spiel/algorithms/best_response.cc
    src/open_spiel/algorithms/expected_returns.cc
    src/open_spiel/algorithms/history_tree.cc
  )
  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)
  add_test(NAME python_runner
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s tests -p test_*.py)
  set_tests_properties(python_runner PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME pbce_smoke COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_pbce.py
    --exe $<TARGET_FILE:run_simultaneous_ltbr_pbce>)
  set_tests_properties(pbce_smoke PROPERTIES TIMEOUT 120)
endif()

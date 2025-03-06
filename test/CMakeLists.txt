set(SMILES_EXECUTABLE "$<TARGET_FILE:smiles-zip>")
set(pythonWrapper ${CMAKE_COMMAND};-E;env;PYTHONPATH=${PROJECT_SOURCE_DIR};${Python_EXECUTABLE})

# Tests CPU version
add_test(NAME cpu-test-compress
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests.py
  --test_data ${TEST_DATA}
  --executable ${SMILES_EXECUTABLE}
  --compress
)

add_test(NAME cpu-test-decompress
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests.py
  --test_data ${TEST_DATA}
  --executable ${SMILES_EXECUTABLE}
  --decompress
)

if(ENABLE_CUDA)
  add_test(NAME cuda-test-compress
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests.py
    --test_data ${TEST_DATA}
    --executable ${SMILES_EXECUTABLE}
    --compress
    --cuda
  )
  add_test(NAME cuda-test-decompress
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests.py
    --test_data ${TEST_DATA}
    --executable ${SMILES_EXECUTABLE}
    --decompress
    --cuda
  )
endif()

if(ENABLE_HIP)
  add_test(NAME cuda-test-compress
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests.py
    --test_data ${TEST_DATA}
    --executable ${SMILES_EXECUTABLE}
    --compress
    --hip
  )
  add_test(NAME cuda-test-decompress
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests.py
    --test_data ${TEST_DATA}
    --executable ${SMILES_EXECUTABLE}
    --decompress
    --hip
  )
endif()

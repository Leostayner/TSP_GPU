cmake_minimum_required(VERSION 3.9)
project (Projeto3 LANGUAGES CXX CUDA)

set(SOURCE_FILES_RDS
    "random-sol.cu"
    )

set(SOURCE_FILES_OP2
    "2-opt-sol.cu"
    )

add_executable(random-sol ${SOURCE_FILES_RDS})
target_compile_options(random-sol PRIVATE -std=c++11 -O3)

add_executable(2-opt-sol ${SOURCE_FILES_OP2})
target_compile_options(2-opt-sol PRIVATE -std=c++11 -O3)
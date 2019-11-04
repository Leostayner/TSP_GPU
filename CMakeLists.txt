cmake_minimum_required(VERSION 3.9)
project (Projeto3 LANGUAGES CXX CUDA)

set(SOURCE_FILES_RDS
    "random_sol.cu"
    )

set(SOURCE_FILES_OP2
    "random_sol_op2.cu"
    )

add_executable(random_sol ${SOURCE_FILES_RDS})
target_compile_options(random_sol PRIVATE -std=c++11 -O3)


add_executable(random_sol_op2 ${SOURCE_FILES_OP2})
target_compile_options(random_sol_op2 PRIVATE -std=c++11 -O3)
################################################################################
# check-dfg-mlir
#
# The dfg-mlir regression test project.
################################################################################

project(check-dfg-mlir)

# Configure the testing site configuration.
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
set(TEST_DEPENDS
    FileCheck count not
    dfg-opt
)

# Create the test suite.
add_lit_testsuite(${PROJECT_NAME}
    "Running the dfg-mlir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)
add_lit_testsuites(DFGMLIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
    ${TEST_DEPENDS}
)
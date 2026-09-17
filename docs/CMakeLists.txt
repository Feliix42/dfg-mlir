find_package(Doxygen)

if (DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # Resolve a sphinx-build command.
    # Prefer the configured Python virtual environment (Python3_EXECUTABLE) so
    # the correct interpreter is used regardless of the system PATH.
    # Fall back to a system-wide sphinx-build if Python3_EXECUTABLE is not set.
    if (Python3_EXECUTABLE)
        set(SPHINX_CMD ${Python3_EXECUTABLE} -m sphinx)
        message(STATUS "Sphinx will run via: ${Python3_EXECUTABLE} -m sphinx")
    else()
        find_program(SPHINX_BUILD_EXE sphinx-build)
        if (SPHINX_BUILD_EXE)
            set(SPHINX_CMD ${SPHINX_BUILD_EXE})
            message(STATUS "Sphinx will run via: ${SPHINX_BUILD_EXE}")
        endif()
    endif()

    if (SPHINX_CMD)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
            ${CMAKE_CURRENT_BINARY_DIR}/conf.py
            @ONLY)

        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
            ${CMAKE_CURRENT_BINARY_DIR}/index.rst
            COPYONLY)

        # Copy the root README and its logos into the build docs tree so Sphinx
        # can include them.  The images live at docs/LAKSA-{Dark,Light}.png
        # relative to the project root, and README.md's <picture> references
        # them with that same relative path. html_extra_path in conf.py is set
        # to "docs", but Sphinx copies the *contents* of that directory to the
        # root of the HTML output (it does not preserve the "docs" folder name
        # itself). So the source needs an extra docs/ nesting here --
        # ${CMAKE_CURRENT_BINARY_DIR}/docs is the html_extra_path source, and
        # the inner docs/ is what survives the copy, landing at
        # <output>/docs/LAKSA-{Dark,Light}.png as required.
        configure_file(
            ${CMAKE_SOURCE_DIR}/WorkWithDfg.md
            ${CMAKE_CURRENT_BINARY_DIR}/WorkWithDfg.md
            COPYONLY)
        configure_file(
            ${CMAKE_SOURCE_DIR}/Install.md
            ${CMAKE_CURRENT_BINARY_DIR}/Install.md
            COPYONLY)
        configure_file(
            ${CMAKE_SOURCE_DIR}/Design.md
            ${CMAKE_CURRENT_BINARY_DIR}/Design.md
            COPYONLY)
        configure_file(
            ${CMAKE_SOURCE_DIR}/README.md
            ${CMAKE_CURRENT_BINARY_DIR}/README.md
            COPYONLY)

        # configure_file(
        #     ${CMAKE_CURRENT_SOURCE_DIR}/LAKSA-Dark.png
        #     ${CMAKE_CURRENT_BINARY_DIR}/docs/docs/LAKSA-Dark.png
        #     COPYONLY)

        # configure_file(
        #     ${CMAKE_CURRENT_SOURCE_DIR}/LAKSA-Light.png
        #     ${CMAKE_CURRENT_BINARY_DIR}/docs/docs/LAKSA-Light.png
        #     COPYONLY)

        # # Also copy the dark logo next to conf.py itself so Sphinx's own
        # # html_logo (the sidebar logo shown across the generated site) can
        # # find it — that path is resolved relative to conf.py's directory,
        # # not through html_extra_path.
        # configure_file(
        #     ${CMAKE_CURRENT_SOURCE_DIR}/LAKSA-Dark.png
        #     ${CMAKE_CURRENT_BINARY_DIR}/LAKSA-Dark.png
        #     COPYONLY)

        set(SPHINX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

        add_custom_target(build-dfg-doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            # Remove stale Exhale RST files so symbols dropped from the XML
            # do not leave orphaned pages that cause Breathe warnings.
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/api
            COMMAND ${SPHINX_CMD}
                    -b html
                    # Ignore the cached doctree/environment so that changes
                    # to conf.py (e.g. source-read hooks) are reliably
                    # applied even when the underlying .md/.rst sources
                    # (e.g. mlir-tblgen output) haven't changed.
                    -E
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${SPHINX_OUTPUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating documentation (Doxygen XML + Sphinx HTML)"
            VERBATIM)

        message(STATUS
            "Run 'cmake --build . --target build-dfg-doc' to build docs")
    else()
        add_custom_target(build-dfg-doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating Doxygen XML (sphinx not found)"
            VERBATIM)

        message(STATUS
            "sphinx not found — set Python3_EXECUTABLE or install sphinx-build")
    endif()

    add_dependencies(build-dfg-doc mlir-doc)
endif()

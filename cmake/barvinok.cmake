include(ExternalProject)

set(BARVINOK_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/barvinok)
file(MAKE_DIRECTORY ${BARVINOK_ROOT_DIR}/include)

ExternalProject_Add(
        Barvinok
        GIT_REPOSITORY "git://repo.or.cz/barvinok.git"
        GIT_TAG "barvinok-0.41.8"
        GIT_SUBMODULES_RECURSE TRUE
        CONFIGURE_COMMAND cd <SOURCE_DIR> && libtoolize && <SOURCE_DIR>/autogen.sh
        COMMAND <SOURCE_DIR>/configure --prefix=${BARVINOK_ROOT_DIR}
        BUILD_COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
        INSTALL_COMMAND make install
        BUILD_IN_SOURCE TRUE
        BUILD_ALWAYS FALSE
        UPDATE_DISCONNECTED TRUE
        BUILD_BYPRODUCTS "${BARVINOK_ROOT_DIR}/lib/libbarvinok.a;${BARVINOK_ROOT_DIR}/lib/libisl.a;${BARVINOK_ROOT_DIR}/lib/libpolylibgmp.a"
)

add_library(isl STATIC IMPORTED)
add_dependencies(isl Barvinok)
set_target_properties(isl PROPERTIES
        IMPORTED_LOCATION "${BARVINOK_ROOT_DIR}/lib/libisl.a"
)

add_library(polylibgmp STATIC IMPORTED)
add_dependencies(polylibgmp Barvinok)
set_target_properties(polylibgmp PROPERTIES
        IMPORTED_LOCATION "${BARVINOK_ROOT_DIR}/lib/libpolylibgmp.a"
)

add_library(barvinok STATIC IMPORTED)
add_dependencies(barvinok Barvinok)
set_target_properties(barvinok PROPERTIES
        IMPORTED_LOCATION "${BARVINOK_ROOT_DIR}/lib/libbarvinok.a"
        INTERFACE_INCLUDE_DIRECTORIES "${BARVINOK_ROOT_DIR}/include"
        INTERFACE_LINK_LIBRARIES "isl;polylibgmp;ntl;gmp"
)
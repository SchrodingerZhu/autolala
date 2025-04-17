get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_library(SLAPExtractor 
  SHARED
   extractor.cpp    
  LINK_LIBS PUBLIC
    MLIRSupport
    MLIRIR
    ${dialect_libs}
    ${extension_libs}
)

target_include_directories(SLAPExtractor PUBLIC ${MLIR_INCLUDE_DIRS} ../include)
target_link_libraries(SLAPExtractor PUBLIC barvinok)
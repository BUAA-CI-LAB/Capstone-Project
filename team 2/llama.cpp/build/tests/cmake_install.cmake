# Install script for directory: /Users/froggy/Documents/peak/llama.cpp/tests

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-tokenizer-0")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-0" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-0")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-0")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-0")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-sampling")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-sampling" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-sampling")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-sampling")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-sampling")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-grammar-parser")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-parser" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-parser")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-parser")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-parser")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-grammar-integration")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-integration" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-integration")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-integration")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-grammar-integration")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-llama-grammar")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-llama-grammar" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-llama-grammar")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-llama-grammar")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-llama-grammar")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-chat")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-json-schema-to-grammar")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-schema-to-grammar" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-schema-to-grammar")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-schema-to-grammar")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-schema-to-grammar")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-quantize-stats")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-stats" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-stats")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-stats")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-stats")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-gbnf-validator")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-gbnf-validator" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-gbnf-validator")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-gbnf-validator")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-gbnf-validator")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-tokenizer-1-bpe")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-bpe" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-bpe")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-bpe")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-bpe")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-tokenizer-1-spm")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-spm" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-spm")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-spm")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-tokenizer-1-spm")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-chat-parser")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-parser" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-parser")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-parser")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-parser")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-chat-peg-parser")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-peg-parser" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-peg-parser")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-peg-parser")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-peg-parser")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-chat-template")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-template" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-template")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-template")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-chat-template")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-json-partial")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-partial" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-partial")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-partial")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-json-partial")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-log")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-log" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-log")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-log")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-log")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-peg-parser")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-peg-parser" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-peg-parser")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-peg-parser")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-peg-parser")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-regex-partial")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-regex-partial" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-regex-partial")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-regex-partial")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-regex-partial")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-thread-safety")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-thread-safety" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-thread-safety")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-thread-safety")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-thread-safety")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-arg-parser")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-arg-parser" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-arg-parser")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-arg-parser")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-arg-parser")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-opt")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-opt" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-opt")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-opt")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-opt")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-gguf")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-gguf" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-gguf")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-gguf")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-gguf")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-backend-ops")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-backend-ops" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-backend-ops")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-backend-ops")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-backend-ops")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-model-load-cancel")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-model-load-cancel" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-model-load-cancel")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-model-load-cancel")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-model-load-cancel")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-autorelease")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-autorelease" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-autorelease")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-autorelease")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-autorelease")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-barrier")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-barrier" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-barrier")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-barrier")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-barrier")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-quantize-fns")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-fns" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-fns")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-fns")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-fns")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-quantize-perf")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-perf" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-perf")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-perf")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-quantize-perf")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-rope")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-rope" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-rope")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-rope")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-rope")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-mtmd-c-api")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-mtmd-c-api" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-mtmd-c-api")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-mtmd-c-api")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-mtmd-c-api")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/froggy/Documents/peak/llama.cpp/build/bin/test-alloc")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-alloc" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-alloc")
    execute_process(COMMAND /opt/homebrew/anaconda3/bin/install_name_tool
      -delete_rpath "/Users/froggy/Documents/peak/llama.cpp/build/bin"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-alloc")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/test-alloc")
    endif()
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/froggy/Documents/peak/llama.cpp/build/tests/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()

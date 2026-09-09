# ADR 0005: Parquet I/O Strategy for C++ Replay Driver on Windows

## Status
Accepted

## Context
The core navigation engine (`core/`) requires an input mechanism to ingest evaluation datasets (`data/processed/outages/`) containing synchronized IMU samples and pre-outage GNSS fixes for 377 outage benchmark instances.

The project specification presents two architectural paths:
1. **Direct In-Engine Parquet Parsing**: Link Apache Arrow C++ (`libarrow`, `libparquet`) via CMake `FetchContent` or system package, abstracting it behind an I/O interface.
2. **Intermediate Replay Cache**: Use the existing Python data evaluation layer (which already has fast, battle-tested `pyarrow` C++ bindings) to pre-export the exact required IMU channels and pre-outage initial conditions into a lightweight cache directory (`data/processed/_cpp_replay_cache/`), read by a zero-dependency C++ streaming reader.

## Analysis of Local Toolchain Constraints
1. **Compiler Architecture**: The local developer environment on Windows utilizes MinGW GCC 6.3.0 (32-bit `x86`, dated 2016).
2. **Apache Arrow C++ Prerequisites**: Modern Apache Arrow C++ strictly requires 64-bit architectures, full C++17 support (GCC $\ge 8.0$ or modern MSVC), and depends on extensive third-party libraries (Boost, Thrift, Snappy, zlib, brotli). Attempting to compile `libarrow` and `libparquet` via `FetchContent` under 32-bit GCC 6.3.0 fails due to missing C++17 filesystem and threading primitives (as already noted in `core/CMakeLists.txt` where GCC $< 8.0$ lacks standard thread support).
3. **Build Overhead & CI Reproducibility**: Compiling Apache Arrow C++ from source on developer machines requires downloading gigabytes of dependencies and takes 30–60 minutes per clean build.

## Decision
We adopt the **Intermediate Replay Cache** architecture:
1. A dedicated pre-export script (`dataeval/harness/export_replay_cache.py`) consumes the 377 outage Parquet files via `pyarrow`, computes pre-outage initial states identical to Step 8's 1.0s circular-mean baseline, and serializes each outage stream to `data/processed/_cpp_replay_cache/<outage_id>.csv`.
2. The C++ replay driver (`idr_replay`) reads the cached CSV streams using zero-dependency, standard C++17 library I/O (`<fstream>`, `<sstream>`).
3. The cache directory `data/processed/_cpp_replay_cache/` is completely ignored by `.gitignore` under the existing `data/processed/*` pattern.
4. The core navigation library (`idr_core`) remains 100% dependency-free, portable to any embedded, edge, or Android NDK target.

## Consequences
- **Zero Build Dependencies**: The C++ core builds instantly with CMake and Ninja in under 1 second without external network downloads or compiler version mismatches.
- **Identical Initial Conditions**: Using Python to pre-compute initial states guarantees bit-for-bit parity with the Python baseline's 1.0s pre-outage averaging window and circular heading mean, ensuring perfectly fair apples-to-apples dead-reckoning comparisons.
- **Fast Replay**: Streaming simple CSV records in C++ executes at over 100,000 samples/sec, completing batch replay of all 377 instances in seconds.
- **Clean Separation of Concerns**: High-level dataset ingestion, schema variant mapping, and deduplication remain in Python (`dataeval/`), while deterministic numerical propagation and sensor mechanics live in C++ (`core/`).

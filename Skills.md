---
name: vscode-clangd-cuda-cutlass
description: Diagnose and repair VS Code clangd support for CUDA and CUTLASS (.cu/.cuh) files, including language activation, Clang resource directories, CUDA CCCL includes, and cascading template or standard-library diagnostics.
---

# VS Code clangd for CUDA and CUTLASS

Use this workflow when VS Code does not call clangd for `.cu`/`.cuh` files, clangd reports missing CUDA/Clang headers, or CUTLASS headers produce implausible errors such as `Gemm` having too few template arguments.

## Core model

- clangd parses code with its embedded Clang frontend. It does not use `nvcc` as its parser.
- An `nvcc` command in clangd logs is a fallback command template. The real parsing args appear after `internal (cc1) args`.
- `CompileFlags.Compiler: nvcc` asks clangd to query `nvcc` for driver information. It does not make `nvcc` provide completion, navigation, or diagnostics.
- CUDA Toolkit and Clang resource headers are separate dependencies. `nvcc` does not require `__clang_cuda_runtime_wrapper.h`; Clang does.
- Recent CUDA toolkits can store CCCL headers under `targets/<target>/include/cccl`. A missing `cuda/std/utility` header can make CUTLASS template parsing fail before `Gemm` is declared correctly.

## Discover environment-specific paths

Do not hard-code usernames, checkout locations, or host architecture in a committed skill. Discover them from the active toolchain:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
CLANGD_BIN="$(command -v clangd)"
CLANG_RESOURCE_DIR="$(clang -print-resource-dir)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CCCL_INCLUDE="$(find "$CUDA_HOME/targets" -type d -path '*/include/cccl' -print -quit)"
```

Verify that every discovered value is non-empty before using it. A resource directory normally contains:

```text
include/__clang_cuda_runtime_wrapper.h
```

## Diagnose in this order

### 1. Verify that clangd is activated for the file

Open the `.cu` file and check the VS Code language mode. It should be `CUDA C++` or another clangd-supported language ID.

The clangd extension activates for `c`, `cpp`, `cuda-cpp`, Objective-C, and Objective-C++. If `.cu` remains plaintext, no LSP request reaches clangd.

Create or update `.vscode/settings.json`:

```json
{
  "files.associations": {
    "*.cu": "cuda-cpp",
    "*.cuh": "cuda-cpp"
  }
}
```

After changing the setting, reopen the file or reload the VS Code window.

To confirm activation, inspect the current extension-host log for:

```text
ExtensionService#_doActivateExtension llvm-vs-code-extensions.vscode-clangd
```

The activation event should be `onLanguage:cuda-cpp`, not only `onCommand:clangd.restart`.

### 2. Capture the full clangd diagnostic stream

Do not filter only for `error:` or names such as `Gemm`. clangd diagnostic codes use forms such as:

```text
E[...] [pp_file_not_found]
```

These lines do not contain `error:` and can be filtered out accidentally.

Run:

```bash
clangd \
  --check=/absolute/path/to/file.cu \
  --log=verbose > /tmp/clangd-cuda.log 2>&1
```

Find the earliest fatal parsing problem:

```bash
rg -n 'Dropped diagnostic|E\[|pp_file_not_found|file not found|resource directory' /tmp/clangd-cuda.log
```

Treat the first missing header or resource error as the root cause. Errors such as these are normally cascades:

- `too few template arguments for class template 'Gemm'`
- `unknown type name 'CutlassGemm'`
- `std::stringstream` shown as `int`
- missing CUDA builtins such as `threadIdx`

Do not work around these by adding CUTLASS template arguments; repair the headers and parse flags first.

### 3. Verify the Clang frontend and resource directory

Check which clangd is running and whether its matching resource directory exists:

```bash
command -v clangd
clangd --version
clang -print-resource-dir
```

If `clang` is unavailable, determine the clangd major version and install the matching Clang package in the same environment. For Conda:

```bash
conda install -c conda-forge clang=<matching-version>
```

Confirm the wrapper exists:

```bash
test -f "$CLANG_RESOURCE_DIR/include/__clang_cuda_runtime_wrapper.h"
```

Pass the matching resource directory both to VS Code and to `--check`:

```json
{
  "clangd.arguments": [
    "--background-index",
    "--resource-dir=/path/to/matching/clang-resource-directory"
  ]
}
```

```bash
clangd \
  --resource-dir="$CLANG_RESOURCE_DIR" \
  --check=/absolute/path/to/file.cu \
  --log=verbose
```

An explicit `--resource-dir` is important when clangd's CUDA fallback command contains `nvcc`: otherwise clangd may infer the nonexistent `<CUDA_HOME>/lib/clang/<version>`.

### 4. Verify CUDA CCCL includes

When `cuda/std/...` headers are missing, locate CCCL for the active CUDA installation:

```bash
find "$CUDA_HOME" \
  -type d \
  -path '*/targets/*/include/cccl'
```

The project `.clangd` configuration must add the directory. In `CompileFlags.Add`, each list entry is an argument. Keep `-isystem` and its path as separate entries:

```yaml
CompileFlags:
  Add:
    - -I/path/to/repository/include/
    - -I/path/to/repository/tools/util/include/
    - -I/path/to/repository/examples/common/
    - -isystem
    - /path/to/cuda/targets/<target>/include/cccl
```

Avoid this form:

```yaml
- -isystem /path/to/cuda/targets/<target>/include/cccl
```

clangd can pass it as one malformed argument with an embedded space. Inspect `internal (cc1) args` in the verbose log if the include appears to be configured but remains ineffective.

### 5. Verify CUDA compiler version macros

CUTLASS features such as the SM90 Collective Builder can depend on CUDA compiler version macros that Clang does not always define during clangd fallback parsing. A typical symptom is:

```text
static assertion failed: Unsupported Toolkit for SM90 Collective Builder
```

Read the toolkit version:

```bash
nvcc --version
```

If the active fallback command does not already define them, add the reported major, minor, and build versions to `.clangd`:

```yaml
    - -D__CUDACC_VER_MAJOR__=<cuda-major>
    - -D__CUDACC_VER_MINOR__=<cuda-minor>
    - -D__CUDACC_VER_BUILD__=<cuda-build>
```

Use the real toolkit version. These macros influence CUTLASS feature selection and can expose or hide library code paths; do not use arbitrary values merely to silence a diagnostic.

### 6. Restart and verify

In VS Code:

1. Run `clangd: Restart language server`.
2. Close and reopen the CUDA file.
3. Confirm the clangd output contains `textDocument/didOpen` for the `.cu` file.

Run the final parser check:

```bash
clangd \
  --resource-dir="$CLANG_RESOURCE_DIR" \
  --check="$REPO_ROOT/examples/00_basic_gemm/basic_gemm.cu" \
  --log=verbose
echo $?
```

Expected result:

- Exit code `0`
- No `pp_file_not_found`
- No `__clang_cuda_runtime_wrapper.h` error
- No cascading `Gemm` or standard-library errors

The following warning is non-fatal:

```text
CUDA version is newer than the latest supported version
```

If newer CUDA headers later cause compatibility failures, use the newest CUDA version supported by the installed clangd release for analysis. Update both `--cuda-path` and the corresponding CCCL include.

## Minimal working reference

`.vscode/settings.json`:

```json
{
  "files.associations": {
    "*.cu": "cuda-cpp",
    "*.cuh": "cuda-cpp"
  },
  "clangd.arguments": [
    "--background-index",
    "--resource-dir=/path/to/matching/clang-resource-directory"
  ]
}
```

`.clangd`:

```yaml
CompileFlags:
  Add:
    - -I/path/to/repository/include/
    - -I/path/to/repository/tools/util/include/
    - -I/path/to/repository/examples/common/
    - -isystem
    - /path/to/cuda/targets/<target>/include/cccl
    - -D__CUDACC_VER_MAJOR__=<cuda-major>
    - -D__CUDACC_VER_MINOR__=<cuda-minor>
    - -D__CUDACC_VER_BUILD__=<cuda-build>
```

These absolute paths are environment-specific. Detect the active clangd resource directory, CUDA home, target directory, and CCCL directory when applying the workflow elsewhere.

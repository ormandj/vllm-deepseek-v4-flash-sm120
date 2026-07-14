# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/contributing/dockerfile/dockerfile.md and
# docs/assets/contributing/dockerfile-stages-dependency.png

# =============================================================================
# VERSION MANAGEMENT
# =============================================================================
# ARG defaults in this Dockerfile are the source of truth for pinned versions.
# docker/versions.json is auto-generated for use with docker buildx bake.
#
# When updating versions:
# 1. Edit the ARG defaults below
# 2. Run: python tools/generate_versions_json.py
#
# To query versions programmatically:
#   jq -r '.variable.CUDA_VERSION.default' docker/versions.json
#
# To build with bake:
#   docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json
# =============================================================================

# SM120 integration: pinned for RTX PRO 6000 Blackwell (sm_120). CUDA 13.3.0 is our
# verified stack standard (dodges the 13.1 MMQ + 13.2.x corruption; runs on
# driver 595.71.05 via CUDA minor-version compat). Ubuntu 24.04 ships Python
# 3.12 natively (no deadsnakes) and matches the rest of the stack.
# NOTE: 26.04 + Python 3.14 was tried and FAILS dependency resolution —
# flashinfer-python's transitive `cuda-tile` has no cp314 wheel compatible with
# torch 2.11's pinned cuda-toolkit==13.0.2 (cp314 is ahead of the CUDA wheel
# ecosystem). 3.12 has full wheel coverage. Revisit 3.13/3.14 when torch bumps.
ARG CUDA_VERSION=13.3.0
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION=24.04

# By parameterizing the base images, we allow third-party to use their own
# base images. One use case is hermetic builds with base images stored in
# private registries that use a different repository naming conventions.
#
# Example:
# docker build --build-arg BUILD_BASE_IMAGE=registry.acme.org/mirror/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

# Important: We build with an old version of Ubuntu to maintain broad
# compatibility with other Linux OSes. The main reason for this is that the
# glibc version is baked into the distro, and binaries built with one glibc
# version are not backwards compatible with OSes that use an earlier version.
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
# Using cuda base image with minimal dependencies necessary for JIT compilation (FlashInfer, DeepGEMM, EP kernels)
ARG FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# OS family of BUILD_BASE_IMAGE. Controls package manager (apt vs dnf) and
# Python bootstrap. Set to "manylinux" alongside a manylinux build base such
# as pytorch/manylinux2_28-builder:cuda13.0 to produce wheels with a glibc
# 2.28 floor (matches PyTorch's own published wheels). Default stays on
# Ubuntu for backwards compatibility.
ARG BUILD_OS=ubuntu

# By parameterizing the Deadsnakes repository URL, we allow third-party to use
# their own mirror. When doing so, we don't benefit from the transparent
# installation of the GPG key of the PPA, as done by add-apt-repository, so we
# also need a URL for the GPG key.
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL

# The PyPA get-pip.py script is a self contained script+zip file, that provides
# both the installer script and the pip base85-encoded zip archive. This allows
# bootstrapping pip in environment where a distribution package does not exist.
#
# By parameterizing the URL for get-pip.py installation script, we allow
# third-party to use their own copy of the script stored in a private mirror.
# We set the default value to the PyPA owned get-pip.py script.
#
# Reference: https://pip.pypa.io/en/stable/installation/#get-pip-py
ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

# PIP supports fetching the packages from custom indexes, allowing third-party
# to host the packages in private mirrors. The PIP_INDEX_URL and
# PIP_EXTRA_INDEX_URL are standard PIP environment variables to override the
# default indexes. By letting them empty by default, PIP will use its default
# indexes if the build process doesn't override the indexes.
#
# Uv uses different variables. We set them by default to the same values as
# PIP, but they can be overridden.
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

# PyTorch provides its own indexes for standard and nightly builds
ARG PYTORCH_CUDA_INDEX_BASE_URL=https://download.pytorch.org/whl

# PIP supports multiple authentication schemes, including keyring
# By parameterizing the PIP_KEYRING_PROVIDER variable and setting it to
# disabled by default, we allow third-party to use keyring authentication for
# their private Python indexes, while not changing the default behavior which
# is no authentication.
#
# Reference: https://pip.pypa.io/en/stable/topics/authentication/#keyring-support
ARG PIP_KEYRING_PROVIDER=disabled
ARG UV_KEYRING_PROVIDER=${PIP_KEYRING_PROVIDER}

# Flag enables built-in KV-connector dependency libs into docker images
ARG INSTALL_KV_CONNECTORS=false

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM ${BUILD_BASE_IMAGE} AS base

ARG TARGETPLATFORM
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG BUILD_OS
ARG USE_SCCACHE
ARG SCCACHE_DOWNLOAD_URL
ARG SCCACHE_ENDPOINT
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0

ENV DEBIAN_FRONTEND=noninteractive

# Environment for uv
# Declared BEFORE the installer + `uv venv` invocations below so the uv
# binary, managed Python, download cache, and /opt/venv all land under
# /opt/uv instead of /root/.local/. Without this, the venv created at
# build time hardlinks back to /root/.local/share/uv/python and
# descendants of this stage (`build`, `dev`, `csrc-build`,
# `extensions-build`) inherit a root-owned, non-root-unreadable layout.
# See #15174, #15359, #31959. Child stages inherit these via Dockerfile
# `ENV` unless they override them explicitly.
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache
ENV UV_INSTALL_DIR=/opt/uv/bin
ENV PATH="/opt/venv/bin:/opt/uv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Install system dependencies including build tools.
# The Ubuntu path uses apt + deadsnakes-via-uv for Python; the manylinux path
# (AlmaLinux 8, e.g. pytorch/manylinux2_28-builder) uses dnf and the Python
# interpreters pre-installed at /opt/python/cpXY-cpXY/.
RUN if [ "${BUILD_OS}" = "manylinux" ]; then \
        # rdma-core-devel provides libibverbs headers; ccache lives in EPEL,
        # which the pytorch manylinux image already enables. git/curl/sudo
        # are typically pre-installed but listed defensively.
        dnf install -y --setopt=install_weak_deps=False \
            ccache \
            git \
            curl \
            sudo \
            rdma-core-devel \
        && dnf clean all \
        && rm -rf /var/cache/dnf; \
    else \
        apt-get update -y \
        && apt-get install -y --no-install-recommends \
            ccache \
            software-properties-common \
            git \
            curl \
            sudo \
            python3-pip \
            libibverbs-dev \
            # GCC 10 was previously pinned to suppress spurious -Wredundant-move warnings
            # from CUTLASS (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=92519). That bug
            # was fixed in GCC 11. GCC >= 11.3 is now required because PyTorch's C++20 headers
            # (pytorch/pytorch#167929) are not compatible with GCC < 11.3.
            gcc-11 \
            g++-11 \
        && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
        # Install python dev headers if available (needed for cmake FindPython on Ubuntu 24.04
        # which ships cmake 3.28 and requires Development.SABIModule; silently skipped on
        # Ubuntu 20.04/22.04 where python3.x-dev is not available without a PPA)
        && (apt-get install -y --no-install-recommends python${PYTHON_VERSION}-dev 2>/dev/null || true) \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Install sccache once in base so Rust and CMake/CUDA build stages share the
# same binary and remote cache configuration.
RUN if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && case "${TARGETPLATFORM}" in \
          linux/arm64) SCCACHE_ARCH="aarch64" ;; \
          linux/amd64) SCCACHE_ARCH="x86_64" ;; \
          *) echo "Unsupported TARGETPLATFORM for sccache: ${TARGETPLATFORM}" >&2; exit 1 ;; \
        esac \
        && export SCCACHE_DOWNLOAD_URL="${SCCACHE_DOWNLOAD_URL:-https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl.tar.gz}" \
        && curl -L -o sccache.tar.gz ${SCCACHE_DOWNLOAD_URL} \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl; \
    fi

ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET_NAME}}
ENV SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION_NAME}}
ENV SCCACHE_S3_NO_CREDENTIALS=${USE_SCCACHE:+${SCCACHE_S3_NO_CREDENTIALS}}
ENV SCCACHE_IDLE_TIMEOUT=${USE_SCCACHE:+0}

# Install uv and bootstrap /opt/venv. Both paths converge on /opt/venv so all
# downstream stages stay distro-agnostic.
RUN mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" "${UV_INSTALL_DIR}" \
    && chmod -R a+rX /opt/uv \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    # `--seed` installs pip/setuptools/wheel into the venv so `python3 -m
    # pip` works regardless of how uv happens to link the venv back to the
    # managed Python install (which, at a non-default UV_PYTHON_INSTALL_DIR,
    # doesn't always expose ensurepip via the default venv layout).
    && if [ "${BUILD_OS}" = "manylinux" ]; then \
           # manylinux images ship Python at /opt/python/cpXY-cpXY/; point uv
           # at the matching interpreter rather than letting it fetch one.
           PYV_NODOT=$(echo ${PYTHON_VERSION} | tr -d '.') \
           && MANYLINUX_PY=/opt/python/cp${PYV_NODOT}-cp${PYV_NODOT}/bin/python${PYTHON_VERSION} \
           && uv venv --seed /opt/venv --python "$MANYLINUX_PY"; \
       else \
           uv venv --seed /opt/venv --python ${PYTHON_VERSION}; \
       fi \
    && rm -f /usr/bin/python3 /usr/bin/python3-config /usr/bin/pip \
    && ln -sf /opt/venv/bin/python3 /usr/bin/python3 \
    && ln -sf /opt/venv/bin/python3-config /usr/bin/python3-config \
    && ln -sf /opt/venv/bin/pip /usr/bin/pip \
    && python3 --version && python3 -m pip --version

# UV_LINK_MODE=copy applies to subsequent `uv pip install` RUNs (avoids
# hardlink failures with BuildKit cache mounts); it must not be set during
# `uv venv` above, which relies on hardlinking /opt/venv back to the
# managed Python source so ensurepip / `python3 -m pip` still resolve.
ENV UV_LINK_MODE=copy

# Verify GCC version
RUN gcc --version

# Enable CUDA forward compatibility by setting '-e VLLM_ENABLE_CUDA_COMPATIBILITY=1'
# Only needed for datacenter/professional GPUs with older drivers.
# See: https://docs.nvidia.com/deploy/cuda-compatibility/
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0

# ============================================================
# SLOW-CHANGING DEPENDENCIES BELOW
# These are the expensive layers that we want to cache
# ============================================================

# Install PyTorch and core CUDA dependencies
# This is ~2GB and rarely changes
ARG PYTORCH_CUDA_INDEX_BASE_URL

WORKDIR /workspace

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install build and runtime dependencies, including PyTorch
# Check whether to install torch nightly instead of release for this build
COPY requirements/common.txt requirements/common.txt
COPY requirements/cuda.txt requirements/cuda.txt
COPY use_existing_torch.py use_existing_torch.py
COPY pyproject.toml pyproject.toml
# nvidia-cutlass-dsl[cu13] installs -libs-base and -libs-cu13 wheels that
# share paths with different content. uv can extract them in either order,
# leaving base files that break CUDA 13 CuTe DSL JIT.
# TODO(mmangkad): Remove this after NVIDIA/cutlass#3259 is fixed.
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "12" ]; then \
        sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' requirements/cuda.txt; \
        sed -i 's/^humming-kernels\[cu13\]/humming-kernels[cu12]/' requirements/cuda.txt; \
    fi \
    && if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing torch nightly..." \
        && uv pip install --python /opt/venv/bin/python3 torch torchaudio torchvision --pre \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        && echo "Installing other requirements..." \
        && /opt/venv/bin/python3 use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/cuda.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        uv pip install --python /opt/venv/bin/python3 -r requirements/cuda.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    fi \
    && if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "13" ]; then \
        CUTLASS_DSL_VERSION=$(uv pip show --python /opt/venv/bin/python3 nvidia-cutlass-dsl 2>/dev/null | awk '/^Version:/{print $2}') && \
        if [ -n "$CUTLASS_DSL_VERSION" ]; then \
            uv pip install --python /opt/venv/bin/python3 --force-reinstall --no-deps \
                "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}"; \
        fi; \
    fi

# Track PyTorch lib versions used during build and match in downstream instances.
# We do this for both nightly and release so we can strip dependencies/*.txt as needed.
# Otherwise library dependencies can upgrade/downgrade torch incorrectly.
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip freeze | grep -i "^torch=\|^torchvision=\|^torchaudio=" > torch_lib_versions.txt \
    && TORCH_LIB_VERSIONS=$(cat torch_lib_versions.txt | xargs) \
    && echo "Installed torch libs: ${TORCH_LIB_VERSIONS}"

# CUDA arch list used by torch
# Explicitly set the list to avoid issues with torch 2.2
# See https://github.com/pytorch/pytorch/pull/123243
# From versions.json: .torch.cuda_arch_list
# Do not add +PTX here: vLLM filters torch's top-level PTX flag when it
# converts global gencode flags into per-kernel arch lists. If a specific
# kernel needs PTX, add +PTX to that kernel's CMake arch list instead.
ARG torch_cuda_arch_list='12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
#################### BUILD BASE IMAGE ####################

#################### RUST BUILD IMAGE ####################
# Build the Rust frontend (`vllm-rs`) in a dedicated stage so the main wheel
# build stage doesn't need the rust toolchain, protoc, or the rust source.
# This stage reuses the Python environment from base and runs in parallel with
# csrc-build/extensions-build.
FROM base AS rust-build
ARG BUILD_OS
ARG USE_SCCACHE
ARG SCCACHE_ENDPOINT

# Install native tools needed only for Rust/protoc builds.
RUN if [ "${BUILD_OS}" = "manylinux" ]; then \
        dnf install -y --setopt=install_weak_deps=False \
            make unzip \
        && dnf clean all && rm -rf /var/cache/dnf; \
    else \
        apt-get update -y \
        && apt-get install -y --no-install-recommends \
            make unzip \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY tools/install_protoc.sh /tmp/install_protoc.sh
RUN /tmp/install_protoc.sh && rm /tmp/install_protoc.sh

WORKDIR /workspace

COPY requirements/build/rust.txt requirements/build/rust.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install --python /opt/venv/bin/python3 -r requirements/build/rust.txt

# Copy only the Rust build inputs; build_rust.sh publishes artifacts needed
# by the wheel build stage.
COPY rust rust
COPY rust-toolchain.toml rust-toolchain.toml
COPY tools/build_rust.py tools/build_rust.py
COPY build_rust.sh build_rust.sh

# Cap cargo parallelism to avoid exhausting the CI host's open-file limit
# (rustc spawns enough concurrent processes to hit RLIMIT_NOFILE otherwise).
# SM120 integration: left at upstream's 4. The Rust stage is a separate, sequential build
# stage that already finishes in ~4min, so it is NOT the bottleneck (the CUDA
# compile is — see max_jobs in build-image.yml). Bumping -j here risks the FD
# ceiling on the runner for ~1-2min of gain; not worth it. Raise the runner's
# nofile limit first if this ever needs to go higher.
ENV CARGO_BUILD_JOBS=4

# BuildKit can run this stage in parallel with csrc-build. Keep Rust on a
# separate local sccache daemon while sharing the same remote cache backend.
ENV SCCACHE_SERVER_PORT=4227

# Build the release artifacts. Cache cargo registry/git, but not target/,
# because stale target metadata can outlive source updates across BuildKit
# cache reuse.
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false \
    if [ "$USE_SCCACHE" = "1" ]; then \
        if [ -n "${SCCACHE_ENDPOINT}" ]; then export SCCACHE_ENDPOINT="${SCCACHE_ENDPOINT}"; fi; \
        export RUSTC_WRAPPER=sccache; \
        sccache --show-stats; \
    fi \
    && bash build_rust.sh \
    && if [ "$USE_SCCACHE" = "1" ]; then \
        sccache --show-stats; \
    fi
#################### RUST BUILD IMAGE ####################

#################### CSRC BUILD IMAGE ####################
FROM base AS csrc-build

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install build dependencies
COPY requirements/build/cuda.txt requirements/build/cuda.txt
COPY use_existing_torch.py use_existing_torch.py
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing build requirements without torch..." \
        && python3 use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt \
        && echo "Installing torch nightly..." \
        && uv pip install --python /opt/venv/bin/python3 $(cat torch_lib_versions.txt | grep -i "^torch=" | xargs) --pre \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "Installing build requirements..." \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    fi

WORKDIR /workspace

COPY pyproject.toml setup.py CMakeLists.txt ./
COPY tools/build_rust.py tools/build_rust.py
COPY cmake cmake/
COPY csrc csrc/
COPY vllm/envs.py vllm/envs.py
COPY vllm/__init__.py vllm/__init__.py

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

ARG USE_SCCACHE
ARG SCCACHE_ENDPOINT

# Flag to control whether to use pre-built vLLM wheels
ARG VLLM_USE_PRECOMPILED=""
ARG VLLM_MERGE_BASE_COMMIT=""
ARG VLLM_MAIN_CUDA_VERSION=""

# Use dummy version for csrc-build wheel (only .so files are extracted, version doesn't matter)
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+csrc.build"

# Use existing torch for nightly builds
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        python3 use_existing_torch.py --prefix; \
    fi

# Provision one bare Python per `requires-python` entry; cmake reads
# DEEPGEMM_PYTHON_INTERPRETERS to build DeepGEMM `_C` for each. See
# cmake/external_projects/deepgemm.cmake for the full picture.
COPY tools/setup_deepgemm_pythons.sh tools/build_deepgemm_C.py tools/
ENV DEEPGEMM_VENV_PREFIX=/opt/dgenv
RUN --mount=type=cache,target=/root/.cache/uv \
    tools/setup_deepgemm_pythons.sh > /tmp/dg_pythons.txt

# Build the vLLM wheel
# if USE_SCCACHE is set, use sccache to speed up compilation
# AWS credentials mounted at ~/.aws/credentials for sccache S3 auth (optional)
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false \
    if [ "$USE_SCCACHE" = "1" ]; then \
        if [ -n "${SCCACHE_ENDPOINT}" ]; then export SCCACHE_ENDPOINT="${SCCACHE_ENDPOINT}"; fi \
        && export CMAKE_BUILD_TYPE=Release \
        && export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" \
        && export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_MERGE_BASE_COMMIT}" \
        && export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION}" \
        && export VLLM_DOCKER_BUILD_CONTEXT=1 \
        && export DEEPGEMM_PYTHON_INTERPRETERS=$(cat /tmp/dg_pythons.txt) \
        && sccache --show-stats \
        && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 \
        && sccache --show-stats; \
    fi

ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,id=vllm-csrc-ccache,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,target=/opt/uv/cache \
    if [ "$USE_SCCACHE" != "1" ]; then \
        # ccache's default temporary directory is inside CCACHE_DIR. Recreate it
        # defensively because an interrupted rootful Podman build can leave a
        # persistent cache mount without the directory.
        mkdir -p "${CCACHE_DIR}/tmp" && \
        # Clean any existing CMake artifacts
        rm -rf .deps && \
        mkdir -p .deps && \
        export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" && \
        export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_MERGE_BASE_COMMIT}" && \
        export VLLM_DOCKER_BUILD_CONTEXT=1 && \
        export DEEPGEMM_PYTHON_INTERPRETERS=$(cat /tmp/dg_pythons.txt) && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi

#################### CSRC BUILD IMAGE ####################

#################### EXTENSIONS BUILD IMAGE ####################
# Build DeepEP - runs in PARALLEL with csrc-build
# This stage is independent and doesn't affect csrc cache
FROM base AS extensions-build
ARG CUDA_VERSION

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

WORKDIR /workspace

# Build DeepEP wheels
COPY tools/ep_kernels/install_python_libraries.sh /tmp/install_python_libraries.sh
# Defaults moved here from tools/ep_kernels/install_python_libraries.sh for centralized version management
ARG DEEPEP_COMMIT_HASH=73b6ea4
ARG NVSHMEM_VER
RUN --mount=type=cache,target=/opt/uv/cache \
    mkdir -p /tmp/ep_kernels_workspace/dist && \
    export TORCH_CUDA_ARCH_LIST='9.0a 10.0a' && \
    /tmp/install_python_libraries.sh \
        --workspace /tmp/ep_kernels_workspace \
        --mode wheel \
        ${DEEPEP_COMMIT_HASH:+--deepep-ref "$DEEPEP_COMMIT_HASH"} \
        ${NVSHMEM_VER:+--nvshmem-ver "$NVSHMEM_VER"} && \
    find /tmp/ep_kernels_workspace/nvshmem -name '*.a' -delete
#################### EXTENSIONS BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install build dependencies
COPY requirements/build/cuda.txt requirements/build/cuda.txt
COPY use_existing_torch.py use_existing_torch.py
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing build requirements without torch..." \
        && python3 use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt \
        && echo "Installing torch nightly..." \
        && uv pip install --python /opt/venv/bin/python3 $(cat torch_lib_versions.txt | grep -i "^torch=" | xargs) --pre \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "Installing build requirements..." \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    fi

WORKDIR /workspace

# Copy pre-built csrc wheel directly
COPY --from=csrc-build /workspace/dist /precompiled-wheels
COPY . .

# Drop the pre-built Rust artifacts into the source tree. setup.py detects
# them and ships them as-is, skipping the local Rust build.
COPY --from=rust-build /workspace/vllm/vllm-rs vllm/vllm-rs
COPY --from=rust-build /workspace/vllm/_rust_*.so vllm/

ARG GIT_REPO_CHECK=0
RUN --mount=type=bind,source=.git,target=.git \
    if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi

ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}

# Skip adding +precompiled suffix to version (preserves git-derived version)
ENV VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1

# Use existing torch for nightly builds
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        python3 use_existing_torch.py --prefix; \
    fi

# Build the vLLM wheel
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,source=.git,target=.git \
    if [ "${vllm_target_device}" = "cuda" ]; then \
        export VLLM_USE_PRECOMPILED=1; \
        export VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /precompiled-wheels/*.whl); \
    fi && \
    python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

# Record the wheel checksum so downstream stages can bust their layer cache
# when the wheel changes, without copying the wheel itself into the image.
RUN sha256sum dist/*.whl > dist/wheel.sha256

# Copy extension wheels from extensions-build stage for later use
COPY --from=extensions-build /tmp/ep_kernels_workspace/dist /tmp/ep_kernels_workspace/dist

# Record the EP kernels wheel checksum for the same cache-busting purpose.
RUN sha256sum /tmp/ep_kernels_workspace/dist/*.whl \
    > /tmp/ep_kernels_workspace/dist/wheels.sha256

# Check the size of the wheel if RUN_WHEEL_CHECK is true
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# sync the default value with .buildkite/check-wheel-size.py
ARG VLLM_MAX_SIZE_MB=500
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=false
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check."; \
    fi

#################### WHEEL BUILD IMAGE ####################

#################### DEV IMAGE ####################
FROM base AS dev

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG BUILD_OS

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

# Install libnuma-dev, required by fastsafetensors (fixes #20384)
RUN if [ "${BUILD_OS}" = "manylinux" ]; then \
        dnf install -y numactl-devel && dnf clean all && rm -rf /var/cache/dnf; \
    else \
        apt-get update && apt-get install -y --no-install-recommends libnuma-dev && rm -rf /var/lib/apt/lists/*; \
    fi


# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install development dependencies
COPY requirements/lint.txt requirements/lint.txt
COPY requirements/test/cuda.in requirements/test/cuda.in
COPY requirements/test/cuda.txt requirements/test/cuda.txt
COPY requirements/dev.txt requirements/dev.txt
COPY use_existing_torch.py use_existing_torch.py
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing dev requirements plus torch nightly..." \
        && python3 use_existing_torch.py --prefix \
        && cat torch_lib_versions.txt >> requirements/test/cuda.in \
        && uv pip compile requirements/test/cuda.in -o requirements/test/cuda.txt --index-strategy unsafe-best-match \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        && uv pip install --python /opt/venv/bin/python3 $(cat torch_lib_versions.txt | xargs) --pre \
        -r requirements/dev.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "Installing dev requirements..." \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/dev.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    fi

#################### DEV IMAGE ####################
#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM ${FINAL_BASE_IMAGE} AS vllm-base

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /vllm-workspace


# Python version string for paths (e.g., "312" for 3.12)
RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Install Python and system dependencies.
# SM120 integration: Ubuntu 24.04 ships Python 3.12 in its default apt repo, so we install
# it directly — no deadsnakes PPA needed. This is the only structural change vs
# upstream's runtime stage.
# SM120 integration: native apt python3-pip already provides pip for 3.12, so upstream's
# `curl ${GET_PIP_URL} | python` bootstrap (needed only for pip-less deadsnakes
# Python) is dropped. It also breaks now: pip 26 made uninstall-no-record-file a
# hard error and can't uninstall Debian's RECORD-less apt pip. uv (below) does all
# real installs, so apt's pip is just the bootstrap.
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        curl \
        sudo \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        libibverbs-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && rm -f /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED \
    && python3 --version && python3 -m pip --version

# Install CUDA development tools for runtime JIT compilation
# (FlashInfer, DeepGEMM, EP kernels all require compilation at runtime)
RUN CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    CUDA_VERSION_SHORT=$(echo $CUDA_VERSION | cut -d. -f1,2) && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvrtc-${CUDA_VERSION_DASH} \
        # SM120 integration: nvrtc.h headers (the -dev pkg), not just the libnvrtc.so runtime.
        # Upstream ships only cuda-nvrtc here — fine for binding-based JIT — but
        # FlashInfer's DeepGEMM cutlass fp4 MoE backend nvcc-compiles a .cu that
        # `#include <nvrtc.h>`. Upstream never hits that path (it relies on prebuilt
        # flashinfer cubins for sm_90/sm_100); THIS image drops the cubin cache AND
        # targets sm_120, so it JITs that kernel and needs the header. Without it:
        # "fatal error: nvrtc.h: No such file or directory" -> TP worker dies at
        # warmup (verified on Step-3.7-Flash-NVFP4, vllm-e, 2026-06-24). With it, the
        # fp4 JIT builds and the VLLM_TEST_FORCE_FP8_MARLIN/VLLM_DISABLED_KERNELS
        # Marlin workaround in deployment-e.yaml can be dropped.
        cuda-nvrtc-dev-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} \
        libcublas-dev-${CUDA_VERSION_DASH} \
        # Required by fastsafetensors (fixes #20384)
        libnuma-dev \
        # numactl CLI for NUMA binding at runtime
        numactl && \
    # Fixes nccl_allocator requiring nccl.h at runtime
    # https://github.com/vllm-project/vllm/blob/1336a1ea244fa8bfd7e72751cabbdb5b68a0c11a/vllm/distributed/device_communicators/pynccl_allocator.py#L22
    # NCCL packages don't use the cuda-MAJOR-MINOR naming convention,
    # so we pin the version to match our CUDA version
    NCCL_VER=$(apt-cache madison libnccl-dev | grep "+cuda${CUDA_VERSION_SHORT}" | head -1 | awk -F'|' '{gsub(/^ +| +$/, "", $2); print $2}') && \
    apt-get install -y --no-install-recommends --allow-change-held-packages libnccl-dev=${NCCL_VER} libnccl2=${NCCL_VER} && \
    rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
RUN python3 -m pip install uv

# Environment for uv
# Redirect uv's managed Python and download cache out of /root/ so downstream
# images (`FROM vllm/vllm-openai` + `USER <uid>`) and direct non-root runs
# (`docker run --user <uid>:<gid>`) can read and execute them. See #15174,
# #15359, #31959.
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache
RUN mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" \
    && chgrp -R 0 /opt/uv \
    && chmod -R g+rwX,a+rX /opt/uv

# ----------------------------------------------------------------------
# Non-root support (opt-in)
# ----------------------------------------------------------------------
# Create a conventional `vllm` user (UID 2000, GID 0) so the image can be
# run under `--user 2000:0` or the opt-in `vllm-openai-nonroot` target.
#
# Design notes:
#   * GID 0 + group-writable cache dirs follow the OpenShift arbitrary-UID
#     pattern, so any UID that is a member of group 0 at runtime can write
#     to /home/vllm and /opt/uv without additional chown work.
#   * The default `vllm-openai` image keeps `USER root`, so every existing
#     `docker run vllm/vllm-openai ...` / K8s manifest / `FROM vllm/vllm-openai`
#     + `RUN uv pip install --system ...` flow is unchanged.
#   * The entrypoint wrapper below is only used by `vllm-openai-nonroot`; it
#     handles the OpenShift arbitrary-UID case (UID not in /etc/passwd).
# See #31959 and docs/deployment/docker.md.
RUN useradd --uid 2000 --gid 0 --create-home --home-dir /home/vllm \
        --shell /bin/bash vllm \
    && mkdir -p /home/vllm/.cache /home/vllm/.config \
    && chown -R 2000:0 /home/vllm \
    && chmod -R g+rwX /home/vllm \
    # Allow the entrypoint wrapper to append a /etc/passwd entry for an
    # arbitrary runtime UID that shares GID 0. Without this, `whoami`, bash's
    # `\u` prompt, `id -un`, and anything else that calls `getpwuid()`
    # directly return "I have no name!" for OpenShift-style arbitrary UIDs.
    # This matches the convention used by Red Hat UBI base images.
    && chgrp 0 /etc/passwd /etc/group \
    && chmod g=u /etc/passwd /etc/group
COPY docker/entrypoints/vllm-nonroot-entrypoint.sh \
    /usr/local/bin/vllm-nonroot-entrypoint.sh
RUN chmod 0755 /usr/local/bin/vllm-nonroot-entrypoint.sh

# Enable CUDA forward compatibility by setting '-e VLLM_ENABLE_CUDA_COMPATIBILITY=1'
# Only needed for datacenter/professional GPUs with older drivers.
# See: https://docs.nvidia.com/deploy/cuda-compatibility/
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0

# ============================================================
# SLOW-CHANGING DEPENDENCIES BELOW
# These are the expensive layers that we want to cache
# ============================================================

# Install PyTorch and core CUDA dependencies
# This is ~2GB and rarely changes
ARG PYTORCH_CUDA_INDEX_BASE_URL
COPY requirements/common.txt /tmp/common.txt
COPY requirements/cuda.txt /tmp/requirements-cuda.txt
# nvidia-cutlass-dsl[cu13] installs -libs-base and -libs-cu13 wheels that
# share paths with different content. uv can extract them in either order,
# leaving base files that break CUDA 13 CuTe DSL JIT.
# TODO(mmangkad): Remove this after NVIDIA/cutlass#3259 is fixed.
#
# SM120 integration: bump flashinfer-python 0.6.13 -> 0.6.14 in the pin before installing.
# vLLM main since #43477 (the SM120 DSv4 enablement, 2026-06-23) calls
# flashinfer.mla trtllm_batch_decode_sparse_mla_dsv4(..., swa_topk_lens=...),
# an argument that ONLY exists in flashinfer >= 0.6.14 (added in
# flashinfer/mla/_core.py; absent in 0.6.13). But requirements/cuda.txt still
# pins 0.6.13, so the DSv4 SM120 decode path crashes at memory-profiling with
# "trtllm_batch_decode_sparse_mla_dsv4() got an unexpected keyword argument
# 'swa_topk_lens'" (verified vllm-c 2026-07-08). 0.6.14 released 2026-07-02 and
# is the latest. Upstream is already bumping its own pin (vllm PR #47669, open
# 2026-07-08), so this override is a stopgap: once that merges the sed no-ops
# and the grep-assert below still passes. The grep asserts the pin is 0.6.14
# after the sed so we fail loudly if upstream jumps PAST it (e.g. 0.6.15 —
# re-verify swa_topk_lens compat then).
#
# Only flashinfer-PYTHON is bumped; flashinfer-cubin stays at upstream's 0.6.13.
# flashinfer published only flashinfer-python==0.6.14 to PyPI — flashinfer-cubin
# has NO 0.6.14 (its latest is 0.6.13, verified 2026-07-14), so pinning cubin to
# 0.6.14 makes uv's resolve unsatisfiable and the build dies. The swa_topk_lens
# fix is a Python-wrapper change (flashinfer/mla/_core.py); the cubin package is a
# separate, optional set of pre-compiled sm_90/sm_100 cubins that this sm_120-only
# runtime-JIT image doesn't use, and flashinfer-python has no hard dep on it. The
# resulting python(0.6.14)/cubin(0.6.13) mismatch trips a hard RuntimeError in
# flashinfer/jit/env.py at import; FLASHINFER_DISABLE_VERSION_CHECK=1 (set in
# vllm-base below) bypasses it. Self-cleans once flashinfer-cubin 0.6.14 ships and
# upstream realigns both pins (then drop that env too).
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "12" ]; then \
        sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' /tmp/requirements-cuda.txt; \
        sed -i 's/^humming-kernels\[cu13\]/humming-kernels[cu12]/' /tmp/requirements-cuda.txt; \
    fi && \
    sed -i 's/^flashinfer-python==0\.6\.13/flashinfer-python==0.6.14/' /tmp/requirements-cuda.txt && \
    grep -q '^flashinfer-python==0.6.14' /tmp/requirements-cuda.txt || { echo "FATAL: flashinfer-python pin is not 0.6.14 after sed — upstream moved the pin; re-verify swa_topk_lens compat before adjusting"; exit 1; } && \
    uv pip install --system -r /tmp/requirements-cuda.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') && \
    if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "13" ]; then \
        CUTLASS_DSL_VERSION=$(uv pip show --system nvidia-cutlass-dsl 2>/dev/null | awk '/^Version:/{print $2}') && \
        if [ -n "$CUTLASS_DSL_VERSION" ]; then \
            uv pip install --system --force-reinstall --no-deps \
                "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}"; \
        fi; \
    fi && \
    rm /tmp/requirements-cuda.txt /tmp/common.txt

# SM120 integration: flashinfer-python is pinned to 0.6.14 (DSv4 SM120 swa_topk_lens, see the
# sed above) but flashinfer-cubin has no 0.6.14 on PyPI (latest 0.6.13, checked
# 2026-07-14), so the two versions intentionally differ. flashinfer/jit/env.py
# hard-raises a RuntimeError at import when they mismatch; bypass it — the
# pre-staged cubins target sm_90/sm_100 and are unused on this sm_120-only
# runtime-JIT image (checksummed cubins are fetched/JIT'd at runtime as needed).
# Set here in vllm-base so every downstream runtime stage inherits it. Remove once
# flashinfer-cubin 0.6.14 ships and the pins realign.
ENV FLASHINFER_DISABLE_VERSION_CHECK=1

# SM120 integration: SKIP upstream's flashinfer-jit-cache install (upstream pins 0.6.13 here).
# It pulls a prebuilt JIT cache from a CUDA-version-specific index
# (https://flashinfer.ai/whl/cuXXX). The cu130 index has no 0.6.14 package and a
# cu133 index is not published (re-verified 2026-07-14), so installing a cache
# matching this Python/CUDA stack cannot resolve. We rely on flashinfer's
# runtime JIT instead, restricted to sm_120 via
# `ENV FLASHINFER_CUDA_ARCH_LIST=12.0f` (set below); kernels compile on first
# use and are cached to disk thereafter. flashinfer-python / -cubin still
# install normally from PyPI via the requirements above.
# https://docs.flashinfer.ai/installation.html

# ============================================================
# OPENAI API SERVER DEPENDENCIES
# Pre-install these to avoid reinstalling on every vLLM wheel rebuild
# ============================================================

# Install gdrcopy (saves ~6s per build)
# TODO (huydhn): There is no prebuilt gdrcopy package on 12.9 at the moment
ARG GDRCOPY_CUDA_VERSION=12.8
ARG GDRCOPY_OS_VERSION=Ubuntu22_04
ARG TARGETPLATFORM
COPY tools/install_gdrcopy.sh /tmp/install_gdrcopy.sh
RUN set -eux; \
    case "${TARGETPLATFORM}" in \
      linux/arm64) UUARCH="aarch64" ;; \
      linux/amd64) UUARCH="x64" ;; \
      *) echo "Unsupported TARGETPLATFORM: ${TARGETPLATFORM}" >&2; exit 1 ;; \
    esac; \
    /tmp/install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "${GDRCOPY_CUDA_VERSION}" "${UUARCH}" && \
    rm /tmp/install_gdrcopy.sh

# Install vllm-openai dependencies (saves ~2.6s per build)
# These are stable packages that don't depend on vLLM itself
# From versions.json: .bitsandbytes.x86_64, .bitsandbytes.arm64
# From versions.json: .openai_server_extras.timm, .openai_server_extras.runai_model_streamer
ARG BITSANDBYTES_VERSION_X86=0.46.1
ARG BITSANDBYTES_VERSION_ARM64=0.42.0
ARG TIMM_VERSION=">=1.0.17"
ARG RUNAI_MODEL_STREAMER_VERSION=">=0.15.7"
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        BITSANDBYTES_VERSION="${BITSANDBYTES_VERSION_ARM64}"; \
    else \
        BITSANDBYTES_VERSION="${BITSANDBYTES_VERSION_X86}"; \
    fi; \
    uv pip install --system accelerate 'modelscope<1.38' \
        "bitsandbytes>=${BITSANDBYTES_VERSION}" "timm${TIMM_VERSION}" "runai-model-streamer[s3,gcs,azure]${RUNAI_MODEL_STREAMER_VERSION}"

# ============================================================
# VLLM INSTALLATION (depends on build stage)
# ============================================================

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install vLLM wheel first, so that torch etc will be installed.
# Check whether to install torch nightly instead of release for this build.
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt
# Copy only the wheel checksum (a few bytes) so a wheel change invalidates this
# install layer. The wheel itself is bind-mounted below and never enters the
# image. Without this the bind mount is not part of the layer cache key, so a
# warm BuildKit agent can skip the install and ship a stale wheel.
COPY --from=build /workspace/dist/wheel.sha256 /tmp/vllm-wheel.sha256
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing torch nightly..." \
        && uv pip install --system $(cat torch_lib_versions.txt | xargs) --pre \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        && echo "Installing vLLM..." \
        && uv pip install --system dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "Installing vLLM..." \
        && uv pip install --system dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    fi

RUN --mount=type=cache,target=/opt/uv/cache \
. /etc/environment && \
uv pip list

# Pytorch now installs NVSHMEM, setting LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install EP kernels wheels (DeepEP) that have been built in the `build` stage.
# As with the vLLM wheel above, copy only the checksum to bust the layer cache
# and bind-mount the wheel for the actual install to keep it out of the image.
COPY --from=build /tmp/ep_kernels_workspace/dist/wheels.sha256 /tmp/ep-kernels-wheels.sha256
RUN --mount=type=bind,from=build,src=/tmp/ep_kernels_workspace/dist,target=/vllm-workspace/ep_kernels/dist \
    --mount=type=cache,target=/opt/uv/cache \
    uv pip install --system ep_kernels/dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# nvidia-cutlass-dsl[cu13] installs -libs-base and -libs-cu13 wheels that
# share paths with different content. Force -libs-cu13 last after runtime
# dependency installs so uv cannot leave base files behind.
# TODO(mmangkad): Remove this after NVIDIA/cutlass#3259 is fixed.
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "13" ]; then \
        CUTLASS_DSL_VERSION=$(uv pip show --system nvidia-cutlass-dsl 2>/dev/null | awk '/^Version:/{print $2}') && \
        if [ -n "$CUTLASS_DSL_VERSION" ]; then \
            uv pip install --system --force-reinstall --no-deps \
                "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}"; \
        fi; \
    fi

# SM120 integration: RE-PIN flashinfer-python to 0.6.14 (DSv4 SM120 swa_topk_lens). The earlier
# sed bumps it in /tmp/requirements-cuda.txt, but that only patches the throwaway copy
# used for the standalone requirements install — NOT the vLLM source tree the wheel is
# built from. vLLM's own requirements/cuda.txt pins flashinfer-python==0.6.13, so that
# pin is baked into the wheel's dependency metadata; `uv pip install dist/*.whl` (and
# the ep-kernels install) then re-resolve and DOWNGRADE flashinfer-python 0.6.14 -> 0.6.13,
# silently reverting the bump. The image then still crashes at memory-profiling with
# "trtllm_batch_decode_sparse_mla_dsv4() got an unexpected keyword argument 'swa_topk_lens'"
# (verified vllm-c 2026-07-08, flashinfer-python=0.6.13 in the shipped image). Force it
# back here, AFTER every install that could re-resolve it; --no-deps so it drags nothing
# else. cubin stays 0.6.13 (no 0.6.14 on PyPI) + FLASHINFER_DISABLE_VERSION_CHECK=1 above.
# Self-cleans once upstream pins 0.6.14: --force-reinstall to the same version is a no-op.
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install --system --force-reinstall --no-deps flashinfer-python==0.6.14 && \
    INSTALLED=$(uv pip show --system flashinfer-python 2>/dev/null | awk '/^Version:/{print $2}') && \
    [ "$INSTALLED" = "0.6.14" ] || { echo "FATAL: flashinfer-python is $INSTALLED, expected 0.6.14 after force-reinstall"; exit 1; }

# SM120 integration: optionally carry up to three open FlashInfer PRs and one resolver
# fix as source patches on the installed 0.6.14 wheel. The build profile stages
# only its selected files (see patches-flashinfer/*.patch provenance headers):
#   * fi-3817 (vedcsolution, community) — TOPK=256 decode-dsv4 instantiation
#     for SM120 sparse MLA; unblocks DSpark draft decode.
#   * fi-3834 (waynehacking8) — TOPK=256 prefill-dsv4 instantiation for SM120
#     sparse MLA, companion to fi-3817 (the DSpark draft's 256-wide indices
#     hit both the decode and prefill kernels).
#   * fi-3903 (yichengj0, NVIDIA) — TensorRT-LLM allreduce support for
#     SM120/SM121 plus corrected legacy Lamport peer-pointer packing.
#   * fi-3930-cuda-runtime-resolver — PR #3930 plus the signed exact-matcher
#     follow-up, so TileLang's libcudart_stub cannot shadow the real CUDA
#     runtime during communication-workspace initialization.
# All are pre-rewritten to the wheel layout (git csrc/ + include/ live under
# flashinfer/data/ when installed) and verified `patch -p1 --dry-run` clean
# against a pristine 0.6.14 wheel; test hunks are stripped (not shipped in the
# wheel). MUST run after the flashinfer-python force-reinstall above — that
# re-extracts the wheel and would wipe anything patched earlier. The patched
# .cu/.cuh sources take effect via flashinfer's runtime JIT (this image builds
# kernels on first use; no prebuilt cubins are involved). Conditional smoke
# tests fail the build if a selected Python-visible patch silently regresses:
# topk-256 decode
# dispatch entries (fi-3817), the SM12x JIT target and cluster-size helper
# (fi-3903), and a synthetic libcudart-stub collision check. fi-3834 is a
# .cu-only dispatch branch with no importable surface; the loud patch-apply
# failure above is its guard.
# NOTE: the build context is the upstream vLLM clone; the workflow stages
# patches-flashinfer/ into it before building (see build-image.yml).
# Drop individual patches when a FlashInfer release contains their PRs.
COPY patches-flashinfer /tmp/patches-flashinfer
RUN apt-get update -y && apt-get install -y --no-install-recommends patch && \
    rm -rf /var/lib/apt/lists/* && \
    SITE_PACKAGES=$(python3 -c 'import flashinfer, os; print(os.path.dirname(os.path.dirname(flashinfer.__file__)))') && \
    HAS_FI_3817=0 && \
    HAS_FI_3903=0 && \
    HAS_FI_RESOLVER=0 && \
    if [ -f /tmp/patches-flashinfer/fi-3817-sm120-topk256-decode.patch ]; then HAS_FI_3817=1; fi && \
    if [ -f /tmp/patches-flashinfer/fi-3903-sm12x-allreduce.patch ]; then HAS_FI_3903=1; fi && \
    if [ -f /tmp/patches-flashinfer/fi-3930-cuda-runtime-resolver.patch ]; then HAS_FI_RESOLVER=1; fi && \
    echo "Patching flashinfer in ${SITE_PACKAGES}" && \
    for p in /tmp/patches-flashinfer/*.patch; do \
        [ -e "$p" ] || continue; \
        echo "Applying $(basename "$p")" && \
        patch -p1 -d "${SITE_PACKAGES}" --forward --no-backup-if-mismatch < "$p" \
            || { echo "FATAL: $(basename "$p") failed to apply"; exit 1; }; \
    done && \
    if [ "$HAS_FI_3903" = 1 ]; then \
        grep -q 'supported_major_versions=\[9, 10, 12\]' "${SITE_PACKAGES}/flashinfer/jit/comm.py" && \
        grep -q 'inline int GetMaxClusterSize' "${SITE_PACKAGES}/flashinfer/data/include/flashinfer/utils.cuh"; \
    fi && \
    if [ "$HAS_FI_3817" = 1 ]; then \
        python3 -c "import flashinfer.mla._sparse_mla_sm120 as m; \
assert {(8, 256), (16, 256), (32, 256), (64, 256), (128, 256)} <= set(m._DECODE_DSV4_DISPATCH), 'fi-3817: topk 256 missing from _DECODE_DSV4_DISPATCH'; \
print('flashinfer sparse MLA patch smoke OK')"; \
    fi && \
    if [ "$HAS_FI_RESOLVER" = 1 ]; then \
        python3 -c "import io; from unittest.mock import patch; \
from flashinfer.comm.cuda_ipc import find_loaded_library; \
maps='1000-2000 r-xp 0 00:00 0 /opt/tilelang/lib/libcudart_stub.so\\n2000-3000 r-xp 0 00:00 0 /usr/local/cuda/lib64/libcudart.so.13\\n'; \
p=patch('builtins.open', return_value=io.StringIO(maps)); p.start(); \
assert find_loaded_library('libcudart') == '/usr/local/cuda/lib64/libcudart.so.13'; \
p.stop(); bad='1000-2000 r-xp 0 00:00 0 /tmp/libcudart-stub.so\\n2000-3000 r-xp 0 00:00 0 /tmp/libcudart-.so\\n3000-4000 r-xp 0 00:00 0 /tmp/libcudart.foo.so\\n'; \
p=patch('builtins.open', return_value=io.StringIO(bad)); p.start(); \
assert find_loaded_library('libcudart') is None; p.stop(); \
good='1000-2000 r-xp 0 00:00 0 /tmp/libcudart-a1b2c3.so.13.0\\n'; \
p=patch('builtins.open', return_value=io.StringIO(good)); p.start(); \
assert find_loaded_library('libcudart') == '/tmp/libcudart-a1b2c3.so.13.0'; \
p.stop(); print('flashinfer CUDA IPC resolver patch smoke OK')"; \
    fi && \
    rm -rf /tmp/patches-flashinfer

# SM120 integration: upstream's `flashinfer download-cubin` RUN block removed here — upstream
# dropped it on main (a 2.5 GB layer-dup fix); sm_120 relies on runtime JIT anyway.

# CUDA image changed from /usr/local/nvidia to /usr/local/cuda in 12.8 but will
# return to /usr/local/nvidia in 13.0 to allow container providers to mount drivers
# consistently from the host (see https://github.com/vllm-project/vllm/issues/18859).
# Until then, add /usr/local/nvidia/lib64 before the image cuda path to allow override.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# SM120 integration: apply pending Ubuntu security updates to the runtime image. The
# nvidia/cuda base image + warm layer cache otherwise freeze whatever deb
# versions existed when the apt layers were first built — the weekly Trivy
# scan (ci/image-scan) failed 2026-07-06 with 48 CRITICAL/HIGH findings, ALL
# in stale base debs (linux-libc-dev kernel-header CVEs, openssl PKCS7 UAF).
# We don't control the base image; refreshing here keeps what we ship
# current. `upgrade` (not full-upgrade) never removes/installs packages, and
# apt-held CUDA packages are skipped. SECURITY_REFRESH is a cache-buster: CI
# passes the build date so this re-runs on every CI build instead of
# cache-hitting a stale upgrade. Deliberately placed at the END of vllm-base,
# after the expensive torch/vLLM layers, so the daily bust only re-runs the
# apt upgrade itself.
ARG SECURITY_REFRESH=0
RUN echo "security refresh: ${SECURITY_REFRESH}" && \
    apt-get update -y && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Copy examples and benchmarks at the end to minimize cache invalidation
COPY examples examples
COPY benchmarks benchmarks
COPY ./vllm/collect_env.py .
#################### vLLM installation IMAGE ####################
#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test

ADD . /vllm-workspace/

ARG PYTHON_VERSION
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

RUN apt-get update -y \
    && apt-get install -y git

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Install development dependencies (for testing)
COPY requirements/lint.txt requirements/lint.txt
COPY requirements/test/cuda.in requirements/test/cuda.in
COPY requirements/test/cuda.txt requirements/test/cuda.txt
COPY requirements/dev.txt requirements/dev.txt
COPY use_existing_torch.py use_existing_torch.py
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    if [ "$CUDA_MAJOR" -ge 12 ]; then \
        if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
            echo "Installing dev requirements plus torch nightly..." \
            && python3 use_existing_torch.py --prefix \
            && cat torch_lib_versions.txt >> requirements/test/cuda.in \
            && uv pip compile requirements/test/cuda.in -o requirements/test/cuda.txt --index-strategy unsafe-best-match \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            && uv pip install --system $(cat torch_lib_versions.txt | xargs) --pre \
            -r requirements/dev.txt \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
        else \
            echo "Installing dev requirements..." \
            && if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
                echo "Recompiling test requirements for arm64..." \
                && uv pip compile requirements/test/cuda.in -o requirements/test/cuda.txt --index-strategy unsafe-best-match \
                --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
            fi \
            && uv pip install --system -r requirements/dev.txt \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
        fi \
    fi

# install development dependencies (for testing)
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install --system -e tests/vllm_test_utils

# enable fast downloads from hf (for testing)
ENV HF_XET_HIGH_PERFORMANCE 1

# increase timeout for hf downloads (for testing)
ENV HF_HUB_DOWNLOAD_TIMEOUT 60

# Copy in the v1 package for testing (it isn't distributed yet)
COPY vllm/v1 /usr/local/lib/python${PYTHON_VERSION}/dist-packages/vllm/v1

# Source code is used in the `python_only_compile.sh` test
# We hide it inside `src/` so that this source code
# will not be imported by other tests
RUN mkdir src
RUN mv vllm src/vllm
#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# base openai image with additional requirements, for any subsequent openai-style images
FROM vllm-base AS vllm-openai-base
ARG TARGETPLATFORM
ARG INSTALL_KV_CONNECTORS=false
ARG CUDA_VERSION
ARG VLLM_BUILD_COMMIT
ARG VLLM_BUILD_PIPELINE
ARG VLLM_BUILD_URL
ARG VLLM_IMAGE_TAG
ARG INTEGRATION_BUILD_PROFILE=unknown
ARG INTEGRATION_PATCH_MANIFEST=unknown
ARG INTEGRATION_BUILD_COMMIT=unknown
ARG INTEGRATION_SOURCE=https://github.com/ormandj/vllm-deepseek-v4-flash-sm120

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

# This timeout (in seconds) is necessary when installing some dependencies via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
ENV UV_HTTP_TIMEOUT=500

# install kv_connectors if requested
# Do not add +PTX here; see the main TORCH_CUDA_ARCH_LIST comment above.
ARG torch_cuda_arch_list='12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,source=requirements/kv_connectors.txt,target=/tmp/kv_connectors.txt,ro \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-'); \
    CUDA_HOME=/usr/local/cuda; \
    # lmcache requires explicit specifying CUDA_HOME
    BUILD_PKGS="libcusparse-dev-${CUDA_VERSION_DASH} \
                libcublas-dev-${CUDA_VERSION_DASH} \
                libcusolver-dev-${CUDA_VERSION_DASH}"; \
    if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then \
        uv pip install --system -r /tmp/kv_connectors.txt --no-build || ( \
            # if the above fails, install from source
            apt-get update -y && \
            apt-get install -y --no-install-recommends --allow-change-held-packages ${BUILD_PKGS} && \
            uv pip install --system -r /tmp/kv_connectors.txt --no-build-isolation && \
            apt-get purge -y ${BUILD_PKGS} && \
            # clean up -dev packages, keep runtime libraries
            rm -rf /var/lib/apt/lists/* \
        ); \
        # Force-reinstall the matching CUDA wheel so the correct nixl_ep_cpp.so is installed.
        uv pip install --system --force-reinstall --no-deps nixl-cu${CUDA_MAJOR}; \
    fi

# Optional override: install mooncake-transfer-engine from a URL instead of the
# PyPI release pulled in above. Use this for wheels built with non-default CMake
# flags (e.g. `STORE_USE_ETCD=ON` for master HA). The URL's manylinux glibc
# floor must be <= the FINAL_BASE_IMAGE's glibc.
ARG MOONCAKE_WHEEL_AARCH64
ARG MOONCAKE_WHEEL_X86_64
RUN if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then \
        if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
            WHEEL="${MOONCAKE_WHEEL_AARCH64}"; \
        else \
            WHEEL="${MOONCAKE_WHEEL_X86_64}"; \
        fi && \
        if [ -n "${WHEEL}" ]; then \
            uv pip install --system "${WHEEL}" && \
            CUDA_MAJOR="${CUDA_VERSION%%.*}" && \
            if [ ! -f /usr/local/cuda/lib64/libcudart.so ] && \
               [ -f "/usr/local/cuda/lib64/libcudart.so.${CUDA_MAJOR}" ]; then \
                ln -s "libcudart.so.${CUDA_MAJOR}" /usr/local/cuda/lib64/libcudart.so; \
            fi; \
        fi; \
    fi

ENV VLLM_USAGE_SOURCE production-docker-image
ENV VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unknown} \
    VLLM_BUILD_PIPELINE=${VLLM_BUILD_PIPELINE:-local} \
    VLLM_BUILD_URL=${VLLM_BUILD_URL:-} \
    VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG:-local/vllm-openai:dev} \
    INTEGRATION_BUILD_PROFILE=${INTEGRATION_BUILD_PROFILE} \
    INTEGRATION_PATCH_MANIFEST=${INTEGRATION_PATCH_MANIFEST} \
    INTEGRATION_BUILD_COMMIT=${INTEGRATION_BUILD_COMMIT}
LABEL org.opencontainers.image.source="${INTEGRATION_SOURCE}" \
      org.opencontainers.image.revision="${INTEGRATION_BUILD_COMMIT}" \
      org.opencontainers.image.version="${VLLM_IMAGE_TAG}" \
      org.opencontainers.image.url="${VLLM_BUILD_URL}" \
      ai.vllm.upstream.commit="${VLLM_BUILD_COMMIT}" \
      ai.vllm.integration.commit="${INTEGRATION_BUILD_COMMIT}" \
      ai.vllm.build.pipeline="${VLLM_BUILD_PIPELINE}" \
      ai.vllm.build.url="${VLLM_BUILD_URL}" \
      ai.vllm.image.tag="${VLLM_IMAGE_TAG}" \
      ai.vllm.build.profile="${INTEGRATION_BUILD_PROFILE}" \
      ai.vllm.build.patches="${INTEGRATION_PATCH_MANIFEST}"

# define sagemaker first, so it is not default from `docker build`
FROM vllm-openai-base AS vllm-sagemaker

COPY examples/deployment/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]

FROM vllm-openai-base AS vllm-openai

# SM120 integration: restrict FlashInfer's runtime JIT/AOT kernels to the sm_120 family
# target (RTX PRO 6000 Blackwell). Avoids a slow/incorrect fallback and cuts
# first-request JIT time. CUDA 13.x enables the `12.0f` family ISA.
ENV FLASHINFER_CUDA_ARCH_LIST=12.0f

# To run the image as non-root, either build the `vllm-openai-nonroot` target
# below, or in a derived Dockerfile uncomment the following line and ensure
# any additional layers chgrp-0 / chmod-g+rwX paths they write to. The `vllm`
# user (UID 2000, GID 0) is already created in the `vllm-base` stage.
# See docs/deployment/docker.md.
# USER vllm
ENTRYPOINT ["vllm", "serve"]
#################### OPENAI API SERVER ####################

#################### OPENAI API SERVER (NON-ROOT, OPT-IN) ####################
# Non-root-ready variant of `vllm-openai`. Built via:
#   docker build --target vllm-openai-nonroot -t vllm:openai-nonroot \
#       -f docker/Dockerfile .
#
# Runtime behavior:
#   * Default USER is `vllm` (UID 2000, GID 0) created in `vllm-base`.
#   * HOME is /home/vllm, pre-created group-0-writable so arbitrary UIDs in
#     group 0 (OpenShift / `--user <uid>:0`) can also use the image.
#   * Entrypoint wrapper handles the "UID not in /etc/passwd" case for truly
#     arbitrary UIDs by falling back HOME/USER to sane writable defaults.
#   * All cache/config envs (HF_HOME, VLLM_CACHE_ROOT, TRITON_CACHE_DIR, ...)
#     remain unset so their library defaults resolve to $HOME/.cache/... ,
#     which is writable.
FROM vllm-openai AS vllm-openai-nonroot

USER vllm
WORKDIR /home/vllm
ENTRYPOINT ["/usr/local/bin/vllm-nonroot-entrypoint.sh"]
#################### OPENAI API SERVER (NON-ROOT, OPT-IN) ####################

#!/usr/bin/env bash
set -euo pipefail

version="${1:?usage: install-chuda.sh VERSION}"
stamp=".tools/.chuda-${version}"

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is required to build Chuda; install a Rust toolchain first" >&2
    exit 1
fi

cargo install \
    --root .tools \
    --version "${version}" \
    --locked \
    chuda

touch "${stamp}"

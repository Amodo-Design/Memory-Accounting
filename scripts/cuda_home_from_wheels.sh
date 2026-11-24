#!/usr/bin/env bash
# A CUDA_HOME for the posm kernel build (torch.utils.cpp_extension) on a host with no CUDA toolkit,
# assembled from NVIDIA's pip wheels in user space. Needs the project venv (uv sync) and uv.
#
#   scripts/cuda_home_from_wheels.sh [dest]          # default dest: ~/cuda-merged
#   export CUDA_HOME=<dest> PATH=<dest>/bin:$PATH     # then the kernels build on first use
#
# Toolkit compatibility notes:
#  * torch's cu13 wheels bring the runtime headers and libraries of the CUDA version torch was built with,
#    but vLLM's dependencies pull a newer nvcc; CCCL refuses a compiler newer than the runtime headers.
#    So nvcc, crt, nvvm and cccl of torch's CUDA version go into a separate directory.
#  * glibc >= 2.42 declares rsqrt/rsqrtf noexcept; CUDA before 13.4 does not, which is a hard error.
#    The private copy of crt/math_functions.h gets CUDA 13.4's guard (only if needed).
#  * the linker wants libcudart.so; the wheels ship libcudart.so.13 only.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-$HOME/cuda-merged}
# DEST is deleted and rebuilt, so it must be a directory this script made: an absolute path whose last
# component names CUDA, not / or $HOME, and either absent, empty, holding our marker file, or in the
# layout earlier versions made (bin, nvvm and lib64 links, include and lib, nothing else)
MARK=.posm-cuda-home
case "$DEST" in /*) ;; *) DEST=$PWD/$DEST ;; esac                 # relative to the repo root (cd above)
while [ "${DEST%/}" != "$DEST" ]; do DEST=${DEST%/}; done
case "$(basename "$DEST")" in *cuda*) ok=1 ;; *) ok=0 ;; esac
if [ -z "$DEST" ] || [ "$DEST" = "${HOME%/}" ] || [ "$ok" = 0 ]; then
    echo "refusing CUDA_HOME '$DEST': give a directory whose name contains 'cuda' (default ~/cuda-merged)" >&2; exit 1
fi
if [ -L "$DEST" ] || { [ -e "$DEST" ] && ! [ -d "$DEST" ]; }; then
    echo "refusing CUDA_HOME '$DEST': exists and is not a directory" >&2; exit 1
fi
if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST")" ] && ! [ -f "$DEST/$MARK" ] && \
   ! { [ -L "$DEST/bin" ] && [ -L "$DEST/nvvm" ] && [ -L "$DEST/lib64" ] && [ "$(ls -A "$DEST" | tr '\n' ' ')" = "bin include lib lib64 nvvm " ]; }; then
    echo "refusing CUDA_HOME '$DEST': not made by this script (no $MARK); remove it yourself or pick another path" >&2; exit 1
fi
PY=${PYTHON:-$PWD/.venv/bin/python}
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
read -r VENV_CU CUDA_VER RT_VER < <("$PY" - <<'EOF'
import importlib.util, importlib.metadata, os, torch
loc = [p for p in importlib.util.find_spec("nvidia").submodule_search_locations if os.path.isdir(os.path.join(p, "cu13"))][0]
print(os.path.join(loc, "cu13"), torch.version.cuda, importlib.metadata.version("nvidia-cuda-runtime"))
EOF
)
TOOLS=${TOOLS:-$HOME/cuda-$CUDA_VER-tools}
echo "venv CUDA files: $VENV_CU (runtime $RT_VER); torch built for CUDA $CUDA_VER; compiler into $TOOLS; CUDA_HOME at $DEST"
"$UV" pip install --python "$PY" ninja >/dev/null
"$UV" pip install --python "$PY" --target "$TOOLS" "nvidia-cuda-nvcc==$CUDA_VER.*" "nvidia-cuda-crt==$CUDA_VER.*" \
    "nvidia-nvvm==$CUDA_VER.*" "nvidia-cuda-cccl==$CUDA_VER.*" "nvidia-cuda-runtime==$RT_VER" >/dev/null
T=$TOOLS/nvidia/cu13
H=$T/include/crt/math_functions.h
if ! grep -q _NV_RSQRT_SPECIFIER "$H"; then
    cp "$H" "$H.tmp" && mv "$H.tmp" "$H"                     # break uv's hard link before editing
    "$PY" - "$H" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
guard = """
/* posm: backported from CUDA 13.4 for glibc >= 2.42 (declares rsqrt/rsqrtf noexcept) */
#if defined(__GLIBC_PREREQ)
# if __GLIBC_PREREQ(2,42)
#  define _NV_RSQRT_SPECIFIER noexcept(true)
# endif
#endif
#ifndef _NV_RSQRT_SPECIFIER
# define _NV_RSQRT_SPECIFIER
#endif
"""
a = "__device_builtin__ double                 rsqrt(double x);"
b = "__device_builtin__ float                  rsqrtf(float x);"
if s.count(a) == 1 and s.count(b) == 1:
    i = s.rfind("\n", 0, s.index(a))
    s = s[:i] + "\n" + guard + s[i:]
    s = s.replace(a, a[:-1] + " _NV_RSQRT_SPECIFIER;").replace(b, b[:-1] + " _NV_RSQRT_SPECIFIER;")
    open(p, "w").write(s); print("patched crt/math_functions.h for glibc >= 2.42")
else:
    print("crt/math_functions.h: declarations not found, left unpatched")
EOF
fi
rm -rf "$DEST"; mkdir -p "$DEST/include" "$DEST/lib"; : > "$DEST/$MARK"   # checked above
ln -s "$T/bin" "$DEST/bin"; ln -s "$T/nvvm" "$DEST/nvvm"
for f in "$VENV_CU"/include/*; do ln -s "$f" "$DEST/include/"; done       # cuBLAS, cuSPARSE, ... headers torch includes
for f in "$T"/include/*; do ln -sfn "$f" "$DEST/include/"; done          # compiler-matched runtime, crt and cccl headers win
for f in "$VENV_CU"/lib/*; do ln -s "$f" "$DEST/lib/"; done
ln -s "$DEST/lib/libcudart.so.13" "$DEST/lib/libcudart.so"
ln -s lib "$DEST/lib64"
"$DEST/bin/nvcc" --version | tail -1
echo "export CUDA_HOME=$DEST PATH=$DEST/bin:$PWD/.venv/bin:\$PATH"

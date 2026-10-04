#!/usr/bin/env bash
# nanorecon @VERSION@ installer for Linux x86_64 with an NVIDIA GPU.
#
#   bash nanorecon-@VERSION@-linux-x86_64.sh                 install for the current user
#   bash nanorecon-@VERSION@-linux-x86_64.sh --extract DIR   only unpack the wheel into DIR
#
# Installs into ~/.local/share/nanorecon/@VERSION@ (its own Python environment; dependencies,
# about 3 GB, are downloaded from PyPI), links ~/.local/bin/nanorecon and downloads the model.
# Uninstall: rm -rf ~/.local/share/nanorecon ~/.local/bin/nanorecon
#
# Built by tools/build_installer.sh: the wheel is appended after the __PAYLOAD__ line.
set -euo pipefail

VERSION="@VERSION@"
WHEEL="@WHEEL@"
ROOT="${HOME}/.local/share/nanorecon"
DEST="${ROOT}/${VERSION}"
BIN="${HOME}/.local/bin"
LOG="${ROOT}/install-${VERSION}.log"
NEED_GB=8  # environment (~5 GB) plus pip's download cache (~3 GB)

say() { printf 'nanorecon: %s\n' "$*"; }
die() { printf 'nanorecon: error: %s\n' "$*" >&2; exit 1; }

extract() {  # unpack the appended payload (the wheel) into directory $1
  local line
  line=$(awk '/^__PAYLOAD__$/ { print NR + 1; exit }' "$0")
  tail -n "+${line}" "$0" | tar -xzf - -C "$1"
}

case "${1:-}" in
  "") ;;
  --extract)
    [ -n "${2:-}" ] || die "usage: bash $0 --extract DIR"
    mkdir -p "$2"
    extract "$2"
    say "unpacked $WHEEL into $2"
    exit 0 ;;
  -h|--help)
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  *) die "unknown option $1 (see --help)" ;;
esac

# ---------------------------------------------------------------- checks (nothing installed yet)

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] ||
  die "this installer is for Linux x86_64 (this machine: $(uname -s) $(uname -m))"

glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{ print $2 }')
if [ -z "$glibc" ] || ! printf '2.27\n%s\n' "$glibc" | sort -V -C; then
  die "glibc ${glibc:-unknown} is too old; JAX needs glibc 2.27 or newer (Ubuntu 18.04+, RHEL/Rocky 8+)"
fi

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys, venv, ensurepip; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null; then
    PYTHON=$(command -v "$candidate")
    break
  fi
done
[ -n "$PYTHON" ] || die "no Python 3.11-3.13 with the venv module was found (Ubuntu: sudo apt install python3.12-venv)"

existing=$ROOT
while [ ! -d "$existing" ]; do existing=$(dirname "$existing"); done
free_kb=$(df -Pk "$existing" | awk 'NR == 2 { print $4 }')
[ "$free_kb" -ge $((NEED_GB * 1024 * 1024)) ] ||
  die "only $((free_kb / 1024 / 1024)) GB free under $existing; about ${NEED_GB} GB are needed"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  say "warning: nvidia-smi not found; compress/decompress need an NVIDIA GPU with a CUDA 12 driver"
else
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)
  if [ -z "$driver" ]; then
    say "warning: nvidia-smi sees no GPU; compress/decompress need an NVIDIA GPU"
  elif [ "${driver%%.*}" -lt 525 ] 2>/dev/null; then
    say "warning: NVIDIA driver $driver is older than 525 and may not support CUDA 12"
  fi
fi

# ---------------------------------------------------------------- install

tmp=$(mktemp -d)
started=0
done_ok=0
# shellcheck disable=SC2329  # called through the EXIT trap
on_exit() {
  rm -rf "$tmp"
  if [ "$started" = 1 ] && [ "$done_ok" = 0 ]; then
    rm -rf "$DEST"
    printf 'nanorecon: installation did not complete; details in %s\n' "$LOG" >&2
  fi
}
trap on_exit EXIT

extract "$tmp"
mkdir -p "$ROOT"
: > "$LOG"
say "installing nanorecon $VERSION into $DEST (Python: $PYTHON)"
started=1
rm -rf "$DEST"
"$PYTHON" -m venv "$DEST" >> "$LOG" 2>&1 || die "could not create the Python environment"
say "downloading and installing dependencies (about 3 GB the first time; this takes a few minutes)"
"$DEST/bin/python" -m pip install --disable-pip-version-check "$tmp/${WHEEL}[cuda12]" >> "$LOG" 2>&1 ||
  die "pip install failed"
"$DEST/bin/nanorecon" --version >> "$LOG" 2>&1 || die "the installed nanorecon command does not start"
done_ok=1

mkdir -p "$BIN"
ln -sfn "$DEST/bin/nanorecon" "$BIN/nanorecon"
say "installed $BIN/nanorecon -> $DEST/bin/nanorecon"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) say "note: $BIN is not on PATH; open a new login shell or add it, e.g. echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc" ;;
esac

say "downloading the model (about 420 MB)"
"$DEST/bin/nanorecon" pull || say "warning: the model download failed; run 'nanorecon pull' later"
say "done. Uninstall with: rm -rf $ROOT $BIN/nanorecon"
exit 0
# shellcheck disable=SC2317  # payload marker, never executed
__PAYLOAD__

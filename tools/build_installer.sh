#!/usr/bin/env bash
# Build dist/nanorecon-<version>-linux-x86_64.sh: packaging/install.sh with the wheel appended.
#
#   tools/build_installer.sh            uses python3 (needs pip); PYTHON=... picks another one
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

version=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' src/nanorecon/__init__.py)
[ -n "$version" ] || { echo "cannot read __version__ from src/nanorecon/__init__.py" >&2; exit 1; }
wheel="nanorecon-${version}-py3-none-any.whl"
out="dist/nanorecon-${version}-linux-x86_64.sh"

rm -rf build "dist/$wheel" "$out"
"$PYTHON" -m pip wheel --no-deps --quiet --disable-pip-version-check -w dist .
rm -rf build
[ -f "dist/$wheel" ] || { echo "dist/$wheel was not built" >&2; exit 1; }

sed -e "s/@VERSION@/${version}/g" -e "s/@WHEEL@/${wheel}/g" packaging/install.sh > "$out"
COPYFILE_DISABLE=1 tar --no-xattrs -czf - -C dist "$wheel" >> "$out"  # no macOS metadata entries
chmod +x "$out"
echo "built $out ($(du -h "$out" | cut -f1)); the wheel alone is dist/$wheel"

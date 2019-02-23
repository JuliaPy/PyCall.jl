#!/bin/bash
thisdir="$(dirname "${BASH_SOURCE[0]}")"
JULIA="${JULIA:-$(cat "$thisdir/_julia_path")}"
exec "${JULIA}" --sysimage="$thisdir/sys.so" "$@"

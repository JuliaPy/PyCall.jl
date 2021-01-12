#!/bin/bash
thisdir="$(dirname "${BASH_SOURCE[0]}")"
exec "$thisdir/julia.sh" --startup-file=no --color=yes --project=$thisdir -e '
using Pkg
Pkg.test("PyCall")
'

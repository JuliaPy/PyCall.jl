#!/bin/bash
# -*- mode: julia -*-
#=
thisdir="$(dirname "${BASH_SOURCE[0]}")"
exec "$thisdir/julia.sh" --startup-file=no "$@" ${BASH_SOURCE[0]}
=#

pkgid = Base.PkgId(Base.UUID("438e738f-606a-5dbb-bf0a-cddfbfd45ab0"), "PyCall")
sysimage_path = unsafe_string(Base.JLOptions().image_file)
if haskey(Base.loaded_modules, pkgid)
    @info "PyCall is compiled in: `$(sysimage_path)`"
else
    error("PyCall is not compiled in: ", sysimage_path)
end

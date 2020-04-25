#!/bin/bash
# -*- mode: julia -*-
#=
exec "${JULIA:-julia}" "$@" ${BASH_SOURCE[0]}
=#

using Libdl
using PackageCompiler
using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(name="PyCall", path=dirname(@__DIR__)))
Pkg.build("PyCall")
Pkg.activate()

sysimage_path = joinpath(@__DIR__, "sys.$(Libdl.dlext)")
create_sysimage(
    [:PyCall],
    sysimage_path = sysimage_path,
    project = @__DIR__(),
    precompile_execution_file = joinpath(@__DIR__, "precompile.jl"),
)

write(joinpath(@__DIR__, "_julia_path"), Base.julia_cmd().exec[1])

@info "System image: $sysimage_path"

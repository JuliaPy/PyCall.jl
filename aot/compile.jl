#!/bin/bash
# -*- mode: julia -*-
#=
exec "${JULIA:-julia}" "$@" ${BASH_SOURCE[0]}
=#

using Pkg
Pkg.activate(@__DIR__)
Pkg.add("MacroTools")
Pkg.develop(PackageSpec(name="PyCall", path=dirname(@__DIR__)))
Pkg.build("PyCall")
Pkg.activate()

using PackageCompiler
sysout, _curr_syso = compile_incremental(
    joinpath(@__DIR__, "Project.toml"),
    joinpath(@__DIR__, "precompile.jl"),
)

pysysout = joinpath(@__DIR__, basename(sysout))
cp(sysout, pysysout, force=true)

write(joinpath(@__DIR__, "_julia_path"), Base.julia_cmd().exec[1])

@info "System image: $pysysout"

# Ahead-of-time compilation for PyCall

This directory contains a set of scripts for testing compatibility of PyCall.jl
with [PackageCompiler.jl](https://github.com/JuliaLang/PackageCompiler.jl).

See `.travis.yml` for how it is actually used.

## How to compile system image

To create a system image with PyCall.jl, run `aot/compile.jl` (which
is executable in *nix):

```sh
aot/compile.jl --color=yes
JULIA=PATH/TO/CUSTOM/julia aot/compile.jl  # to specify a julia binary
```

Resulting system image is stored at `aot/sys.so`.

## How to use compiled system image

To use compiled system image, run `aot/julia.sh`, e.g.:

```sh
aot/julia.sh --compiled-modules=no --startup-file=no
```

Note that Julia binary used for compiling the system image is cached
and automatically picked by `aot/julia.sh`.  You don't need to specify
the Julia binary.

Since Julia needs to re-compile packages when switching system images,
it is recommended to pass `--compiled-modules=no` if you are using it
in your machine with a standard Julia setup.

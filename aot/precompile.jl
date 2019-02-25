# Activate ./Project.toml.  Excluding `"@v#.#"` from `Base.LOAD_PATH`
# to make compilation more reproducible.
using Pkg
empty!(Base.LOAD_PATH)
append!(Base.LOAD_PATH, ["@", "@stdlib"])
Pkg.activate(@__DIR__)

# Manually invoking `__init__` to workaround:
# https://github.com/JuliaLang/julia/issues/22910

import MacroTools
isdefined(MacroTools, :__init__) && MacroTools.__init__()

using PyCall
PyCall.__init__()
PyCall.pyimport("sys")[:executable]

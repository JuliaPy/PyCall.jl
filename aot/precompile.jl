# Manually invoking `__init__` to workaround:
# https://github.com/JuliaLang/julia/issues/22910

import MacroTools
isdefined(MacroTools, :__init__) && MacroTools.__init__()

using PyCall
PyCall.__init__()
PyCall.pyimport("sys")[:executable]

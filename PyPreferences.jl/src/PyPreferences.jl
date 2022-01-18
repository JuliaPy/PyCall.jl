baremodule PyPreferences

using Logging

function use_system end
function use_conda end
# function use_jll end
function use_inprocess end
function recompile end

# API to be used from PyCall
function assert_configured end
function instruction_message end

# function diagnose end
function status end

module Implementations

    module PythonUtils
        include("python_utils.jl")
    end
    include("which.jl")
    include("core.jl")
    include("api.jl")
end

let prefs = Implementations.setup_non_failing()
    global const python = prefs.python
    global const inprocess = prefs.inprocess
    global const conda = prefs.conda
    global const python_fullpath = prefs.python_fullpath
    global const libpython = prefs.libpython
    global const python_version = prefs.python_version
    global const PYTHONHOME = prefs.PYTHONHOME
end

const pyprogramname = python_fullpath
const pyversion_build = python_version

end  # baremodule PyPreferences

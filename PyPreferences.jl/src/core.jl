using ..PyPreferences: PyPreferences
using .PythonUtils: find_libpython, python_version_of, pythonhome_of, conda_python_fullpath

using Preferences: @set_preferences!, @load_preference, @delete_preferences!

struct PythonPreferences
    python::Union{Nothing,String}
    inprocess::Bool
    conda::Bool
    # jll::Bool
end

function Base.show(io::IO,x::PythonPreferences)
    print(io, "PythonPreferences(python=$(x.python), inprocess=$(x.inprocess), conda=$(x.conda))")
end


set(; python = nothing, inprocess = false, conda = false) =
    set(PythonPreferences(python, inprocess, conda))

function set(prefs::PythonPreferences)
    @debug "setting new Python Preferences" prefs
    if prefs.python === nothing
        @delete_preferences!("python")
    else
        @set_preferences!("python" => prefs.python)
    end
    if prefs.inprocess
        @set_preferences!("inprocess" => prefs.inprocess)
    else
        @delete_preferences!("inprocess")
    end
    if prefs.conda
        @set_preferences!("conda" => prefs.conda)
    else
        @delete_preferences!("conda")
    end
    PyPreferences.recompile()
    return prefs
end

PythonPreferences(rawprefs::AbstractDict) = PythonPreferences(
    get(rawprefs, "python", nothing),
    get(rawprefs, "inprocess", false),
    get(rawprefs, "conda", false),
)

function _load_python_preferences()
    # TODO: lookup v#.#?
    _python = @load_preference("python", nothing)
    _inprocess = @load_preference("inprocess", false)
    _conda = @load_preference("conda", false)
    #isempty(rawprefs) && return nothing

    # default value
    # if !_inprocess && !_conda && _python === nothing
    #     _python = get_python_fullpath(get_default_python())
    #     @info "Setting default Python interpreter to $(_python)"
    #     return set(python=_python)
    # end
    return PythonPreferences(_python, _inprocess, _conda)
end

function load_pypreferences_code()
    return """
     $(Base.load_path_setup_code())
     PyPreferences = Base.require(Base.PkgId(
         Base.UUID("cc9521c6-0242-4dda-8d66-c47a9d9eec02"),
         "PyPreferences",
     ))
     """
end

function include_stdin_cmd()
    return ```
    $(Base.julia_cmd())
    --startup-file=no
    -e "include_string(Main, read(stdin, String))"
    ```
end

function PyPreferences.recompile()
    code = """
    $(load_pypreferences_code())
    PyPreferences.assert_configured()
    """
    cmd = include_stdin_cmd()
    open(cmd; write = true) do io
        write(io, code)
    end
    return
end

"""
Returns the default python executable used by PyCall. This defaults to
`python3`, and can be overridden by `ENV["PYTHON"]` if it is desired.
"""
get_default_python() = get(ENV,"PYTHON", "python3")

function get_python_fullpath(python)
    python_fullpath = nothing
    if python !== nothing
        python_fullpath = _which(python)
        if python_fullpath === nothing
            @error "Failed to find a binary named `$(python)` in PATH."
        else
            @debug "Found path for command $(python)" python_fullpath
        end
    end
    return python_fullpath
end

function setup_non_failing()
    python = nothing
    inprocess = false
    conda = false
    python_fullpath = nothing
    libpython = nothing
    python_version = nothing
    PYTHONHOME = nothing

    prefs = _load_python_preferences()
    @debug "Loaded python preferences" prefs
    python = prefs.python
    inprocess = prefs.inprocess
    conda = prefs.conda

    if !inprocess 
        if conda
            python = python_fullpath = conda_python_fullpath()
        elseif python === nothing
            python = get_default_python()
        end

        @debug "Python binary selected. Attempting to find the path" python 

        try
            if python !== nothing
                python_fullpath = _which(python)
                if python_fullpath === nothing
                    @error "Failed to find a binary named `$(python)` in PATH."
                else
                    @debug "Found path for command $(python)" python_fullpath
                end
            end

            if python_fullpath !== nothing
                libpython, = find_libpython(python_fullpath)
                python_version = python_version_of(python_fullpath)
                PYTHONHOME = pythonhome_of(python_fullpath)
            end
        catch err
            @error(
                "Failed to configure for `$python`",
                exception = (err, catch_backtrace())
            )
        end

        @debug "Determined python binary path" python_fullpath libpython python_version PYTHONHOME
    end
    if python === nothing
        python = python_fullpath
    end

    return (
        python = python,
        inprocess = inprocess,
        conda = conda,
        python_fullpath = python_fullpath,
        libpython = libpython,
        python_version = python_version,
        PYTHONHOME = PYTHONHOME,
    )
end

function status_inprocess()
    print("""
    python         : $(PyPreferences.python)
    inprocess      : $(PyPreferences.inprocess)
    conda          : $(PyPreferences.conda)
    python_fullpath: $(PyPreferences.python_fullpath)
    libpython      : $(PyPreferences.libpython)
    python_version : $(PyPreferences.python_version)
    PYTHONHOME     : $(PyPreferences.PYTHONHOME)
    """)
end

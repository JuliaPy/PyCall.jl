# This file vendors the utility `Sys.which` for Julia versions prior to 1.7,
# as in 1.6 it called `realpath` and therefore breaks virtual environment
# detection. (Without this, all `test/test_venv.jl` would fail).

@static if VERSION >= v"1.7.0"
    const _which = Sys.which
else
    function _which(program_name::String)
        if isempty(program_name)
           return nothing
        end
        # Build a list of program names that we're going to try
        program_names = String[]
        base_pname = basename(program_name)
        if Sys.iswindows()
            # If the file already has an extension, try that name first
            if !isempty(splitext(base_pname)[2])
                push!(program_names, base_pname)
            end

            # But also try appending .exe and .com`
            for pe in (".exe", ".com")
                push!(program_names, string(base_pname, pe))
            end
        else
            # On non-windows, we just always search for what we've been given
            push!(program_names, base_pname)
        end

        path_dirs = String[]
        program_dirname = dirname(program_name)
        # If we've been given a path that has a directory name in it, then we
        # check to see if that path exists.  Otherwise, we search the PATH.
        if isempty(program_dirname)
            # If we have been given just a program name (not a relative or absolute
            # path) then we should search `PATH` for it here:
            pathsep = Sys.iswindows() ? ';' : ':'
            path_dirs = abspath.(split(get(ENV, "PATH", ""), pathsep))

            # On windows we always check the current directory as well
            if Sys.iswindows()
                pushfirst!(path_dirs, pwd())
            end
        else
            push!(path_dirs, abspath(program_dirname))
        end

        # Here we combine our directories with our program names, searching for the
        # first match among all combinations.
        for path_dir in path_dirs
            for pname in program_names
                program_path = joinpath(path_dir, pname)
                # If we find something that matches our name and we can execute
                if isfile(program_path) && Sys.isexecutable(program_path)
                    return program_path
                end
            end
        end

        # If we couldn't find anything, don't return anything
        nothing
    end
    
    _which(program_name::AbstractString) = _which(String(program_name))
end
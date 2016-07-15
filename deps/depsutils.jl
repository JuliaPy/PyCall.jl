hassym(lib, sym) = Libdl.dlsym_e(lib, sym) != C_NULL

# call dlsym_e on a sequence of symbols and return the symbol that gives
# the first non-null result
function findsym(lib, syms...)
    for sym in syms
        if hassym(lib, sym)
            return sym
        end
    end
    error("no symbol found from: ", syms)
end

# need to be able to get the version before Python is initialized
Py_GetVersion(libpy) = unsafe_string(ccall(Libdl.dlsym(libpy, :Py_GetVersion), Ptr{UInt8}, ()))

using PyCall
PyCall.pyimport("sys")[:executable]

has_numpy = try
    PyCall.pyimport("numpy")
    true
catch
    false
end
if has_numpy
    # Invoke numpy support (which mutates various states in PyCall) to
    # test that it does not introduce any run-time bugs:
    PyCall.npyinitialize()
end

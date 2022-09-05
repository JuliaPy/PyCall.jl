import jnumpy as jnp
import os
import ctypes

os.environ['PYCALL_INPROC_LIBPYPTR'] = hex(ctypes.pythonapi._handle)
os.environ['PYCALL_INPROC_PROCID'] = str(os.getpid())

jnp.init_jl()

jnp.exec_julia("Pkg.activate()")
jnp.exec_julia("using PyCall")
jnp.exec_julia("println(PyCall.python)")
jnp.exec_julia("println(PyCall.start_python_from_julia())")

CPYTHON_OBJECTS = [
	# types
	:PyLong_Type,
	:PyUnicode_Type,
	:PyTuple_Type,
	:PyList_Type,
	:PyBool_Type,
	:PyFloat_Type,
	:PyDict_Type,
	:PySlice_Type,
	# objects
	:_Py_NoneStruct => :Py_None,
	:_Py_TrueStruct => :Py_True,
	:_Py_FalseStruct => :Py_False,
	:_Py_EllipsisObject => :Py_Ellipsis,
	:_Py_NotImplementedStruct => :Py_NotImplemented,
]

for name in CPYTHON_OBJECTS
	jname = Symbol(:C, name isa Pair ? name[2] : name)
	@eval const $jname = Ref{PyPtr}(C_NULL)
end

@eval function capi_init()
	$([begin
	    cname, jname = name isa Pair ? name : name=>name
	    jname = Symbol(:C, jname)
	    cnamesym = QuoteNode(cname)
	    :($jname[] = @pyglobalobj($cnamesym))
	end for name in CPYTHON_OBJECTS]...)
end


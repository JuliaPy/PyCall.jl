const scipysparse_ = PyNULL()

function scipysparse()
    if ispynull(scipysparse_)
        copy!(scipysparse_, pyimport_conda("scipy.sparse", "scipy"))
    end
    return scipysparse_
end

using SparseArrays

function PyObject(S::SparseMatrixCSC)
    scipysparse()["csc_matrix"]((S.nzval, S.rowval .- 1, S.colptr .- 1), shape=size(S))
end

function convert(::Type{SparseMatrixCSC}, o::PyObject)
    I, J, V = scipysparse().find(o)
    return sparse(I .+ 1, J .+ 1, V)
end

using JLD
using Gunrock

function init_parts()
    a = load("../graph.jld")["T"]
    I, J, V = findnz(a)
    len = Int(length(I)/2)
    i1 = I[1:len]
    j1 = J[1:len]
    v1 = V[1:len]
    a1 = sparse(i1, j1, v1)
    a1 = makesquare(a1)
    i2 = I[len+1:end]
    j2 = J[len+1:end]
    v2 = V[len+1:end]
    a2 = sparse(i2, j2, v2)
    a2 = makesquare(a2)
    a1, a2
end

function makesquare(a::SparseMatrixCSC)
    colptr = deepcopy(a.colptr)
    final = colptr[end]
    extra = fill(final, a.m - length(a.colptr))
    append!(colptr, extra)
    amod = SparseMatrixCSC(a.m, a.m, colptr, a.rowval, a.nzval)
    amod = amod + amod'
end

function main()
    info("Reading input . . .")
    a1, a2 = init_parts()
    info("Done.")
    scores = Array(Float32, size(a1, 1))
    info("Calculating pagerank for the first timestep")
    n1, r1 = pagerank(a1)
    info("Updating scores.")
    scores[n1] += r1
    info("Calculating pagerank for the second timestep")
    n2, r2 = pagerank(a2)
    info("Updating scores.")
    scores[n2] += r2
    scores
end

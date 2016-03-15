using Gunrock
function vulnerability(a::SparseMatrixCSC, n::Integer)
    l = size(a, 1)
    d = Dict{Int,Vector{Int}}()
    dists = sssp(a, n)
    max = maximum(dists)
    for i = 0:max
        setindex!(d, Int[], i)
    end
    for i = 1:l
        push!(d[dists[i]], i)
    end
    d
end

function cluster(a::SparseMatrixCSC)
    n = size(a, 1)
    G = map(Int, map(Bool,a) | map(Bool, speye(n)))
    val = maximum(a,2)
    leader = Array(Int, n)
    for i = 1:n
        leader[i] = findin(a[i,:], val[i])[1]
    end
    S = G * sparse(collect(1:n), leader, 1)
    val = maximum(a,2)
    leader = Array(Int, n)
    for i = 1:n
        leader[i] = findin(a[i,:], val[i])[1]
    end
    leader
end

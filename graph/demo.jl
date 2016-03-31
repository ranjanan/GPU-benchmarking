using JLD
using Gunrock

function init_parts(a::SparseMatrixCSC)
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

function setup()
    a = load("../graph.jld")["T"]
    info("Reading input . . .")
    a1, a2 = init_parts(a)
    info("Done.")
    scores = Array(Float32, size(a, 1))
    info("Calculating pagerank for the first timestep")
    n1, r1 = pagerank(a1)
    info("Updating scores.")
    scores[n1] += r1
    info("Calculating pagerank for the second timestep")
    n2, r2 = pagerank(a2)
    info("Updating scores.")
    scores[n2] += r2
    amod = makesquare(a)
    amod , scores
end

function generate_new_graph(l::Integer)
    a = sprand(l, l, 1/l)
    a + a'
end

function monitor(a_original::SparseMatrixCSC, scores::Vector{Float32})
    l = size(a_original, 1)
    i = 3
    while i<=20
        #Normalize scores
        scores = scores / sum(scores)
        
        #Iteration
        info("Time step $i")

        #New set of activity
        a_new = generate_new_graph(l)
    
        #Update state
        a_updated = a_original + a_new
        
        #perform pagerank
        n, r = pagerank(a_updated)

        #Normalize scores
        r = r / sum(r)
        
        #Difference and normalize
        diff = abs(scores[n] - r)

        #Find max diff
        val, pos = findmax(diff)
        @show val

        #Check if diff is too much
        if val > 0.15
            node = n[pos]
            info("Alert at node $node !")
            break
        end

        #Update scores
        scores = scores + diff
    
        #Update Iteration
        i += 1

    end
    a_original, scores
end

function driver()
    a_original, scores = setup()
    a, s = monitor(a_original, scores)
end

a,s = driver() #Set up 20 time steps
s[100] = 1 #Set dramatic deviation
monitor(a,s) #Start monitoring again

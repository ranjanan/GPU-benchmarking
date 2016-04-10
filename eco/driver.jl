include("laplacian.jl")
include("solveCG.jl")
using CUSPARSE, CUSOLVER, CUDArt
using JLD
function main()

	#a = sprand(10^6,10^6,1e-6)
	#a = a + a'
	a = load("../graph.jld")["T"]
	colptr = deepcopy(a.colptr)
	final = colptr[end]
	extra = fill(final, a.m - 473)
	append!(colptr, extra)
	amod = SparseMatrixCSC(a.m, a.m, colptr, a.rowval, a.nzval)
	amod = amod + amod'
	lap = laplacian(amod)
	b = zeros(size(amod,1))
	lap = round(Float64, lap)

	#source is 1
	b[1] = 1
	
	#since destination is 2, delete 2nd row and second column from matrix
	i,j,v = findnz(lap)
	indy = findin(j, 2)
	deleteat!(i, indy)
	deleteat!(j, indy)
	deleteat!(v, indy)
	for k = 1:size(j,1)
		if j[k] > 2
			j[k] -= 1
		end
	end
	indx = findin(i, 2)
	deleteat!(j, indx)
	deleteat!(i, indx)
	deleteat!(v, indx)
	for k = 1:size(i,1)
		if i[k] > 2
			i[k] -= 1
		end
	end
	lap_mod = sparse(i,j,v)
	splice!(b, size(amod, 1) - 1)
	
	#Transfer to GPU
	lapd = CudaSparseMatrixCSR(lap_mod)
	bd  = CudaArray(b)
	#cd = CudaArray(zeros(10^2 - 1))
	
	#Perform solve
	#@time CUSOLVER.csrlsvchol!(lapd, bd, cd, 1e-6, zero(Cint), 'O')
	@time cd = solveCG(lapd, bd, bd, 1e-6, 200)

end
	

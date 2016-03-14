using FactCheck, JLD
include("funk.jl")
function benchmark(k)

	println("-------------- k = $k ------------") 

	a = sprand(10^7, 10^7, k*1e-7)
	
	println("Time taken to transfer matrix to device:")
	@time ad = CudaSparseMatrixCSR(a)

	b = rand(10^7)
	println("Time taken to transfer vector: ")
	@time bd = CudaArray(b)

	println("Time taken for multiplication: ")
	@time cd = ad * bd

	println("Time taken to transfer back to host: ")
	@time c = to_host(cd)

	println("Time taken for multiplication on CPU")
	@time ch = a * b

	facts("Checking for correctness for k = $k") do
		@fact c => roughly(ch)
	end
end

function run_benchmark()
	for k = 1:9
		benchmark(k)
	end
end

function warmup()
	a = sprand(100,100,0.1)
	b = rand(100)
	ad = CudaSparseMatrixCSR(a)
	bd = CudaArray(b)
	ad * bd
	info("warmup done.")
end

function realdata_benchmark()
	warmup()
	a = load("../graph.jld")["T"]
	colptr = deepcopy(a.colptr)
	final = colptr[end]
	extra = fill(final, a.m - 473)
	append!(colptr, extra)
	amod = SparseMatrixCSC(a.m, a.m, colptr, a.rowval, a.nzval)
	amod = round(Float64, amod)
	b = zeros(size(amod,1))
	b[1] = 1
	@time amod * b
	ad = CudaSparseMatrixCSR(amod)
	bd  = CudaArray(b)
	t1 = @elapsed ad * bd
	println("Time taken for csrmv = $t1")
end 

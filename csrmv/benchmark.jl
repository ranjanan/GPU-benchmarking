using FactCheck
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

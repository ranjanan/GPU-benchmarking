#using FactCheck
#include("funk.jl")
function benchmark(k)

	println("-------------- k = $k ------------") 

	a = sprand(10^7, 10^7, k*1e-7)
	
	b = sprand(10^7, 1, k*1e-5)
	bs = SparseVector(b)
	bf = full(b)
	
	println("Time taken for spm - spm mult: ")
	@time a * b

	println("Time taken for spm - spv mult: ")
	@time a * bs

	println("Time taken for spm - dv mult: ")
	@time a * bf
end
function run_benchmark()
	for k = 1:9
		benchmark(k)
	end
end

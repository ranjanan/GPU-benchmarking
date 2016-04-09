using CUBLAS, CUDArt
function benchmark()
	for k in [0.5, 0.75, 1, 1.25]
		info("--- k = $k ---")
		dim = Int(k * 10^4)
		a = rand(dim, dim)
		b = deepcopy(a)
		@time ad = CudaArray(a)
		@time bd = CudaArray(b)
		t1 = @elapsed a * b
		t2 = @elapsed CUBLAS.gemm('N', 'N', ad, bd)
		t3 = @elapsed CUBLAS.gemm('N', 'N', a, b)
		println("Time taken for CPU version = $t1")
		println("Time taken for GPU version = $t2")
		println("Time taken for GPU with tranfer version = $t3")
		CUBLAS.free(ad)
		CUBLAS.free(bd)
	end
end

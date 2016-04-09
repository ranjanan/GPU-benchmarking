using CUBLAS, CUDArt
function benchmark()
	warmup()
	for k in [0.5, 0.75, 1, 1.25]
		info("--- k = $k ---")
		dim = Int(k * 10^4)
		a = rand(dim, dim)
		a = a + a'
		b = deepcopy(a)
		@time ad = CudaArray(a)
		@time bd = CudaArray(b)
		t1 = @elapsed a * b
		t2 = @elapsed CUBLAS.symm('L', 'U', ad, bd)
		t3 = @elapsed CUBLAS.symm('L', 'U', a, b)
		println("Time taken for CPU version = $t1")
		println("Time taken for GPU version = $t2")
		println("Time taken for GPU with tranfer version = $t3")
		CUBLAS.free(ad)
		CUBLAS.free(bd)
	end
end

function warmup()
	a = rand(100,100)
	a = a + a'
	b = deepcopy(a)
	ad = CudaArray(a)
	bd = CudaArray(b)
	CUBLAS.symm('L', 'U', ad, bd)
	c2 = CUBLAS.symm('L', 'U', a, b)
	c1 = a * b
	if sum(c1 - c2) < 1e-6
		info("Test passed")
	end 
	println("Warm up done.")
end

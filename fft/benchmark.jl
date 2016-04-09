using CUBLAS, CUDArt, CUFFT
function benchmark()
	warmup()
	for k in [0.5, 0.75, 1, 1.25]
		info("--- k = $k ---")
		dim = Int(k * 10^4)
		a = rand(dim, dim)
		@time ad = CudaArray(a)
		
		gfft = CudaArray(Complex{Float64}, div(size(a,1), 2) + 1, size(a,2))

		t2 = @elapsed begin
		p = plan(gfft, ad)
		p(gfft, ad, true)
		end
		t1 = @elapsed rfft(a)
		
		println("Time taken for CPU version = $t1")
		println("Time taken for GPU version = $t2")
		CUBLAS.free(ad)
		CUBLAS.free(gfft)
	end
end

function warmup()
	a = rand(100,100)
	ad = CudaArray(a)
	gfft = CudaArray(Complex{Float64}, 51, 100)
	p = plan(gfft, ad)
	p(gfft, ad, true)
	c2 = to_host(gfft)
	c1 = rfft(a)
	if abs(sum(c1 - c2)) < 1e-6
		info("Test passed")
	end 
	println("Warm up done.")
end

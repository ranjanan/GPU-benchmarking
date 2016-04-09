using CUSPARSE, CUDArt, CUSOLVER
function benchmark()
	warmup()
	for k in [10^2, 10^3, 10^4, 10^5]
		info("--- dim = $k X $k ---")
		dim = k
		a = sprand(dim, dim, 1/dim)
		a = a + a'
		a = a + dim*speye(dim)
		b = rand(dim)
		x = zeros(dim)
		ad = CudaSparseMatrixCSR(a)
		bd = CudaArray(b)
		xd = CudaArray(x)
		t1 = @elapsed CUSOLVER.csrlsvchol!(ad, bd, xd, 1e-6, zero(Cint), 'O')
		t2 = @elapsed a \ b
		println("Time for CPU version = $t1")
		println("Time for GPU version = $t2")
		#CUDArt.free(ad)
		#CUDArt.free(bd)
	end
end

function warmup()
	a = sprand(100,100,1e-2)
	a = a + a'
	a = a + 100*speye(100)
	b = rand(100)
	x = zeros(100)
	ad = CudaSparseMatrixCSR(a)
	bd = CudaArray(b)
	xd = CudaArray(x)
	CUSOLVER.csrlsvchol!(ad, bd, xd, 1e-6, zero(Cint), 'O')
	x = to_host(xd)
	xcpu = a \ b
	if sum(x - xcpu) < 1e-6
		info("Test passed")
	end
end

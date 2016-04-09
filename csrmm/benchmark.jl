using CUSPARSE, CUDArt, JLD
function benchmark()
	warmup()
	for dim in [10^6, 50*10^6, 10^7]
		info(" Dim = $dim")
		a = sprand(dim, dim, 1/dim)
		ad = CudaSparseMatrixCSR(a)
		t1 = @elapsed bd = CUSPARSE.gemm('N', 'N', ad, ad, 'O', 'O', 'O')
		t2 = @elapsed a * a
		println("Time taken for GPU : $t1")
		println("Time taken for CPU : $t2")
	end
	realdata_benchmark()
end

function warmup()
	a = sprand(100,100,0.1)
	ad = CudaSparseMatrixCSR(a)
	bd = CUSPARSE.gemm('N', 'N', ad, ad, 'O', 'O', 'O')
	b1 = a * a
	b2 = to_host(bd)
	if sum(b1 - b2) < 1e-6
		info("Test passed")
	end
end

function realdata_benchmark()
	a = load("/home/ubuntu/GPU-benchmarking/graph.jld")["T"]
	colptr = deepcopy(a.colptr)
	final = colptr[end]
	extra = fill(final, a.m - 473)
	append!(colptr, extra)
	amod = SparseMatrixCSC(a.m, a.m, colptr, a.rowval, a.nzval)
	amod = round(Float64, amod)
	ad = CudaSparseMatrixCSR(amod)
	t1 = @elapsed CUSPARSE.gemm('N', 'N', ad, ad, 'O', 'O', 'O')
	println("Time taken for realdata = $t1")
end	

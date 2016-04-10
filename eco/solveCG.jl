include("utils.jl")
function solveCG(A, f, s, tol, maxiter)

	#Declare on GPU
	A = CudaSparseMatrixCSR(A)
	f = CudaArray(f)
	s = CudaArray(s)

	u = s
	t = A*s
	r = CUBLAS.copy(f)
	sub!(t, r)
	p = CUBLAS.copy(r)
	rho = dot(r,r)
	niter = 0
	flag = 0
	normf = norm(f)	
	if normf < eps()
		normf = 1
	end
	
	while (norm(r) / normf ) > tol
		a = A * p
		alpha = rho / dot(a,p)
		CUBLAS.axpy!(alpha, p, u)
		CUBLAS.axpy!(-alpha, a, r)
		rho_new = dot(r,r)
		#p = r + (rho_new/rho) * p
		temp = rho_new/rho
		r1 = CUBLAS.copy(r)
		CUBLAS.axpy!(temp, p, r1)
		p = r1
		rho = rho_new
		niter = niter + 1
		if niter == maxiter 
			flag = 1
			break
		end
	end
	#to_host(u), niter, flag
	to_host(u)
end

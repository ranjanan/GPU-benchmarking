using CUSPARSE, CUDArt, CUBLAS
import Base.*, Base.norm, Base.dot

#b is replaced by a + b
function add!(a::CudaArray, b::CudaArray)
	CUBLAS.axpy!(length(a), 1.0, a, 1, b, 1)
end

#b is replaced by b - a
function sub!(a::CudaArray, b::CudaArray)
	CUBLAS.axpy!(length(a), -1.0, a, 1, b, 1)
end

#dot product
function dot(a::CudaArray, b::CudaArray)
	CUBLAS.dot(a,b)
end

#norm
function norm(a::CudaArray)
	CUBLAS.nrm2(a)
end

function *(a::CudaSparseMatrixCSR, b::CudaArray)
    CUSPARSE.csrmv('N', a, b, 'O')
end

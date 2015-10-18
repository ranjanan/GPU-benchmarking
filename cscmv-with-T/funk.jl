using CUSPARSE, CUDArt

import Base.*
function *(a::CudaSparseMatrixCSR, b::CudaArray)
	CUSPARSE.csrmv('T', a, b, 'O')
end

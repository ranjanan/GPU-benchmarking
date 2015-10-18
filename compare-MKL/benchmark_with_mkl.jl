using MKLSparse
function benchmark(k)

	println("----------- k = $k ---------") 

	a = sprand(10^7, 10^7, k*1e-7)
	
	b = rand(10^7)

	println("Time taken for multiplication:")
	@time ch = a * b

end

function run_benchmark()
	println("----------- MKL --------------")
	for k = 1:9
		benchmark(k)
	end
end

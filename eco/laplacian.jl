function laplacian(a)
	a = -abs(a)
	a = spones(a)
	a = a - diagm(sparse(diag(a)))
	a = a - diagm(sparse(sum(a,2)))
	sparse(-a)
end

function laplacian2(a)
	a = spones(a)
	diags = sum(a,2)
	@time for i = 1:size(a,1)
		a[i,i] = -diags[i]
	end
	-a
end


#
#	apply Heff
#
function applyHeff(x::AbstractArray{ComplexF64}, localMPO::MPO, L::Array{<:Any,1}, R::Array{<:Any,1})	
	
	res = zeros(ComplexF64, size(L[1])[1], size(x)[2], size(R[1])[1])
	tmp = zeros(ComplexF64, size(L[1])[2], size(R[1])[1])
	
	
	@inbounds for op in eachoperation(localMPO) # potentially possible to run in parallel 
		
		@inbounds r = @view R[op[3][2]][:, :]
		@inbounds l = @view L[op[3][1]][:, :]
		c = ComplexF64(op[1])	

		m = @view x[:, op[2][1], :]
		new_m = @view res[:, op[2][2], :] 

		LinearAlgebra.BLAS.gemm!('N','T', c, m, r, ComplexF64(0.0), tmp)
		LinearAlgebra.BLAS.gemm!('N', 'N', ComplexF64(1.0), l, tmp, ComplexF64(1.0), new_m)
	 end

	return res #(α_bra, d_bra, β_bra)
end

function applyHeff2(x::AbstractArray{<:Number}, localMPO1::MPO, localMPO2::MPO, L, R)

	res = zeros(ComplexF64, size(x))

	mpo1 = getMPOTensor(localMPO1)
	mpo2 = getMPOTensor(localMPO2)

	@tensor bondOp[α, d1, d2, d1', d2', β] := mpo1[d1, α, γ, d2]*mpo2[d1', γ, β, d2']

	for (i, l) in enumerate(L)

		@tensor tmp[α, d1, d2, β] := l[α, γ]*x[γ, d1, d2, β]

		for (j, r) in enumerate(R)
			op = @view bondOp[i,:,:,:,:,j]	
			@tensor tmp2[α, d1, d2, β] := tmp[α, d1, d2, γ]*r[β, γ]

			@tensor tmp2[α, d2, d2', β] := op[d1, d2, d1', d2']*tmp2[α, d1, d1', β]

			res+=tmp2
		end
	end

	return res

end



function applyHCeff(x::AbstractArray{<:Any}, L::Array{<:Any,1}, R::Array{<:Any,1})			
	res = zeros(ComplexF64, size(L[1])[1], size(R[1])[1])
	tmp = similar(res)
	for diag_iter = 1:length(L)
		
		@inbounds l = @view L[diag_iter][:,:]
		@inbounds r = @view R[diag_iter][:,:]

		LinearAlgebra.BLAS.gemm!('N', 'N', ComplexF64(1.0), l, x, ComplexF64(0.0), tmp)
		LinearAlgebra.BLAS.gemm!('N', 'T', ComplexF64(1.0), tmp, r, ComplexF64(1.0), res)

 	end
	
	return res
end

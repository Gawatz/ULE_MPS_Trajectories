using LinearAlgebra
using JLD
using TensorOperations
using MPOmodule


function get_MPOTensor(MPO)
	bin = 0; bout = 0
	try 
		bin = MPO.bDim[1]
		bout = MPO.bDim[2]
	catch
		bin = MPO.bDim
		bout = bin
	end

	ddim = size(MPO.Operator[1])[1]
	mpo_tensor = zeros(Complex{Float64}, ddim, bin, bout, ddim)

	# load mpo_tensor
	Idx = Array(Transpose(LinearIndices(fill(1,bout,bin))))
	#@show Idx
	#@show MPO.Op_index
	
	Idx = [findall(x->x == idx, Idx)[1] for idx in MPO.Op_index]
	#@show Idx
	for (i,x) in enumerate(Idx)
		mpo_tensor[:, x[1], x[2], :] = MPO.Operator[i]
	end

	return mpo_tensor #(d, bin, bout, d′) 
end

function Op_to_MPO(N, Op; tol = 1e-7)	
	MPOvec = []
	bout_pre = 1
	d1, d2 = size(Op)
	tmp = zeros((1,d1,d2))
	tmp[1,:,:] = Op
	for site in 1:N

		bout_pre, d1, d2 = size(tmp)
		tmp = reshape(tmp, bout_pre, 2, Int(d1/2), 2, Int(d2/2)) # ( d_first_site, d_rest, d′_first_site, d′_rest)

		tmp = permutedims(tmp, [1, 2, 4, 3, 5]) # (bin, d_first_site, d′_first_site, d_rest, d′_rest)

		tmp = reshape(tmp, 4*bout_pre, Int(d1/2)*Int(d2/2))

		F = svd(tmp)

		S = F.S
		S = S[S.>= tol]
		U = F.U[:,1:size(S)[1]]

		

		V = F.V'[1:size(S)[1],:]	
		tmp = diagm(0=>S)*V #(bout_new, d_rest)


		tmp = reshape(tmp, size(tmp)[1], Int(d1/2), Int(d2/2)) #(bin, d_rest, d′_rest)

		mpo = permutedims(reshape(U, bout_pre, 2, 2, Int(size(U)[2])), [2,1,4,3])
		mpo = site == N ? tmp[1]*mpo : mpo
		push!(MPOvec, mpo)
	end
	return MPOvec
end

function creatOp_from_MPO(MPOvec::Vector{AbstractArray{<:Number,4}})
	Op = ones(Complex{Float64},1,1,1,1)
	for mpo in MPOvec
		bin = size(Op)[2]
		bout = size(mpo)[end-1]
	
		@tensor new_Op[d1, d1_next, bin, bout, d2, d2_next] := Op[d1,bin, γ, d2]*mpo[d1_next, γ, bout, d2_next]
		
		Op = reshape(new_Op, *(size(new_Op)[1:2]...), bin, bout, *(size(new_Op)[end-1:end]...))
		
	
	end

	return Op
end

function creatOp_from_MPO(MPOvec::Vector{MPO})
	Op = ones(Complex{Float64},1,1,1,1)
	for mpo in MPOvec
		mpo = get_MPOTensor(mpo)
		bin = size(Op)[2]
		bout = size(mpo)[end-1]
	
		@tensor new_Op[d1, d1_next, bin, bout, d2, d2_next] := Op[d1,bin, γ, d2]*mpo[d1_next, γ, bout, d2_next]
		
		Op = reshape(new_Op, *(size(new_Op)[1:2]...), bin, bout, *(size(new_Op)[end-1:end]...))
		
	
	end

	return Op
end

function MPOTensor_to_MPO(MPOvec)
	new_MPOvec = Vector{MPO}([])
	for mpo in MPOvec
		
		d, bin, bout, __ = size(mpo)

		bDim = bin == bout ? bin : (bin, bout)

		Idx = LinearIndices((bout, bin))

		OpVec = []
		IdxVec = []
		for b_in in 1:bin
			for b_out in 1:bout
				
				if isapprox(mpo[:,b_in, b_out, :], zeros(2,2), atol = 1e-15) == false
					
					push!(IdxVec, Idx[b_out, b_in])
					push!(OpVec, mpo[:, b_in, b_out, :])

				end
			end
		end
		
		push!(new_MPOvec, MPO(bDim, OpVec, IdxVec))

	end

	return new_MPOvec
end

import Base.*
function (*)(MPOvecA::Vector{MPO}, MPOvecB::Vector{MPO})

	@assert size(MPOvecA) == size(MPOvecB) "you are trying to mulitply
					to MPO with different length"

	new_MPOvec = []
	for i in 1:size(MPOvecA)[1]

		mpoA = get_MPOTensor(MPOvecA[i])
		mpoB = get_MPOTensor(MPOvecB[i])

		@tensor new_mpo[α, α′, d, d′, β, β′] := mpoA[d, α, β, γ]*mpoB[γ, α′, β′, d′]
		
		new_mpo = reshape(new_mpo, *(size(new_mpo)[1:2]...), size(new_mpo)[3], size(new_mpo)[4], *(size(new_mpo)[5:end]...))
		new_mpo = permutedims(new_mpo, [2,1,4,3])
		push!(new_MPOvec, new_mpo)
	end
	
	return MPOTensor_to_MPO(new_MPOvec)

end

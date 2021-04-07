using TensorOperations, KrylovKit, LinearAlgebra
using MPOmodule2

#
#	apply TM 
#
function applyTM_OP(MPS_ket::AbstractArray{<:Number}, MPS_bra::AbstractArray{<:Number}, Op::localOp, 
			x::AbstractArray{<:Number}; left::Bool = true)
	
	res = zeros(eltype(x), size(MPS_bra)[[1,3][left+1]], size(MPS_ket)[[1,3][left+1]])
	tmp = zeros(eltype(x), size(x)[2], size(MPS_bra)[[1,3][left+1]])
	if left == true 
		@inbounds for op in eachoperation(Op)

			c = ComplexF64(op[1])
			mket = @view MPS_ket[:, op[2], :]	#α_ket, β_ket
			mbra = @view MPS_bra[:, op[3], :]	#α_bra, β_bra

			LinearAlgebra.BLAS.gemm!('T','N', c, x, conj.(mbra), ComplexF64(0.0), tmp)  #α_ket, β_bra
			LinearAlgebra.BLAS.gemm!('T', 'N', ComplexF64(1.0), tmp, mket, ComplexF64(1.0), res)
		end

	else
		
		@inbounds for op in eachoperation(Op)
			
			c = ComplexF64(op[1])
			mket = @view MPS_ket[:, op[2], :]	#α_ket, β_ket
			mbra = @view MPS_bra[:, op[3], :]	#α_bra, β_bra

			LinearAlgebra.BLAS.gemm!('T','T', c, x, conj.(mbra), ComplexF64(0.0), tmp)  #β_ket, α_bra
			LinearAlgebra.BLAS.gemm!('T', 'T', ComplexF64(1.0), tmp, mket, ComplexF64(1.0), res)
		end



	end
	#=
	Op = Array(Op)
	
	if left == true
		@tensor tmp[αket, dbra, βbra] := conj(MPS_bra)[γ, dbra, βbra]*x[γ, αket] #scales with dD³
		@tensor tmp[αket, dket, βbra] := Op[dket, dbra]*tmp[αket, dbra, βbra] #scales with d²D²
		@tensor res[βbra, βket] := MPS_ket[αket, dket, βket]*tmp[αket, dket, βbra] #scales with dD³
	else	
		@tensor tmp[αbra, dbra, βket] := conj(MPS_bra)[αbra, dbra, γ]*x[γ, βket] #scales with dD³		
                @tensor tmp[αbra, dket, βket] := Op[dket, dbra]*tmp[αbra, dbra, βket] #scales with d²D²
                @tensor res[αbra, αket] := MPS_ket[αket, dket, βket]*tmp[αbra, dket, βket] #scales with dD³

	end
	=#
	return res
end

function applyTM_OP(MPS_ket::Vector{<:Any}, MPS_bra::Vector{<:Any}, Op_string::NTuple{N,Int}, vecMPO::Vector{MPO}, 
 		  x::AbstractArray{<:Number}; left::Bool = true) where {N}
	
	# TODO check that MPS_ket is the same length as MPS_bra and that phys dim match
	unitCellLength = length(MPS_ket)

	if left == true 
		A = MPS_ket[1]
        	B = MPS_bra[1]
		local_op = vecMPO[1][Op_string[1]]
	else 
		A = MPS_ket[end]
		B = MPS_bra[end]
		local_op = vecMPO[end][Op_string[end]]
	end


	res = left == true ? applyTM_OP(A, B, local_op, x; left = true) : applyTM_OP(A, B, local_op, x; left = false)

	# go thorugh unit cell
	if unitCellLength > 1
		if left == true
			res = applyTM_OP(MPS_ket[2:end], MPS_bra[2:end], Op_string[2:end], vecMPO[2:end], res; left = true)
		else
			res = applyTM_OP(MPS_ket[1:end-1], MPS_bra[1:end-1], Op_string[1:end-1], vecMPO[1:end-1], res; left = false)
		end
	end

		
	return res # (α_bra , α_ket) 
end

function applyTM_MPO(MPSket::Vector{<:Any}, MPSbra::Vector{<:Any}, MPOvec::Vector{MPO}, x::Vector{<:Any}; left::Bool = true)
	MPS_ChainLength = length(MPSket)

	#check if same length then MPOvec
	
	A = left == true ? MPSket[1] : MPSket[end]
	B = left == true ? MPSbra[1] : MPSbra[end]
	local_MPO = left == true ? MPOvec[1] : MPOvec[end]
	b_dim = length(local_MPO.bDim) == 1 ? (local_MPO.bDim, local_MPO.bDim) : local_MPO.bDim
	Op_idx = local_MPO.Op_index
	d_phys = size(MPSket[1])[2]
	
	if left == true
		YLa = Vector{Any}(undef,b_dim[2])
	
		for diag_idx = b_dim[2]:-1:1 #go through outgoing bond dim

                	a = diag_idx
			YLa[a] = zeros(ComplexF64,size(B)[3],size(A)[3])
		
			for b = 1:b_dim[1]
				
				Op = local_MPO[b, a]

				if Op != nothing

					tmp = x[b]		
					tmp = applyTM_OP(A, B, Op, tmp; left = true)
					YLa[a] = YLa[a] + tmp #(α_bra ,α_ket)	
				else

					continue
					#YLa[a] = YLa[a] #(α_bra ,α_ket)
				end
                		
                                                                                                                              
                	end
                end
	else
		YRa = Vector{Any}(undef,b_dim[1])
		for diag_idx = 1:b_dim[1] 
			a = diag_idx
			YRa[a] = zeros(ComplexF64,size(B)[1],size(A)[1])
			
			for b = b_dim[2]:-1:1
				
				
				Op = local_MPO[a, b]
				
				if Op != nothing	
					
					tmp = x[b]		
					tmp = applyTM_OP(A, B, Op, tmp; left = false)
					YRa[a] = YRa[a] + tmp #(α_bra ,α_ket)	
				else
					continue
					#YRa[a] = YRa[a] #(α_bra ,α_ket)
				end

			end		                                                                                 
		end                                                                                                      	
	
	end



	if MPS_ChainLength > 1
		if left == true
			YLa = applyTM_MPO(MPSket[2:end], MPSbra[2:end], MPOvec[2:end], YLa; left = true)
		else
			YRa = applyTM_MPO(MPSket[1:end-1], MPSbra[1:end-1], MPOvec[1:end-1], YRa; left = false)
		end
	
	end

	return res = left == true ? YLa : YRa
end

applyTM_MPO(MPS::Vector{<:Any}, MPOvec::Vector{MPO}, x::Vector{<:Any}; left::Bool = true) = applyTM_MPO(MPS, MPS, MPOvec, x; left = left)


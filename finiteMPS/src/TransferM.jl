#
#	Transfer Matrix routines
#

#
#	To DO: 
#	- eltype dependent transfer matricies 
#



"""
    applyTMop(Mket, Mbra, local_op, x; left = true)

Applies a transfer matrix from the left repectively from the right depending on option left.

!!! ADD DESCRIPTION !!!

# Arguments
- Mket:
- Mbra:
- local_op:
- x:
- left: if true applies transfer matrix from the left if false from the right

return:

"""
function applyTM_OP(Mket::AbstractArray{<:Number}, Mbra::AbstractArray{<:Number}, local_op::localOp, 
			x::AbstractArray{<:Number}; left::Bool = true)
	
	res = zeros(eltype(x), size(Mbra)[[1,3][left+1]], size(Mket)[[1,3][left+1]])
	tmp = zeros(eltype(x), size(x)[2], size(Mbra)[[1,3][left+1]])
	if left == true 
		@inbounds for op in eachoperation(local_op)

			op_coef = ComplexF64(op[1])
			mket = @view Mket[:, op[2], :]	#α_ket, β_ket
			mbra = @view Mbra[:, op[3], :]	#α_bra, β_bra

			LinearAlgebra.BLAS.gemm!('T','N', op_coef, x, conj.(mbra), ComplexF64(0.0), tmp)  #α_ket, β_bra
			LinearAlgebra.BLAS.gemm!('T', 'N', ComplexF64(1.0), tmp, mket, ComplexF64(1.0), res)
		end

	else
		
		@inbounds for op in eachoperation(local_op)
			
			op_coef = ComplexF64(op[1])
			mket = @view Mket[:, op[2], :]	#α_ket, β_ket
			mbra = @view Mbra[:, op[3], :]	#α_bra, β_bra

			LinearAlgebra.BLAS.gemm!('T','T', op_coef, x, conj.(mbra), ComplexF64(0.0), tmp)  #β_ket, α_bra
			LinearAlgebra.BLAS.gemm!('T', 'T', ComplexF64(1.0), tmp, mket, ComplexF64(1.0), res)
		end



	end

	#=
	#	
	Op = Array(Op)
	
	if left == true
		@tensor tmp[αket, dbra, βbra] := conj(MPSbra)[γ, dbra, βbra]*x[γ, αket] #scales with dD³
		@tensor tmp[αket, dket, βbra] := Op[dket, dbra]*tmp[αket, dbra, βbra] #scales with d²D²
		@tensor res[βbra, βket] := MPSket[αket, dket, βket]*tmp[αket, dket, βbra] #scales with dD³
	else	
		@tensor tmp[αbra, dbra, βket] := conj(MPSbra)[αbra, dbra, γ]*x[γ, βket] #scales with dD³		
                @tensor tmp[αbra, dket, βket] := Op[dket, dbra]*tmp[αbra, dbra, βket] #scales with d²D²
                @tensor res[αbra, αket] := MPSket[αket, dket, βket]*tmp[αbra, dket, βket] #scales with dD³

	end
	=#
	
	return res
end



"""
    applyTMop(MPSket, MPSbra, op_string, MPOvec, x; left = true)

Applies a transfer matrix from the left repectively from the right depending on option left.

!!! ADD DESCRIPTION !!!

# Arguments
- MPSket:
- MPSbra:
- op_string:
- MPOvec:
- x:
- left: if true applies transfer matrix from the left if false from the right

return:

"""
function applyTM_OP(MPSket::Vector{<:Any}, MPSbra::Vector{<:Any}, op_string::NTuple{N,Int}, MPOvec::Vector{MPO}, 
 		  x::AbstractArray{<:Number}; left::Bool = true) where {N}

	# to do: make these checks obsolet by defining structures for MPS and MPO
	length(MPSket) == length(MPSbra) || throw(DomainError("MPS ket and bra are not of the same length"))
	length(MPSket) == length(MPOvec) || throw(DomainError("MPSs and MPO are not of the same length"))
	N == length(MPOvec) || throw(DomainError("op_string not of the same length as MPOvec"))	

	system_size = length(MPSket)

	if left == true 
		A = MPSket[1]
        	B = MPSbra[1]
		local_op = MPOvec[1][op_string[1]]
	else 
		A = MPSket[end]
		B = MPSbra[end]
		local_op = MPOvec[end][op_string[end]]
	end


	res = left == true ? applyTM_OP(A, B, local_op, x; left = true) : applyTM_OP(A, B, local_op, x; left = false)

	# go through MPS chain
	if system_size > 1
		if left == true
			res = applyTM_OP(MPSket[2:end], MPSbra[2:end], op_string[2:end], MPOvec[2:end], res; left = true)
		else
			res = applyTM_OP(MPSket[1:end-1], MPSbra[1:end-1], op_string[1:end-1], MPOvec[1:end-1], res; left = false)
		end
	end

		
	return res # (α_bra , α_ket) 
end

"""
    applyTM_MPO(MPSket, MPSbra, op_string, MPOvec, x; left = true)

Applies a transfer matrix from the left repectively from the right depending on option left.

!!! ADD DESCRIPTION !!!

# Arguments
- MPSket:
- MPSbra:
- MPOvec:
- x:
- left: if true applies transfer matrix from the left if false from the right

return:

"""
function applyTM_MPO(MPSket::Vector{<:Any}, MPSbra::Vector{<:Any}, MPOvec::Vector{MPO}, x::Vector{<:Any}; left::Bool = true)	
	length(MPSket) == length(MPSbra) || throw(DomainError("MPS ket and bra are not of the same length"))
	length(MPSket) == length(vecMPO) || throw(DomainError("MPSs and MPO are not of the same length"))


	system_size = length(MPSket)
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



	if system_size > 1
		if left == true
			YLa = applyTM_MPO(MPSket[2:end], MPSbra[2:end], MPOvec[2:end], YLa; left = true)
		else
			YRa = applyTM_MPO(MPSket[1:end-1], MPSbra[1:end-1], MPOvec[1:end-1], YRa; left = false)
		end
	
	end

	return res = left == true ? YLa : YRa
end

applyTM_MPO(MPS::Vector{<:Any}, MPOvec::Vector{MPO}, x::Vector{<:Any}; left::Bool = true) = applyTM_MPO(MPS, MPS, MPOvec, x; left = left)


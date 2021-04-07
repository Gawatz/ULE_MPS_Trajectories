
"""
    getOverlap(MPSA, MPSB)

calculates the between to MPS encoded in MPSA and MPSB: < MPSB | MPSA >

# Arguments
- MPSA: vector containing the ket of the overlap
- MPSB: vector containing the bra of the overlap

return:


"""
function getOverlap(MPSA::Vector{<:Any}, MPSB::Vector{<:Any})
	Length = Int(size(MPSA)[1])

        @tensor T_multi[α_bra, α_ket, β_bra, β_ket] := (MPSA[1][α_ket, d, β_ket] * 
							conj(MPSB[1][α_bra, d, β_bra]))

        for site = 2:Length
        	#println("add ",site)		
        	@tensor begin
        		
        		E_local[a_in',a_in,a_out',a_out] := (MPSA[site][a_in, d, a_out] * 
        						     conj(MPSB[site][a_in', d, a_out']))
        	
        		T_multi[α_bra, α_ket, β_bra, β_ket] := (T_multi[α_bra, α_ket, γ_bra, γ_ket] * 
        							E_local[γ_bra, γ_ket, β_bra, β_ket])
        	end                                                  
        end

	return T_multi #(α_bra, α_ket, β_bra, β_ket)
end


"""
    getExpValue(MPSvec, MPOvec)

calculates the expectation value of an MPO : < MPSvec | MPOvec | MPSvec >

# Arguments
- MPSvec: vector of M's
- MPOvec: vector of MPO's

return:


"""
function getExpValue(MPSvec::Vector{<:Any}, MPOvec::Vector{MPO})
	OpString = stringMPO(MPOvec)
	expValue = 0
	for (i, x) in enumerate(OpString[1])
		test = applyTM_OP(MPSvec, MPSvec, x, MPOvec, Array{ComplexF64}(I,1,size(MPSvec[1])[1]))
		expValue += tr(test)
	end

	return expValue
end


"""
    singleSiteExpValue(MPSvec, C, Op, site; can_left = true)

calculates the expectation value of a single site Operator Op given
the local MPS-matrix MPSvec[site] and singular-values between the next(previous) bond
depending on the left(right) canonical condition of M specified by can_left.

# Arguments
- MPSvec: containing the local MPS-matrices 
- C: singular values missing to bring M into mixed canonical form
- Op: local single site operator
- site: site on which Op should act
- can_left = true : is MPSvec[site] in left 0r right can. form.

return:
	
	expectation values of < MPSvec | Op_site | MPSvec >

"""
singleSiteExpValue(MPSvec::Vector{<:Any}, C::AbstractArray{<:Number}, 
		   Op::AbstractArray{<:Number}, site::Int; can_left=true) = singleSiteExpValue(MPSvec[site], C, Op, can_left = can_left)
	
function singleSiteExpValue(M::AbstractArray{<:T}, C::AbstractArray{<:T}, Op::AbstractArray{<:T}; can_left = true) where T<:Number	
	if can_left
		@tensor tmp[β_ket,α_ket,d] := C[γ, β_ket] * M[α_ket, d, γ]
	else
		@tensor tmp[β_ket,α_ket,d] := C[α_ket, γ] * M[γ, d, β_ket]
	end

	@tensor tmp[d_ket,d_bra] := tmp[β,α,d_ket] * conj(tmp[β,α,d_bra])  
	@tensor expValue[] := Op[d_bra,d_ket] * tmp[d_ket,d_bra] 
	
	#expVlaue = 2.0
	return expValue
end



function getCoef(MPSvec::Vector{<:Any})
	Coef = Array{ComplexF64}(I,1,1)
	for localMPS in MPSvec
		@tensor Coef[dpre, d, β] := Coef[dpre, γ]*localMPS[γ, d, β]
		Coef = reshape(Coef, *(size(Coef)[1:2]...), size(Coef)[3])
	end

	return Coef[:,1]
end


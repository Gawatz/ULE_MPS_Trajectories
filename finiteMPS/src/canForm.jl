using TensorOperations

#
#	Can forms
#

"""
    leftCanSite(M; optSVD = false)

brings M into left canonical form. Either by QR-decomposition or iff optSVD = true
bs signular-value decomposition

# Arguments
- 


"""
function leftCanSite(M::AbstractArray{<:Number}; optSVD = false)	
	α_dim, d_dim, β_dim = size(M)
	M = reshape(M, *(size(M)[1:2]...), size(M)[3])
	
	if optSVD == true
		U,S,V = svd(M)
		AL = reshape(U, α_dim, d_dim, size(S)[1])
		
		return AL, diagm(0 => S), V'
	
	else
		Q, R = qr!(M)
		γ_dim = min(size(M)...)	
		Q = Q*Matrix(I, size(Q)[2], γ_dim)
		AL = reshape(Q, α_dim, d_dim, γ_dim)
		
		return AL, R	
	end
end


"""
    leftCanMPS(MPSvec; Cpre = nothing)

takes Cpre and brings the MPS chain Cpre - M.... M - stored in Mvec into left-canonical form.

# Arguments
- MPSvec: Vector cotaining the M's of the MPS which have to be transformed into the left-canonical form
- Cpre: singular values which come before the first element in MPSvec


return:
	Cnext, Remainder, Cvec


	perfoms SVD on last site so M = U, S, V were the last M in left can. form is formed from U
	S is returned as Cnext, and the V is returned as the Remainder.
	For all intermidiate steps S of the svd decomposition has been added to a vector Cvec.
	Note the last Cnext has not been added to Cvec by default! (Cvec just contains all intermidiate Cpre of each A) 


"""
function leftCanMPS(MPSvec::SubArray{<:Any}; Cpre::Union{AbstractArray{<:Number}, Nothing} = nothing)
	if Cpre != nothing 
		@tensor MPSvec[1][α, d, β] := Cpre[α, γ]*MPSvec[1][γ, d, β]
	end
	Cvec = []
	
	for site in 1:size(MPSvec[1:end-1])[1]
		F = leftCanSite(MPSvec[site]; optSVD = true)
		MPSvec[site] = F[1]
		@tensor newMPSnext[α, d, β] := F[2][α, γ]*F[3][γ, γ̃]*MPSvec[site+1][γ̃, d, β]
		MPSvec[site+1] = newMPSnext
		push!(Cvec, F[2])
	end
	
	F = leftCanSite(MPSvec[end]; optSVD = true)
	MPSvec[end] = F[1]
	Cnext = F[2]
	Remainder = F[3]
	
	return Cnext, Remainder, Cvec
end


"""
    leftCanSite(M; optSVD = false)

brings M into left canonical form. Either by QR-decomposition or iff optSVD = true
bs signular-value decomposition

# Arguments
- 


"""
function rightCanSite(M::AbstractArray{<:Number}; optSVD = false)
	α_dim, d_dim, β_dim = size(M)
	M = reshape(M, size(M)[1], *(size(M)[2:3]...))
	
	if optSVD == true
		U, S, V = svd(M)
		AR = reshape(V', size(S)[1], d_dim, β_dim)
		
		return AR, diagm( 0 => S), U
	else
		L, Q = lq!(M)

		γ_dim = min(size(M)...)

		Q = Matrix(I, γ_dim, size(Q)[1])*Q

		AR = reshape(Q, γ_dim, d_dim, β_dim)
	
		return AR, L
	end
end


"""
    leftCanSite(M; optSVD = false)

brings M into left canonical form. Either by QR-decomposition or iff optSVD = true
bs signular-value decomposition

# Arguments
- 


"""
function rightCanMPS(MPSvec::SubArray{<:Any}; Cnext::Union{AbstractArray{<:Number}, Nothing} = nothing)
	if Cnext != nothing
	
		@tensor MPSvec[end][α, d, β] = Cnext[γ, β] * MPSvec[end][α, d, γ]
	end
	

	Cvec = []
	for site in size(MPSvec)[1]:-1:2
		F = rightCanSite(MPSvec[site]; optSVD = true)
		MPSvec[site] = F[1]

		@tensor newMPSnext[α, d, β] := MPSvec[site-1][α, d, γ̃] * F[3][γ̃, γ]*F[2][γ, β]
		MPSvec[site-1] = newMPSnext 
		push!(Cvec, F[2])
	end
	
	F = rightCanSite(MPSvec[1]; optSVD = true)
	
	MPSvec[1] = F[1]
	Cpre = F[2]
	Remainder = F[3]
	#push!(Cvec, Cpre)
	return Cpre, Remainder, Cvec[end:-1:1]
end

rightCanMPS(MPSvec::Vector{<:Any}; Cnext::Union{AbstractArray{<:Number}, Nothing} = nothing, 
	    from::Int = 1, to::Int = size(MPSvec)[1]) = rightCanMPS(@view MPSvec[from:to]; Cnext = Cnext)
leftCanMPS(MPSvec::Vector{<:Any}; Cpre::Union{AbstractArray{<:Number}, Nothing} = nothing, 
	   from::Int = 1, to::Int = size(MPSvec)[1]) = leftCanMPS(@view MPSvec[from:to]; Cpre = Cpre)
rightCanMPS(MPSvec::Vector{<:Any}, i::Int, j::Int) = rightCanMPS(MPSvec; Cnext = nothing, from = i, to = j)
leftCanMPS(MPSvec::Vector{<:Any}, i::Int, j::Int) = leftCanMPS(MPSvec; Cpre = nothing, from = i, to = j)


"""
    mixedCanMPS(MPSvec, site)

brings the MPSvec into mixed canonical from. That means all sites 1-site are in left can. form
while all sites from site+1-end are in right can. form.

# Arguments
- 


return:

	Schmidt-Coefficients between left and right part.

"""
function mixedCanMPS(MPSvec::Vector{<:Any}, site::Int)

	MPS_size = size(MPSvec)[1]
	Cpre, Rr, Cvec = rightCanMPS(MPSvec, site+1, MPS_size)
	Cnext, Rl, Cvec = leftCanMPS(MPSvec, 1, site)

	F = svd(Cnext*Rl*Rr*Cpre)

	# multiply F.U and F.V into left respectively right MPS block
	@tensor res[α, d, β_new] := MPSvec[site][α, d, β]*F.U[β, β_new]
	MPSvec[site] = res

	@tensor res[α_new, d, β] := MPSvec[site+1][α, d, β] * F.V'[α_new, α]
	MPSvec[site+1] = res

	return F.S
end


"""
    getSchmidtVec(MPSvec, site)


# Arguments
- 


return:

	Schmidt-Coefficients between left and right part.

"""
function getSchmidtVec(MPSvec::Vector{<:Any}, site::Int)
	S = mixedCanMPS(MPSvec, site)

	leftHalf = @view MPSvec[1:site]
	rightHalf = @view MPSvec[site+1:end]

	Svec_left = mergeMPS(leftHalf) #(phys_dim, ingoing, outgoing)
	@show size(Svec_left)
	Svec_left = Svec_left[:,1,:]
	Svec_right = mergeMPS(rightHalf)
	@show size(Svec_right)
	Svec_right = Svec_right[:,:,1]

	return S, Svec_left, Svec_right
end

"""************************ initialization schemes **************************"""

"""
    randmps(N, d, maxD)

Constructs a random finite MPS for a system with N sites with local physical dimension
d and max. bond dimension maxD. Bond dimension will increas dynamically with
system size but wont exceed maxD. Will return a vector containing all local matrices M[i] and
a vector containing singular values Cvec[i]:

M[1] - Cvec[1] - M[2] - Cvec[2] - M[3] - ... Cvec[N-1] - M[N] - Cvec[N]

# Arguments
- N: system size
- d: local physical dimension
- maxD: maximal bond dimension

return:
	- MPSvec: vector storing local matrices M[i] for each site i
	- Cvec: vector storing singular values between site and site+1
	  The last entry of Cvec is basically the norm of the sate
 	  and is of dimension 1 x 1. 
"""
function randmps(N::Int, d::Int, maxD::Int)
	MPSvec = []
	Cvec = []


	N_half = round(N/2, RoundDown)
	for i = 1:N_half

		α_dim = d^(i-1) > maxD ? maxD : Int(d^(i-1))
		β_dim = d^i > maxD ? maxD : Int(d^i)
		push!(MPSvec, randn(ComplexF64, (α_dim, d, β_dim)))
		push!(Cvec, randn(ComplexF64, (β_dim, β_dim)))
	end


	if N-2*N_half != 0
		bDim = d^N_half > maxD ? maxD : Int(d^N_half)
		push!(MPSvec, randn(ComplexF64, (bDim, d, bDim)))
		push!(Cvec, randn(ComplexF64, (bDim, bDim)))
	end
	
	for i = N_half:-1:1
		
		β_dim = d^(i-1) > maxD ? maxD : Int(d^(i-1))
		α_dim = d^i > maxD ? maxD : Int(d^i)
		
		push!(MPSvec, randn(ComplexF64, (α_dim, d, β_dim)))
		push!(Cvec, randn(ComplexF64, (β_dim, β_dim)))
	
	end
	
	return MPSvec, Cvec
end


"""
    randombloch(N, maxD)

!!!ADD DESCRIPTION!!!

# Arguments
- N: system size
- maxD: maximal bond dimension

return:
	- MPSvec: vector storing local matrices M[i] for each site i
	- Cvec: vector storing singular values between site and site+1
	  The last entry of Cvec is basically the norm of the sate
 	  and is of dimension 1 x 1. 
"""
function randombloch(N::Int, maxD::Int)
		MPSvec = []
		C = []

		
		N_half = round(N/2, RoundDown)
		
		for i = 1:N_half

			ϕ = rand(Uniform(0.0,2.0*pi))
			θ = rand(Uniform(0.0,2.0*pi))
			α_dim = 2^(i-1) > maxD ? maxD : Int(2^(i-1))
			β_dim = 2^i > maxD ? maxD : Int(2^i)
			
			A = zeros(ComplexF64,α_dim, 2, β_dim)
			c = zeros(Float64, β_dim, β_dim)
			A[1,1,1] = cos(θ/2.0)
			A[1,2,1] = sin(θ/2.0)*exp(1.0im*ϕ)
			c[1,1] = 1.0
			
			push!(MPSvec, A)
			push!(C, c)
		end
		
		if N-2*N_half != 0
			bDim = 2^N_half > maxD ? maxD : Int(2^N_half)
			
			ϕ = rand(Uniform(0.0,2.0*pi))
			θ = rand(Uniform(0.0,2.0*pi))
			
			A = zeros(ComplexF64, bDim, 2, bDim)
			c = zeros(Float64, bDim, bDim)
			A[1,1,1] = cos(θ/2.0)
			A[1,2,1] = sin(θ/2.0)*exp(1.0im*ϕ)
			c[1,1] = 1.0
			
			push!(MPSvec, A)
			push!(C, c)
		end
		
		
		for i = N_half:-1:1
			
			β_dim = 2^(i-1) > maxD ? maxD : Int(2^(i-1))
			α_dim = 2^i > maxD ? maxD : Int(2^i)
			
			ϕ = rand(Uniform(0.0,2.0*pi))
			θ = rand(Uniform(0.0,2.0*pi))
			
			A = zeros(ComplexF64,α_dim, 2, β_dim)
			c = zeros(Float64, β_dim, β_dim)
			
			push!(MPSvec, A)
			push!(C, c)
		
		end

		return MPSvec, C
 end
"""
    coeftomps(N, Coef)
!!!ADD DESCRIPTION!!! so far just for local hilbertspace dimension d = 2

# Arguments
- N: system size
- Coef: Coefficient vector 

return:
	- MPSvec: vector storing local matrices M[i] for each site i
"""
function coeftomps(N::Int, Coef::Vector{<:Number})
	MPSvec = []
	
	tmp = zeros(ComplexF64, 1,size(Coef)[1])
	tmp[1,:] = Coef
	for site in 1:N 
		
		bout_pre, d_left = size(tmp)
		tmp =  reshape(tmp, bout_pre *2, Int(d_left/2))
		F = svd(tmp)
		mps = reshape(F.U, bout_pre, 2, size(F.U)[2])

		tmp = diagm(0=>F.S)*F.V'

		push!(MPSvec, mps)

	end
	
	return MPSvec
end




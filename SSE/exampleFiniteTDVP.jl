using MPOmodule2
using finiteMPS
using LinearAlgebra
using TensorOperations

Id(n) = Array{ComplexF64}(I, n, n)

sz = localOp([1.0 0.0; 0.0 -1.0])
sx = localOp([0.0 1.0; 1.0 0.0])
sy = localOp([0.0 -1.0im; 1.0im 0.0])
id = localOp(Id(2))



cr = localOp([0.0 1.0; 0.0 0.0]) 
ca = localOp([0.0 0.0; 1.0 0.0])
n = localOp([0.0 0.0; 0.0 1.0])

function startMPS(N::Int, d::Int, maxD::Int)
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

function TDVP(MPSvec::Vector{<:Any}, MPOvec::Vector{MPO}, Tsteps::Int, dτ::Union{Float64, ComplexF64})
	#the inital
	leftCanMPS(MPSvec)
	Cpre, R, Cvec = rightCanMPS(MPSvec)

	
	Lenv = [Array{ComplexF64,2}(I,1,1)]
	@show  Lenv = applyTM_MPO(MPSvec[1:end], MPOvec[1:end], Lenv; left = false)



	@show size(Cvec)
	#
	# build up right environment blocks
	#
	Renv = [Array{ComplexF64,2}(I,1,1)]
	RBlocks = [Renv]
	for site = 1:length(MPSvec)-1
		Renv = applyTM_MPO([MPSvec[end-(site-1)]],[MPOvec[end-(site-1)]], RBlocks[site]; left = false)
		push!(RBlocks, Renv)#[init,N,.....,2]
	end
	

	
	measureMPS = deepcopy(MPSvec)
	Cnext, Remainder, __ = leftCanMPS(measureMPS, 1, Int(length(MPSvec)/2)-1)



	measurelist = [] 
	
	for i in 1:Tsteps
		println("\n\n")
		@show i
		Cpre, RBlocks, Cvec = evo_sweep(MPSvec, Cpre, MPOvec, RBlocks, dτ)
		
		Lenv = [Array{ComplexF64,2}(I,1,1)]
		expSz= applyTM_MPO(deepcopy(MPSvec[1:end]), MPOvecSz[1:end], Lenv; left = false)
		@show expSz./2
		push!(measurelist, (expSz./2)[1,1])

	end
	

	@show measurelist
	return measurelist

end


""" transver Ising """

# creat MPO representation for transver ising
#
#	 1  	0	0 
#	-J*sx	λ	0
#	-h*sz	sx	1
#

J = 1.0
h = -0.5
N = 10


# for now λ = 0 
Op_vec_TIsing(J, h) = Vector{localOp}([id, sx, h*sz, J*sx, id])
Op_idx_TIsing = Vector{Int}([1, 2, 3, 6, 9])
Op_vec_TIsingStart(J, h) = Vector{localOp}([h*sz, J*sx, id])
Op_vec_TIsingEnd(J, h) = Vector{localOp}([id, sx, h*sz])


TIsing = MPO(3,Op_vec_TIsing(J, h),Op_idx_TIsing)

MPOvecTIsing = [MPO((1,3),Op_vec_TIsingStart(J,h),[1,2,3]), [TIsing for i = 1:N-2]...,MPO((3,1), Op_vec_TIsingEnd(J,h),[1,2,3])]


Op_vec_Id = Vector{localOp}([id, id, id])
Op_vec_Sz = Vector{localOp}([id, sz, id])
Op_idx_Sz = Vector{Int}([1,2,4])

SzMPO = MPO(2, Op_vec_Sz, Op_idx_Sz)

MPOvecSz = [MPO((1,2),Vector{localOp}([sz, id]), Vector{Int}([1,2])),[SzMPO for i = 1:N-2]..., MPO((2,1),Vector{localOp}([id, sz]),Vector{Int}([1,2]))]



"""****************************** init MPS ********************************"""
MPSvec, C = startMPS(N,2,4)

A = zeros(ComplexF64, 20, 2, 20)
A[1, 1, 1] = 1.0
B = zeros(ComplexF64, 1, 2, 20)
B[1, 1, 1] = 1.0
D = zeros(ComplexF64, 20, 2, 1)
D[1, 1, 1] = 1.0

MPSvec[1] = B
C[1] = Id(2)
for i = 2:1:N-1
	MPSvec[i] = A
	C[i] = Id(2)
end
MPSvec[end] = D
C[end] = Id(1)



rightCanMPS(MPSvec)
leftCanMPS(MPSvec)

Lenv = [Array{ComplexF64,2}(I,1,1)]
Lenv = applyTM_MPO(MPSvec[2:end], MPOvecSz[2:end], Lenv; left = false)
@show  "exp right" , applyTM_MPO([MPSvec[1]], [MPOvecSz[1]], Lenv; left = false)



Lenv = [Array{ComplexF64,2}(I,1,1)]
Lenv = applyTM_MPO(MPSvec[1:end-1], MPOvecSz[1:end-1], Lenv; left = true)
@show  "exp left", applyTM_MPO([MPSvec[end]], [MPOvecSz[end]], Lenv; left = true)



rightCanMPS(MPSvec)
measureMPS = deepcopy(MPSvec)
Cnext, Remainder, __ = leftCanMPS(measureMPS, 1, Int(length(MPSvec)/2)-1)
@tensor C[α, β] := Cnext[α, γ]*Remainder[γ, β]

@tensor MPS[α, d, β] := C[α, γ]*MPSvec[Int(length(MPSvec)/2)][γ, d, β]

@tensor expvalue[] := MPS[α, d, β]*Array(sz)[d, d']*conj(MPS)[α, d', β]
@show "here ",expvalue



"""****************************** init Ham ********************************"""

measurelist = TDVP(MPSvec, MPOvecTIsing, 1000, 0.01im)

measurelist =[x[1,1] for x in measurelist]


#using PyCall, PyPlot
#pygui(:tk)
#PyPlot.plot(measurelist)
#PyPlot.show()


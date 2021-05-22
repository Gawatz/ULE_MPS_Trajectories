using LinearAlgebra
using MPOmodule


sz = Array{ComplexF64}([1.0 0.0; 0.0 -1.0])
sy = Array{ComplexF64}([0.0 -1.0im; 1.0im 0.0])
sx = Array{ComplexF64}([0.0 1.0; 1.0 0.0])
Id(n) = Array{ComplexF64}(I,n,n)
Basis = [Id(2), sx, sy, sz]


ca = Array{ComplexF64}([0.0 1.0; 0.0 0.0]) 
cr = Array{ComplexF64}([0.0 0.0; 1.0 0.0])
#n = Array{ComplexF64}([1.0 0.0; 0.0 0.0]) #changed this 
n = Array{ComplexF64}([0.0 0.0; 0.0 1.0]) #changed this 

#=========================================================

		MPO models

=========================================================#

""" SZ - measurement """
Op_vec_Id = Vector{localOp}(localOp.([Id(2),Id(2),Id(2)]))
Op_vec_Sz = Vector{localOp}(localOp.([Id(2),sz,Id(2)]))
Op_idx_Sz = Vector{Int}([1,2,4])

SzMPO = MPO(2, Op_vec_Sz, Op_idx_Sz)

"""occupation"""

#
#	1	0
#	n	1
#

Op_vec_n = Vector{localOp}(localOp.([Id(2),n,Id(2)]))
Op_idx_n = Vector{Int}([1,2,4])

Op_vec_n_start = Vector{localOp}(localOp.([n, Id(2)]))
Op_idx_n_start = Vector{Int}([1,2])

Op_vec_n_end = Vector{localOp}(localOp.([Id(2), n]))
Op_idx_n_end = Vector{Int}([1,2])


""" transver Ising """

# creat MPO representation for transver ising
#
#	 1  	0	0 
#	-J*sx	0	0
#	-h*sz	sx	1
#

Op_vec_TIsing(J, h) = Vector{localOp}(localOp.([Id(2), -J*sx, -h*sz, sx, Id(2)]))
Op_idx_TIsing = Vector{Int}([1, 2, 3, 6, 9])

Op_vec_TIsing_start(J, h) = Vector{localOp}(localOp.([-h*sz, sx, Id(2)]))
Op_idx_TIsing_start = Vector{Int}([1, 2, 3])

Op_vec_TIsing_end(J, h) = Vector{localOp}(localOp.([Id(2), -J*sx, -h*sz]))
Op_idx_TIsing_end = Vector{Int}([1, 2, 3])



""" Rice - Mele with intra cell repulsion """


#
#
#	1		 0	  0	0	0
#	c^†		 0	  0	0	0
#	c		 0	  0	0	0
#	n		 0	  0	0	0
#	-(V - μ)n 	-Jc	-Jc^†	Un	1
#

Op_vec_RM(J, V, U, μ) = Vector{localOp}(localOp.([Id(2), cr, ca, n, -(V-μ)*n, -J*ca, -J*cr, U*n, Id(2)]))
Op_idx_RM = Vector{Int}([1, 2, 3, 4, 5, 10, 15, 20, 25])

Op_vec_RM_start(J, V, U, μ) = Vector{localOp}(localOp.([-(V-μ)*n, -J*ca, -J*cr, U*n, Id(2)]))
Op_idx_RM_start = Vector{Int}([1,2,3,4,5])

Op_vec_RM_end(V, μ) = Vector{localOp}(localOp.([Id(2), cr, ca, n, -(V-μ)*n]))
Op_idx_RM_end = Vector{Int}([1,2,3,4,5])

""" Cij matrix """

function CMPO(i,j,N)

	Id_local = localOp(Id(2))
	CijVec = [MPO(1, Vector{localOp}([Id_local]), Vector{Int}([1])) for k in 1:N]
	CijVec[i] = MPO(1, localOp.([cr]), Vector{Int}([1]))
	#CijVec[i] = MPO(1,[ca],[1])
	if i!=j 
		CijVec[j] = MPO(1, localOp.([ca]), Vector{Int}([1]))
		#CijVec[j] = MPO(1,[cr],[1])
	else
		CijVec[i] = MPO(1,  localOp.([cr*ca]), Vector{Int}([1]))
		#CijVec[i] = MPO(1,[ca*cr],[1])
	end
	
	for k in Int(i+1):Int(j-1)
		CijVec[k] = MPO(1, localOp.([(Id(2)-2*cr*ca)]), Vector{Int}([1]))#[(Id(2)-2*cr*ca)], [1])
		#CijVec[k] = MPO(1, [(Id(2)-2*ca*cr)], [1])#[(Id(2)-2*cr*ca)], [1])
	end

	return Vector{MPO}(CijVec)
end

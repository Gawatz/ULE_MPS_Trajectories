module MPOmodule3
using LinearAlgebra
using TensorOperations

export MPO, stringMPO, localOp
export eachoperation, eachOp
export parallelOp
export getBondDim
export getMPOTensor, getMPOFromTensor
#
#	Local Operator
#

struct localOp{DT<:Number, D}
	c::Vector{DT}
	in::Vector{Int}
 	out::Vector{Int}

 	function localOp(A::AbstractArray{<:Number,2})
		idx = findall(x->x!=0, A)
		DT = eltype(A)

		in = [x[1] for x in idx]
		out = [x[2] for x in idx]
		c = A[idx]

		return new{DT, size(A)[1]}(c, in, out)
	end

	function localOp(c::Vector{<:Number}, in::Vector{Int}, out::Vector{Int}, D::Int)
		return new{eltype(c), D}(c, in, out)
	end
end


import Base.convert

function convert(::Type{Array{DT, 2}}, op::localOp{DT, D}) where {DT<:Number, D}
	OP = zeros(DT, D, D)
	@inbounds for x in eachoperation(op)
		OP[x[2],x[3]] = x[1]
	end

	return OP
end

import Base.Array

Array(op::localOp{DT, D}) where {DT<:Number, D} = convert(Array{DT,2}, op)


import Base.*

*(a::Number, b::localOp{DT,D}) where {DT<:Number, D}= localOp(a.*b.c, b.in, b.out, D)


import Base.+

function +(a::localOp{DT1, D}, b::localOp{DT2, D}) where {DT1<:Number, DT2<:Number, D} 
	
	#dirty way replace
	return localOp(Array(a)+Array(b))


end

import Base.eltype

eltype(a::localOp{DT, D}) where {DT<:Number, D}= (DT,D)


function eachoperation(a::localOp)
	return zip(a.c, a.in, a.out)
end

# check if two localOP are parallel 
function parallelOp(a::localOp, b::localOp)
	
	parallel = false
	scaling = 0
	if a.in == b.in && a.out == b.out
		
		
		if iszero(diff(a.c./b.c))
			parallel = true
			scaling = (a.c./b.c)[1]
		end
	end

	return parallel, scaling
end

#
#	MPO structure
#

struct MPO
	bDim::Union{Int, Tuple{Int,Int}}
	Operator::Vector{localOp}
	Op_index::Vector{Int}
end

getBondDim(mpo::MPO) = typeof(mpo.bDim) == Int ? (mpo.bDim, mpo.bDim) : mpo.bDim



import Base.getindex


function getindex(mpo::MPO, i::Int) 
 	
     	idx = findfirst(x->x == i, mpo.Op_index)


	if idx == nothing
		return nothing
	else
		return mpo.Operator[idx]
	end
end

getindex(mpo::MPO, i::Int, j::Int) = getindex(mpo, LinearIndices((getBondDim(mpo)[1], getBondDim(mpo)[2]))[i,j])
getindex(mpo::MPO, i::Int, j::Colon) = [getindex(mpo, x) for x in LinearIndices((getBondDim(mpo)[1], getBondDim(mpo)[2]))[i,:]]
getindex(mpo::MPO, i::Colon, j::Int) = [getindex(mpo, x) for x in LinearIndices((getBondDim(mpo)[1], getBondDim(mpo)[2]))[:,j]]

"""
	iteration over alll localOp

"""
function eachLocalOp(mpo::MPO)
	idx = CartesianIndices(getBondDim(mpo))[mpo.Op_index]
	idx = [(x[1],x[2]) for x in idx]

	return zip(mpo.Operator, idx)
end

#
#	this should be changed
#
"""
	iteration over all terms in MPO
"""
function eachoperation(mpo::MPO)
	tmp = []
	@inbounds for x in eachLocalOp(mpo)
		for y in eachoperation(x[1])
			push!(tmp, [y[1], Tuple(y[2:3]), x[2]])
		end
	
	end

	return tmp
end

#
#	Operator String routines (To Do : Comment/ Documentation)
#

struct OpString 
	idx::String
end

#
# overriding base operators for OpString matrix multiplication 
#

# in case either a or b is empty "" creat new string by just combining otherwise add a "→" between them 
import Base.+
(+)(a::OpString, b::OpString) = a.idx=="" || b.idx=="" ? OpString(string(a.idx, b.idx)) : OpString(string(a.idx,"→ ",b.idx))

# in case either a or b is empty "" the new string is also empty otherwise split operator string and add a ⋅ inbetween 
# note that a an b OpStrings have to have same length
import Base.*
(*)(a::OpString, b::OpString) = a.idx == "" || b.idx == "" ? OpString("") : OpString(string.(split(a.idx,"→ "),"⋅", split(b.idx,"→ ")))

import Base.zero
zero(a::OpString) = OpString("")

function OpString(a::Vector{String})
	if length(a)>1
		res = OpString(a[1]) + OpString(a[2:end])
	else
		res = OpString(a[1])
	end

	return OpString(res.idx)
end

function StringArray_to_OpStringArray(A::AbstractArray{<:String})
	B = Array{OpString}
	B = fill(OpString(""),size(A)...)

	B[:,:] = OpString.(A[:,:])
	

	return B
end

function MPO_to_StringArray(mpo_A::MPO)	
	bDim = mpo_A.bDim
	bin = 0
	bout = 0
	try 
		bin = bDim[1]
		bout = bDim[2]
	catch
		bin = bDim
		bout = bDim
	end
	
	B = Array{String}
	B = fill("", bin, bout)
	Op_index = mpo_A.Op_index
	
	Idx = Array(Transpose(LinearIndices(fill(1,bout,bin))))
	mpo_idx = [findfirst(x-> x == idx, Idx) for idx in Op_index]


	B[mpo_idx] .= [string(i) for i in 1:size(mpo_idx)[1]]

	return B
end

MPO_to_OpString(mpoA::MPO) = StringArray_to_OpStringArray(MPO_to_StringArray(mpoA))

# matrix multiplication

function stringM_mul(A::AbstractArray{OpString}, B::AbstractArray{OpString})	
	return A*B
end

stringM_mul(A::AbstractArray{String}, B::AbstractArray{String}) = stringM_mul(StringArray_to_OpStringArray(A), StringArray_to_OpStringArray(B))

"""
   convert_OpStringArrayToTuple(A::AbstractArray{<:String})

converts and Array respresenting the MPO product as strings to an Array containing
tuples where each int corresponds to the operator index of the local MPO

#


"""
function convert_OpStringArrayToTuple(A::AbstractArray{OpString})
	bin, bout = size(A)
	B = Array{Any,2}
	B = fill([],bin,bout)
	for i in 1:bin
		row = []
		for j in 1:bout

			
			try
				#decompose string
				terms = split(A[i,j].idx,"→ ")
				terms = [Tuple(parse.(Int, split(x,"⋅"))) for x in terms]
				
				B[i,j] = terms
			catch
				continue
			end
			#push!(row, terms)

		end
		#push!(B, row)
	end

	return B
end


function stringMPO(MPOvec::Vector{MPO}; op_String::Union{Array{OpString,2},Nothing} = nothing)
	mpo = MPOvec[1]
	B = MPO_to_OpString(mpo)

	op_String = op_String == nothing ? B : stringM_mul(op_String,B)
	
	if size(MPOvec[2:end])[1] > 0
		stringMPO(MPOvec[2:end], op_String = op_String)
	else
		return convert_OpStringArrayToTuple(op_String) 
	
	end
end





#
#	some MPO algebra
#

(*)(a::Number, MPOvec::Vector{MPO}) = Vector{MPO}([MPO(MPOvec[1].bDim, a .* MPOvec[1].Operator, MPOvec[1].Op_index),MPOvec[2:end]...])
(+)(mpo_a::Vector{MPO}, mpo_b::Vector{MPO}) = add_MPO(mpo_a, mpo_b)

function add_MPO(MPOvecA::Vector{MPO}, MPOvecB::Vector{MPO})
	@assert size(MPOvecA) == size(MPOvecB) "you are trying to add
					to MPO with different length"

	newMPOvec = Vector{MPO}([])				
	
	for i in 1:size(MPOvecA)[1]
		
		MPOA = MPOvecA[i]
		MPOB = MPOvecB[i]
		
		bDim_A = MPOA.bDim
		bDim_B = MPOB.bDim
		
		bDim_A = typeof(bDim_A) == Int ? (bDim_A, bDim_A) : bDim_A
		bDim_B = typeof(bDim_B) == Int ? (bDim_B, bDim_B) : bDim_B

		# take care of last and first site 
		new_bin = i != 1 ? bDim_A[1]+bDim_B[1] : 1
		new_bout = i != size(MPOvecA)[1] ? bDim_A[2]+bDim_B[2] : 1

		OP_vec_A = MPOA.Operator
		OP_idx_A = MPOA.Op_index
		
		OP_vec_B = MPOB.Operator
		OP_idx_B = MPOB.Op_index

		#
		#	bring into block diagonal form
		#
		
		# adjust OP_idx_A
		idx = CartesianIndices((bDim_A[1],bDim_A[2]))[OP_idx_A]
		OP_idx_A = LinearIndices((new_bin, new_bout))[idx]

		# adjust Op_idx_B
		idx = CartesianIndices((bDim_B[1], bDim_B[2]))[OP_idx_B]
		idx = [Array([Tuple(x)...]).+[new_bin-bDim_B[1], new_bout-bDim_B[2]] for x in idx]	
		OP_idx_B = [LinearIndices((new_bin, new_bout))[x...] for x in idx]
		
		
		new_Op_vec = Vector{Any}([OP_vec_A..., OP_vec_B...])
		new_Op_idx = Vector{Int}([OP_idx_A..., OP_idx_B...])
		
		newMPO = MPO((new_bin, new_bout), new_Op_vec, new_Op_idx)
		push!(newMPOvec, newMPO)
	end

	return newMPOvec
end

# !!! Multiplications of two MPO

function getMPOTensor(mpo::MPO)

	list_op = eachLocalOp(mpo)

	operators = [x[1] for x in list_op]
	phys_dim = eltype(operators[1])[2]
	data_type = promote_type([x[1] for x in eltype.(operators)]...)

	idx = [x[2] for x in list_op]
	dims = getBondDim(mpo)
	
	MPOTensor = zeros(data_type, (phys_dim, dims..., phys_dim))

	for (j,i) in enumerate(idx)
		MPOTensor[:, i..., :] = Array(operators[j])
	end

	return MPOTensor
end


function getMPOFromTensor(mpoTensor::Array{DT, 4}) where {DT<:Number}

	bond_dim = size(mpoTensor)[2:3]

	operator = localOp[]
	idx = Int[]
	for col in 1:bond_dim[2]
		for row in 1:bond_dim[1]

			op = mpoTensor[:,row,col,:]

			if iszero(op)!=true
				push!(operator, localOp(op))
				push!(idx, LinearIndices(bond_dim)[row, col])
			end
		end
	end


	mpo = MPO(bond_dim, operator, idx)

	return mpo

end

function (*)(MPOvecA::Vector{MPO}, MPOvecB::Vector{MPO})

	@assert length(MPOvecA) == length(MPOvecB) "you are trying to mulitply
						    to MPO with different length"

	new_MPOvec = MPO[]
	for i in 1:length(MPOvecA)

		mpoA = getMPOTensor(MPOvecA[i])
		mpoB = getMPOTensor(MPOvecB[i])

		@tensor new_mpo[α, α′, d, d′, β, β′] := mpoA[d, α, β, γ]*mpoB[γ, α′, β′, d′]
		
		new_mpo = reshape(new_mpo, *(size(new_mpo)[1:2]...), size(new_mpo)[3], size(new_mpo)[4], *(size(new_mpo)[5:end]...))
		new_mpo = permutedims(new_mpo, [2,1,4,3])
		new_mpo = getMPOFromTensor(new_mpo)
		push!(new_MPOvec, new_mpo)
	end
	
	return new_MPOvec

end


#
#	function for deparalization of MPO (shrinks complexity)
#

#=
function checkPara(a::Vector{localOp}, b::Vector{localOp})
	parallel = true


	vec_a = [[x.Coef for x in a]...][1]
	vec_b = [[x.Coef for x in b]...][1]

	if isapprox(abs(vec_a'*vec_b)/(norm(vec_a)*norm(vec_b)), 1, atol = 1e-12) == false
		parallel = false
	end
	
	
	scale_fac =0
	
	if parallel   # this might lead to bad numeric rounding 
		vec_a = vec_a[vec_a.!=0]
		vec_b = vec_b[vec_b.!=0]
		scale_fac = vec_a./vec_b
		#@show vec_a, vec_b
		scale_fac = scale_fac[scale_fac.!=0][1]
	end
	
	#@show scale_fac, parallel
	return parallel, scale_fac
end


function findParaCol(mpo)
	bin, bout = size(mpo)

	# init T
	a = zeros(Complex{Float64}, bout)
	a[1] = 1
	T = [a]
	
	# init K
	col = mpo[:,1]
	K = [col]

	for col_idx in size(K)[1]+1:bout
		para = false

		for k_idx in 1:size(K)[1]

			col = mpo[:,col_idx]
			k = K[k_idx]

			para, scale_fac = checkPara(col,k)
			
			if para == true

				T[k_idx][col_idx] = scale_fac	
				break
			end

		end
		
		if para != true
			push!(K, col)
			a = zeros(Complex{Float64},bout)
			a[col_idx] = 1
			push!(T,a)	
		end

	end

	return hcat(K...), transpose(hcat(T...))
end

function findParaRow(mpo)
	bin, bout = size(mpo)

	# init T
	a = zeros(Complex{Float64}, bin)
	a[1] = 1
	T = [a]
	
	# init K
	row = mpo[1,:]
	K = [row]

	for row_idx in size(K)[1]+1:bin
		para = false

		for k_idx in 1:size(K)[1]

			row = mpo[row_idx,:]
			k = K[k_idx]

			para, scale_fac = checkPara(row, k)
			
			if para == true

				T[k_idx][row_idx] = scale_fac	
				break
			end

		end
		
		if para != true
			push!(K, row)
			a = zeros(Complex{Float64},bin)
			a[row_idx] = 1
			push!(T,a)	
		end

	end

	return Transpose(hcat(K...)), hcat(T...)
end


function depara(MPOvec::Vector{MPO})
		
	for site in length(MPOvec):-1:1

		mpo = MPOvec[site]
		mpo, T = findParaRow(mpo)
		localOP_MPO[site] = mpo

		if site-1 >= 1


			mpo_pre = MPOvec[site-1]

			mpo_pre = mpo_pre*T

			MPOvec[site-1] = mpo_pre


		end
	end
	
	
	for site in 1:size(localOP_MPO)[1]

		mpo = localOP_MPO[site]
		
		mpo, T = findParaCol(mpo)
		localOP_MPO[site] = mpo

		#@show T	
		if site+1 <= size(localOP_MPO)[1]


			mpo_next = localOP_MPO[site+1]

			mpo_next = T*mpo_next

			localOP_MPO[site+1] = mpo_next


		end



	end
	
	for site in size(localOP_MPO)[1]:-1:1

		mpo = localOP_MPO[site]
		mpo, T = findParaRow(mpo)
		localOP_MPO[site] = mpo

		if site-1 >= 1


			mpo_pre = localOP_MPO[site-1]

			mpo_pre = mpo_pre*T

			localOP_MPO[site-1] = mpo_pre


		end
	end




	return localOP_MPO
end
=#








end # module

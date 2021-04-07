using MPOmodule2
include("PauliComutator.jl")

#
#	checks if two rows or columns of a mpo are
#	parallel to each other
#
function checkPara(a, b)

	@assert length(a) == length(b)

	parallel = false
	c = 0
	if typeof.(a) != typeof.(b)
		return (false, 0.0)
	end

	idx = findall(x->typeof(x) != Nothing, a)
	tmp = [parallelOp(a[x], b[x]) for x in idx]
	if length(unique([x[2] for x in tmp])) != 1
		return (false, 0.0)
	end


	return (tmp[1][1], tmp[1][2])

end


function findParaRow1(mpo)

	a = zeros(ComplexF64, getBondDim(mpo)[1])
	a[1] = 1.0
	T = [a]

	mpo_row = mpo[1,:]
	K = Any[mpo_row]

	for row_i in length(K)+1:getBondDim(mpo)[1]
		para = false

		for k_idx in 1:length(K)

			mpo_row = mpo[row_i, :]
			k = K[k_idx]

			para, scale = checkPara(mpo_row, k)

			if para == true

				T[k_idx][row_i] = scale
				break
			end
		end

		if para != true
			push!(K, mpo_row)
			a = zeros(ComplexF64, getBondDim(mpo)[1])
			a[row_i] = 1.0
			push!(T, a)
		end

	end

	# recat K into mpo
	
	K = hcat(K...)
	K = K[Transpose(LinearIndices(size(K)))]
	idx = findall(x->typeof(x) != Nothing, K)
	op_idx = LinearIndices(size(K))[idx]
	K = MPO(size(K), K[idx], op_idx)

	
	return K, hcat(T...)
end

function findParaCol1(mpo)
	
	a = zeros(ComplexF64, getBondDim(mpo)[2])
	a[1] = 1.0
	T = [a]
	mpo_col = mpo[:,1]
	K = Any[mpo_col]

	for col_i in length(K)+1:getBondDim(mpo)[2]
		para = false

		for k_idx in 1:length(K)
			mpo_col = mpo[:,col_i]
			k = K[k_idx]

			para, scale = checkPara(mpo_col, k)
			if para == true

				T[k_idx][col_i] = scale
				break
			end

		end

		if para != true
			
			push!(K, mpo_col)
			a = zeros(ComplexF64, getBondDim(mpo)[2])
			a[col_i] = 1.0
			push!(T,a)



		end

	end


	# recat K into mpo
	K = hcat(K...)
	idx = findall(x->typeof(x) != Nothing, K)
	op_idx = LinearIndices(size(K))[idx]
	K = MPO(size(K), K[idx], op_idx)

	return K, Transpose(hcat(T...))
end


#
#	multiplication of MPO with Array 
#	(multiplication acts as matrix multiplication only on the bond dimension)
#

import Base.(*)

function (*)(mpo::MPO, a::AbstractArray{D, 2}) where {D<:Number}

	tmp = []

	#multiplication from the right
	@inbounds for col in 1:size(a)[2]
		for row in 1:getBondDim(mpo)[1]
			array_col = @view a[:,col]  # get column
			idx = findall(x->x!=0, array_col) # find nz elements in column
			if idx != [] 
				mpo_row = mpo[row,:]  # eventually replace with a view
				idx_nothing = findall(x->typeof(x) == Nothing, mpo_row) #check if there is a localOp
				idx = setdiff!(idx,idx_nothing) #set difference of idx: make sure that element of idx is not in idx_nothing
				if idx != [] 
					mpo_row = mpo_row[idx] #[mpo[row,i] for i in idx] # get elements which are at nz column index
					coef = array_col[idx] # [array_col[i] for i in idx]
					new_localOp = sum(coef.*mpo_row) #obtain actual new localOp
					push!(tmp, (new_localOp, (row,col)))
				end
			end
		end

	end

	# construct new OP
	Operator = [x[1] for x in tmp]
	Op_index = [LinearIndices((getBondDim(mpo)[1], size(a)[2]))[x[2]...] for x in tmp]
	
	# this would just be necessary if we would loop first over row
	#idx = sortperm(Op_index)
	#Op_index = Op_index[idx]
	#Operator = Operator[idx]


	return MPO((getBondDim(mpo)[1], size(a)[2]), Operator, Op_index)
end

function (*)(a::AbstractArray{D, 2}, mpo::MPO) where {D<:Number}

	tmp = []

	#multiplication from the left
	@inbounds for col in 1:getBondDim(mpo)[2]
		for row in 1:size(a)[1]

			array_row = @view a[row,:]  # get row
			idx = findall(x->x!=0, array_row) # find nz elements in row
			if idx != [] 
				mpo_col = mpo[:,col]
				idx_nothing = findall(x->typeof(x) == Nothing, mpo_col) #check if there is a localOp
				idx = setdiff!(idx,idx_nothing)
				if idx != [] 
					mpo_col = mpo_col[idx] #[mpo[i, col] for i in idx] # get elements which are at nz column index
					coef = array_row[idx] #[array_row[i] for i in idx]
					new_localOp = sum(coef.*mpo_col) #obtain actual new localOp
					push!(tmp, (new_localOp, (row,col)))
				end
			end
		end

	end

	# construct new OP
	Operator = [x[1] for x in tmp]
	Op_index = [LinearIndices((size(a)[1], getBondDim(mpo)[2]))[x[2]...] for x in tmp]
	
	# this would just be necessary if we would loop first over row
	#idx = sortperm(Op_index)
	#Op_index = Op_index[idx]
	#Operator = Operator[idx]


	return MPO((size(a)[1], getBondDim(mpo)[2]), Operator, Op_index)
end



#
#	deparallizes mpo-chain
#
function deparaMPOChain!(MPOvec::Vector{MPO})
	

	for site in 1:length(MPOvec)
		#@show "dePara col", site
		mpo = MPOvec[site]

		@time mpo, T = findParaCol1(mpo)
		MPOvec[site] = mpo

		if site+1 <= length(MPOvec)
			mpo_next = MPOvec[site+1]
			@time mpo_next = T*mpo_next
			MPOvec[site+1] = mpo_next
		end
	end
	
	for site in length(MPOvec):-1:1
		#@show "dePara row", site
		mpo = MPOvec[site]
		mpo, T = findParaRow1(mpo)
		MPOvec[site] = mpo
		if site-1 > 1
			mpo_pre = MPOvec[site-1]
			mpo_pre = mpo_pre*T
			MPOvec[site-1] = mpo_pre
		end
	end


	for site in 1:length(MPOvec)
		#@show "dePara col", site	
		mpo = MPOvec[site]

		mpo, T = findParaCol1(mpo)
		MPOvec[site] = mpo

		if site+1 <= length(MPOvec)
			mpo_next = MPOvec[site+1]
			mpo_next = T*mpo_next
			MPOvec[site+1] = mpo_next
		end
	end

	return MPOvec
end


function constructMPO(L::Vector{pauliString}, Basis)

	N = L[1].N
	bondDim = length(L)

	mpo_ID_start = MPO((1,bondDim), [localOp(Array{Float64}(I,2,2)) for i in 1:bondDim], [i for i in 1:bondDim])
	mpo_ID = MPO(bondDim, [localOp(Array{Float64}(I,2,2)) for i in 1:bondDim], [1+bondDim*i+i for i in 0:bondDim-1])
	mpo_ID_end = MPO((bondDim, 1), [localOp(Array{Float64}(I,2,2)) for i in 1:bondDim], [i for i in 1:bondDim]) 

	MPOvec = [mpo_ID_start, [deepcopy(mpo_ID) for i in 2:N-1]..., mpo_ID_end]

	for (i, pString) in enumerate(L)
	
		l_op = localOp.(Basis[pString.baseIdx.+1]) # +1 since Basis contains ID at index 1 and pString.baseIdx is 0 for id
		l_op[1] = coef(pString)*l_op[1]

		mpos = MPOvec[pString.sites]
		for (j, op) in enumerate(l_op)
			MPOvec[pString.sites[j]].Operator[i] = op

		end

	end

	MPOvec = deparaMPOChain!(MPOvec)
	return MPOvec
end

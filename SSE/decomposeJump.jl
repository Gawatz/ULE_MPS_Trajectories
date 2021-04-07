using MPOmodule2
using Combinatorics
using SparseArrays
using LinearAlgebra



#
#	localPauliOp struct
#

struct localPauliOp
	Coef::Vector{Complex{Float64}}
end

function localPauliOp(idx::Int, coef::Number)
	Coef = zeros(Complex{Float64},4)
	Coef[idx+1] = coef
	return localPauliOp(Coef) 

end

function translate_localPauliOp(a::localPauliOp, basis::Vector{String})
	idx = findall(x->x!=0, a.Coef)
	op = basis[idx]
	coef = string.(Real.(a.Coef[idx]))
	
	if size(coef)[1] > 0
		return *([coef[i]*x for (i,x) in enumerate(op)]...)
	else
		return ""
	end
end


import Base.*
import Base.+
import Base.zero
import Base.transpose

(*)(a::Number, b::localPauliOp) = a == 0 ? localPauliOp([0,0,0,0]) : localPauliOp(b.Coef*a)
(*)(b::localPauliOp, a::Number) = a == 0 ? localPauliOp([0,0,0,0]) : localPauliOp(b.Coef*a)
(+)(a::localPauliOp, b::localPauliOp) = localPauliOp(a.Coef+b.Coef)

transpose(a::localPauliOp) = a
zero(a::localPauliOp) = localPauliOp([0,0,0,0])
zero(a::Type{localPauliOp}) = localPauliOp([0,0,0,0])


function getPauliCoef(Op::AbstractArray{<:Number,2}, pauli_combi, sites)
	N = Int(log(size(Op)[1])/log(2))
	n_qbits =  size([Tuple(pauli_combi)...])[1]
	n_sites = size([sites...])[1]	
	pauli_string = Matrix{ComplexF64}(I, size(Op)[1], size(Op)[1])
	
	for i in 1:n_sites
	
		pauli_indicator = pauli_combi[i]
		
		site = sites[i]

		if  pauli_indicator == 1
			σ = get_Sx(N, site)	
		elseif pauli_indicator == 2
			σ = get_Sy(N, site)
		else
			σ = get_Sz(N, site)
		end
				
		pauli_string = pauli_string * σ

	end

	#show pauli_string
	coef = tr(pauli_string*Op)/Int(2^N)

	#@show coef
	#println("\n")
	
	return coef
end


"""
    expandString(OpStrings)


# Arguments

return:
	- (min_site, max_site)
	- Array of operator strings where each row corresponds to a 
	  new operator string. 0 correspods to identity while
	  (1, 2, 3) correspond to σx, σy, σz.

"""
function expandString(OpStrings)
	max_site = maximum([maximum(x[1]) for x in OpStrings])
	min_site = minimum([minimum(x[1]) for x in OpStrings])
	lengthGraph = Int(max_site-min_site+1)
	
	OpStrings_Expanded = []

	for x in OpStrings
		new_op_string = zeros(Int16, lengthGraph)
		new_op_string[x[1].-min_site.+1] = [x[2]...]
		push!(OpStrings_Expanded, new_op_string)	

	end

	return (min_site, max_site), hcat(OpStrings_Expanded...)'
end


#
#	depara routines depends on localOP
#

function checkPara(a::Vector{localPauliOp}, b::Vector{localPauliOp})
	parallel = true


	vec_a = [[x.Coef for x in a]...][1]
	vec_b = [[x.Coef for x in b]...][1]

	if isapprox(abs(vec_a'*vec_b)/(norm(vec_a)*norm(vec_b)), 1, atol = 1e-12) == false
		parallel = false
	end
	
	#=
	for (i,x) in enumerate(a)

		# comapre x and b[i]
		if isapprox(x.Coef'*b[i].Coef, norm(x.Coef)*norm(b[i].Coef), atol = 1e-12) == false
			parallel = false
			break
		end

	end
	=#
	
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

function depara(localOP_MPO)
	
	
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
	
	
	for site in 1:size(localOP_MPO)[1]

		#@show "deparallize MPO at site: $(site)"
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


#
#	function converting decomposed Op into MPO
#	with deparalized bond-dim
#

function to_MPO(localPauliOp_MPO, Basis)
	MPOvec = Vector{MPO}([])

	for mpo in localPauliOp_MPO
		Op_vec = []
		Op_idx = []
		bin, bout = size(mpo)
	
		ddim, __ = size(Basis[1])
	
		Idx = Array(LinearIndices(fill(1,bin,bout)))
		for i in 1:bin
			for j in 1:bout
				
				op = mpo[i,j].Coef
				op_idx = findall(x->x!=0, op)
				
				if op_idx != []
					Op = op[op_idx].*Basis[op_idx]
					Op = localOp(sum(Op))
					
					push!(Op_vec, Op)

					push!(Op_idx, Idx[i, j])
				end
			end
		end

		bdim = bin == bout ? bin : (bin, bout)
		mpo = MPO(bdim, Op_vec, Op_idx) 

		push!(MPOvec, mpo)
	end

	return MPOvec
end

function construct_MPO(L::Vector{pauliString}, N, Basis)	
	op_string = zeros(Int, size(L)[1], N) # number pauli strings x system size
	for (i,x) in enumerate(L)
		op_string[i,x.sites] .= x.baseIdx # fill in pauli strings
	end

	localPauliOp_MPO = []

	
	localMPO = fill(zero(localPauliOp), 1, size(op_string)[1]) 
	# for each pauli string one localMPO (come up with a better name)
	
	
	# construct localPauliOp for each pauli string according to the first site
	for (i,x) in enumerate(op_string[:,1])
		localMPO[1,i] = localPauliOp(Int(x), coef(L[i]))
	end
	push!(localPauliOp_MPO, localMPO)

	# add sites in the middle
	for i in 2:size(op_string)[2]-1
		localMPO = fill(zero(localPauliOp), size(op_string)[1], 
			       size(op_string)[1])
		for (i,x) in enumerate(op_string[:,i])
			localMPO[i,i] = localPauliOp(Int(x),1.0)
		end
		
		push!(localPauliOp_MPO, localMPO)
	end

	# take care of last site
	localMPO = fill(zero(localPauliOp), size(op_string)[1], 1)
	for (i,x) in enumerate(op_string[:,end])
		localMPO[i,1] = localPauliOp(Int(x),1.0)
	end
	push!(localPauliOp_MPO, localMPO)


	# deparallize localMPO
	localPauliOpMPO = depara(localPauliOp_MPO)
	
	op_string_range = [1,N] 
	# add missing localMPO
	for i in 1:op_string_range[1]-1
		localMPO = fill(zero(localPauliOp), 1, 1)
		localMPO[1,1] = localPauliOp([1.0, 0, 0, 0])
		pushfirst!(localPauliOp_MPO, localMPO)
	end

	for i in op_string_range[2]+1:N
		localMPO = fill(zero(localPauliOp), 1, 1)
		localMPO[1,1] = localPauliOp([1.0, 0, 0, 0])
		push!(localPauliOp_MPO, localMPO)
	end

	# construct MPO from localPauliOp_MPO
	MPOvec = to_MPO(localPauliOp_MPO, Basis)

	return MPOvec
end

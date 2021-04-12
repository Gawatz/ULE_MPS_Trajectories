using Combinatorics
using SparseArrays
using LinearAlgebra

#
#	many particle pauli matrices for operator decomposition
#
function get_Sz(N::Int, site::Int)
	# Important convention we start chain with site : 1 ..... and end with site N
	site = site - 1 # so site 1 corresponds to a bitshift of 0  

	
	vec_nidx = [0:1:(2^N-1)...] # vector with all nidx labels 0,1,2,3 for 0..0, 10..0, 010..0, 110..0,...

	vec = vec_nidx.>>site  # shift bit by site to the left so if there was a 1 at site: 
			 	# site in orginal it would be at leading bit now.
	
	vec = Vector{Int}(-1 .*(2 .*(vec.%2).-1)) # check if there is a 1 at bit at site: site it is now at the leading bit.
	 				    # we can check if vec entries are now odd or not .. if odd the bit at site :site was 
					    # previously 1 otherwise it was 0 ---> for 1 assigne +1 for 0 assigne -1.


	#vec = Vector{Int}(1 .*(2 .*(vec.%2).-1)) # check if there is a 1 at bit at site: site it is now at the leading bit
	 				    # we can check if vec entries are now odd or not .. if odd the bit at site :site was 
					    # previously 1 otherwise it was 0 ---> for 1 assigne +1 for 0 assigne -1



	return sparse(vec_nidx.+1, vec_nidx.+1, vec) # .+1 since julia starts array count with 1
end

function get_Sy(N::Int, site::Int)
	# Important convention we start chain with site : 1 ..... and end with site N
	site = site - 1 # so site 1 corresponds to a bitshift of 0  
	
	vec_nidx = [0:1:(2^N-1)...] # vector with all nidx labels 0,1,2,3 for 0..0, 10..0, 010..0, 110..0,...

	# well Sy ≈ S+ - S-
	# so we can incorporate this with an xor operation:
	
	vec_applied = vec_nidx .⊻ Int(2^site)
	# if there is a bit 1 at site in vec_nidx it will remove that bit 1 but if there isnt it will creat one! 
	
	# according to if their was a spin or not we have -1 or +1 
	vec_data = vec_nidx.>>site # see comment at the beginning
	vec_data = 2 .*(vec_data.%2).-1
	vec_data = 1im.*vec_data
	#vec_data = -1im.*vec_data

	 return sparse(vec_nidx.+1, vec_applied.+1, vec_data)
end

function get_Sx(N::Int, site::Int)
	# Important convention we start chain with site : 1 ..... and end with site N
	site = site - 1 # so site 1 corresponds to a bitshift of 0  
	
	vec_nidx = [0:1:(2^N-1)...] # vector with all nidx labels 0,1,2,3 for 0..0, 10..0, 010..0, 110..0,...
	
	# well Sx ≈ S+ + S-
	# so we can incorporate this with an xor operation:
	vec_applied = vec_nidx .⊻ Int(2^site)
	
	vec_data = ones(2^N)

	return sparse(vec_nidx.+1, vec_applied.+1, vec_data)
end


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

	coef = tr(pauli_string*Op)/Int(2^N)

	return coef
end



#
# pauliString struct
#

struct pauliString
	N::Int  # length 
	sites::Vector{Int} # vector which contains site indicator where the string has non-trivial operators 
	baseIdx::Vector{Int} # indicates the operator (σx, σy, σz)
	coef::Base.RefValue{ComplexF64}
end


getSupport(p::pauliString)  = (minimum(p.sites),maximum(p.sites))

pauliString(N::Int, sites::Vector{Int}, baseIdx::Vector{Int}, coef::Number) = pauliString(N, sites, baseIdx, Ref(convert(ComplexF64,coef)))
coef(p::pauliString) = p.coef[]
set_coef(p::pauliString, a::Number) = p.coef[] = a

import Base.==
import Base.+
import Base.*
import Base.conj
(==)(a::pauliString, b::pauliString) = a.sites == b.sites && a.baseIdx == b.baseIdx ? true : false
(+)(a::pauliString, b::pauliString) = a == b ? pauliString(a.N, a.sites, a.baseIdx, coef(a)+coef(b)) : [a,b]
(*)(a::pauliString, c::Number) = pauliString(a.N, a.sites, a.baseIdx, coef(a)*c)
(*)(c::Number, a::pauliString) = pauliString(a.N, a.sites, a.baseIdx, coef(a)*c)

function (*)(a::pauliString, b::pauliString)

	p1_sites = a.sites
	p2_sites = b.sites
	p1_idx = a.baseIdx.+1 #  in order that 0 (id) are not get cut out later on 
	p2_idx = b.baseIdx.+1
	#@show p1_sites, p2_sites

	pArray = sparse([[1 for i in p1_sites]...,
			[2 for i in p2_sites]...],
			[p1_sites...,p2_sites...],[p1_idx..., p2_idx...])
	#@show pArray	
	new_σ = []
	sites = [Set([x[2] for x in  findall(x->x!=0,pArray)])...][end:-1:1]
	#@show sites	
	for col in sites
		#@show pArray[1,col], pArray[2,col]
		i = pArray[1,col] == 0 ? 1 : pArray[1,col]
		j = pArray[2,col] == 0 ? 1 : pArray[2,col]
		push!(new_σ, σσ(i, j))  # +1 to match the convention 
     					   # 0 = σ⁰, 1 = σˣ , ... 
	end
	#@show a.N, sites, new_σ[1][1], new_σ[1][2], coef(a)	
	
	new_idx = [Int(new_σ[i][1])-1 for i in 1:size(sites)[1]]
	c = *([new_σ[i][2] for i in 1:size(sites)[1]]...)
	#@show [new_σ[i][2] for i in 1:size(sites)[1]]
	#@show c, coef(a), coef(b)
	
	#@show new_idx
	idx = sortperm(sites)
	sites  = sites[idx]
	new_idx = new_idx[idx]

	return pauliString(a.N, sites, new_idx, c*coef(a)*coef(b))
end

function Base.conj(a::pauliString)
	
	b = deepcopy(a)
	set_coef(b, (coef(b))')

	return b
end

function rev_sites(a::pauliString)

	return pauliString(a.N, [a.N.-(a.sites.-1)...], a.baseIdx, coef(a))

end


function isevenperm(p::Vector{T}) where {T<:Any}
	@assert isperm(p)
	n = length(p)
	used = falses(n)
	even = true

	for k = 1:n
		if used[k]; continue; end

		# Each even cycle flips even (an odd number of times)
		used[k] = true
		j = p[k]
		
		while !used[j]
			used[j] = true
			j = p[j]
			even = !even
		end
	end
	
	return even 
end
isoddperm(p) = !isevenperm(p)

"""
    σσ(i,j)

calculates σᵢσⱼ = δᵢⱼ I   - ϵᵢⱼₗσₗ

"""
function σσ(i::Int, j::Int)
	coef = 1.0
	#@show i, j
	# input is σ⁰ = 1, σˣ = 2, ...
	i-=1; j-=1 # change to σ⁰ = 0, σˣ = 1, ...

	if i == 0 || j == 0 #either i or j is σ⁰
		return i+j+1, coef
	elseif i == j 
		return 1, coef
	else

		# determin σ	
		σ_res = filter(x->x ∉ [i,j],[1,2,3])[1]
		
		# determin coeff
		coef *= isoddperm([i,j,σ_res]) ? -1.0im : 1.0im 
	
		return σ_res+1, coef
	end
end


"""
	[σᵢ , σⱼ] = 2*im*ϵᵢⱼₖ σₖ


"""
function com_σσ(i::Int, j::Int)
	i-=1; j-=1
	coef = 1
	if i == 0 || j == 0
		return 0,0
	elseif i == j
		return 0,0
	else
		# determin σ	
		σ_res = filter(x->x ∉ [i,j],[1,2,3])[1]
	
		# determin coeff
		coef *= isoddperm([i,j,σ_res]) ? -1.0im : 1.0im 
	
		return σ_res+1, 2*coef
	end
end

function commutePauli(p1::pauliString, p2::pauliString)	
	p1_sites = p1.sites
	p2_sites = p2.sites
	p1_idx = p1.baseIdx
	p2_idx = p2.baseIdx

	pArray = sparse([[1 for i in p1_sites]...,
			[2 for i in p2_sites]...],
			[p1_sites...,p2_sites...],[p1_idx..., p2_idx...])

	sites = [Set([x[2] for x in  findall(x->x!=0,pArray)])...][end:-1:1]
	new_σσ = []	
	for x in sites

		o1 = pArray[1,x]+1
		o2 = pArray[2,x]+1
		
		o1_o2 = σσ(o1, o2)
		o2_o1 = σσ(o2, o1)
		push!(new_σσ,(o1_o2, o2_o1))
	end


	c = *([x[1][2] for x in new_σσ]...)-*([x[2][2] for x in new_σσ]...)
	
	if c != 0
		new_p = []
		new_site = []
		for (i,x) in enumerate(new_σσ)
			new_o = x[1][1]-1
			if new_o != 0
				push!(new_p, new_o)
				push!(new_site, [sites...][i])
			end
		end
		res_p = pauliString(p1.N, Vector{Int}(new_site), Vector{Int}(new_p), c*coef(p1)*coef(p2))
		
		return res_p
	
	end
end


#=
#
# fermionicString struct
#

struct fermionString
	N::Int 
	sites::Vector{Int}
	baseIdx::Vector{Bool}
	coef::Base.RefValue{ComplexF64}
end

fermionString(N::Int, sites::Vector{Int}, baseIdx::Vector{Bool}, coef::Number) = fermionString(N, sites, baseIdx, Ref(convert(ComplexF64,coef)))
coef(p::fermionString) = p.coef[]
set_coef(p::fermionString, a::Number) = p.coef[] = a

(==)(a::fermionString, b::fermionString) = a.sites == b.sites && a.baseIdx == b.baseIdx ? true : false
(+)(a::fermionString, b::fermionString) = a == b ? fermionString(a.N, a.sites, a.baseIdx, coef(a)+coef(b)) : "cannot be added"


function (*)(a::fermionString, b::fermionString)

	sites = unique(vcat([a.sites, b.sites]...))

	for x in sites

		if x in a.sites && x in b.sites

		
		else
		

		
		end

	end
	


end
=#


using FFTW

#
#	get frequency bins
#
function fftfreq(n,d=1)
	if n%2 == 0
		bins = [0:1:Int(n/2-1)...,(Int(-n/2):1:-1)...]
	else
		bins = [0:1:Int((n-1)/2)...,(Int(-(n-1)/2):1:-1)...]
	end
	
	return bins/(d*n)
end

function get_time_domain_func(f, freqvec; sorted = false)
	df = freqvec[2]-freqvec[1]
	N = size(freqvec)[1]

	f_vec = f.(freqvec)
	
	fft_val = fft(f_vec)
	tvec = fftfreq(N, df)

	fft_val = fft_val.*[(-1)^i for i in 0:N-1]

	dt = tvec[2]-tvec[1]

	fft_val = fft_val*df

	if sorted == true
	
		idx = sortperm(tvec)
		fft_val = fft_val[idx]
		tvec = tvec[idx]
	end

	return fft_val, tvec
end

function get_jump_operator_static(X, H, J::Any, gamma::Real)	
	F = eigen(Array(H), )
	eig_energies = F.values
	eig_vec = F.vectors
	E_mat = eig_energies*ones(size(eig_energies)[1])'
	
	E_diff_mat = E_mat'.-E_mat .+ 1e-14
	
	R = (2*pi*gamma).*J.(E_diff_mat)
	try
		R = sqrt.(R)
	catch
		E_diff_mat = E_mat'.-E_mat .+ 1e-12
		R = (2*pi*gamma).*J.(E_diff_mat)
		R = sqrt.(R)
	end
	
	X_diag = eig_vec'*X*eig_vec
	L_diag = R.*X_diag

	#L = L_diag
	L = eig_vec*L_diag*eig_vec'

	return L
end

get_jump_operator_static(X::Vector{<:AbstractArray}, H::AbstractArray, Jlist::Vector{<:Any}, gamma_list::Vector{<:Real}) = [get_jump_operator_static(X[i], H, Jlist[i], gamma_list[i]) for i in 1:size(X)[1]]




function decompose_Jump(Op::AbstractArray{<:Number,2}; qbit_max = 2, treshold = 1e-10)	
	N = Int(log(size(Op)[1])/log(2))

	PauliList = []
	for n_qbits in 0:qbit_max
		site_combi = combinations([1:N...], n_qbits)
		size = binomial(N, n_qbits)
	
		# vector for all site combinations 
		nqbits_pauli = []

		for (site_combi_idx, sites) in enumerate(site_combi)
			
			# for each site combination creat a vector for all nqbit combinations
			coef = Vector{Complex{Float64}}(zeros(Int(3^n_qbits)))
			shape = Tuple([3 for i in 1:n_qbits])
			
			for coef_idx in 1:Int(3^n_qbits)
				pauli_combi = CartesianIndices(shape)[coef_idx] 
				# Cartesian index of a nqbit ranked tensor with each dimension = 3
				# corresponds to all possible combinations
				coef[coef_idx] = getPauliCoef(Op, pauli_combi, sites)

			end
			
			
			
			for idx in findall(x->abs.(x) >= treshold, coef)
				pauli_combi = CartesianIndices(shape)[idx]
				if sites == [] # for const term Id ⊗ Id ⊗ ... ⊗ Id
					sites = [1]  
					pauli_combi = [0] 
				end
				push!(nqbits_pauli, pauliString(N, sites, [Tuple(pauli_combi)...], coef[idx]))
			end
		end
		


		push!(PauliList, nqbits_pauli)
	end

	return PauliList
end





######################################################################
######################################################################
######################################################################



"""
    adH(H, p, order)
calculates all elements of adH up to order n

!!! To do put output in a dict specifying each order
like this it is pretty much useless sofar. 

"""
function adH(H, p, order)
	pauliList = Vector{pauliString}([deepcopy(p)])
	new_pauliList = []
	for n in 1:order
		for y in pauliList			
			
			# get operator strings in H with non-trivial support overlap with y
			#ySup = getSupport(y)
			#H_accesible = filter(x-> getSupport(x)[1]<=ySup[2] && getSupport(x)[2]>=ySup[1], H)
			#res = filter(x->x!=nothing, [commutePauli(x,y)  for x in H_accesible])
			
			
			res = filter(x->x!=nothing, [commutePauli(x,y)  for x in H])
			
			push!(new_pauliList, res...)

		end
		pauliList = Vector{pauliString}(new_pauliList)
	end

	if new_pauliList != []
		return pauliList
	else
		return nothing
	end
end



function simplify_stringList(L::Vector{pauliString}; tol = nothing)
	filter!(x->coef(x) != 0, L)
	sort!(L, by = x -> size(x.sites)[1]) #sort number of sites

	c = 1
	new_L = Vector{pauliString}([])
	while L != []

		idx = findlast(x->size(x.sites)[1] == c, L)
		tmp = idx != nothing ? [popfirst!(L) for i in 1:idx] : []

		while tmp != []

			idx = findall(x->x == tmp[1], tmp) # find all similar pauliStrings
			if sum([coef.(tmp[idx])...]) != 0

				# erase zeros from string
				non_zero_idx = findall(x->x!=0,tmp[idx][1].baseIdx)
				if non_zero_idx != []
					new_base_idx = tmp[idx][1].baseIdx[non_zero_idx]
					new_sites = tmp[idx][1].sites[non_zero_idx]
			
					sort_sites = sortperm(new_sites)
					new_sites = new_sites[sort_sites]
					new_base_idx = new_base_idx[sort_sites]
				
				else
				
					new_base_idx = [0]
					new_sites = [1]
				end

				#push!(new_L, pauliString(tmp[idx][1].N, tmp[idx][1].sites, tmp[idx][1].baseIdx, sum([coef.(tmp[idx])...])))
				push!(new_L, pauliString(tmp[idx][1].N, new_sites, new_base_idx, sum([coef.(tmp[idx])...])))
			end
			deleteat!(tmp, idx)

		end
		
		c += 1

	end

	if tol !=nothing new_L = filter(x->abs(coef(x))>=tol,new_L) end

	return new_L
end

function getJump_from_Comutator(a::Vector{pauliString}, H, G_time, time_vec, order)
	@show a[1].sites
	pauli_string = a
	flip_sign(x) = x != 0 ? sign(x) : 1
	L = Vector{pauliString}([])
	dt = time_vec[2]-time_vec[1]
	@show dt
	for n in 0:order
		#  ((-1i)ⁿ/n!) * ∫g(t)⋅tⁿ dt 
		c =(2*pi)^(n+1)*sum(flip_sign.(time_vec).*G_time.*(time_vec.^n).*dt)*(((-1im)^n)/factorial(n))
		# overall 2*pi + 2*pi for freq. convertion 

		#c = 1.0
		@show c  # c should be all real but somehow I still get imaginary parts .... :/ 
		push!(L, [pauliString(x.N, x.sites, x.baseIdx, coef(x)*c) 
			  for x in pauli_string]...)
		pauli_string = simplify_stringList(vcat([adH(H,p,1) for p in pauli_string]...))
		#pauli_string = vcat([adH(H,p,1) for p in pauli_string]...)
	end
	L = simplify_stringList(L)

	return L
end

#
#	creat Operator from pauliString
#
function construct_OP(PauliList::Vector{pauliString})
	N = PauliList[1].N
	Op = spzeros(ComplexF64, Int(2^N), Int(2^N))

	for x in PauliList
		
		c = coef(x)
		pos = x.sites
		pauli_idx = x.baseIdx

		op = spdiagm(0=>[1 for i in 1:Int(2^N)])
		for (i,site) in enumerate(pos)

			p = pauli_idx[i]
			
			if p == 0
				continue
				#σ = Id(Int(2^N))
				σ = spdiagm(0=>[1 for i in 1:Int(2^N)])
			else
				if  p == 1
		
					#σ = get_Sx(N, (N-(site-1)))
					σ = get_Sx(N, site)
				elseif p == 2
				
					#σ = get_Sy(N, (N-(site-1)))
					σ = get_Sy(N, site)
				else
					
					#σ = get_Sz(N, (N-(site-1)))
					σ = get_Sz(N, site)
				end

			
				op *= σ
			end
		
		end

		Op = Op + c*op

	end

	return Op
end





function estimateLoc(pvec::Vector{pauliString})

	norm = 1/sum(abs.(coef.(pvec)))

	loc = norm*sum(abs.(coef.(pvec)).*[x[2]-x[1] for x in getSupport.(pvec)])

	return loc

end




####################################################################
####################################################################
####################################################################
#=

#
#	Heisenberg Spin Chain
#
N = 8; 
J = -0.5; B = 2.0
#V = 1.5; J = 0.5; J′ = 1.0
order = 5


Hspin = Vector{pauliString}([])
for site in 1:N-1 # just has to be build up to order
	push!(Hspin, pauliString(N,[site,site+1],[1,1],J))
	push!(Hspin, pauliString(N,[site,site+1],[2,2],J))
	push!(Hspin, pauliString(N,[site,site+1],[3,3],J))
	push!(Hspin, pauliString(N,[site],[3],B))
end
push!(Hspin, pauliString(N,[N],[3],B))


#=
Hspin = Vector{pauliString}([])
for site in 1:N-1 # just has to be build up to order
	push!(Hspin, pauliString(N, [site, site+1],[1,1],[-J′,-J][Int(isodd(site))+1]*0.25))
	push!(Hspin, pauliString(N, [site, site+1],[2,2],[-J′,-J][Int(isodd(site))+1]*0.25))
	#push!(Hspin, pauliString(N, [site, site+1],[2,1],1im*[-J′,-J][Int(isodd(site))+1]*0.25))
	#push!(Hspin, pauliString(N, [site, site+1],[1,2],-1im*[-J′,-J][Int(isodd(site))+1]*0.25))
	push!(Hspin, pauliString(N, [site],[3],0.5*[-V,V][Int(isodd(site))+1]))
	push!(Hspin, pauliString(N,[site],[0],0.5*[-V,V][Int(isodd(site))+1]))
end
push!(Hspin, pauliString(N,[N],[3],0.5*[-V,V][Int(isodd(N))+1]));
push!(Hspin, pauliString(N,[N],[0],0.5*[-V,V][Int(isodd(N))+1]))
=#

#
#	physical units
#
GHz = 1 
MHz = 0.001
THz = 1000
ns  = 1 
mus = 1/MHz 
ps  = 1/THz 
K   = 20.837
Tesla  = 1

#
#	Bath parameters
#
Temp1 = 0.5*K
γ1 = 10*MHz
Λ = 40*GHz

#
#	Bath spectral function 	
#

function J_CutOff(ω, Λ, T)	
	res = ω == 0 ?  T/GHz : (ω/(1*GHz))*exp(-ω^2/(2*Λ^2)) / (1-exp(-ω/T))
	return res
end

function J1(x) return  J_CutOff(x, Λ, Temp1) end

#
#	freq_vec for FT
#
df = 100*MHz 
freqvec = -20*Λ:df:20*Λ

g(x) = sqrt(J1(x)/(2*pi))
			
G_time, time_vec = get_time_domain_func(g, freqvec, sorted = true) # get g in time

dt = time_vec[2]-time_vec[1]
#tmp = ones(ComplexF64, size(G_time))
tmp = 1
L = Vector{pauliString}([])
p2 = pauliString(N, [N], [1], sqrt(γ1))
pauli_string = [p2]
x1 = construct_OP(pauli_string)


L = getJump_from_Comutator(pauli_string, Hspin, G_time, time_vec, order)



#
#	get jump operator from eq.(31)
#
H = construct_OP(Hspin)
F = eigen(Array(H))
eig_energies = F.values
eig_vec = F.vectors
H_diag = diagm(0=>eig_energies)

x1 = Array(x1)
x1_eig = eig_vec'*x1*eig_vec


L_list = get_jump_operator_static([x1_eig], H_diag, [J1], [γ1])
L_list = [eig_vec*x*eig_vec' for x in L_list]



op = construct_OP(L)

#=
include("decomposeJump.jl")
sz = Array{ComplexF64}([1.0 0.0; 0.0 -1.0])
sy = Array{ComplexF64}([0.0 -1.0im; 1.0im 0.0])
sx = Array{ComplexF64}([0.0 1.0; 1.0 0.0])
Basis = [[1 0; 0 1], sx, sy, sz]
L = simplify_stringList(L)
#mpo = construct_MPO(L, N, Basis)
=#

=#

using Distributed
using exactQmodule
using Arpack
using PauliStrings
using MPOmodule
using finiteMPS
using FFTW
#@everywhere using ProgressMeter
@everywhere using DelimitedFiles
@everywhere include("finiteTDVP_SSE.jl")

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

function constructMPO(L::Vector{PauliString}, Basis)

	N = L[1].N
	bondDim = length(L)

	mpo_ID_start = MPO((1,bondDim), [localOp(Array{Complex{Float64}}(I,2,2)) for i in 1:bondDim], [i for i in 1:bondDim])
	mpo_ID = MPO(bondDim, [localOp(Array{Complex{Float64}}(I,2,2)) for i in 1:bondDim], [1+bondDim*i+i for i in 0:bondDim-1])
	mpo_ID_end = MPO((bondDim, 1), [localOp(Array{Complex{Float64}}(I,2,2)) for i in 1:bondDim], [i for i in 1:bondDim]) 

	MPOvec = [mpo_ID_start, [deepcopy(mpo_ID) for i in 2:N-1]..., mpo_ID_end]

	for (i, pString) in enumerate(L)
	
		l_op = localOp.(Basis[pString.baseIdx.+1]) # +1 since Basis contains ID at index 1 and pString.baseIdx is 0 for id
		l_op[1] = coef(pString)*l_op[1]

		mpos = MPOvec[pString.sites]
		for (j, op) in enumerate(l_op)
			mpo = mpos[j]
			mpo.Operator[i] = op

		end

	end

	MPOvec = depara!(MPOvec)
	return MPOvec
end

function getJump_from_Comutator(a::Vector{PauliString}, H, G_time, time_vec, order)
	@show a[1].sites
	pauli_string = a
	flip_sign(x) = x != 0 ? sign(x) : 1
	L = Vector{PauliString}([])
	dt = time_vec[2]-time_vec[1]
	@show dt
	for n in 0:order
		#  ((-1i)ⁿ/n!) * ∫g(t)⋅tⁿ dt 
		c =(2*pi)^(n+1)*sum(flip_sign.(time_vec).*G_time.*(time_vec.^n).*dt)*(((-1im)^n)/factorial(n))

		@show c #just for sanity check to see if real
		push!(L, [PauliString(x.N, x.sites, x.baseIdx, coef(x)*c) 
			  for x in pauli_string]...)
		pauli_string = simplify_stringList(vcat([adH(H,p,1) for p in pauli_string]...))
		#pauli_string = vcat([adH(H,p,1) for p in pauli_string]...)
	end
	L = simplify_stringList(L)

	return L
end


#
#
#
include("MPOcollection.jl")

#
#	sys size
N = 10
Np = 5
"""****************************** initialize ham  ************************************"""
V = 1.5
U = 0.5
J = 1.0 #intra
J′ = 0.5 #inter
μ = 0.0
Temp = 2.0

"""**************************** get Jump Operator  *********************************"""
#
#	get Ham as pauli string
#
HRM = Vector{PauliString}([])
for site in 1:N-1 # just has to be build up to order
	    
	push!(HRM, PauliString(N,[site,site+1],[1,1],[-J′,-J][Int(isodd(site))+1]*0.5))
	push!(HRM, PauliString(N,[site,site+1],[2,2],[-J′,-J][Int(isodd(site))+1]*0.5))
		    
	push!(HRM, PauliString(N,[site,site+1],[2,1],1im*[-J′,-J][Int(isodd(site))+1]*0.25))
	push!(HRM, PauliString(N,[site,site+1],[1,2],-1im*[-J′,-J][Int(isodd(site))+1]*0.25))

	push!(HRM, PauliString(N,[site,site+1],[1,2], 1im*[-J′,-J][Int(isodd(site))+1]*0.25))
	push!(HRM, PauliString(N,[site,site+1],[2,1], -1im*[-J′,-J][Int(isodd(site))+1]*0.25))

	push!(HRM, PauliString(N,[site],[3],0.5*[-V, V][Int(isodd(site))+1]))
	push!(HRM, PauliString(N,[site],[0],0.5*[-V, V][Int(isodd(site))+1]))
	
	# unit cell repulsive interaction  U*nᵢnᵢ₊1 = 0.25*U*(sᶻᵢsᶻᵢ₊₁ + sᶻᵢIᵢ₊₁ + Iᵢsᶻᵢ₊₁ + IᵢIᵢ₊₁)
	if isodd(site)
		push!(HRM, PauliString(N, [site, site+1], [3,3], 0.25*U))
		push!(HRM, PauliString(N, [site, site+1], [3,0], 0.25*U))
		push!(HRM, PauliString(N, [site, site+1], [0,3], 0.25*U))
		push!(HRM, PauliString(N, [site, site+1], [0,0], 0.25*U))  # is irrelevant for expansion
	end

end
push!(HRM, PauliString(N,[N],[3],0.5*[-V, V][Int(isodd(N))+1]))
push!(HRM, PauliString(N,[N],[0],0.5*[-V, V][Int(isodd(N))+1]))



#for Temp in 0.4:0.05:6.0
@show Temp	
GHz = 1.0
function J_CutOff(ω, Λ, T)	
	res = ω == 0 ?  T/GHz : (ω/(1*GHz))*exp(-ω^2/(2*Λ^2)) / (1-exp(-ω/T))
	return res
end

Λ = 40
γ = 0.02
function S(x) return  J_CutOff(x, Λ, Temp) end
g(x) = sqrt(S(x)/(2*pi))
freqvec = -20*Λ:0.01:20*Λ
G_time, time_vec = get_time_domain_func(g, freqvec, sorted = true) # get g in time


order = 4
tol = nothing #10e-10
jump_pauli = []
jump_op = []
#for i in 1:2:N
for i in 1:2:Int(N)
#for i in Int(N/2)	
	
	@show i
	pauli_string = [PauliString(N, [i], [3], 1.0)]

	#
	#	from commutator
	#
	L = getJump_from_Comutator(pauli_string, HRM, G_time, time_vec, order)
	L .*= sqrt(γ)
	L = simplify_stringList(Vector{PauliString}(L), tol = tol)
	L = simplify_stringList(Vector{PauliString}(L), tol = tol)
	
	push!(jump_pauli, Vector{PauliString}(L))

end

save("pauli_string_T$(Temp)_γ$(γ)_order$(order).jld", "string", jump_pauli)
#end


sz = Array{ComplexF64}([-1.0 0.0; 0.0 1.0])
sy = Array{ComplexF64}([0.0 -1.0im; 1.0im 0.0])
sx = Array{ComplexF64}([0.0 1.0; 1.0 0.0])
Id(n) = Array{ComplexF64}(I,n,n)
Basis = [Id(2), sx, sy, sz]
constructMPO(x) = constructMPO(x, Basis)
Jump_MPO_List = []
for x in jump_pauli
	@show length(x)


	#testMPS = MPSvec
	#leftCanMPS(testMPS)
	#rightCanMPS(testMPS)
	x = constructMPO(x)
	#Lenv = [Array{ComplexF64,2}(I,1,1)]
	#@show applyTM_MPO(testMPS[1:end], x, Lenv; left = false)
	
	
	push!(Jump_MPO_List, x)
end
#construct_MPO(x) =  MPOTensor_to_MPO(Op_to_MPO(N, x; tol = 1e-4))
#Jump_MPO_List = construct_MPO.(jump_op)


# check if particle number conserving 
for x in Jump_MPO_List
	@show [y.bDim for y in x]
	#testMPS, __ = finiteMPS(N, 2, 10)
	#testMPS = MPSvec
	#leftCanMPS(testMPS)
	#rightCanMPS(testMPS)
	#Lenv = [Array{ComplexF64,2}(I,1,1)]
	#Nbefore = applyTM_MPO(testMPS[1:end], MPOvec_TotalN[1:end], Lenv; left = false)
	#testMPS = new_apply_MPO(testMPS, x)
	#testMPS = iter_applyMPO(testMPS, x, 10)
	#leftCanMPS(testMPS)
	#rightCanMPS(testMPS)
	#Lenv = [Array{ComplexF64,2}(I,1,1)]
	#Nafter = applyTM_MPO(testMPS[1:end], MPOvec_TotalN[1:end], Lenv; left = false)
	#@show Nbefore, Nafter
end

"""**************************** get Decay Operator  *********************************"""
Decay_MPO_List = []
Decay_pauli_List = []
for i in 1:size(jump_pauli)[1]



	#
	#	construct by commutator
	#
	
	L_decay = []
	for x in Base.product(conj.(jump_pauli[i]), jump_pauli[i])
		push!(L_decay, *(x...))

	end
	L_decay = simplify_stringList(Vector{PauliString}(L_decay))
	L_decay = simplify_stringList(Vector{PauliString}(L_decay))
	filter!(x->abs.(coef(x))>1e-4,L_decay)
	push!(Decay_pauli_List, L_decay)
	#decay_mpo = constructMPO(L_decay)
	
	decay_mpo = constructMPO(L_decay)



	#
	#	which one should be used
	#
	#push!(Decay_MPO_List, Decay_1_MPO) #exact
	push!(Decay_MPO_List, decay_mpo) # commutator
	println("size of Decay operators: ", [size(m.Op_index) for m in Decay_MPO_List[end]])
	
	
	# check if particle number conserving 
	#testMPS, __ = finiteMPS(N, 2, 10)
	#testMPS = MPSvec
	#rightCanMPS(testMPS)
	#Lenv = [Array{ComplexF64,2}(I,1,1)]
	#Nbefore = applyTM_MPO(testMPS[1:end], MPOvec_TotalN[1:end], Lenv; left = false)
	#testMPS = iter_applyMPO(testMPS, Decay_MPO_List[end], 100)
	#rightCanMPS(testMPS)
	#Lenv = [Array{ComplexF64,2}(I,1,1)]
	#Nafter = applyTM_MPO(testMPS[1:end], MPOvec_TotalN[1:end], Lenv; left = false)
	#@show Nbefore, Nafter
end

test = vcat([-0.5im*vcat(Decay_pauli_List...)...]...)
test = simplify_stringList(Vector{PauliString}(test))
test = simplify_stringList(Vector{PauliString}(test))
heff_mpo = constructMPO(test)
#heff_mpo = sum(Decay_MPO_List)
heff_mpo = depara!(heff_mpo)
#heff_mpo = H_MPOvec
@show [size(m.Op_index) for m in heff_mpo]

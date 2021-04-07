using Distributed
using exactQmodule
using Arpack
using MPOmodule2
using finiteMPS
#@everywhere using ProgressMeter
@everywhere using DelimitedFiles
@everywhere include("finiteTDVP_SSE.jl")



#
#
#
include("MPOcollection.jl")
include("PauliComutator.jl")
include("decomposeJump.jl")
include("dePara.jl")

#
#	sys size
N = 50
Np = 5
"""****************************** initialize ham  ************************************"""
V = 1.5
U = 0.5
J = 1.0 #intra
J′ = 0.5 #inter
μ = 0.0

#
#
#	1		 0	  0	0	0
#	c^†		 0	  0	0	0
#	c		 0	  0	0	0
#	n		 0	  0	0	0
#	-(V - μ)n 	-Jc	-Jc^†	Un	1
#
H_MPOvec = [MPO((1,5),Op_vec_RM_start(J, -V, U, μ), Op_idx_RM_start),
	    [MPO(5, Op_vec_RM([J, J′][Int(isodd(i))+1], [-V, V][Int(isodd(i))+1], [U,0.0][isodd(i)+1], μ), Op_idx_RM) for i in 1:N-2]..., 
	   MPO((5,1), Op_vec_RM_end(V, μ), Op_idx_RM_end)]

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
#	Bath spectral function 	
#
bDim = 8
#Temp = 0.05*K
Temp = 0.1*K
γ = 20*MHz
Λ = 30*GHz
df = 100*MHz 
freqvec = -20*Λ:df:20*Λ

"""***************************** initialize measurements  ******************************"""


measureDict = Dict()

n = Array{ComplexF64}([0.0 0.0; 0.0 1.0]) #changed this 
#n = Array{ComplexF64}([1.0 0.0; 0.0 0.0]) #changed this   # with this definition first physical dimension is occupied, second unoccupied!



MPOvec_TotalN = [MPO((1,2),Vector{localOp}(localOp.([n,Id(2)])), Vector{Int}([1,2])),[MPO((2,2), Vector{localOp}(localOp.([Id(2),n,Id(2)])),
		Vector{Int}([1,2,4])) for i = 1:N-2]..., MPO((2,1),Vector{localOp}(localOp.([Id(2), n])),Vector{Int}([1,2]))]

MPOvec_TotalNsquare = MPOvec_TotalN*MPOvec_TotalN


#=
MPOvec_TotalSz = [MPO((1,2),Vector{Any}([sz,Id(2)]), Vector{Int}([1,2])),[SzMPO for i = 1:N-2]..., 
		  MPO((2,1),Vector{Any}([Id(2), sz]),Vector{Int}([1,2]))]
measureDict["total-Sz"] = MPOvec_TotalSz

for i in 1:N
	measureDict["sz-$(i)"] = [i, sz]
end
=#

#=
for i in 1:N
	measureDict["n-$(i)"] = [i, n]
end
=#

measureDict["Energy"] = H_MPOvec
measureDict["Energy_square"] = H_MPOvec*H_MPOvec

measureDict["Np"] = MPOvec_TotalN
measureDict["Np_square"] = MPOvec_TotalNsquare

"""****************************** load MPS ************************************"""
MPSvec = load("MPS_N$(N)_Np$(Np)_J$(J)_Jprime$(0.5)_V$(V)_U$(0.0).jld")["mps"]
leftCanMPS(MPSvec)
global Cpre, __, Cvec = rightCanMPS(MPSvec)
Lenv = [Array{ComplexF64,2}(I,1,1)]
Number_N = applyTM_MPO(MPSvec[1:end], MPOvec_TotalN, Lenv; left = false)[1,1][1]
@show Number_N
Lenv = [Array{ComplexF64,2}(I,1,1)]
E = applyTM_MPO(MPSvec[1:end], H_MPOvec, Lenv; left = false)[1,1][1]
@show E
Lenv = [Array{ComplexF64,2}(I,1,1)]

E = applyTM_MPO(MPSvec[1:end], H_MPOvec*H_MPOvec, Lenv; left = false)[1,1][1]
@show E

#=
"""****************************** construct MPS ************************************"""
MPSvec = []
# fist site
A = zeros(ComplexF64, 1, 2, bDim)
A[1,2,1] = 1.0
push!(MPSvec, A)

# site in the middle
for i in 2:5
	A = rand(ComplexF64, bDim, 2, bDim)
	#A = zeros(ComplexF64, bDim, 2, bDim)
	#A[1,2,1] = 1.0
	push!(MPSvec, A)
end
for i in 6:N-1
	A = rand(ComplexF64, bDim, 2, bDim)
	#A = zeros(ComplexF64, bDim, 2, bDim)
	#A[1,1,1] = 1.0
	push!(MPSvec, A)
end

# last site
A = rand(ComplexF64, bDim, 2, 1)
#A = zeros(ComplexF64, bDim, 2, 1)
#A[1,1,1] = 1.0
push!(MPSvec, A)
=#

leftCanMPS(MPSvec)
global Cpre, __, Cvec = rightCanMPS(MPSvec)
@show size(diag(Cvec[Int(round(N/2, RoundDown))]))
pushfirst!(Cvec, Cpre)
@show "initial site occupation", [singleSiteExpValue(MPSvec, Cvec[i], n, i; can_left=false)[1] for i in 1:N]

#=
#for i in 1:N
#
#	@show getCoef(MPSvec)'*0.5*(Array{ComplexF64}(I,2^N, 2^N)+get_Sz(N,i))*getCoef(MPSvec)
#
#end

"""**************************** run vMPS for groundstate ***********************************"""
Lenv = [Array{ComplexF64,2}(I,1,1)]
Number_N = applyTM_MPO(MPSvec[1:end], MPOvec_TotalN, Lenv; left = false)[1,1][1]
@show Number_N

R = Array{ComplexF64}(I,1,1)
Renv = [R]
RBlocks = [Renv]
for site = 1:length(MPSvec)-1
	Renv = applyTM_MPO([MPSvec[end-(site-1)]], [H_MPOvec[end-(site-1)]], RBlocks[site]; left = false)
	push!(RBlocks, Renv)#[init,N,.....,2]
end


for i in 1:4
	global Cpre, RBlocks, __ = vmps_sweep(MPSvec, Cpre, H_MPOvec, RBlocks)
end


__, __, Cvec = rightCanMPS(MPSvec)
leftCanMPS(MPSvec)
__, __, Cvec = rightCanMPS(MPSvec)
@show diag(Cvec[Int(round(N/2, RoundDown))])

#save("init_state.jld", "coef", getCoef(MPSvec))
Lenv = [Array{ComplexF64,2}(I,1,1)]
Number_N= applyTM_MPO(MPSvec[1:end], MPOvec_TotalN, Lenv; left = false)[1,1][1]
@show Number_N
Lenv = [Array{ComplexF64,2}(I,1,1)]
E = applyTM_MPO(MPSvec[1:end], H_MPOvec, Lenv; left = false)[1,1][1]
@show E
Lenv = [Array{ComplexF64,2}(I,1,1)]
E2 = applyTM_MPO(MPSvec[1:end], H_MPOvec*H_MPOvec, Lenv; left = false)[1,1][1]
@show E2
=#
#=
"""**************************** get Jump Operator  *********************************"""
#
#	get Ham as pauli string (we will deal with the problem as hardcore bosons!) 
#
HRM = Vector{pauliString}([])
for site in 1:N-1 # just has to be build up to order
	    
	push!(HRM, pauliString(N,[site,site+1],[1,1],[-J′,-J][Int(isodd(site))+1]*0.5))
	push!(HRM, pauliString(N,[site,site+1],[2,2],[-J′,-J][Int(isodd(site))+1]*0.5))
		    
	push!(HRM, pauliString(N,[site,site+1],[2,1],1im*[-J′,-J][Int(isodd(site))+1]*0.25))
	push!(HRM, pauliString(N,[site,site+1],[1,2],-1im*[-J′,-J][Int(isodd(site))+1]*0.25))

	push!(HRM, pauliString(N,[site,site+1],[1,2], 1im*[-J′,-J][Int(isodd(site))+1]*0.25))
	push!(HRM, pauliString(N,[site,site+1],[2,1], -1im*[-J′,-J][Int(isodd(site))+1]*0.25))

	push!(HRM, pauliString(N,[site],[3],0.5*[-V, V][Int(isodd(site))+1]))
	push!(HRM, pauliString(N,[site],[0],0.5*[-V, V][Int(isodd(site))+1]))
	
	# unit cell repulsive interaction  U*nᵢnᵢ₊1 = 0.25*U*(sᶻᵢsᶻᵢ₊₁ + sᶻᵢIᵢ₊₁ + Iᵢsᶻᵢ₊₁ + IᵢIᵢ₊₁)
	if isodd(site)
		push!(HRM, pauliString(N, [site, site+1], [3,3], 0.25*U))
		push!(HRM, pauliString(N, [site, site+1], [3,0], 0.25*U))
		push!(HRM, pauliString(N, [site, site+1], [0,3], 0.25*U))
		push!(HRM, pauliString(N, [site, site+1], [0,0], 0.25*U))  # is irrelevant for expansion
	end

end
push!(HRM, pauliString(N,[N],[3],0.5*[-V, V][Int(isodd(N))+1]))
push!(HRM, pauliString(N,[N],[0],0.5*[-V, V][Int(isodd(N))+1]))



#for Temp in 0.4:0.05:6.0
@show Temp	

function J_CutOff(ω, Λ, T)	
	res = ω == 0 ?  T/GHz : (ω/(1*GHz))*exp(-ω^2/(2*Λ^2)) / (1-exp(-ω/T))
	return res
end

function S(x) return  J_CutOff(x, Λ, Temp) end
g(x) = sqrt(S(x)/(2*pi))			
G_time, time_vec = get_time_domain_func(g, freqvec, sorted = true) # get g in time


order = 4
tol = nothing #10e-10
jump_pauli = []
jump_op = []
#for i in 1:2:N
for i in 1:2:Int(N)
#for i in Int(N/2)	
	
	@show i
	pauli_string = [pauliString(N, [i], [3], 1.0)]

	#
	#	from commutator
	#
	L = getJump_from_Comutator(pauli_string, HRM, G_time, time_vec, order)
	L .*= sqrt(γ)
	L = simplify_stringList(Vector{pauliString}(L), tol = tol)
	L = simplify_stringList(Vector{pauliString}(L), tol = tol)
	
	#
	#	construct Jump operator form eigenstates
	#
	#=
	Htest =  HRM_op
	F = eigen(Array(Htest))
	eig_energies = F.values
	eig_vec = F.vectors
        H_diag = diagm(0=>eig_energies)

	x1 = construct_OP(pauli_string)
        xL_eig = eig_vec'*x1*eig_vec
	L_list = get_jump_operator_static([xL_eig], H_diag, [S], [γ])
        L_list = [eig_vec*x*eig_vec' for x in L_list]
	push!(jump_op, L_list[1])
	=#

	#
	# from operator decomposition
	#	
	#println("decompose Jump operator: ")
	#pauli_list = vcat(decompose_Jump(L_list[1], qbit_max = 3, treshold = 1e-5)...)

	#
	#	save an check for quality
	#
	#println("check quality!")
	println("comutator")
	#@show maximum(abs.(sparse(op-L_list[1])))
	#@show  opnorm(abs.(Array(op-L_list[1]))), opnorm(L_list[1]), opnorm(Array(op))
	push!(jump_pauli, Vector{pauliString}(L))

	#println("decomposition")
	#@show opnorm(Array(abs.(L_list[1] - construct_OP(Vector{pauliString}(pauli_list))))), opnorm(Array(abs.(L_list[1])))
	#push!(jump_pauli, Vector{pauliString}(pauli_list))
	
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


	testMPS = MPSvec
	leftCanMPS(testMPS)
	rightCanMPS(testMPS)
	x = constructMPO(x)
	Lenv = [Array{ComplexF64,2}(I,1,1)]
	@show applyTM_MPO(testMPS[1:end], x, Lenv; left = false)
	
	
	push!(Jump_MPO_List, x)
end
#construct_MPO(x) =  MPOTensor_to_MPO(Op_to_MPO(N, x; tol = 1e-4))
#Jump_MPO_List = construct_MPO.(jump_op)


# check if particle number conserving 
for x in Jump_MPO_List
	@show [y.bDim for y in x]
	#testMPS, __ = finiteMPS(N, 2, 10)
	testMPS = MPSvec
	leftCanMPS(testMPS)
	rightCanMPS(testMPS)
	Lenv = [Array{ComplexF64,2}(I,1,1)]
	Nbefore = applyTM_MPO(testMPS[1:end], MPOvec_TotalN[1:end], Lenv; left = false)
	#testMPS = new_apply_MPO(testMPS, x)
	testMPS = iter_applyMPO(testMPS, x, 10)
	leftCanMPS(testMPS)
	rightCanMPS(testMPS)
	Lenv = [Array{ComplexF64,2}(I,1,1)]
	Nafter = applyTM_MPO(testMPS[1:end], MPOvec_TotalN[1:end], Lenv; left = false)
	@show Nbefore, Nafter
end

"""**************************** get Decay Operator  *********************************"""
Decay_MPO_List = []
Decay_pauli_List = []
for i in 1:size(jump_pauli)[1]


	#
	#	construct by decomposition of actuall operator
	#
	#=
	J1 = jump_op[i]
	Decay_1 = J1'*J1
	Decay_1_tmp = vcat(decompose_Jump(Decay_1, qbit_max = 2, treshold = 1e-5)...)
	println("qualitiy of pauli-decomposition (norm comparrison): ", opnorm(abs.(Decay_1-construct_OP(Vector{pauliString}(Decay_1_tmp)))), "  ", opnorm(abs.(Decay_1)))
	Decay_1_MPO = construct_MPO(Vector{pauliString}(Decay_1_tmp))
	=#


	#
	#	construct by commutator
	#
	
	L_decay = []
	for x in Base.product(conj.(jump_pauli[i]), jump_pauli[i])
		push!(L_decay, *(x...))

	end
	L_decay = simplify_stringList(Vector{pauliString}(L_decay))
	L_decay = simplify_stringList(Vector{pauliString}(L_decay))
	filter!(x->abs.(coef(x))>1e-4,L_decay)
	push!(Decay_pauli_List, L_decay)
	#decay_mpo = constructMPO(L_decay)









	testMPS = MPSvec
	leftCanMPS(testMPS)
	rightCanMPS(testMPS)
	decay_mpo = constructMPO(L_decay)
	Lenv = [Array{ComplexF64,2}(I,1,1)]
	@show applyTM_MPO(testMPS[1:end], decay_mpo, Lenv; left = false)






	#d1 = construct_OP(Vector{pauliString}(jump_pauli[i]))
	#D = d1'*d1
	#@show opnorm(abs.(D-Decay_1))
	#println("qualitiy of commutator expansion (norm comparrison): ", opnorm(abs.(Array(D-construct_OP(Vector{pauliString}(L_decay))))), "  ",opnorm(abs.(Array(D))))

	#
	#	which one should be used
	#
	#push!(Decay_MPO_List, Decay_1_MPO) #exact
	push!(Decay_MPO_List, decay_mpo) # commutator
	println("size of Decay operators: ", [size(m.Op_index) for m in Decay_MPO_List[end]])
	
	
	# check if particle number conserving 
	#testMPS, __ = finiteMPS(N, 2, 10)
	testMPS = MPSvec
	rightCanMPS(testMPS)
	Lenv = [Array{ComplexF64,2}(I,1,1)]
	Nbefore = applyTM_MPO(testMPS[1:end], MPOvec_TotalN[1:end], Lenv; left = false)
	testMPS = iter_applyMPO(testMPS, Decay_MPO_List[end], 100)
	rightCanMPS(testMPS)
	Lenv = [Array{ComplexF64,2}(I,1,1)]
	Nafter = applyTM_MPO(testMPS[1:end], MPOvec_TotalN[1:end], Lenv; left = false)
	@show Nbefore, Nafter
end

test = vcat([-0.5im*vcat(Decay_pauli_List...)...]...)
test = simplify_stringList(Vector{pauliString}(test))
test = simplify_stringList(Vector{pauliString}(test))
heff_mpo = constructMPO(test)
#heff_mpo = sum(Decay_MPO_List)
#heff_mpo = deparaMPOChain!(heff_mpo)
heff_mpo = heff_mpo+H_MPOvec
#heff_mpo = H_MPOvec
@show [size(m.Op_index) for m in heff_mpo]

=#
"""**************************** run SSE-TDVP ***********************************"""
N_realize = 1
dτ = 0.0001
Dmax = bDim 
Tmax = 150
""" load previous state """


""" init MPS_SSE """
#A = MPS_SSE(MPSvec, H_MPOvec, heff_mpo, Decay_MPO_List[1:1:end], Jump_MPO_List[1:1:end], 0.025, (0, Tmax), Dmax, measureDict)
#save MPS_SSE instance 
#saveMPS_SSE("MPS_SSE_instance_N$(N)_T$(Temp)_gamma$(γ)_order$(order)_U$(U).jld", A)


#load instance 
A = loadMPS_SSE("./MPS_SSE_instance_N$(N)_T$(Temp)_gamma$(γ)_order$(order)_U$(U).jld")
#@show A
#A = A["init"]
# rearrage with new MPSvec 
A = MPS_SSE(MPSvec, A.H_MPO, A.Heff, A.Decay_MPO, A.Jump_MPO, A.n_rel, (0,Tmax), Dmax, A.measureDict)


Lenv = [Array{ComplexF64,2}(I,1,1)]
init_total_sz = applyTM_MPO(MPSvec[1:end], MPOvec_TotalN[1:end], Lenv; left = false)
@show init_total_sz


Lenv = [Array{ComplexF64,2}(I,1,1)]
init_energy =  applyTM_MPO(MPSvec[1:end], H_MPOvec[1:end], Lenv; left = false)
@show  init_energy





#
#	Distributed work
#



println("start evolution")

# creat dir to save runs! 
#dir_name = "./results/N$(N)_Np$(Int(N/2))_Temp$(Temp)_γ$(γ)_dτ$(dτ)_U$(U)"
dir_name = "./results/N$(N)_Np$(Np)_Temp$(Temp)_γ$(γ)_dτ$(dτ)_U$(U)_order$(order)_dmax$(A.maxDim)"

try 
	mkdir(dir_name)
	mkdir(string(dir_name,"/inter_states"))
catch
	println(string("director ",dir_name," already exists."))
end
start_time = time()

@everywhere global f = open("error_message.txt", "w+")
@everywhere redirect_stderr(f)
#name = (dir_name, 999)
#evo_TDVP_SSE(A, MPOvec = A.Heff, dτ = dτ, name = name)
runtime = pmap([i for i in 26:25+N_realize]) do x
	println("started job $x")
	name = (dir_name, x)
	time, __ = evo_TDVP_SSE(A, MPOvec = A.Heff, dτ = dτ, name = name, saveState = false)
	return time
	#return evo_TDVP_SSE(x)
end
@show time()-start_time

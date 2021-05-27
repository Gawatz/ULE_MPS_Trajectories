using Distributed
using exactQmodule
using Arpack
using MPOmodule2
using finiteMPS
#@everywhere using ProgressMeter
@everywhere using DelimitedFiles
@everywhere include("finiteTDVP_SSE.jl")
@everywhere include("MPOcollection.jl")
include("PauliComutator.jl")
include("decomposeJump.jl")
include("dePara.jl")

#
#	sys size
N = 16
Np = 4


V = 1.5
U = 0.5
J = 1.0 #intra
J′ = 0.5 #inter
μ = 0.0




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
Temp = 0.1*K
γ = 20*MHz
Λ = 30*GHz
df = 100*MHz 
freqvec = -20*Λ:df:20*Λ


function J_CutOff(ω, Λ, T)	
	res = ω == 0 ?  T/GHz : (ω/(1*GHz))*exp(-ω^2/(2*Λ^2)) / (1-exp(-ω/T))
	return res
end

function S(x) return  J_CutOff(x, Λ, Temp) end
g(x) = sqrt(S(x)/(2*pi))			
G_time, time_vec = get_time_domain_func(g, freqvec, sorted = true) # get g in time



"""**************************** run SSE-TDVP ***********************************"""
N_realize = 30
dτ = 0.01
dmax = 256
order = 4

#load instance 
A = loadMPS_SSE("./MPS_SSE_instance_N$(N)_T$(Temp)_gamma$(γ)_order$(order)_U$(U).jld")




println("start evolution")

# creat dir to save runs! 
dir_name = "./results/N$(N)_Np$(Np)_Temp$(Temp)_γ$(γ)_dτ$(dτ)_U$(U)_order$(order)_dmax$(dmax)"
try 
	mkdir(dir_name)
	mkdir(string(dir_name,"/inter_states"))
catch
	println(string("directory",dir_name," already exists."))
end

@everywhere global f = open("error_message.txt", "w+")
@everywhere redirect_stderr(f)

f = open("datalist.txt", "r")
f = readlines(f)

#=
mps = load(f[1])["MPSvec"]	
i = findlast("id", f[1])
x = f[1][i[2]+1:end]
i = findfirst(".", x)
x = x[1:i[1]-1]
@show x
name = (dir_name, x)
x = MPS_SSE(mps, A.H_MPO, A.Heff, A.Decay_MPO[1:1:end], A.Jump_MPO[1:1:end], 0.025, (250,260), 150, A.measureDict)
evo_TDVP_SSE(x, MPOvec = A.Heff, dτ = dτ, name = name)
=#

pmap([1:length(f)...]) do x
	mps = load(f[x])["MPSvec"]	
	
	i = findlast("id", f[x])
	x = f[x][i[2]+1:end]
	i = findfirst(".", x)
	x = x[1:i[1]-1]
	@show x
	name = (dir_name, string(x,"_continued"))
	x = MPS_SSE(mps, A.H_MPO, A.Heff, A.Decay_MPO[1:1:end], A.Jump_MPO[1:1:end], 0.025, (150,170), dmax, A.measureDict)
	evo_TDVP_SSE(x, MPOvec = A.Heff, dτ = dτ, name = name, saveState = true)
end

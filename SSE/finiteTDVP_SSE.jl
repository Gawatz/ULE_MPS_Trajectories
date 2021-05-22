using Distributed
using TensorOperations, KrylovKit, LinearAlgebra
using Combinatorics
using Base.Threads
using MPOmodule
using finiteMPS
using JLD

############################################################################
#
#	TDVP functions
#
############################################################################

function simple_taylor(Heff, x::AbstractArray{<:Any}, dτ::Number, order::Int)
	res = x
	tmp = x
	C = 1.0
	for i in 1:order
		tmp = Heff(tmp)
		C/= i
		res .+= C*tmp*(dτ^i)
	end

	return res

end

function simple_taylor_Heff!(MPO, Lenv, Renv, x::AbstractArray{<:Any}, dτ::Number, order::Int)
	res = x
	tmp = x
	C = 1.0
	for i in 1:order
		tmp = Heff(tmp)
		C/= i
		res .+= C*tmp*(dτ^i)
	end

	return res

end

function evo_sweep_TDVP(MPSvec::Vector{<:Any}, Cpre::AbstractArray{<:Number}, MPOvec::Vector{MPO}, 
			RBlocks::Vector{<:Any}, dτ::Union{Float64,ComplexF64}) 	
	
	order = 3
	taylor_t = []	
	env_t = []

	#
	#	foward sweep
	#
	Lenv = [Array{ComplexF64,2}(I,1,1)] 
	LBlocks = [Lenv]
	for site in 1:length(MPSvec)-1
		
		#
		# evolve A_site forward in time
		#
		localMPO = MPOvec[site]
		Renv = RBlocks[end-(site-1)]
		Lenv = LBlocks[site]
		Heff(x) = applyHeff(x, localMPO, Lenv, Renv)
		
		
		AR_site = MPSvec[site]
		@tensor MPSvec[site][α, d, β] := Cpre[α, γ]*AR_site[γ, d, β]

		

		ACnew = simple_taylor(Heff, MPSvec[site], -0.5*dτ, order)
		#ACnew, info = exponentiate(x->Heff(x), -0.5*dτ, MPSvec[site]; ishermitian = false, tol = 1e-14)

		#
		# split A_new_site and left can it
		#
		ALnew, S, Vdagger = leftCanSite(ACnew; optSVD = true)
		S = S./sign(S[1,1])	
		S = S./(sqrt(sum(S.^2))) #normalize S
		C = S*Vdagger
		MPSvec[site] = ALnew

		
		#
		# build up Lenv
		#
		Lenv = applyTM_MPO([MPSvec[site]], [localMPO], LBlocks[end]; left=true)
		push!(LBlocks, Lenv) # (int,1,....,N-1)
		
		
		#
		# evolve C back in time 
		#
		Hceff(x) = applyHCeff(x, Lenv, Renv)
	
		Cpre = simple_taylor(Hceff, C, 0.5*dτ, order)
		#Cpre, info = exponentiate(x->Hceff(x), 0.5*dτ, C; ishermitian = false, tol = 1e-14)
			
	end
	#
	#	evolve site N
	#
	Lenv=LBlocks[end]
	Renv=RBlocks[1]
	localMPO = MPOvec[end]
	Heff(x) = applyHeff(x, localMPO, Lenv, Renv)
	AR_site = MPSvec[end]
	@tensor v_0[α, d, β] := Cpre[α, γ]*AR_site[γ, d, β]

	ACnew = simple_taylor(Heff, v_0, -1.0*dτ, order)
	
	MPSvec[end] = ACnew 

	RBlocks = [Renv]


	#
	#	evolve back 
	#
	Cvec = []
	for site in length(MPSvec):-1:2
		
		AR, S, U = rightCanSite(MPSvec[site];optSVD=true)
		MPSvec[site] = AR
		S = S./sign(S[1,1])
		snorm = sqrt(sum(S.^2))
		S = S./snorm
		C = U*S
		push!(Cvec, C)
		
		#
		# add RBlock
		#
		Renv = applyTM_MPO([MPSvec[site]],[MPOvec[site]], RBlocks[end]; left = false)
		push!(RBlocks, Renv)#[int,N,.....,2]


		#
		#	evolve C backwards
		#
		Lenv = LBlocks[site]
		Hceff(x) = applyHCeff(x, Lenv, Renv)
		#Cnext, info = exponentiate(Hceff, 0.5*dτ, C; ishermitian = false, tol = 10e-16)
		Cnext = simple_taylor(Hceff, C, 0.5*dτ, order)	

		#
		#	evolve AC forward
		#
		@tensor v_0[α, d, β] := MPSvec[site-1][α, d, γ]*Cnext[γ, β]
		Lenv = LBlocks[site-1]
		Renv = RBlocks[end]
		localMPO = MPOvec[site-1]
		Heff(x) = applyHeff(x, localMPO, Lenv, Renv)
		

		#ACnew, info = exponentiate(Heff, -0.5*dτ, v_0; ishermitian = false, tol=10e-16, verbosity = 0, maxiter = 20)
		ACnew = simple_taylor(Heff, v_0, -0.5*dτ, order)
		
		MPSvec[site-1] = ACnew 
	
	end

	AR, S, U = rightCanSite(MPSvec[1]; optSVD = true)
	MPSvec[1] = AR

	S = S./sign(S[1,1])
	snorm = sqrt(sum(S.^2))
	S = S./snorm
	Cpre = U*S
	
	return Cpre, RBlocks, Cvec[end:-1:1]
end


function get1RDM(MPSvec::Vector{<:Any})
	γ = zeros(ComplexF64, length(MPSvec), length(MPSvec))
	for i in 1:length(MPSvec)
	 	for j in i:length(MPSvec)
			
	 		mpo = CMPO(i,j, length(MPSvec))
			c = getExpValue(MPSvec, mpo)
			γ[i,j] = c

			γ[j,i] = conj(c)

		end
	end

	return γ
end

############################################################################
#
#	MPS_SEE struct
#
############################################################################

struct MPS_SSE
	init_MPS::Vector{Array{ComplexF64,3}}
	H_MPO::Vector{MPO}
	Heff::Vector{MPO}
	Decay_MPO::Vector{Vector{MPO}}
	Jump_MPO::Vector{Vector{MPO}}
	n_rel::Float64 # expected number of jumps per time step 
	evoT::Tuple{Int, Int}
	maxDim::Int
	measureDict::Dict
end

function saveMPS_SSE(name, A::MPS_SSE)
	
	save(name, "MPS_SSE", A)

end

function loadMPS_SSE(name)
	
	return load(name)["MPS_SSE"]
	
end

function get_dτ(A::MPS_SSE, Decay_sum::Vector{MPO})	
	Γ_act = opnorm(creatOp_from_MPO(Decay_sum)[:,1,1,:])   # how do I change this for large systems 
	dτ = A.n_rel/Γ_act
	return dτ
end

function get_Heff(A::MPS_SSE; depara_opt = false)	
	H_MPO = A.H_MPO
	Decay_MPO = A.Decay_MPO
	
	Heff = H_MPO
	for decay_mpo in Decay_MPO
		
		Heff = Heff + (-0.5im*decay_mpo)
	end
	
	if depara_opt == true
		# deparallize mpo
		# add further MPO comperssion methods
		println("not implemented")
	end


	return Heff
end

function get_Decay_MPO(A::MPS_SSE)
	Decay_Sum_MPO = 0
	for (i,decay) in enumerate(A.Decay_MPO)
		Decay_Sum_MPO = i == 1 ? decay : Decay_Sum_MPO + decay

	end

	return Decay_Sum_MPO
end

function get_measurements(MPSvec::Vector{<:Any}, C::Vector{<:Any}, T::Float64, measureDict::Dict)
	N = size(MPSvec)[1]
	
	
	# ensure normalization 
	leftCanMPS(MPSvec)
	Cpre, __, Cvec = rightCanMPS(MPSvec)
	measure_results = []
	push!(measure_results, T) # time 
	Cvec = Cvec[end:-1:1]
	push!(Cvec, Cpre)
	Cvec = Cvec[end:-1:1]
	
	for x in measureDict
		#@show "measure: $(x[1])"

		if typeof(x[2]) == Vector{MPO}

			Renv = [Array{ComplexF64,2}(I,1,1)]
			measure_x= applyTM_MPO(MPSvec[1:end], x[2][1:end], Renv; left = false)[1,1][1]
			#@show x[1]
			#@show measure_x

		# eventuall add routine for single site measurement
		else
			measure_x = singleSiteExpValue(MPSvec, Cvec[x[2][1]], x[2][2],x[2][1]; can_left=false)
		end

		push!(measure_results, measure_x)
	end
	
	#entanglement
	for i in 0:Int(round(N/2, RoundDown))-1
		
		#@show Cvec[Int(round(N/2, RoundDown))-i]
		λ = diag(Cvec[Int(round(N/2, RoundDown))-i])
		#push!(measure_results, diag(Cvec[Int(round(N/2, RoundDown))]))
		#@show -2.0*sum((λ.^2).*log.(λ))
		push!(measure_results , -2.0*sum((λ.^2).*log.(λ)))


	end

	return measure_results
end

function evo_TDVP_SSE(MPS_SSE::MPS_SSE; MPOvec::Union{Nothing, Vector{MPO}} = nothing, dτ::Union{Number,Nothing} = nothing, name = nothing, saveState = false)
	
	
	#
	#	initialize result txt
	#
	if name != nothing 
		dir_name = name[1]
		id = name[2]
		name_measurement_files = string(dir_name,"/res_id$(id).txt")

		test =[x[1] for  x in MPS_SSE.measureDict]
		measurements = Array{Any,2}(undef, 1, length(test))
		measurements[1,:] = test[:]
		writedlm(name_measurement_files, measurements)
	end
	
	
	
	# initlize run
	MPSvec = MPS_SSE.init_MPS
	MPOvec = MPOvec == nothing ? get_Heff(MPS_SSE) : MPOvec
	maxDim = MPS_SSE.maxDim
	
	
	MPOvec_decay = deparaMPOChain!(get_Decay_MPO(MPS_SSE))
	dτ = dτ == nothing ?  get_dτ(MPS_SSE, MPOvec_decay) : dτ
	(Tinit, Tmax) = MPS_SSE.evoT	

	
	#right can. initial state
	Cpre, R, Cvec = rightCanMPS(MPSvec)
	
	#
	# build up right environment blocks
	#
	Renv = [Array{ComplexF64,2}(I,1,1)]
	RBlocks = [Renv]
	
	for site = 1:length(MPSvec)-1
		Renv = applyTM_MPO([MPSvec[end-(site-1)]],[MPOvec[end-(site-1)]], RBlocks[site]; left = false)
		push!(RBlocks, Renv)#[init,N,.....,2]
	end



	@show Tmax
	T = Tinit
	n_jumps = 0
	dτ = dτ
	t_measure = 20*dτ

	measure_list = []
	#@showprogress for iter in 0:dτ:(Tmax+dτ) #while T < Tmax 
	
	runtime = @elapsed while T < Tmax 
	
		#@show T	
		T += dτ; t_measure += dτ
		
		#
		#	TDVP - evo
		#
		preMPS = deepcopy(MPSvec)
		Cpre, RBlocks, Cvec = evo_sweep_TDVP(MPSvec, Cpre, MPOvec, RBlocks, im*dτ)	
		#
		#	stochastic jump
		#
		number_channels = length(MPS_SSE.Jump_MPO)
		
		#
		#	get decay prob.
		#
		#@time begin	
		R = [Array{ComplexF64,2}(I,1,1)]
		decay_prob = applyTM_MPO(MPSvec, MPOvec_decay, R;
					 left = false)[1,1][1]*dτ
		
		if abs(decay_prob) > 1 @error "decay prob > 1" end

		# get random number
		rand_jump = rand(1)[1]
		
		
		
		# perform jump
		if rand_jump < abs(decay_prob)
		
			# rand_jump is uniformly distributed between 
			# [0,decay_prob) we can use it to select channel
			rand_channel = rand_jump
			sum_d_channel = 0
			nop = 1
			Jump_MPO = nothing


			for n_channel in 1:number_channels
				D_MPO = MPS_SSE.Decay_MPO[n_channel]	
				
				R = [Array{ComplexF64,2}(I,1,1)]
				new_decay_contribution = applyTM_MPO(MPSvec, D_MPO, R; left = false)[1,1][1]
				sum_d_channel += new_decay_contribution*dτ
				#@show sum_d_channel, rand_channel, nop
				# check in which channel rand_channel lies
				if abs(sum_d_channel) > rand_channel

 					Jump_MPO =  MPS_SSE.Jump_MPO[nop]
					#println("*********jump occured at channel $(nop) ****************")
					break
				else
					nop += 1
				end
			
				#@show sum_d_channel, rand_channel
				@assert nop <= size(MPS_SSE.Decay_MPO)[1] "prob. didn't add up"
			end
			#
			#	variational application
			#
			#measure_results = get_measurements(deepcopy(MPSvec), pushfirst!(Cvec,Cpre), T, MPS_SSE.measureDict)
			
			
			#MPSvec = iter_applyMPO(MPSvec, Jump_MPO, maxDim; Niter = 1)		
			MPSvec = iter_applyMPO(preMPS, Jump_MPO, maxDim; Niter = 1)	
			
			#
			#	variational truncation 
			#
			#MPSvec = iter_trunc(MPSvec, 140; Niter = 1)
			
			
			leftCanMPS(MPSvec)
			Cpre, R, Cvec = rightCanMPS(MPSvec)
			n_jumps += 1	
			#@show n_jumps	
			
			#
			#	svd trunc
			#
			
			#=
			Renv = [Array{ComplexF64,2}(I,1,1)]
			@show applyTM_MPO(newMPSvec, measureDict["Np"], Renv; left = false)
			
			# show particle number in Schmidt vector 
			s = mixedCanMPS(newMPSvec, Int(round(N/2, RoundDown)))	
			#println("schmidt coefficients: $(s)")


			Renv = [Array{ComplexF64,2}(I,1,1)]
			np_half = [measureDict["Np"][1] ,measureDict["Np"][Int(round(N/2, RoundDown)+2):end]...]
			Rblock_total_N = applyTM_MPO(newMPSvec[Int(round(N/2, RoundDown)+1):end], np_half, Renv; left = false)
			#Rblock_total_N = sum(Rblock_total_N)
			N_λ_right = [Rblock_total_N[1][i,i] for i in 1:size(Rblock_total_N[1])[1]]
			#println("particle number right: $(N_λ_right)")

			
			Renv = [Array{ComplexF64,2}(I,1,1)]
			np_half = [measureDict["Np"][1:Int(round(N/2, RoundDown)-1)]..., measureDict["Np"][end]]
			Lblock_total_N = applyTM_MPO(newMPSvec[1:Int(round(N/2, RoundDown))], np_half, Renv; left = true)
			#Lblock_total_N = sum(Lblock_total_N)
			N_λ_left = [Lblock_total_N[1][i,i] for i in 1:size(Lblock_total_N[1])[1]]
			#println("particle number left: $(N_λ_left)")
			
			Np_s = Dict()
			for (i,x) in enumerate(s)
				Np_s[x] = (N_λ_left[i], N_λ_right[i])
			end
			
			for x in sort!([x for x in Np_s], by = x->x[1])
				println(x)
			end
			
			@tensor res[a,b,c] := newMPSvec[Int(round(N/2, RoundDown))][a,b, i]*diagm(0=>s)[i,c]
			newMPSvec[Int(round(N/2, RoundDown))] = res
			leftCanMPS(newMPSvec)
			rightCanMPS(newMPSvec)
			=#
			
			#
			#	you have to build a new right block
			#	bcs. MPS has changed from jump
			#
			Renv = [Array{ComplexF64,2}(I,1,1)]
			RBlocks = [Renv]
			for site = 1:length(MPSvec)-1
				Renv = applyTM_MPO([MPSvec[end-(site-1)]],[MPOvec[end-(site-1)]], RBlocks[site]; left = false)
				push!(RBlocks, Renv)#[init,N,.....,2]
			end
			
		end
		
		#
		# measure stuff MPS comes in right can. form
		#
		
		if t_measure >= 20*dτ
			@show T
			T = Float64(T)
			measure_results = get_measurements(deepcopy(MPSvec), pushfirst!(Cvec,Cpre), T, MPS_SSE.measureDict)
			t_measure = 0.0
			push!(measure_results, n_jumps)
			push!(measure_results, abs(decay_prob))
			push!(measure_list, measure_results)

			#
			#	save intermidiate state 
			#
			if name != nothing 
				dir_name = name[1]
				id = name[2]
				io = open(name_measurement_files, "a")
				write_res = Array{Any,2}(undef, 1, length(measure_results))
				write_res[1,:] = measure_results[:]
				writedlm(io, write_res)
				close(io)
				if saveState == true	
					name_inter_state = string(dir_name,"/inter_states/inter_state_id$(id).jld")
					name_1RDM = string(dir_name,"/inter_states/1RDM/1RDM_id$(id)_T$(T).jld")
					C = get1RDM(MPSvec)
					save(name_1RDM, "1RDM", C, "T", T)
					save(name_inter_state, "time", T, "MPSvec", MPSvec)
				end
			end
		end
	
	end
	
	
	rightCanMPS(MPSvec)
	name_inter_state = string(dir_name,"/inter_states/inter_state_id$(id).jld")
	save(name_inter_state, "time", T, "MPSvec", MPSvec)
	return runtime, measure_list#, getCoef(MPSvec)
end




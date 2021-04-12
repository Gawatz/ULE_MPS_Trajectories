function evo_sweep(MPSvec::Vector{<:Any}, Cpre::AbstractArray{<:Number}, MPOvec::Vector{MPO}, RBlocks::Vector{<:Any}, dτ::Union{Float64,ComplexF64}) 
	
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

		
		#MPS = Heff(MPSvec[site])
		#@tensor energyTest[] := MPS[a,b,c]*conj(MPSvec)[site][a,b,c]
		#@show energyTest


		ACnew, info = exponentiate(x->Heff(x), -0.5*dτ, MPSvec[site]; ishermitian = true, tol=10e-16, verbosity = 0, maxiter = 20)
		
		#
		# split A_new_site and left can it
		#
		
		ALnew, S, Vdagger = leftCanSite(ACnew; optSVD = true)
		S = S./sign(S[1,1])	
		snorm = sqrt(sum(S.^2)) #normalize S
		S = S./snorm
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
		Cpre, info = exponentiate(x->Hceff(x), 0.5*dτ, C; ishermitian = true, tol = 10e-16)
			
	end
	#@show "at N-1"
	#
	#	evolve site N
	#
	Lenv=LBlocks[end]
	Renv=RBlocks[1]
	localMPO = MPOvec[end]
	Heff(x) = applyHeff(x, localMPO, Lenv, Renv)
	AR_site = MPSvec[end]
	@tensor v_0[α, d, β] := Cpre[α, γ]*AR_site[γ, d, β]

	ACnew, info = exponentiate(Heff, -1.0*dτ, v_0; ishermitian = true, tol=10e-16, verbosity = 0, maxiter = 20)
	
	MPSvec[end] = ACnew 

	RBlocks = [Renv]

	#@show  "energy last site ",applyTM_MPO(MPSvec[1:end], MPOvec[1:end], [Array{ComplexF64,2}(I,1,1)]; left = false)
	
	#@show "at N"
	#
	#	evolve back 
	#
	Cvec = []
	for site in length(MPSvec):-1:2
		
		#here MPS hast to be reshaped 
		
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
		#
		Renv = applyTM_MPO([MPSvec[site]],[MPOvec[site]], RBlocks[end]; left = false)
		push!(RBlocks, Renv)#[int,N,.....,2]


		#
		#	evolve C backwards
		#
		Lenv = LBlocks[site]
		Hceff(x) = applyHCeff(x, Lenv, Renv)
		Cnext, info = exponentiate(Hceff, 0.5*dτ, C)
	

		#
		#	evolve AC forward
		#
		@tensor v_0[α, d, β] := MPSvec[site-1][α, d, γ]*Cnext[γ, β]
		Lenv = LBlocks[site-1]
		Renv = RBlocks[end]
		localMPO = MPOvec[site-1]
		Heff(x) = applyHeff(x, localMPO, Lenv, Renv)

		ACnew, info = exponentiate(Heff, -0.5*dτ, v_0; ishermitian = true, tol=10e-16, verbosity = 0, maxiter = 20)

		MPSvec[site-1] = ACnew 
	
	end

	AR, S, U = rightCanSite(MPSvec[1]; optSVD = true)
	MPSvec[1] = AR
	S = S./sign(S[1,1])
	snorm = sqrt(sum(S.^2))
	S = S./snorm
	Cpre = U*S
	#@show Cpre

	return Cpre, RBlocks, Cvec[end:-1:1]
end

function evo_sweep_2Site(MPSvec::Vector{<:Any}, Cpre::AbstractArray{<:Number}, MPOvec::Vector{MPO}, RBlocks::Vector{<:Any}, dτ::Union{Float64,ComplexF64}; maxDim::Int = 200) 
	
	#
	#	foward sweep
	#
	Lenv = [Array{ComplexF64,2}(I,1,1)] 
	LBlocks = [Lenv]
	AC = MPSvec[1]
	for site in 2:length(MPSvec)
		


		#
		# 	evolve 2 sites forward
		#
		
		# 	get MPOs
		localMPO1 = MPOvec[site-1]
		localMPO2 = MPOvec[site]

		#	get Environment
		Renv = RBlocks[end-(site-1)]
		Lenv = LBlocks[site-1]
		# change to pop! 
		#Renv = pop!(RBlocks)
		#Lenv = LBlocks[end]



		Heff2(x) = applyHeff2(x, localMPO1, localMPO2, Lenv, Renv)
		@tensor AC2[α, d1, d2, β] := AC[α, d1, γ]*MPSvec[site][γ, d2, β]
		AC2new, info = exponentiate(x->Heff2(x), -0.5*dτ, AC2; ishermitian = true, tol=10e-16, verbosity = 0, maxiter = 20)
		


		#
		#	split AC2new
		#
		(a,d1, d2, b) = size(AC2new)
		AC2 = reshape(AC2new, a*d1, d2*b)
		U, S, V = svd(AC2)
		
		if length(S) > maxDim
			S = S[1:maxDim] # truncate
		end
		sdim = length(S)


		S = diagm(0=>S)
		S = S./sign(S[1,1])	
		snorm = sqrt(sum(S.^2)) #normalize S
		S = S./snorm


		AL = reshape(U[:,1:sdim], a, d1, size(S)[1])
		MPSvec[site-1] = AL
		AC = reshape(S*(V')[1:sdim,:], size(S)[2], d2, b)
		

		#
		#	evolve AC backwards
		#


		#
		#	build up Lenv
		#
		Lenv = applyTM_MPO([MPSvec[site-1]], [localMPO1], LBlocks[end]; left=true)
		push!(LBlocks, Lenv) # (int,1,....,N-1)
		
	
		#
		#	evolve C back in time 
		#
		Heff(x) = applyHeff(x, localMPO2, Lenv, Renv)
		AC, info = exponentiate(x->Heff(x), 0.5*dτ, AC; ishermitian = true, tol=10e-16, verbosity = 0, maxiter = 20)
			

		


	end


	#pop!(LBlokcs)
	Renv=RBlocks[1]
	RBlocks = [Renv]
	#
	#	evolve back 
	#
	Cvec = []
	for site in length(MPSvec)-1:-1:1
		
		
		# 	get MPOs
		localMPO1 = MPOvec[site]
		localMPO2 = MPOvec[site+1]
		
		#	get Environment
		Renv = RBlocks[end]
		Lenv = LBlocks[site]
		# with pop!
		#Lenv = pop!(LBlocks)
		#Renv = RBlocks[end]
	
		#
		#	evolve AC forward
		#
		Heff2(x) = applyHeff2(x, localMPO1, localMPO2, Lenv, Renv)
		@tensor AC2[α, d1, d2, β] := AC[γ, d2, β]*MPSvec[site][α, d1, γ]
		AC2new, info = exponentiate(x->Heff2(x), -0.5*dτ, AC2; ishermitian = true, tol=10e-16, verbosity = 0, maxiter = 20)
	

		#
		#	split AC2new
		#
		(a,d1, d2, b) = size(AC2new)
		AC2 = reshape(AC2new, a*d1, d2*b)
		U, S, V = svd(AC2)
		
		if length(S) > maxDim
			S = S[1:maxDim] # truncate
		end
		sdim = length(S)
		
		
		S = diagm(0=>S)
		S = S./sign(S[1,1])	
		snorm = sqrt(sum(S.^2)) #normalize S
		S = S./snorm

		AC = reshape(U[:,1:sdim]*S, a, d1, size(S)[1])
		AR = reshape((V')[1:sdim, :], size(S)[2], d2, b)
		MPSvec[site+1] = AR




		#
		# add RBlock
		#
		#
		Renv = applyTM_MPO([MPSvec[site+1]],[localMPO2], RBlocks[end]; left = false)
		push!(RBlocks, Renv)#[int,N,.....,2]


		#
		#	evolve C backwards
		#
		Lenv = LBlocks[site]
		Heff(x) = applyHeff(x, localMPO1, Lenv, Renv)
		AC, info = exponentiate(x->Heff(x), 0.5*dτ, AC; ishermitian = true, tol=10e-16, verbosity = 0, maxiter = 20)
		
	end

	MPSvec[1] = AC
	AR, S, U = rightCanSite(MPSvec[1]; optSVD = true)
	MPSvec[1] = AR
	S = S./sign(S[1,1])
	snorm = sqrt(sum(S.^2))
	S = S./snorm
	Cpre = U*S
	#@show Cpre

	return Cpre, RBlocks

end

function vmps_sweep(MPS::Vector{<:Any}, Cpre::AbstractArray, MPOvec::Vector{MPO}, RBlocks::Vector{<:Any}) 
	
	#
	#	foward sweep
	#
	L = Array{ComplexF64}(I,1,1)
	Lenv = [L] 
	LBlocks = [Lenv]
	for site in 1:length(MPS)-1
		
		#
		# evolve A_site forward in time
		#
		
		localMPO = MPOvec[site]
		Renv = RBlocks[end-(site-1)]
		Lenv = LBlocks[site]
		Heff(x) = applyHeff(x, localMPO, Lenv, Renv)
		AR_site = MPS[site]
		@tensor MPS[site][α, d, β] := Cpre[α, γ]*AR_site[γ, d, β]

		

		E, ACnew, info = eigsolve(x->Heff(x), MPS[site], 1, :SR, ishermitian = true, tol= 10e-14)
		@show E
		ACnew = ACnew[1]
		#@show info
		
		#
		# split A_new_site and left can it
		#
		
		ALnew, S, Vdagger = leftCanSite(ACnew; optSVD = true)

		
		S = S/sign(S[1,1])
		S = S/(sqrt(dot(S,S))) #normalize S
		Cpre = S*Vdagger
		
		MPS[site] = ALnew

		
		#
		# build up Lenv
		#
		Lenv = applyTM_MPO([MPS[site]], [localMPO], LBlocks[end]; left=true)
		push!(LBlocks, Lenv) # (int,1,....,N-1)
		
			
	end
	#@show "at N-1"
	#
	#	evolve site N
	#
	Lenv=LBlocks[end]
	Renv=RBlocks[1]
	localMPO = MPOvec[end]
	Heff(x) = applyHeff(x, localMPO, Lenv, Renv)
	AR_site = MPS[end]
	
	@tensor v_0[α, d, β] := Cpre[α, γ]*AR_site[γ, d, β]

	#ACnew, info = exponentiate(Heff, -1.0*dτ, SparseTensor(v_0); ishermitian = true, tol=10e-16, verbosity = 0, maxiter = 20)
	E, ACnew, info = eigsolve(x->Heff(x), v_0, 1, :SR, ishermitian = true, tol= 10e-14)
	@show E

	MPS[end] = ACnew[1] 
	RBlocks = [Renv]

	#
	#	evolve back 
	#
	Cvec = []
	for site in length(MPS):-1:2
		
		#here MPS hast to be reshaped 
		
		AR, S, U = rightCanSite(MPS[site]; optSVD=true)
		MPS[site] = AR
		S = S/sign(S[1,1])
		snorm = sqrt(dot(S,S))
		S = S/snorm
		Cnext = U*S
		push!(Cvec, Cnext)
		
		
		#
		# add RBlock
		#
		Renv = applyTM_MPO([MPS[site]], [MPOvec[site]], RBlocks[end]; left = false)
		push!(RBlocks, Renv)#[int,N,.....,2]

		#
		#	evolve AC forward
		#
		@tensor v_0[α, d, β] := MPS[site-1][α, d, γ]*Cnext[γ, β]
		Lenv = LBlocks[site-1]
		Renv = RBlocks[end]
		localMPO = MPOvec[site-1]
		Heff(x) = applyHeff(x, localMPO, Lenv, Renv)

		E, ACnew, info = eigsolve(x->Heff(x), v_0, 1, :SR, ishermitian = true, tol= 10e-14)
		@show E
		MPS[site-1] = ACnew[1]
	
	end
	
	AR, S, U = rightCanSite(MPS[1]; optSVD = true)
	MPS[1] = AR

	S = S/sign(S[1,1])
	snorm = sqrt(dot(S,S))
	S = S/snorm
	Cpre = U*S
	
	return Cpre, RBlocks, Cvec[end:-1:1]
end


function applyMPO_sweep(MPSvec, new_MPSvec, MPOvec, RBlocks)	
	Lenv = [Array{ComplexF64,2}(I,1,1)]
	LBlocks = [Lenv]
	Cpre = Array{ComplexF64,2}(I,1,1)	
	N = size(MPSvec)[1]
	
	for site in 1:N
		
		
		# multiply B(site) with Cpre to form M
		localMPO = MPOvec[site]
		Renv = RBlocks[end-(site-1)]
		Lenv = LBlocks[site]
		
		Heff(x) = applyHeff(x, localMPO, Lenv, Renv)
		AR_site = MPSvec[site]
		@tensor MPSvec[site][α, d, β] := Cpre[α, γ]*AR_site[γ, d, β]
	
		#
		#	apply Heff
		#
		new_M = Heff(MPSvec[site])
		#move center
		A, S, V = leftCanSite(new_M, optSVD=true)
		
		# update site
		new_MPSvec[site] = A


		A, S, V = leftCanSite(MPSvec[site], optSVD=true)
		S = S./sign(S[1,1])	
		S = S./(sqrt(sum(S.^2))) #normalize S
		Cpre = S*V

		# update site
		MPSvec[site] = A

		if site < N
			#
			# build up Lenv
			#
			Lenv = applyTM_MPO([MPSvec[site]],[new_MPSvec[site]], [localMPO], LBlocks[end]; left=true)
			push!(LBlocks, Lenv) # (int,1,....,N-1)
		end
	end


	Renv = [Array{ComplexF64,2}(I,1,1)]
	RBlocks = [Renv]

	Cnext = Cpre #Array{ComplexF64, 2}([[1]])
	for site in N:-1:1
	
		AL_site = MPSvec[site]
		@tensor MPSvec[site][α, d, β] := AL_site[α, d, γ]*Cnext[γ, β]
		Lenv = LBlocks[site]
		Renv = RBlocks[end]
		
		localMPO = MPOvec[site]
		Heff(x) = applyHeff(x, localMPO, Lenv, Renv)

		# multiply A(site) with Cpre to form M
		new_M = Heff(MPSvec[site])
		
		#move center
		B, __, __ = rightCanSite(new_M, optSVD=true)
		#B, S, U = rightCanSite(new_M, optSVD=true)
		#S = S./sign(S[1,1])
		#snorm = sqrt(sum(S.^2))
		#S = S./snorm
		#Cnext = U*S

		# update site
		new_MPSvec[site] = B

		B, S, U = rightCanSite(MPSvec[site], optSVD=true)
		S = S./sign(S[1,1])
		snorm = sqrt(sum(S.^2))
		S = S./snorm
		Cnext = U*S

		MPSvec[site] = B


		# update right environment
		if site > 1
			Renv = applyTM_MPO([MPSvec[site]], [new_MPSvec[site]], [MPOvec[site]], RBlocks[end]; left = false)
			push!(RBlocks, Renv)#[int,N,.....,2]
		end
	end

	return new_MPSvec, RBlocks
end

function iter_applyMPO(MPSvec::Vector{<:Any}, MPOvec::Vector{MPO}, new_MaxD::Int; Niter = 1)	
	N = size(MPSvec)[1]
	d = size(MPSvec[1])[2]

	new_MPSvec, Cvec = randMPS(N, d, new_MaxD)

	# bring both MPS in right can form 
	rightCanMPS(MPSvec)
	rightCanMPS(new_MPSvec)

	
	Renv = [Array{ComplexF64,2}(I,1,1)]
	RBlocks = [Renv]
	for site = 1:length(MPSvec)-1
		Renv = applyTM_MPO([MPSvec[end-(site-1)]], [new_MPSvec[end-(site-1)]], [MPOvec[end-(site-1)]], RBlocks[site]; left = false)
		push!(RBlocks, Renv)#[init,N,.....,2]
	end


	# sweep through the system and minimize || |ψ_old> - |ψ_new> ||^2
	for i in 1:Niter	
		new_MPSvec, RBlocks = applyMPO_sweep(MPSvec, new_MPSvec, MPOvec, RBlocks)
	end

	return new_MPSvec
end

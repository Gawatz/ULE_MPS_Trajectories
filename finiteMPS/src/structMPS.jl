# DT can be block sparse structure or normal tensor 
struct MPS{DT}

	Mvec::Vector{DT}
end





import Base.iterate
iterate(mps::MPS) = iterate(mps.Mvec)
iterate(mps::MPS, state) = iterate(mps.Mvec, state)


import Base.getindex
getindex(mps, idx::Int) = mps.Mvec[idx]

import Base.setindex
function setindex(mps::MPS, v, idx::Int)
	mps.Mvec[idx] = v
end



#
#	simple arimetric 
#

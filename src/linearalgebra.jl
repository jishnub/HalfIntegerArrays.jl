import LinearAlgebra: wrapperop

for AT in [:AdjOrTransAbsHalfIntMatrix, :AdjOrTransAbsHalfIntVector]
	for IT in [:Integer, :Real]
		@eval function Base.isassigned(A::$AT, I::$IT...)
			checkbounds(Bool, A, I...) || return false
		    K = trimtoN(I,Val(2))
		    isassigned(parent(A), reverse(K)...)
		end
	end
	for IT in [:Int, :HalfInt]
		@eval @propagate_inbounds function Base.getindex(A::$AT, i::$IT, j::$IT)
			@boundscheck checkbounds(A, i, j)
			wrapperop(A)(A.parent[j,i])
		end

		@eval @propagate_inbounds function Base.setindex!(A::$AT, val, i::$IT, j::$IT)
			@boundscheck checkbounds(A, i, j)
			A.parent[j,i] = wrapperop(A)(val)
			A
		end
	end
end

@propagate_inbounds function Base.getindex(a::AdjOrTransAbsHalfIntVecOrMat, i::Int)
	@boundscheck checkbounds(a, i)
	c = CartesianIndicesHalfInt(axes(a))
	inds = Tuple(c[i])
	a[inds...]
end

@propagate_inbounds function Base.setindex!(a::AdjOrTransAbsHalfIntVecOrMat, val, i::Int)
	@boundscheck checkbounds(a, i)
	c = CartesianIndicesHalfInt(axes(a))
	inds = Tuple(c[i])
	a[inds...] = val
	a
end

function LinearAlgebra.diagind(h::AbstractHalfIntegerMatrix, k::Integer=0)
	diagind(size(h,1),size(h,1),k)
end
# These methods fall back to the parent by default, assuming that parent(x) != x 
# Subtypes should specialize them if custom behaviour is desired
for f in [:isdiag, :isbanded, :issymmetric, :ishermitian, :isposdef, :det]
	@eval LinearAlgebra.$f(h::AbstractHalfIntegerMatrix) = $f(parent(h))
end

for f in [:istriu, :istril]
	@eval LinearAlgebra.$f(h::AbstractHalfIntegerMatrix, k::Integer=0) = $f(parent(h), k)
end
LinearAlgebra.diag(h::AbstractHalfIntegerMatrix, k::Integer=0) = parent(h)[diagind(h,k)]

function LinearAlgebra.inv(h::HalfIntArray)
	HalfIntArray(inv(parent(h)), h.offsets)
end
function LinearAlgebra.inv(s::SpinMatrix)
	SpinMatrix(inv(parent(s)), s.j)
end

function Base.:(*)(h1::HalfIntMatrix, h2::HalfIntMatrix)
	axes(h1,2) == axes(h2,1) || 
	throw(DimensionMismatch("A has dimension $(axes(h1)) but B has dimension $(axes(h2))"))

	y1, y2 = map(unwraphalfint,(h1,h2))
	HalfIntArray(y1 * y2, h1.offsets)
end
function Base.:(*)(S1::SpinMatrix, S2::SpinMatrix)
	S1.j == S2.j || 
	throw(DimensionMismatch("A has dimension $(axes(S1)) but B has dimension $(axes(S2))"))

	y1, y2 = map(unwraphalfint,(S1,S2))
	SpinMatrix(y1 * y2, S1.j)
end
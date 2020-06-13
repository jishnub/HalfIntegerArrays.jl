"""
	HalfIntSubArray{T,N,P,I,L} <: AbstractHalfIntegerArray{T,N}

`N`-dimensional view into a parent array (of type `P <: AbstractHalfIntegerArray`) 
with an element type `T`, restricted by a tuple of indices (of type `I`). 
`L` is `true` for types that support fast linear indexing, and false otherwise.

Their behavior is analogous to `SubArray`s, except they enable indexing with 
half-integers.

Construct `HalfIntSubArray`s using the `view` function.

# Example
```jldoctest
julia> h = HalfIntArray(reshape(1:9, 3, 3), -1:1, -1:1)
3×3 HalfIntArray(reshape(::UnitRange{Int64}, 3, 3), -1:1, -1:1) with eltype Int64 with indices -1:1×-1:1:
 1  4  7
 2  5  8
 3  6  9

julia> hv = @view h[-1:1, 0]
3-element HalfIntSubArray(view(reshape(::UnitRange{Int64}, 3, 3), 1:3, 2), -1:1) with eltype Int64 with indices -1:1:
 4
 5
 6

julia> hv[0]
5

julia> h = HalfIntArray(reshape(collect(1:4), 2, 2), -1//2:1//2, -1//2:1//2)
2×2 HalfIntArray(::Array{Int64,2}, -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> hv = @view h[:, 1//2]
2-element HalfIntSubArray(view(::Array{Int64,2}, :, 2), -1/2:1/2) with eltype Int64 with indices -1/2:1/2:
  3
 10

julia> hv[1//2] = 10
10

julia> h
2×2 HalfIntArray(::Array{Int64,2}, -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1   3
 2  10
```
"""
struct HalfIntSubArray{T,N,P,I,L} <: AbstractHalfIntegerArray{T,N}
	parent :: SubArray{T,N,P,I,L}
	offsets :: NTuple{N,HalfInt} # only relevant for Cartesian Indexing
end

Base.parent(h::HalfIntSubArray) = h.parent

parenttype(::Type{HalfIntSubArray{T,N,P,I,L}}) where {T,N,P,I,L} = SubArray{T,N,P,I,L}

subarrayoffsets(::Tuple{}) = ()
subarrayoffsets(inds::Tuple{Real,Vararg{Any}}) = subarrayoffsets(Base.tail(inds))
function subarrayoffsets(inds::Tuple{AbstractRange, Vararg{Any}})
	(zero(HalfInt), subarrayoffsets(Base.tail(inds))...)
end
function subarrayoffsets(inds::Tuple{Base.Slice{<:IdOffsetRange}, Vararg{Any}})
	(first(inds).indices.offset, 
		subarrayoffsets(Base.tail(inds))...)
end

function HalfIntSubArray(A::AbstractArray{<:Any,N}, inds::NTuple{N,Union{AbstractRange, Real}}) where {N}
	@boundscheck checkbounds(A, inds...)
	parentinds = to_parentindices(axes(A), inds)
	indsoffset = subarrayoffsets(inds)
	s = @view A[parentinds...]
	HalfIntSubArray(s, indsoffset)
end
function HalfIntSubArray(h::AbstractHalfIntegerArray{<:Any,N}, inds::NTuple{N,Union{AbstractRange, Real}}) where {N}
	#=
	inds are the indices of the subarray. We translate these to the indices of the parent array using the axes.
	Given an IdOffsetRange axis ro = (rp,ro), an index i is mapped to rp[i - ro]
	=#
	@boundscheck checkbounds(h, inds...)
	parentinds = to_parentindices(axes(h), inds)
	indsoffset = subarrayoffsets(inds)
	s = @view parent(h)[parentinds...]
	HalfIntSubArray(s, indsoffset)
end
function HalfIntSubArray(h::AbstractHalfIntegerArray{<:Any,N}, I::NTuple{N,Any}) where N
	@boundscheck checkbounds(h, I...)
	HalfIntSubArray(h,to_indices(h,I))
end
function HalfIntSubArray(h::AbstractHalfIntegerArray{<:Any,N}, I::NTuple{M,Any}) where {N,M}
	@boundscheck checkbounds(h, I...)
	h′ = reshape(h, Val(M))
	HalfIntSubArray(h′, to_indices(h,I))
end

# Avoid a level of indirection

function HalfIntSubArray(h::HalfIntSubArray{<:Any,N}, inds::NTuple{N,Union{AbstractRange, Real}}) where {N}
	#=
	inds are the indices of the subarray. We translate these to the indices of the parent array using the axes.
	Given an IdOffsetRange axis ro = (rp,ro), an index i is mapped to rp[i - ro]
	=#
	@boundscheck checkbounds(h, inds...)
	indsoffset = subarrayoffsets(inds)
	parentinds = to_parentindices(axes(h), inds)
	s = @view parent(h)[parentinds...]
	HalfIntSubArray(s, indsoffset)
end

# AbstractArray implementation

Base.size(h::HalfIntSubArray) = size(parent(h))
Base.axes(h::HalfIntSubArray) = map(IdOffsetRange,axes(parent(h)), h.offsets)
function Base.axes(h::HalfIntSubArray, d)
	d <= ndims(h) ? IdOffsetRange(axes(parent(h),d), h.offsets[d]) : IdOffsetRange(Base.OneTo(1))
end

@propagate_inbounds function Base.getindex(h::HalfIntSubArray, I::HalfInt...)
	@boundscheck checkbounds(h, I...)
	J = to_parentindices(axes(h), I)
	parent(h)[J...]
end

# 1D views use Cartesian indexing
for T in [:HalfInt,:Real,:Int,:Integer,:HalfInteger]
	@eval @propagate_inbounds function Base.getindex(h::HalfIntSubArray{<:Any,1}, i::$T)
		@boundscheck checkbounds(h, i)
		ip = parentindex(axes(h,1),i)
		parent(h)[ip]
	end
end

# Indexing with a single Int/HalfInt forces linear indexing
@propagate_inbounds Base.getindex(A::HalfIntSubArray, i::Int)  = parent(A)[i]

@propagate_inbounds function Base.setindex!(A::HalfIntSubArray, val, I::HalfInt...)
    @boundscheck checkbounds(A, I...)
    J = to_parentindices(axes(A), I)
    parent(A)[J...] = val
    A
end

# 1D arrays use Cartesian indexing
for DT in [:HalfInt,:Real,:Int,:HalfInteger,:Integer]
    @eval @propagate_inbounds function Base.setindex!(A::HalfIntSubArray{<:Any,1}, val, i::$DT)
        @boundscheck checkbounds(A, i)
        J = parentindex(Base.axes1(A), i)
        parent(A)[J] = val
        A
    end
end

# Linear indexing
@propagate_inbounds function Base.setindex!(A::HalfIntSubArray, val, i::Int)
    parent(A)[i] = val
    A
end

# Hash

function Base.hash(A::HalfIntSubArray, h::UInt)
	h = hash(A.offsets, h)
	h = hash(parent(A), h)
	h = hash(:HalfIntSubArray, h)
	return h
end

# View 

Base.view(h::AbstractHalfIntegerArray, inds...) = HalfIntSubArray(h,inds)

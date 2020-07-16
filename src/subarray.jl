"""
	HalfIntSubArray{T,N,P,I,L}

`N`-dimensional view into a parent array (of type `P`) with an element type `T`, 
restricted by a tuple of indices (of type `I`). 
`L` is `true` for types that support fast linear indexing, and false otherwise.

Their behavior is analogous to `SubArray`s, except that they enable indexing with 
half-integers.

Construct `HalfIntSubArray`s from [`AbstractHalfIntegerArray`](@ref)s 
using the `view` function, or equivalently 
the `@view` macro.

# Example
```jldoctest
julia> h = HalfIntArray(reshape(1:9, 3, 3), -1:1, -1:1)
3×3 HalfIntArray(reshape(::UnitRange{Int64}, 3, 3), -1:1, -1:1) with eltype Int64 with indices -1:1×-1:1:
 1  4  7
 2  5  8
 3  6  9

julia> hv = @view h[-1:1, 0]
3-element view(HalfIntArray(reshape(::UnitRange{Int64}, 3, 3), -1:1, -1:1), -1:1, 0) with eltype Int64:
 4
 5
 6

julia> hv[1]
4

julia> h = HalfIntArray(reshape(collect(1:4), 2, 2), -1//2:1//2, -1//2:1//2)
2×2 HalfIntArray(::Array{Int64,2}, -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> hv = @view h[:, 1//2]
2-element view(HalfIntArray(::Array{Int64,2}, -1/2:1/2, -1/2:1/2), :, 1/2) with eltype Int64 with indices -1/2:1/2:
 3
 4

julia> hv[1//2] = 10
10

julia> h
2×2 HalfIntArray(::Array{Int64,2}, -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1   3
 2  10
```
"""
struct HalfIntSubArray{T,N,P,I,O,L} <: AbstractHalfIntegerArray{T,N}
	parent :: P
	indices :: I
	offset1:: O
	stride1:: Int

	function HalfIntSubArray{T,N,P,I,O,L}(parent, indices, offset1, stride1) where {T,N,P,I,O,L}
        Base.check_parent_index_match(parent, indices)
        new(parent, indices, offset1, stride1)
    end
end

# Compute the linear indexability of the indices, and combine it with the linear indexing of the parent
function HalfIntSubArray(parent::AbstractArray, indices::Tuple)
    HalfIntSubArray(IndexStyle(Base.viewindexing(indices), IndexStyle(parent)), parent, 
    	Base.ensure_indexable(indices), Base.index_dimsum(indices...))
end
function HalfIntSubArray(::IndexCartesian, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    HalfIntSubArray{eltype(P), N, P, I, Int, false}(parent, indices, 0, 0)
end
function HalfIntSubArray(::IndexLinear, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    # Compute the stride and offset
    stride1 = Base.compute_stride1(parent, indices)
    offset1 = Base.compute_offset1(parent, stride1, indices)
    HalfIntSubArray{eltype(P), N, P, I, typeof(offset1), true}(parent, indices, offset1, stride1)
end

# Half-integer vectors use Cartesian indexing, the offset will be half-integer in general
function Base.compute_offset1(parent::AbstractHalfIntegerVector, stride1::Integer, I::Tuple{Real})
    HalfInt(I[1]) - stride1
end
Base.compute_stride1(s, inds, I::Tuple{AbstractUnitRange{HalfInt}, Vararg{Any}}) = s

Base.parent(h::HalfIntSubArray) = h.parent

Base.size(V::HalfIntSubArray) = map(n->Int(Base.unsafe_length(n)), axes(V))

Base.similar(V::HalfIntSubArray, T::Type, dims::Dims) = similar(V.parent, T, dims)

Base.sizeof(V::HalfIntSubArray) = length(V) * sizeof(eltype(V))

Base.copy(V::HalfIntSubArray) = V.parent[V.indices...]

Base.dataids(v::HalfIntSubArray) = Base.dataids(parent(v))

@static if VERSION >= v"1.2"
    reindex(V, I) = Base.reindex(V.indices, I)
else
    reindex(V, I) = Base.reindex(V, V.indices, I)
end

_maybeInt(::AbstractHalfIntegerArrayOrWrapper, i::Real) = i
_maybeInt(::AbstractArray, i::Real) = Int(i)

# In general, we simply re-index the parent indices by the provided ones
@inline function Base.getindex(V::HalfIntSubArray{T,N}, I::Vararg{Real,N}) where {T,N}
    @boundscheck checkbounds(V, I...)
    @inbounds r = V.parent[reindex(V, I)...]
    r
end
@propagate_inbounds function Base.getindex(V::HalfIntSubArray{T,N}, I::Vararg{Real}) where {T,N}
	@boundscheck checkbounds(V, I...)
	IN = trimtoN(I, Val(N))
	V[IN...]
end

# linear indexing for slow subarrays is through Cartesian indexing
@inline function Base.getindex(V::HalfIntSubArray{T,N}, i::Real) where {T,N}
	@boundscheck checkbounds(V, i)
    @boundscheck ensureInt(i)
	@inbounds V[CartesianIndicesHalfInt(V)[unsafeInt(i)]]
end

# But SubArrays with fast linear indexing pre-compute a stride and offset
FastSubArray{T,N,P,I,O} = HalfIntSubArray{T,N,P,I,O,true}
@inline function Base.getindex(V::FastSubArray, i::Real)
    @boundscheck checkbounds(V, i)
    ensureInt(i)
    @inbounds r = V.parent[_maybeInt(V.parent, V.offset1 + V.stride1*i)]
    r
end

# We can avoid a multiplication if the first parent index is a Colon or AbstractUnitRange,
# or if all the indices are scalars, i.e. the view is for a single value only
FastContiguousSubArray{T,N,P,I<:Union{Tuple{Union{Base.Slice, AbstractUnitRange}, Vararg{Any}},
                                      Tuple{Vararg{Real}}},O} = HalfIntSubArray{T,N,P,I,O,true}
@inline function Base.getindex(V::FastContiguousSubArray, i::Real)
    @boundscheck checkbounds(V, i) 
    ensureInt(i)
    @inbounds r = V.parent[_maybeInt(V.parent, V.offset1 + i)]
    r
end
# For vector views with linear indexing, we disambiguate to favor the stride/offset
# computation as that'll generally be faster than (or just as fast as) re-indexing into a range.
@inline function Base.getindex(V::FastSubArray{<:Any, 1}, i::Real)
    @boundscheck checkbounds(V, i)
    @inbounds r = V.parent[_maybeInt(V.parent, V.offset1 + V.stride1*i)]
    r
end
@inline function Base.getindex(V::FastContiguousSubArray{<:Any, 1}, i::Real)
    @boundscheck checkbounds(V, i)
    @inbounds r = V.parent[_maybeInt(V.parent, V.offset1 + i)]
    r
end

# Indexed assignment follows the same pattern as `getindex` above
@inline function Base.setindex!(V::HalfIntSubArray{T,N}, x, I::Vararg{Real,N}) where {T,N}
    @boundscheck checkbounds(V, I...)
    @inbounds V.parent[reindex(V, I)...] = x
    V
end
@propagate_inbounds function Base.setindex!(V::HalfIntSubArray{T,N}, val, I::Vararg{Real}) where {T,N}
	@boundscheck checkbounds(V, I...)
	IN = trimtoN(I, Val(N))
	V[IN...] = val
	V
end
# linear indexing for slow subarrays is through Cartesian indexing
@inline function Base.setindex!(V::HalfIntSubArray{T,N}, val, i::Real) where {T,N}
	@boundscheck checkbounds(V, i)
	ensureInt(i)
	@inbounds V[CartesianIndicesHalfInt(V)[unsafeInt(i)]] = val
	V
end

@inline function Base.setindex!(V::FastSubArray, x, i::Real)
    @boundscheck checkbounds(V, i)
    ensureInt(i)
    @inbounds V.parent[_maybeInt(V.parent, V.offset1 + V.stride1*i)] = x
    V
end
@inline function Base.setindex!(V::FastContiguousSubArray, x, i::Real)
    @boundscheck checkbounds(V, i)
    ensureInt(i)
    @inbounds V.parent[_maybeInt(V.parent, V.offset1 + i)] = x
    V
end
@inline function Base.setindex!(V::FastSubArray{<:Any, 1}, x, i::Real)
    @boundscheck checkbounds(V, i)
    @inbounds V.parent[_maybeInt(V.parent, V.offset1 + V.stride1*i)] = x
    V
end
@inline function Base.setindex!(V::FastContiguousSubArray{<:Any, 1}, x, i::Real)
    @boundscheck checkbounds(V, i)
    @inbounds V.parent[_maybeInt(V.parent, V.offset1 + i)] = x
    V
end

Base.IndexStyle(::Type{<:FastSubArray}) = IndexLinear()
Base.IndexStyle(::Type{<:HalfIntSubArray}) = IndexCartesian()

# indices are taken from the range/vector
# Since bounds-checking is performance-critical and uses
# indices, it's worth optimizing these implementations thoroughly
Base.axes(S::HalfIntSubArray) = _indices_sub(S.indices...)
_indices_sub(::Real, I...) = _indices_sub(I...)
_indices_sub() = ()
function _indices_sub(i1::AbstractArray, I...)
    (Base.unsafe_indices(i1)..., _indices_sub(I...)...)
end

Base.unaliascopy(A::HalfIntSubArray) = typeof(A)(Base.unaliascopy(A.parent), map(Base.unaliascopy, A.indices), A.offset1, A.stride1)

# View 

## SubArray creation
# We always assume that the dimensionality of the parent matches the number of
# indices that end up getting passed to it, so we store the parent as a
# ReshapedArray view if necessary. The trouble is that arrays of `CartesianIndex`
# can make the number of effective indices not equal to length(I).
_maybe_reshape_parent(A::AbstractArray, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::AbstractArray{<:Any,1}, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape_parent(A::AbstractArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A
_maybe_reshape_parent(A::AbstractArray, ::NTuple{N, Bool}) where {N} = reshape(A, Val(N))

function Base.view(A::AbstractHalfIntegerArrayOrWrapper, I::Vararg{Any,N}) where {N}
    J = map(i->Base.unalias(A,i), to_indices(A, I))
    @boundscheck checkbounds(A, J...)
    unsafe_view(_maybe_reshape_parent(A, Base.index_ndims(J...)), J...)
end

ViewIndex = Union{Real, AbstractArray}
function unsafe_view(A::AbstractArray, I::Vararg{ViewIndex})
    HalfIntSubArray(A, I)
end

# When we take the view of a view, it's often possible to "reindex" the parent
# view's indices such that we can "pop" the parent view and keep just one layer
# of indirection. But we can't always do this because arrays of `CartesianIndex`
# might span multiple parent indices, making the reindex calculation very hard.
# So we use _maybe_reindex to figure out if there are any arrays of
# `CartesianIndex`, and if so, we punt and keep two layers of indirection.
unsafe_view(V::HalfIntSubArray, I::Vararg{ViewIndex,N}) where {N} =
    _maybe_reindex(V, I)
_maybe_reindex(V, I) = _maybe_reindex(V, I, I)
_maybe_reindex(V, I, ::Tuple{AbstractArray{<:Base.AbstractCartesianIndex}, Vararg{Any}}) =
    HalfIntSubArray(V, I)
# But allow arrays of CartesianIndex{1}; they behave just like arrays of Ints
_maybe_reindex(V, I, A::Tuple{AbstractArray{<:Base.AbstractCartesianIndex{1}}, Vararg{Any}}) =
    _maybe_reindex(V, I, Base.tail(A))
_maybe_reindex(V, I, A::Tuple{Any, Vararg{Any}}) = _maybe_reindex(V, I, Base.tail(A))
function _maybe_reindex(V, I, ::Tuple{})
    @inbounds idxs = to_indices(V.parent, reindex(V, I))
    HalfIntSubArray(V.parent, idxs)
end

# hash

function Base.hash(v::HalfIntSubArray, h::UInt)
	h = hash(v.parent, h)
	h = hash(axes(v), h)
	h = hash(v.offset1, h)
	h = hash(v.stride1, h)
	h = hash(:HalfIntSubArray, h)
	h
end
"""
    CartesianIndexHalfInt(i, j, k...) -> I
    CartesianIndexHalfInt((i, j, k...)) -> I

Create a multidimensional index `I`, which can be used for
indexing a multidimensional `AbstractHalfIntegerArray`.  
In particular, for an array `A`, the operation `A[I]` is
equivalent to `A[i,j,k...]`.  One can freely mix integer, half-integer and
`CartesianIndex` indices; for example, `A[Ipre, i, Ipost]` (where
`Ipre` and `Ipost` are `CartesianIndex` indices and `i` is an
integer or a half-integer) 
can be a useful expression when writing algorithms that
work along a single dimension of an array of arbitrary dimensionality.

# Examples
```jldoctest
julia> h = HalfIntArray(reshape(1:4, 2, 2), -1//2:1//2, -1//2:1//2)
2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> h[CartesianIndexHalfInt(-1//2, 1//2)]
3

julia> h[CartesianIndexHalfInt(-1//2), 1//2]
3
```
"""
struct CartesianIndexHalfInt{N} <: Base.AbstractCartesianIndex{N}
    parent :: CartesianIndex{N}
    offsets :: NTuple{N,HalfInt}
end

Base.parent(c::CartesianIndexHalfInt) = c.parent

CartesianIndexHalfInt(::Tuple{}) = CartesianIndexHalfInt{0}(())
function CartesianIndexHalfInt(index::NTuple{N,Int}, 
    offsets::NTuple{N,HalfInt} = Base.fill_to_length((),zero(HalfInt),Val(N))) where {N}

    CartesianIndexHalfInt{N}(index, offsets)
end
function CartesianIndexHalfInt{N}(index::NTuple{N,Int}, 
    offsets::NTuple{N,HalfInt} = Base.fill_to_length((),zero(HalfInt),Val(N))) where {N}

    CartesianIndexHalfInt{N}(CartesianIndex(index), offsets)
end
function CartesianIndexHalfInt(offsets::NTuple{N,HalfInt}) where {N}
    CartesianIndexHalfInt{N}(offsets)
end
function CartesianIndexHalfInt{N}(offsets::NTuple{N,HalfInt}) where {N}
    index = Base.fill_to_length((),0,Val(N))
    CartesianIndexHalfInt{N}(CartesianIndex(index), offsets)
end
# Specifying Ints populates the parent CaratesianIndex
CartesianIndexHalfInt() = CartesianIndexHalfInt{0}((),())
CartesianIndexHalfInt(index::Int...) = CartesianIndexHalfInt(index)
# Specifying HalfInts populates the offsets
CartesianIndexHalfInt(offsets::HalfInt...) = CartesianIndexHalfInt(offsets)
CartesianIndexHalfInt(offsets::Real...) = CartesianIndexHalfInt(HalfInt.(offsets))
# # Allow passing tuples smaller than N
CartesianIndexHalfInt{0}(index::Tuple{}) = CartesianIndexHalfInt{0}((),())
CartesianIndexHalfInt{N}(index::Tuple{}) where {N} = CartesianIndexHalfInt{N}(Base.fill_to_length(index, 1, Val(N)))
CartesianIndexHalfInt{N}(index::Tuple{Vararg{Int}}) where {N} = CartesianIndexHalfInt{N}(Base.fill_to_length(index, 1, Val(N)))
CartesianIndexHalfInt{N}(index::Integer...) where {N} = CartesianIndexHalfInt{N}(index)
CartesianIndexHalfInt{N}(offsets::Tuple{Vararg{HalfInt}}) where {N} = CartesianIndexHalfInt{N}(Base.fill_to_length(offsets, one(HalfInt), Val(N)))
CartesianIndexHalfInt{N}(offsets::Tuple{Vararg{Real}}) where {N} = CartesianIndexHalfInt{N}(HalfInt.(offsets))
CartesianIndexHalfInt{N}(offsets::HalfInt...) where {N} = CartesianIndexHalfInt{N}(offsets)
CartesianIndexHalfInt{N}(offsets::Real...) where {N} = CartesianIndexHalfInt{N}(HalfInt.(offsets))
CartesianIndexHalfInt{N}() where {N} = CartesianIndexHalfInt{N}(())
# Un-nest passed CartesianIndexes
CartesianIndexHalfInt(index::Union{Integer, HalfInteger, CartesianIndexHalfInt}...) = CartesianIndexHalfInt(index)
CartesianIndexHalfInt(index::Tuple{Vararg{Union{Integer, HalfInteger, CartesianIndexHalfInt}}}) = CartesianIndexHalfInt(flatten(index))

flatten(I::Tuple{}) = I
flatten(I::Tuple{Any}) = I
flatten(I::Tuple{HalfIntOrInt}) = (HalfInt(I[1]),)
flatten(I::Tuple{<:CartesianIndexHalfInt}) = Tuple(I[1])
@inline flatten(I) = _flatten(I...)
@inline _flatten() = ()
@inline _flatten(i, I...) = (HalfInt(i), _flatten(I...)...)
@inline _flatten(i::CartesianIndexHalfInt, I...) = (Tuple(i)..., _flatten(I...)...)

# length
Base.length(::CartesianIndexHalfInt{N}) where {N} = N

# indexing
Base.getindex(index::CartesianIndexHalfInt, i::Integer) = parent(index)[i] .+ index.offsets[i]
Base.eltype(::CartesianIndexHalfInt) = HalfInt

# access to index tuple
Base.Tuple(index::CartesianIndexHalfInt) = map(+,Tuple(parent(index)), index.offsets)

# equality
Base.:(==)(a::CartesianIndexHalfInt{N}, b::CartesianIndexHalfInt{N}) where N = Tuple(a) == Tuple(b)

# zeros and ones
Base.zero(::CartesianIndexHalfInt{N}) where {N} = zero(CartesianIndexHalfInt{N})
Base.zero(::Type{CartesianIndexHalfInt{N}}) where {N} = CartesianIndexHalfInt(ntuple(x -> 0, Val(N)))
Base.oneunit(::CartesianIndexHalfInt{N}) where {N} = oneunit(CartesianIndexHalfInt{N})
Base.oneunit(::Type{CartesianIndexHalfInt{N}}) where {N} = CartesianIndexHalfInt(ntuple(x -> 1, Val(N)))

# arithmetic, min/max
@inline Base.:(-)(index::CartesianIndexHalfInt{N}) where {N} =
    CartesianIndexHalfInt{N}(-parent(index), map(-,index.offsets))
@inline Base.:(+)(index1::CartesianIndexHalfInt{N}, index2::CartesianIndexHalfInt{N}) where {N} =
    CartesianIndexHalfInt{N}(parent(index1) + parent(index2), map(+, index1.offsets, index2.offsets))
@inline Base.:(-)(index1::CartesianIndexHalfInt{N}, index2::CartesianIndexHalfInt{N}) where {N} =
    CartesianIndexHalfInt{N}(parent(index1) - parent(index2), map(-, index1.offsets, index2.offsets))

for f in [:min, :max]
    @eval @inline function Base.$f(index1::CartesianIndexHalfInt{N}, index2::CartesianIndexHalfInt{N}) where {N}
        index = map($f, Tuple(parent(index1)), Tuple(parent(index2)))
        of = map($f, index1.offsets, index2.offsets)
        CartesianIndexHalfInt{N}(index, of)
    end
end

@inline Base.:(*)(a::Integer, index::CartesianIndexHalfInt{N}) where {N} = CartesianIndexHalfInt{N}(a*parent(index), map(x->a*x, index.offsets))
@inline Base.:(*)(index::CartesianIndexHalfInt, a::Integer) = *(a,index)

# comparison
@inline Base.isless(I1::CartesianIndexHalfInt{N}, I2::CartesianIndexHalfInt{N}) where {N} = _isless(0, Tuple(I1), Tuple(I2))
@inline function _isless(ret, I1::NTuple{N,HalfInt}, I2::NTuple{N,HalfInt}) where N
    newret = ifelse(ret==0, icmp(I1[N], I2[N]), ret)
    _isless(newret, Base.front(I1), Base.front(I2))
end
_isless(ret, ::Tuple{}, ::Tuple{}) = ifelse(ret==1, true, false)
icmp(a, b) = ifelse(isless(a,b), 1, ifelse(a==b, 0, -1))

# conversions
Base.convert(::Type{T}, index::CartesianIndexHalfInt{1}) where {T<:Number} = convert(T, index[1])
Base.convert(::Type{T}, index::CartesianIndexHalfInt) where {T<:Tuple} = convert(T, Tuple(index))

# hashing
const cartindexhash_seed = UInt == UInt64 ? 0xd60ca92f8284b8b0 : 0xf2ea7c2e
function Base.hash(ci::CartesianIndexHalfInt, h::UInt)
    h += cartindexhash_seed
    h = hash(ci.offsets, h)
    h = hash(parent(ci), h)
    h = hash(:CartesianIndexHalfInt, h)
    return h
end

Base.iterate(::CartesianIndexHalfInt) =
    error("iteration is deliberately unsupported for CartesianIndexHalfInt. Use `I` rather than `I...`, or use `Tuple(I)...`")

"""
    CartesianIndicesHalfInt((istart:istop, jstart:jstop, ...)) -> R

Define a region `R` spanning a multidimensional rectangular range
of integer indices. These are most commonly encountered in the
context of iteration, where `for I in R ... end` will return
[`CartesianIndexHalfInt`](@ref) indices `I` equivalent to the nested loops

    for j = jstart:jstop
        for i = istart:istop
            ...
        end
    end

Consequently these can be useful for writing algorithms that
work in arbitrary dimensions.

A `CartesianIndicesHalfInt` type is equivalent to a `CartesianIndices` type 
for integer ranges. The difference is that a `CartesianIndicesHalfInt` allows the 
ranges to be half-integer `AbstractUnitRange`s. They are therefore suitable for 
indexing into `AbstractHalfIntegerArray`s.

    CartesianIndicesHalfInt(A::AbstractArray) -> R

As a convenience, constructing a `CartesianIndicesHalfInt` from an array makes a
range of its indices.

# Examples
```jldoctest
julia> h = HalfIntArray(reshape(1:4, 2, 2), -1//2:1//2, -1//2:1//2)
2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> c = CartesianIndicesHalfInt(h)
2×2 CartesianIndicesHalfInt{2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}} with indices -1/2:1/2×-1/2:1/2:
 CartesianIndexHalfInt(-1/2, -1/2)  CartesianIndexHalfInt(-1/2, 1/2)
 CartesianIndexHalfInt(1/2, -1/2)   CartesianIndexHalfInt(1/2, 1/2)

julia> c[1/2,1/2]
CartesianIndexHalfInt(1/2, 1/2)
```
"""
struct CartesianIndicesHalfInt{N,R} <: AbstractHalfIntegerWrapper{CartesianIndexHalfInt{N},N}
    cartinds :: CartesianIndices{N,R}
    offsets :: NTuple{N,HalfInt}
end

Base.parent(iter::CartesianIndicesHalfInt) = CartesianIndicesHalfIntParent(iter.cartinds, iter.offsets)

struct CartesianIndicesHalfIntParent{N,R} <: AbstractArray{CartesianIndexHalfInt{N},N}
    cartinds :: CartesianIndices{N,R}
    offsets :: NTuple{N,HalfInt}
end

Base.parent(iter::CartesianIndicesHalfIntParent) = iter.cartinds

const CartesianIndicesHalfIntAny{N,R} = Union{CartesianIndicesHalfInt{N,R},CartesianIndicesHalfIntParent{N,R}}

_offsets(iter::Tuple{IdOffsetRange,Vararg{IdOffsetRange}}) = (iter[1].offset,_offsets(Base.tail(iter))...)
_offsets(iter::Tuple{IdOffsetRange}) = (iter[1].offset,)
function _offsets(iter::Tuple{AbstractUnitRange{HalfInt},Vararg{AbstractUnitRange{HalfInt}}})
    x = first(iter[1])
    (x - one(x), _offsets(Base.tail(iter))...)
end
_offsets(iter::Tuple{AbstractUnitRange{HalfInt}}) = (x = first(iter[1]); (x - one(x),))
_parents(iter::Tuple{IdOffsetRange,Vararg{IdOffsetRange}}) = (parent(iter[1]),_parents(Base.tail(iter))...)
_parents(iter::Tuple{IdOffsetRange}) = (parent(iter[1]),)
function _parents(iter::Tuple{AbstractUnitRange{HalfInt},Vararg{AbstractUnitRange{HalfInt}}})
    (Base.OneTo(length(iter[1])), _parents(Base.tail(iter))...)
end
_parents(iter::Tuple{AbstractUnitRange{HalfInt}}) = (Base.OneTo(length(iter[1])),)

CartesianIndicesHalfInt() = CartesianIndicesHalfInt(())
function CartesianIndicesHalfInt(::Tuple{})
    CartesianIndicesHalfInt(CartesianIndices(()), ())
end
function CartesianIndicesHalfInt(iter::NTuple{N,AbstractUnitRange{HalfInt}}) where {N}
    of = _offsets(iter)
    c = CartesianIndices(_parents(iter))
    CartesianIndicesHalfInt(c, of)
end
function CartesianIndicesHalfInt(iter::NTuple{N,AbstractUnitRange{Int}},
    of = ntuple(x->zero(HalfInt),Val(N))) where {N}
    CartesianIndicesHalfInt(CartesianIndices(iter), of)
end

# Mix of integer and half-integer axes, promote to half-integer
function CartesianIndicesHalfInt(iter::Tuple{Vararg{AbstractUnitRange}})
    iterIdOfR = map(IdOffsetRange, iter)
    CartesianIndicesHalfInt(iterIdOfR)
end

CartesianIndicesHalfInt(A::AbstractArray) = CartesianIndicesHalfInt(axes(A))

Base.IndexStyle(::Type{<:CartesianIndicesHalfIntAny{N,R}}) where {N,R} = IndexCartesian()

_convertIntOrHalfInt(t::NTuple{N,Int}) where {N} = t
_convertIntOrHalfInt(t::NTuple{N,Real}) where {N} = map(HalfInt, t)
_convertIntOrHalfInt(i::Int) = i
_convertIntOrHalfInt(i::Real) = HalfInt(i)

@propagate_inbounds function Base.getindex(iter::CartesianIndicesHalfInt{N}, I::Real...) where {N}
    @boundscheck checkbounds(iter, I...)
    J = trimtoN(I, Val(N))
    CartesianIndexHalfInt(_convertIntOrHalfInt(J))
end

@propagate_inbounds function Base.getindex(iter::CartesianIndicesHalfIntParent, I::Int...)
    @boundscheck checkbounds(iter, I...)
    CartesianIndexHalfInt(iter.cartinds[I...], iter.offsets)
end

# 1D arrays use Cartesian Indexing
for DT in [:Int, :Real]
    @eval @propagate_inbounds function Base.getindex(iter::CartesianIndicesHalfInt{1}, i::$DT)
        @boundscheck checkbounds(iter, i)
        CartesianIndexHalfInt(_convertIntOrHalfInt(i))
    end
end

# Linear Indexing
@propagate_inbounds function Base.getindex(A::CartesianIndicesHalfInt, i::Real)
    @boundscheck checkbounds(A, i)
    @boundscheck ensureInt(i)
    CartesianIndexHalfInt(A.cartinds[unsafeInt(i)], A.offsets)
end

Base.IteratorSize(::Type{<:CartesianIndicesHalfInt{N}}) where {N} = Base.HasShape{N}()

@inline function _iterate(iter, newstate)
    ind,_ = newstate
    cind = CartesianIndexHalfInt(ind, iter.offsets)
    cind, ind
end
@inline _iterate(iter, ::Nothing) = nothing

@inline function Base.iterate(iter::CartesianIndicesHalfIntAny)
    newstate = iterate(iter.cartinds)
    _iterate(iter, newstate)
end
@inline function Base.iterate(iter::CartesianIndicesHalfIntAny, state)
    newstate = iterate(iter.cartinds, state)
    _iterate(iter, newstate)
end

# 0-d cartesian ranges are special-cased to iterate once and only once
function Base.iterate(iter::CartesianIndicesHalfIntAny{0}, done=false)
    done ? nothing : (CartesianIndexHalfInt(), true)
end

@inline function Base.index_ndims(i1::CartesianIndexHalfInt, I...)
    (map(x->true, parent(i1).I)..., Base.index_ndims(I...)...)
end
@inline function Base.index_ndims(i1::AbstractArray{CartesianIndexHalfInt{N}}, I...) where N
    (ntuple(x->true, Val(N))..., Base.index_ndims(I...)...)
end

for f in [:size, :length]
    @eval @inline function Base.$f(iter::CartesianIndicesHalfIntAny)
        $f(iter.cartinds)
    end
end

Base.axes(c::CartesianIndicesHalfInt) = map(IdOffsetRange, axes(c.cartinds), c.offsets)
Base.axes(c::CartesianIndicesHalfInt, d) = d <= ndims(c) ? IdOffsetRange(axes(c.cartinds, d), c.offsets[d]) : IdOffsetRange(axes(c.cartinds, d))

for f in [:first, :last]
    @eval @inline function Base.$f(iter::CartesianIndicesHalfIntAny)
        CartesianIndexHalfInt($f(iter.cartinds), iter.offsets)
    end
end

@inline function Base.in(i::CartesianIndexHalfInt{N}, r::CartesianIndicesHalfInt{N}) where {N}
    _in(true, Tuple(i), Tuple(first(r)), Tuple(last(r)))
end
_in(b, ::Tuple{}, ::Tuple{}, ::Tuple{}) = b
@inline function _in(b, i, start, stop)
    _in(b & (start[1] <= i[1] <= stop[1]), 
        Base.tail(i), Base.tail(start), Base.tail(stop))
end

function Base.hash(c::CartesianIndicesHalfInt, h::UInt)
    h = hash(c.offsets, h)
    h = hash(c.cartinds, h)
    h = hash(:CartesianIndicesHalfInt, h)
    return h
end

@inline Base.to_indices(A, I::Tuple{Vararg{Union{HalfInteger, CartesianIndexHalfInt}}}) = to_indices(A, axes(A), I)
@inline Base.to_indices(A::AbstractHalfIntegerArrayOrWrapper, inds::Tuple{AbstractRange{Int},Vararg{Any}}, I::Tuple{CartesianIndexHalfInt, Vararg{Any}}) =
    to_indices(A, inds, (Tuple(first(I))..., Base.tail(I)...))
@inline Base.to_indices(A::AbstractHalfIntegerArrayOrWrapper, inds, I::Tuple{CartesianIndexHalfInt, Vararg{Any}}) =
    to_indices(A, inds, (Tuple(first(I))..., Base.tail(I)...))

# In general assume that arrays have integer indices
@inline Base.to_indices(A::AbstractArray, inds::Tuple{AbstractRange{Int},Vararg{Any}}, I::Tuple{CartesianIndexHalfInt, Vararg{Any}}) =
    to_indices(A, inds, (map(Int, Tuple(first(I)))..., Base.tail(I)...))

# But for arrays of CartesianIndex, we just skip the appropriate number of inds
@inline function Base.to_indices(A, inds, I::Tuple{AbstractArray{CartesianIndexHalfInt{N}}, Vararg{Any}}) where N
    _, indstail = Base.IteratorsMD.split(inds, Val(N))
    (Base.to_index(A, first(I)), to_indices(A, indstail, Base.tail(I))...)
end

function Base.checkbounds(::Type{Bool}, A::AbstractArray, i::Union{CartesianIndexHalfInt, AbstractArray{<:CartesianIndexHalfInt}})
    Base.checkbounds_indices(Bool, axes(A), (i,))
end

@inline Base.checkbounds_indices(::Type{Bool}, ::Tuple{}, I::Tuple{CartesianIndexHalfInt,Vararg{Any}}) =
    Base.checkbounds_indices(Bool, (), (Tuple(first(I))..., Base.tail(I)...))
@inline Base.checkbounds_indices(::Type{Bool}, IA::Tuple{Any}, I::Tuple{CartesianIndexHalfInt,Vararg{Any}}) =
    Base.checkbounds_indices(Bool, IA, (Tuple(first(I))..., Base.tail(I)...))
@inline Base.checkbounds_indices(::Type{Bool}, IA::Tuple, I::Tuple{CartesianIndexHalfInt,Vararg{Any}}) =
    Base.checkbounds_indices(Bool, IA, (Tuple(first(I))..., Base.tail(I)...))

# Support indexing with an array of CartesianIndex{N}s
# Here we try to consume N of the indices (if there are that many available)
# The first two simply handle ambiguities
@inline function Base.checkbounds_indices(::Type{Bool}, ::Tuple{},
        I::Tuple{AbstractArray{CartesianIndexHalfInt{N}},Vararg{Any}}) where N
    checkindex(Bool, (), I[1]) & Base.checkbounds_indices(Bool, (), Base.tail(I))
end
@inline function Base.checkbounds_indices(::Type{Bool}, IA::Tuple{Any},
        I::Tuple{AbstractArray{CartesianIndexHalfInt{0}},Vararg{Any}})
    Base.checkbounds_indices(Bool, IA, Base.tail(I))
end
@inline function Base.checkbounds_indices(::Type{Bool}, IA::Tuple{Any},
        I::Tuple{AbstractArray{CartesianIndexHalfInt{N}},Vararg{Any}}) where N
    checkindex(Bool, IA, I[1]) & Base.checkbounds_indices(Bool, (), Base.tail(I))
end
@inline function Base.checkbounds_indices(::Type{Bool}, IA::Tuple,
        I::Tuple{AbstractArray{CartesianIndexHalfInt{N}},Vararg{Any}}) where N
    
    IA1, IArest = Base.IteratorsMD.split(IA, Val(N))
    checkindex(Bool, IA1, I[1]) & Base.checkbounds_indices(Bool, IArest, Base.tail(I))
end

function Base.checkindex(::Type{Bool}, inds::Tuple, I::AbstractArray{<:CartesianIndexHalfInt})
    b = true
    for i in I
        b &= Base.checkbounds_indices(Bool, inds, (i,))
    end
    b
end

# It might be possible to construct a LinearIndices range for integer indices
function Base.CartesianIndices(A::AbstractHalfIntegerArrayOrWrapper)
    CartesianIndices(IdentityUnitRange.(UnitRange{Int}.(axes(A))))
end

"""
    LinearIndicesHalfInt(A::AbstractArray)

Return a `LinearIndicesHalfInt` array with the same shape and `axes` as `A`,
holding the linear index of each entry in `A`. Indexing this array with
cartesian indices allows mapping them to linear indices.

For arrays with conventional indexing (indices start at 1), or any multidimensional
array, linear indices range from 1 to `length(A)`. However, for `AbstractVector`s
linear indices are `axes(A, 1)`, and therefore do not start at 1 for vectors with
unconventional indexing.

Calling this function is the "safe" way to write algorithms that
exploit linear indexing.

`LinearIndicesHalfInt` are equivalent to `LinearIndices` for arrays with integer axes,
however `LinearIndicesHalfInt` also support arrays with half-integer axes. They are 
therefore the suitable for working with `AbstractHalfIntegerArray`s.

# Examples
```jldoctest
julia> h = HalfIntArray(reshape(1:4, 2, 2), -1//2:1//2, -1//2:1//2)
2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> linds = LinearIndicesHalfInt(h)
2×2 LinearIndicesHalfInt{2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}} with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> linds[1/2, 1/2]
4

julia> linds[CartesianIndexHalfInt(1/2, 1/2)]
4

julia> v = HalfIntArray([1,2,3], -1:1)
3-element HalfIntArray(::Array{Int64,1}, -1:1) with eltype Int64 with indices -1:1:
 1
 2
 3

julia> lindsv = LinearIndicesHalfInt(v)
3-element LinearIndicesHalfInt{1,Tuple{Base.OneTo{Int64}}} with indices -1:1:
 -1
  0
  1

julia> lindsv[0]
0
```
"""
struct LinearIndicesHalfInt{N,R<:NTuple{N,AbstractUnitRange{Int}}} <: AbstractHalfIntegerWrapper{HalfInt,N}
    lininds :: LinearIndices{N,R}
    offsets :: NTuple{N,HalfInt}
end

Base.parent(l::LinearIndicesHalfInt) = LinearIndicesHalfIntParent(l.lininds, l.offsets)

struct LinearIndicesHalfIntParent{N,R<:NTuple{N,AbstractUnitRange{Int}}} <: AbstractArray{HalfInt,N}
    lininds :: LinearIndices{N,R}
    offsets :: NTuple{N,HalfInt}
end

Base.parent(l::LinearIndicesHalfIntParent) = l.lininds

function LinearIndicesHalfInt(iter::LinearIndices{N,R},
    offsets = first.(iter.indices) .- 1) where {N,R}

    LinearIndicesHalfInt{N,R}(iter, offsets)
end

function LinearIndicesHalfInt(inds::NTuple{N,AbstractUnitRange{Int}},
    offsets = Base.fill_to_length((),zero(HalfInt),Val(N))) where {N}

    LinearIndicesHalfInt(LinearIndices(inds), offsets)
end

function LinearIndicesHalfInt(inds::Tuple{Vararg{IdOffsetRange}})
    LinearIndicesHalfInt(parent.(inds), offset.(inds))
end

# Mix of integer and half-integer axes, promote to half-integer
function LinearIndicesHalfInt(iter::Tuple{Vararg{AbstractUnitRange}})
    iterIdOfR = map(IdOffsetRange, iter)
    LinearIndicesHalfInt(iterIdOfR)
end

LinearIndicesHalfInt(::Tuple{}) = LinearIndicesHalfInt{0,Tuple{}}(LinearIndices(()),())

LinearIndicesHalfInt(A::AbstractArray) = LinearIndicesHalfInt(axes(A))

# AbstractArray implementation
Base.IndexStyle(::Type{<:LinearIndicesHalfInt}) = IndexLinear()
Base.axes(iter::LinearIndicesHalfInt) = map(IdOffsetRange,axes(iter.lininds), iter.offsets)
Base.axes(iter::LinearIndicesHalfInt, d) = d <= ndims(iter) ? IdOffsetRange(axes(iter.lininds,d), iter.offsets[d]) : IdOffsetRange(axes(iter.lininds, d))
for f in [:length, :size]
    @eval function Base.$f(iter::Union{LinearIndicesHalfInt, LinearIndicesHalfIntParent})
        $f(iter.lininds)
    end
end

# Linear indexing
function Base.getindex(iter::LinearIndicesHalfInt, i::Real)
    @boundscheck checkbounds(iter, i)
    ensureInt(i)
    unsafeInt(i)
end

# Cartesian indexing
@propagate_inbounds function Base.getindex(iter::LinearIndicesHalfInt, I::Real...)
    @boundscheck checkbounds(iter, I...)
    J = to_parentindices(axes(iter),I)
    iter.lininds[J...]
end

function Base.getindex(iter::LinearIndicesHalfIntParent, i::Int)
    @boundscheck checkbounds(iter, i)
    i 
end
@propagate_inbounds function Base.getindex(iter::LinearIndicesHalfIntParent, I::Int...)
    @boundscheck checkbounds(iter, I...)
    iter.lininds[I...]
end

@inline function Base.getindex(iter::LinearIndicesHalfInt{1}, i::Real)
    @boundscheck checkbounds(iter, i)
    i
end
@inline function Base.getindex(iter::LinearIndicesHalfInt{1}, i::Real, I::Real...)
    @boundscheck checkbounds(iter, i, I...)
    @inbounds iter[i]
end
@inline function Base.getindex(iter::LinearIndicesHalfIntParent{1}, i::Int)
    @boundscheck checkbounds(iter, i)
    @inbounds i + iter.offsets[1]
end
@inline function Base.getindex(iter::LinearIndicesHalfIntParent{1}, i::Int, I::Int...)
    @boundscheck checkbounds(iter, i, I...)
    @inbounds iter[i]
end

@propagate_inbounds function Base.getindex(iter::LinearIndicesHalfInt, i::AbstractRange{<:HalfIntegerOrInteger})
    @boundscheck checkbounds(iter, i)
    (first(iter):last(iter))[i]
end
@propagate_inbounds function Base.getindex(iter::LinearIndicesHalfInt{1}, i::AbstractRange{<:HalfIntegerOrInteger})
    @boundscheck checkbounds(iter, i)
    j = unsafeUnitRangeInt(i .- iter.offsets[1])
    IdOffsetRange(iter.lininds.indices[1][j], iter.offsets[1])
end
@propagate_inbounds function Base.getindex(iter::LinearIndicesHalfIntParent{1}, i::AbstractRange{<:HalfIntegerOrInteger})
    @boundscheck checkbounds(iter, i)
    iter.lininds.indices[1][i] .+ iter.offsets[1]
end

Base.iterate(iter::LinearIndicesHalfInt{1}, s...) = iterate(Base.axes1(iter), s...)
Base.iterate(iter::LinearIndicesHalfInt, i=1) = i > length(iter) ? nothing : (i, i+1)

# Needed since firstindex and lastindex are defined in terms of LinearIndicesHalfInt
Base.first(iter::LinearIndicesHalfInt) = 1
Base.first(iter::LinearIndicesHalfInt{1}) = first(iter.lininds.indices[1]) + iter.offsets[1]
Base.last(iter::LinearIndicesHalfInt) = length(iter)
Base.last(iter::LinearIndicesHalfInt{1}) = last(iter.lininds.indices[1]) + iter.offsets[1]

function Base.hash(c::LinearIndicesHalfInt, h::UInt)
    h = hash(c.offsets, h)
    h = hash(c.lininds, h)
    h = hash(:LinearIndicesHalfInt, h)
    return h
end

# It might be possible to construct a LinearIndices range for integer indices
function Base.LinearIndices(A::AbstractHalfIntegerArrayOrWrapper)
    LinearIndices(IdentityUnitRange.(UnitRange{Int}.(axes(A))))
end
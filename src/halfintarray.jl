"""
    HalfIntArray(A::AbstractArray, indices::AbstractUnitRange...)

Return a `HalfIntArray` that shares that shares element type and size with the first argument, 
but used the given `indices`, which are checked for compatible size.

The `indices` may be `Integer` or `HalfInteger` ranges with a unit step. 
`Rational` ranges over integer or half-integer values may also be provided.

# Examples
```jldoctest
julia> h = HalfIntArray(reshape(1:9,3,3), -1:1, -1:1)
3×3 HalfIntArray(reshape(::UnitRange{Int64}, 3, 3), -1:1, -1:1) with eltype Int64 with indices -1:1×-1:1:
 1  4  7
 2  5  8
 3  6  9

julia> h[-1, -1]
1

julia> h = HalfIntArray(reshape(1:4,2,2), -1//2:1//2, -1//2:1//2)
2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> h[1//2, 1//2]
4
```
"""
struct HalfIntArray{T,N,A<:AbstractArray} <: AbstractHalfIntegerWrapper{T,N}
    parent :: A
    offsets :: NTuple{N,HalfInt}
end

function checknonnegativemomentum(j)
    j >= zero(j) || 
    throw(ArgumentError("Invalid angular momentum j=$j, must be ≥ 0"))
end

"""
    SpinMatrix(A::AbstractMatrix, [j::Real])

Return a `SpinMatrix` that allows the underlying matrix `A` to be indexed using 
integers or half-integers, depending on the value of the angular momentum `j`. 
The value of `j` needs to be either an `Integer` a half-integral `Real` number.
If it is not provided it is automatically inferred from the size of the array `A`.
The parent matrix `A` needs to use `1`-based indexing, and have a size of `(2j+1, 2j+1)`.

The axes of the `SpinMatrix` for an angular momentum `j` will necessarily be `(-j:j, -j:j)`.
 
# Examples
```jldoctest
julia> SpinMatrix(reshape(1:4,2,2))
2×2 SpinMatrix(reshape(::UnitRange{Int64}, 2, 2), 1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> SpinMatrix(zeros(ComplexF64,3,3), 1)
3×3 SpinMatrix(::Array{Complex{Float64},2}, 1) with eltype Complex{Float64} with indices -1:1×-1:1:
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im
```

See also: [`HalfIntArray`](@ref)
"""
struct SpinMatrix{T,A<:AbstractMatrix} <: AbstractHalfIntegerWrapper{T,2}
    parent :: A
    j :: HalfInt

    function SpinMatrix{T,A}(arr::A, j) where {T,A<:AbstractMatrix}
        checknonnegativemomentum(j)
        !Base.has_offset_axes(arr) || throw(ArgumentError("the parent array must have axes starting from 1"))
        N = twice(j) + 1
        size(arr) == (N,N) || 
        throw(ArgumentError("size $(size(arr)) of the parent array is not compatible with the angular momentum $j"))

        new{T,A}(arr,j)
    end
end

const HalfIntVector{T,A<:AbstractArray} = HalfIntArray{T,1,A}
const HalfIntMatrix{T,A<:AbstractArray} = HalfIntArray{T,2,A}
const HIAorSM = Union{HalfIntArray,SpinMatrix}

# unwrap

unwraphalfint(a::AbstractArray) = unwraphalfint(a, parent(a))
unwraphalfint(a, b) = unwraphalfint(b, parent(b))
unwraphalfint(a::AbstractHalfIntegerArray, b) = b

offset(axparent::AbstractUnitRange, ax::AbstractUnitRange) = HalfInt(first(ax) - first(axparent))
offset(axparent::AbstractUnitRange, ax::Integer) = HalfInt(one(first(axparent)) - first(axparent))

function HalfIntArray(arr::AbstractArray{T,N}, offsets::NTuple{N,HalfInt}) where {T,N}
    HalfIntArray{T,N,typeof(arr)}(arr, offsets)
end
HalfIntArray(A::AbstractArray{T,0}, offsets::Tuple{}) where T =
    HalfIntArray{T,0,typeof(A)}(A, ())

HalfIntArray(A::AbstractArray{T,N}, offsets::Vararg{HalfInt,N}) where {T,N} =
    HalfIntArray(A, offsets)
HalfIntArray(A::AbstractArray{T,0}) where {T} = HalfIntArray(A, ())

const ArrayInitializer = Union{UndefInitializer, Missing, Nothing}

"""
    HalfIntArray{T}(init, indices::AbstractUnitRange...)

Return a `HalfIntArray` having elements of type `T` and axes as specified by `indices`. 
The initializer `init` may be one of `undef`, `missing` or `nothing`.

# Examples
```jldoctest
julia> HalfIntArray{Union{Int,Missing}}(undef, 0:1, 0:1)
2×2 HalfIntArray(::Array{Union{Missing, Int64},2}, 0:1, 0:1) with eltype Union{Missing, Int64} with indices 0:1×0:1:
 missing  missing
 missing  missing
```
"""
HalfIntArray{T,N}(init::ArrayInitializer, inds::Indices{N}) where {T,N} =
    HalfIntArray(Array{T,N}(init, map(indexlength, inds)), map(indexoffset, inds))
HalfIntArray{T}(init::ArrayInitializer, inds::Indices{N}) where {T,N} = HalfIntArray{T,N}(init, inds)
HalfIntArray{T,N}(init::ArrayInitializer, inds::Vararg{AbstractUnitRange,N}) where {T,N} = HalfIntArray{T,N}(init, inds)
HalfIntArray{T}(init::ArrayInitializer, inds::Vararg{AbstractUnitRange,N}) where {T,N} = HalfIntArray{T,N}(init, inds)

function HalfIntArray(A::AbstractArray{T,N}, inds::NTuple{N,AbstractUnitRange}) where {T,N}
    axparent = axes(A)
    lA = map(length, axparent)
    lI = map(length, inds)
    lA == lI || throw(DimensionMismatch("supplied axes do not agree with the size of the array (got size $lA for the array and $lI for the indices"))
    HalfIntArray(A, map(offset, axparent, inds))
end
HalfIntArray(A::AbstractArray{T,N}, inds::Vararg{AbstractUnitRange,N}) where {T,N} =
    HalfIntArray(A, inds)

# avoid a level of indirection when nesting HalfIntArrays
function HalfIntArray(A::HalfIntArray, offsets::NTuple{N,HalfInt}) where {N}
    HalfIntArray(parent(A), offsets .+ A.offsets)
end
function HalfIntArray(A::HalfIntArray, offsets::NTuple{N,HalfIntOrInt}) where {N}
    HalfIntArray(parent(A), offsets .+ A.offsets)
end
HalfIntArray(A::HalfIntArray{T,0}, inds::Tuple{}) where {T} = HalfIntArray(parent(A), ())

function SpinMatrix(A::AbstractMatrix{T}, j::Real) where {T}
    jh = HalfInt(j)
    SpinMatrix{T,typeof(A)}(A, jh)
end

"""
    SpinMatrix{T}(init, j)

Create a `SpinMatrix` of type `T` for the angular momentum `j`. 
An underlying `Matrix` of size `(2j+1, 2j+1)` is allocated in the process, with values 
set by the initializer `init` that may be one of `undef`, `missing` or `nothing`.

# Examples
```jldoctest
julia> SpinMatrix{Union{Int,Missing}}(undef, 1//2)
2×2 SpinMatrix(::Array{Union{Missing, Int64},2}, 1/2) with eltype Union{Missing, Int64} with indices -1/2:1/2×-1/2:1/2:
 missing  missing
 missing  missing
```
"""
function SpinMatrix{T}(init::ArrayInitializer, j::Real) where {T}
    checknonnegativemomentum(j)
    jh = HalfInt(j)
    N = twice(jh) + 1
    A = Array{T}(init, N, N)
    SpinMatrix(A, jh)
end

function SpinMatrix(h::AbstractMatrix)
    s = size(h)
    s[1] == s[2] || 
        throw(ArgumentError("only a square matrix may be converted to a SpinMatrix"))
    j = half(s[1] - one(s[1]))
    SpinMatrix(h,j)
end

SpinMatrix(h::HalfIntArray) = SpinMatrix(parent(h))
SpinMatrix(h::SpinMatrix) = h

# Equality for SpinMatrix
Base.:(==)(a::SpinMatrix, b::SpinMatrix) = a.j == b.j && parent(a) == parent(b)

function Base.hash(A::HalfIntArray, h::UInt)
    h = hash(A.offsets, h)
    h = hash(parent(A), h)
    h = hash(:HalfIntArray, h)
    return h
end

function Base.hash(A::SpinMatrix, h::UInt)
    h = hash(A.j, h)
    h = hash(parent(A), h)
    h = hash(:SpinMatrix, h)
    return h
end

parenttype(::Type{HalfIntArray{T,N,AA}}) where {T,N,AA} = AA
parenttype(::Type{SpinMatrix{T,AA}}) where {T,AA} = AA
Base.IndexStyle(::Type{H}) where {H<:Union{HalfIntArray,SpinMatrix}} = IndexStyle(parenttype(H))

Base.parent(A::HIAorSM) = A.parent

Base.size(A::HalfIntArray) = size(parent(A))
Base.size(A::SpinMatrix) = (twice(A.j) + 1, twice(A.j) + 1)

Base.axes(A::HalfIntArray) = map(IdOffsetRange, axes(parent(A)), A.offsets)
Base.axes(A::HalfIntArray, d) = d <= ndims(A) ? IdOffsetRange(axes(parent(A), d), A.offsets[d]) : IdOffsetRange(axes(parent(A), d))

function Base.axes(A::SpinMatrix)
    ax = IdOffsetRange(Base.OneTo(twice(A.j) + 1), -A.j-one(A.j))
    (ax, ax)
end
function Base.axes(A::SpinMatrix, d)
    ax = IdOffsetRange(Base.OneTo(twice(A.j) + 1), -A.j-one(A.j))
    d <= 2 ? ax : IdOffsetRange(axes(parent(A),d))
end

## Indexing

# Note this gets the index of the parent *array*, not the index of the parent *range*
# Assuming that the parent has 1-based indexing
function parentindex(r::IdOffsetRange, i::HalfInt)
    unsafeInt(i - r.offset)
end
parentindex(r::IdOffsetRange, i::Real) = parentindex(r, HalfInt(i))
parentindex(r::AbstractUnitRange{Int}, i::Int) = i - first(r) + 1
parentindex(r::AbstractUnitRange{Int}, i::Real) = parentindex(r, Int(i))
function parentindex(r::AbstractUnitRange, I::AbstractUnitRange)
    parentindex(r, first(I)):parentindex(r, last(I))
end
function parentindex(::AbstractUnitRange, I::Base.Slice{<:IdOffsetRange})
    Base.Slice(parent(I.indices))
end
function parentindex(r::AbstractUnitRange, I::AbstractRange)
    parentindex(r, first(I)):Int(step(I)):parentindex(r, last(I))
end

function to_parentindices(ax::Tuple, inds::Tuple)
    (parentindex(first(ax),first(inds)),
        to_parentindices(Base.tail(ax),Base.tail(inds))...)
end
# drop extra indices
to_parentindices(ax::Tuple{}, inds::Tuple) = ()
# zero-dim arrays
to_parentindices(ax::Tuple{}, inds::Tuple{}) = ()
# append ones if fewer indices are specified
function to_parentindices(ax::NTuple{N,Any}, inds::Tuple{}) where {N}
    Base.fill_to_length((),1,Val(N))
end

# Cartesian Indexing for nD arrays
@propagate_inbounds function Base.getindex(A::HIAorSM, I::Real...)
    @boundscheck checkbounds(A, I...)
    J = to_parentindices(axes(A), I)
    parent(A)[J...]
end

for DT in [:Real, :Int]
    # 1D arrays always use Cartesian Indexing
    @eval @propagate_inbounds function Base.getindex(A::HalfIntArray{<:Any,1}, i::$DT)
        @boundscheck checkbounds(A, i)
        J = parentindex(axes(A,1), i)
        parent(A)[J]
    end
end

# Indexing with a single index forces linear indexing
# for arrays with >1 dimensions
@propagate_inbounds function Base.getindex(A::HIAorSM, i::Real)
    ensureInt(i)
    parent(A)[unsafeInt(i)]
end

@propagate_inbounds function Base.setindex!(A::HIAorSM, val, I::Real...)
    @boundscheck checkbounds(A, I...)
    J = to_parentindices(axes(A), I)
    parent(A)[J...] = val
    A
end

for DT in [:Real, :Int]
    # 1D arrays use Cartesian indexing
    @eval @propagate_inbounds function Base.setindex!(A::HalfIntArray{<:Any,1}, val, i::$DT)
        @boundscheck checkbounds(A, i)
        J = parentindex(Base.axes1(A), i)
        parent(A)[J] = val
        A
    end

end
# Indexing with a single Int/HalfInt forces linear indexing
# for arrays with >1 dimensions
@propagate_inbounds function Base.setindex!(A::HIAorSM, val, i::Real)
    ensureInt(i)
    parent(A)[unsafeInt(i)] = val
    A
end

### Some mutating functions defined only for HalfIntVector ###

Base.resize!(A::HalfIntVector, nl::Integer) = (resize!(A.parent, nl); A)
Base.push!(A::HalfIntVector, x...) = (push!(A.parent, x...); A)
Base.pop!(A::HalfIntVector) = pop!(A.parent)
Base.empty!(A::HalfIntVector) = (empty!(A.parent); A)

### Low-level utilities ###

indexoffset(r::AbstractRange) = HalfInt(first(r) - one(eltype(r)))
indexoffset(i::Integer) = zero(HalfInt)
indexoffset(i::Colon) = zero(HalfInt)
indexlength(r::AbstractRange) = length(r)
indexlength(i::Integer) = i
indexlength(i::Colon) = Colon()
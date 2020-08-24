module HalfIntegerArrays

using HalfIntegers
import HalfIntegers: HalfIntegerOrInteger
const HalfIntOrInt = Union{HalfInt,Int}
using LinearAlgebra
using Base: Indices, @propagate_inbounds
@static if !isdefined(Base, :IdentityUnitRange)
    const IdentityUnitRange = Base.Slice
else
    using Base: IdentityUnitRange
end
using OffsetArrays
import Base: throw_boundserror

export HalfIntArray
export SpinMatrix
export CartesianIndicesHalfInt
export CartesianIndexHalfInt
export LinearIndicesHalfInt

"""
	AbstractHalfIntegerArray{T,N}

Supertype for `N`-dimensional arrays (or array-like types) with elements of type `T`. 
Such arrays may be indexed using integers as well as half-integers.
"""
abstract type AbstractHalfIntegerArray{T,N} <: AbstractArray{T,N} end

"""
    AbstractHalfIntegerWrapper{T,N} <: AbstractHalfIntegerArray{T,N}

Supertype for `N`-dimensional arrays (or array-like types) with elements of type `T` 
that wrap a parent array that has integer indices. The wrapper acts as a map between the 
indices of the array and those of the parent.

Subtypes of `AbstractHalfIntegerWrapper` must define a method for `Base.parent` that returns 
the underlying array.

[`HalfIntArray`](@ref), [`SpinMatrix`](@ref) and other types are subtypes of this.
"""
abstract type AbstractHalfIntegerWrapper{T,N} <: AbstractHalfIntegerArray{T,N} end

const AbstractHalfIntegerVector{T} = AbstractHalfIntegerArray{T,1}
const AbstractHalfIntegerMatrix{T} = AbstractHalfIntegerArray{T,2}
const AbstractHalfIntegerVecOrMat{T} = Union{AbstractHalfIntegerVector{T},
	AbstractHalfIntegerMatrix{T}}
const AdjOrTransAbsHalfIntVecOrMat{T} = LinearAlgebra.AdjOrTrans{T,<:AbstractHalfIntegerVecOrMat}
const AdjOrTransAbsHalfIntMatrix{T} = LinearAlgebra.AdjOrTrans{T,<:AbstractHalfIntegerMatrix}
const AdjOrTransAbsHalfIntVector{T} = LinearAlgebra.AdjOrTrans{T,<:AbstractHalfIntegerVector}

#= This type is useful to extend Base functions for wrapper types
around AbstractHalfIntegerArrays that are defined in Base or stdlib.
This is because Base functions often rely on CartesianIndices and LinearIndices,
whereas this package uses CartesianIndicesHalfInt and LinearIndicesHalfInt
=# 
const AbstractHalfIntegerArrayOrWrapper = Union{AbstractHalfIntegerArray,
	AdjOrTransAbsHalfIntVecOrMat}

unsafeInt(d) = floor(Int, d)

include("axes.jl")
include("lincartindexing.jl")
include("halfintarray.jl")
include("linearalgebra.jl")
include("broadcast.jl")
include("subarray.jl")
include("arrayshow.jl")

# Define functions common to the supertype

Base.keys(h::AbstractHalfIntegerArrayOrWrapper) = CartesianIndicesHalfInt(axes(h))
Base.keys(h::AbstractHalfIntegerArray{<:Any,1}) = LinearIndicesHalfInt(h)
Base.keys(r::IdOffsetRange) = LinearIndicesHalfInt(r)

Base.axes1(A::AbstractHalfIntegerArray{<:Any,0}) = Base.OneTo(1)

_maybetail(::Tuple{}) = ()
_maybetail(t::Tuple) = Base.tail(t)

throw_lininderr() = throw(ArgumentError("linear indexing requires an integer index"))
ensureInt(i) = isinteger(i) || throw_lininderr()
ensureInt(::Int) = true

parenttype(A::AbstractHalfIntegerArray) = parenttype(typeof(A))

function trimtoN(I::Tuple, ::Val{N}) where {N}
    J,_ = Base.IteratorsMD.split(I,Val(N))
    Base.fill_to_length(J,one(Int),Val(N))
end

# Convert Real indices to HalfInt
function Base.to_indices(A::AbstractHalfIntegerArrayOrWrapper, inds, I::Tuple{Real, Vararg{Any}})
    (HalfInt(first(I)), to_indices(A, _maybetail(inds), Base.tail(I))...)
end
# Conversely convert HalfInt indices to Integers if the axis has integer indices
# This should really only happen if the index is already an integer in value
function Base.to_indices(A::AbstractHalfIntegerArrayOrWrapper, inds::Tuple{AbstractUnitRange{Int},Vararg{Any}}, I::Tuple{Real, Vararg{Any}})
    (Int(first(I)), to_indices(A, Base.tail(inds), Base.tail(I))...)
end

# zero dim arrays may be indexed with ones
function Base.to_indices(A::AbstractHalfIntegerArrayOrWrapper, inds::Tuple{}, I::Tuple{Real, Vararg{Any}})
    (Int(first(I)), to_indices(A, (), Base.tail(I))...)
end

# AbstractRanges may be handled by to_index
Base.to_index(::AbstractHalfIntegerArrayOrWrapper, a::UnitRange{Int}) = a
Base.to_index(::AbstractHalfIntegerArrayOrWrapper, a::UnitRange{<:Real}) = UnitRange{HalfInt}(a)

for DT in [:Int, :Real]
    @eval Base.error_if_canonical_getindex(::IndexLinear, A::AbstractHalfIntegerArray, ::$DT) =
        error("getindex not defined for ", typeof(A))
    @eval Base.error_if_canonical_getindex(::IndexCartesian, A::AbstractHalfIntegerArray{T,N}, ::Vararg{$DT,N}) where {T,N} =
        error("getindex not defined for ", typeof(A))

    @eval Base.error_if_canonical_setindex(::IndexLinear, A::AbstractHalfIntegerArray, ::$DT) =
        error("setindex! not defined for ", typeof(A))
    @eval Base.error_if_canonical_setindex(::IndexCartesian, A::AbstractHalfIntegerArray{T,N}, ::Vararg{$DT,N}) where {T,N} =
        error("setindex! not defined for ", typeof(A))
end

@propagate_inbounds function Base.getindex(A::AbstractHalfIntegerArrayOrWrapper, 
    I::Union{Real,CartesianIndex,CartesianIndexHalfInt}...)
    
    Base.error_if_canonical_getindex(IndexStyle(A), A, I...)
    J = trimtoN(to_indices(A, I), Val(ndims(A)))
    A[J...]
end

@propagate_inbounds function Base.setindex!(A::AbstractHalfIntegerArrayOrWrapper, v, 
    I::Union{Real, CartesianIndex, CartesianIndexHalfInt}...)

    Base.error_if_canonical_setindex(IndexStyle(A), A, I...)
    J = trimtoN(to_indices(A, I), Val(ndims(A)))
    A[J...] = v
    A
end

#=
indicescompatible is meant to be called after checkbounds. This ensures that arrays
with half-integer axes are indexed with half-integers, and arrays with integer axes are
indexed with integers.
=#
indicescompatible(::Tuple{}, ::Tuple{Real, Vararg{Any}}) = true
indicescompatible(::Tuple{HalfInt}, ::Tuple{}) = true
indicescompatible(::Tuple{}, ::Tuple{}) = true
function indicescompatible(offsets::Tuple{HalfInt,Vararg{Any}}, I::Tuple{Real, Vararg{Any}})
    isinteger(first(offsets) + first(I)) && indicescompatible(Base.tail(offsets), Base.tail(I))
end
throw_indicesincompatible() = throw(ArgumentError("Indices provided are incompatible with the array's axes"))

# Linear indexing
for DT in [:Integer, :Real]
    @eval function Base.isassigned(a::AbstractHalfIntegerArrayOrWrapper, i::$DT...)
        try
            a[i...]
            true
        catch e
            if isa(e, BoundsError) || isa(e, UndefRefError) || isa(e,InexactError)
                return false
            else
                rethrow()
            end
        end
    end
end

Base.pairs(::IndexLinear, A::AbstractHalfIntegerArrayOrWrapper) = Iterators.Pairs(A, LinearIndicesHalfInt(A))
Base.pairs(::IndexCartesian, A::AbstractHalfIntegerArrayOrWrapper) = Iterators.Pairs(A, CartesianIndicesHalfInt(axes(A)))

Base.eachindex(::IndexCartesian, A::AbstractHalfIntegerArrayOrWrapper) = CartesianIndicesHalfInt(axes(A))

function _all_match_first_shadow(f::F, inds, A, B...) where F<:Function
    (inds == f(A)) & _all_match_first_shadow(f, inds, B...)
end
_all_match_first_shadow(f::F, inds) where F<:Function = true

@inline function Base.eachindex(::IndexCartesian, A::AbstractHalfIntegerArrayOrWrapper, B::AbstractHalfIntegerArrayOrWrapper...)
    axsA = axes(A)
    _all_match_first_shadow(axes, axsA, B...) || Base.throw_eachindex_mismatch(IndexCartesian(), A, B...)
    CartesianIndicesHalfInt(axsA)
end

function Base.collect(h::AbstractHalfIntegerArrayOrWrapper)
    b = Array{eltype(h)}(undef,size(h))
    @inbounds for (bi,hi) in zip(eachindex(b), eachindex(h))
        b[bi] = h[hi]
    end
    b
end
Base.Array{T,N}(h::AbstractHalfIntegerArray{T,N}) where {T,N} = collect(h)

# Similar and reshape

Base.similar(A::AbstractHalfIntegerArrayOrWrapper, ::Type{T}, dims::Dims) where T =
    similar(unwraphalfint(A), T, dims)

# Need to import OffsetAxis to get around the type-piracy in OffsetArrays
# See https://github.com/JuliaArrays/OffsetArrays.jl/issues/87#issuecomment-581391453
import OffsetArrays: OffsetAxis, OffsetAxisKnownLength
# This lets us avoid ambiguities with Base
const ReshapedAxis = Union{Integer,Base.OneTo}
const ReshapedAxisOneTo = Union{ReshapedAxis, OneTo}
# Axis types internal to this package
const HalfIntAxis = Union{IdOffsetRange,OneTo}
const HalfIntOffsetAxisKnownLength = Union{OffsetAxisKnownLength, HalfIntAxis}
const HalfIntOffsetAxis = Union{OffsetAxis, HalfIntAxis}

# Similar with non-offset axes returns an Array
# OneTo is treated as a non-offset axis
for ax in [:ReshapedAxis, :OneTo, :ReshapedAxisOneTo]
	@eval function Base.similar(A::AbstractHalfIntegerArrayOrWrapper, ::Type{T}, inds::Tuple{$ax,Vararg{$ax}}) where T
	    similar(parent(A), T, map(indexlength,inds))
	end
end

# Similar with offset axes returns a HalfIntArray
for ax in [:HalfIntOffsetAxisKnownLength, :OffsetAxisKnownLength]
    @eval function Base.similar(A::AbstractHalfIntegerArrayOrWrapper, ::Type{T}, inds::Tuple{$ax,Vararg{$ax}}) where T
        B = similar(parent(A), T, map(indexlength, inds))
        return HalfIntArray(B, map(offset, axes(B), inds))
    end
end

for DT in [:OneTo, :ReshapedAxis, :HalfIntOffsetAxisKnownLength, :OffsetAxisKnownLength]
	@eval function Base.similar(A::AbstractHalfIntegerArrayOrWrapper, inds::Tuple{$DT,Vararg{$DT}})
	    similar(A, eltype(A), inds)
	end
end

for DT in [:AbstractArray, :AbstractHalfIntegerArrayOrWrapper]
	@eval function Base.similar(A::$DT, ::Type{T}, inds::Tuple{HalfIntAxis,Vararg{HalfIntAxis}}) where {T}
		P = similar(A, T, map(indexlength, inds))
		HalfIntArray(P, map(offset, axes(P), inds))
	end
end

function Base.similar(::Type{T}, inds::Tuple{HalfIntAxis,Vararg{HalfIntOffsetAxisKnownLength}}) where {T<:AbstractArray}
    P = T(undef, map(indexlength, inds))
    HalfIntArray(P, map(offset, axes(P), inds))
end

for ax in [:HalfIntOffsetAxis, :OffsetAxis]
    @eval Base.reshape(A::AbstractHalfIntegerArrayOrWrapper, inds::Tuple{$ax,Vararg{$ax}}) =
        HalfIntArray(reshape(parent(A), map(indexlength, inds)), map(indexoffset, inds))
end

Base.reshape(A::AbstractHalfIntegerArray, inds::Tuple{OneTo,Vararg{OneTo}}) = reshape(parent(A), Base.OneTo.(inds))

# And for non-offset axes, we can just return a reshape of the parent directly
Base.reshape(A::AbstractHalfIntegerArray, inds::Vararg{Int}) = reshape(A, inds)
Base.reshape(A::AbstractHalfIntegerArray, inds::Vararg{Union{Colon,Int}}) = reshape(A, inds)
Base.reshape(A::AbstractHalfIntegerArray, inds::Vararg{Union{Int,AbstractUnitRange}}) = reshape(A, inds)
Base.reshape(A::AbstractHalfIntegerArray, ::Colon) = A

for DT in [:AbstractHalfIntegerArray, :AbstractHalfIntegerArrayOrWrapper]
	@eval Base.reshape(A::$DT, inds::Tuple{ReshapedAxis,Vararg{ReshapedAxis}}) = reshape(parent(A), inds)
	@eval Base.reshape(A::$DT, inds::Tuple{Int,Vararg{Int}}) = reshape(parent(A), inds)
	@eval Base.reshape(A::$DT, inds::Tuple{Union{Int,Colon},Vararg{Union{Int,Colon}}}) = reshape(parent(A), inds)
	@eval Base.reshape(A::$DT, inds::Tuple{Union{Int,Colon},Vararg{Union{Int,Colon,Base.OneTo}}}) = reshape(parent(A), inds)
	
	@eval Base.reshape(A::$DT, inds::Tuple{Colon,Vararg{Colon}}) = 
    	throw(DimensionMismatch("new dimensions $inds may have at most one omitted dimension specified by `Colon()`"))
	@eval Base.reshape(A::$DT, inds::Tuple{}) = 
	    throw(DimensionMismatch("new dimensions () must be consistent with array size $(length(A))"))
end

Base.sizeof(a::AbstractHalfIntegerWrapper) = sizeof(parent(a))

@static if isdefined(Base, :copyto_unaliased!)
    function Base.copyto_unaliased!(deststyle::IndexStyle, dest::AbstractHalfIntegerArrayOrWrapper, srcstyle::IndexStyle, src::AbstractHalfIntegerArrayOrWrapper)
        isempty(src) && return dest
        destinds, srcinds = LinearIndicesHalfInt(dest), LinearIndicesHalfInt(src)
        idf, isf = first(destinds), first(srcinds)
        Δi = idf - isf
        (checkbounds(Bool, destinds, isf+Δi) & checkbounds(Bool, destinds, last(srcinds)+Δi)) ||
            throw(BoundsError(dest, srcinds))
        if deststyle isa IndexLinear
            if srcstyle isa IndexLinear
                # Single-index implementation
                @inbounds for i in srcinds
                    dest[i + Δi] = src[i]
                end
            else
                # Dual-index implementation
                i = idf - 1
                @inbounds for a in src
                    dest[i+=1] = a
                end
            end
        else
            iterdest, itersrc = eachindex(dest), eachindex(src)
            if iterdest == itersrc
                # Shared-iterator implementation
                for I in iterdest
                    @inbounds dest[I] = src[I]
                end
            else
                # Dual-iterator implementation
                ret = iterate(iterdest)
                @inbounds for a in src
                    idx, state = ret
                    dest[idx] = a
                    ret = iterate(iterdest, state)
                end
            end
        end
        return dest
    end
end

end

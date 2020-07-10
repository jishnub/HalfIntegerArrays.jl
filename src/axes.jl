"""
    HalfIntegerArrays.OneTo(n)

Define an AbstractUnitRange that behaves like `1:n`, with the added distinction that the lower limit is guaranteed (by the type
system) to be `1`. The elements are integers, but of a `HalfInteger` type.
"""
struct OneTo{T<:HalfInteger} <: AbstractUnitRange{T}
    stop :: T

    OneTo{T}(stop) where {T<:HalfInteger} = new(max(zero(T), stop))
    function OneTo{T}(r::AbstractRange{Q}) where {T<:HalfInteger, Q}
        throwstart(r) = throw(ArgumentError("first element must be 1, got $(first(r))"))
        throwstep(r)  = throw(ArgumentError("step must be 1, got $(step(r))"))
        first(r) == one(Q) || throwstart(r)
        step(r)  == one(Q) || throwstep(r)
        return new(max(zero(T), last(r)))
    end
end
OneTo(stop::T) where {T<:Integer} = OneTo{Half{T}}(stop)
OneTo(stop::T) where {T<:HalfInteger} = OneTo{T}(stop)
OneTo(r::AbstractRange{T}) where {T<:Integer} = OneTo{Half{T}}(r)
OneTo(r::AbstractRange{T}) where {T<:HalfInteger} = OneTo{T}(r)
OneTo(r::OneTo{T}) where {T<:HalfInteger} = r
OneTo{T}(r::OneTo{T}) where {T<:HalfInteger} = r
OneTo{T}(r::Union{OneTo,Base.OneTo}) where {T<:HalfInteger} = OneTo{T}(r.stop)

Base.first(::OneTo{Q}) where {Q} = oneunit(Q)
Base.step(::OneTo{Q}) where {Q} = oneunit(Q)
Base.last(b::OneTo) = b.stop

Base.unsafe_length(r::OneTo) = unsafeInt(r.stop)
Base.length(r::OneTo) = Base.unsafe_length(r)

@inline function Base.getindex(v::OneTo{T}, i::Integer) where T
    @boundscheck ((i > 0) & (i <= v.stop)) || throw_boundserror(v, i)
    convert(T, i)
end
@inline function Base.getindex(r::OneTo{T}, s::Union{OneTo,Base.OneTo}) where T
    @boundscheck checkbounds(r, s)
    OneTo(T(s.stop))
end

# conversions
Base.convert(::Type{AbstractUnitRange{Q}}, b::Base.OneTo) where {Q<:HalfInteger} = OneTo{Q}(b.stop)
Base.convert(::Type{AbstractUnitRange{Q}}, b::OneTo) where {Q<:HalfInteger} = OneTo{Q}(b.stop)
Base.convert(::Type{AbstractUnitRange{Q}}, b::OneTo) where {Q<:Integer} = Base.OneTo{Q}(b)

Base.OneTo{Q}(b::OneTo) where {Q<:Integer} = Base.OneTo{Q}(unsafeInt(b.stop))
Base.OneTo(b::OneTo) = Base.OneTo(unsafeInt(b.stop))

"""
    ro = IdOffsetRange(r::AbstractUnitRange, offset::HalfInteger)

Construct an "identity offset range". 
Numerically, `collect(ro) == collect(r) .+ offset`, with the additional property that 
`axes(ro,1) = axes(r, 1) .+ offset`. 
When `r` starts at `1`, then `ro[i] == i` and even `ro[ro] == ro`, i.e., it's the "identity," which is the 
origin of the "Id" in `IdOffsetRange`. The element type is a `HalfInteger`.
"""
struct IdOffsetRange{T<:HalfInteger,I<:AbstractUnitRange{<:Integer}} <: AbstractUnitRange{T}
    parent::I
    offset::T

    function IdOffsetRange{T,I}(r::I, o::T) where {T<:HalfInteger,I<:AbstractUnitRange{<:Integer}}
        new{T,I}(r,o)
    end
end

# Construction/coercion from arbitrary AbstractUnitRanges
function IdOffsetRange{T,I}(r::AbstractUnitRange, offset::HalfIntegerOrInteger = zero(HalfInt)) where {T<:HalfInteger,I<:AbstractUnitRange{<:Integer}}
    rc, o = offset_coerce(I, r)
    return IdOffsetRange{T,I}(rc, convert(T, o+offset))
end
function IdOffsetRange{T}(r::AbstractUnitRange{<:Integer}, offset::HalfIntegerOrInteger = zero(HalfInt)) where {T<:HalfInteger}
    return IdOffsetRange{T,typeof(r)}(r, convert(T, offset))
end
function IdOffsetRange(r::AbstractUnitRange{Q}, offset::HalfIntegerOrInteger) where {Q<:Integer}
    of = HalfInteger(offset)
    IdOffsetRange{typeof(of),typeof(r)}(r, of)
end

IdOffsetRange(r::AbstractUnitRange{T}) where {T<:Integer} = IdOffsetRange(r, zero(Half{T}))

# Coercion from other IdOffsetRanges
IdOffsetRange{T,I}(r::IdOffsetRange{T,I}) where {T,I} = r
function IdOffsetRange{T,I}(r::IdOffsetRange) where {T<:HalfInteger,I<:AbstractUnitRange{<:Integer}}
    rc, offset = offset_coerce(I, parent(r))
    IdOffsetRange{T,I}(rc, r.offset+offset)
end
function IdOffsetRange{T}(r::IdOffsetRange) where {T<:HalfInteger}
    IdOffsetRange{T}(parent(r), convert(T,offset(r)))
end
IdOffsetRange(r::IdOffsetRange) = r

function offset_coerce(::Type{R}, r::AbstractUnitRange{T}) where {T,R<:Union{Base.OneTo,OneTo}}
    o = first(r) - one(eltype(r))
    return R(floor(Int,last(r) - o)), T(o)
end

# Fallback, specialze this method if `convert(I, r)` doesn't do what you need
offset_coerce(::Type{I}, r::AbstractUnitRange) where I<:AbstractUnitRange{T} where {T} =
    convert(I, r), zero(T)

Base.eltype(::IdOffsetRange{T,<:AbstractUnitRange{Q}}) where {T,Q} = promote_type(T,Q)

@inline offset(iter::IdOffsetRange) = iter.offset
@inline Base.parent(r::IdOffsetRange) = r.parent
@inline Base.axes(r::IdOffsetRange) = (Base.axes1(r),)
@inline Base.axes1(r::IdOffsetRange) = IdOffsetRange(Base.axes1(parent(r)), offset(r))
@inline Base.unsafe_indices(r::IdOffsetRange) = (r,)
@inline Base.unsafe_indices(r::Base.Slice{<:IdOffsetRange}) = (r.indices,)
@inline Base.length(r::IdOffsetRange) = length(parent(r))
@inline Base.compute_offset1(parent, stride1::Integer, dims::Tuple{Int}, inds::Tuple{IdOffsetRange}, I::Tuple) =
    Base.compute_linindex(parent, I) - stride1*first(axes(parent, dims[1]))
Base.reduced_index(i::IdOffsetRange) = typeof(i)(first(parent(i)):first(parent(i)),offset(i))
Base.reduced_index(i::IdentityUnitRange{<:IdOffsetRange}) = Base.reduced_index(i.indices)

@inline function Base.iterate(r::IdOffsetRange, i...)
    ret = iterate(parent(r), i...)
    ret === nothing && return nothing
    return (ret[1] + r.offset, ret[2])
end

@inline Base.first(r::IdOffsetRange) = first(r.parent) + r.offset
@inline Base.last(r::IdOffsetRange) = last(r.parent) + r.offset

function unsafeUnitRangeInt(r::AbstractUnitRange{HalfInt})
    unsafeInt(first(r)):unsafeInt(last(r))
end

@propagate_inbounds function Base.getindex(r::IdOffsetRange{HalfInt}, i::HalfInt)
    r.parent[unsafeInt(i - r.offset)] + r.offset
end
@propagate_inbounds function Base.getindex(r::IdOffsetRange{HalfInt}, s::AbstractUnitRange{HalfInt})
    return r.parent[unsafeUnitRangeInt(s .- r.offset)] .+ r.offset
end
@propagate_inbounds function Base.getindex(r::IdOffsetRange{HalfInt}, s::IdOffsetRange{HalfInt})
    p = r.parent[unsafeUnitRangeInt(s .- r.offset)]
    return IdOffsetRange(p, r.offset)
end
for DT in [:Integer,:HalfInteger]
    @eval @propagate_inbounds function Base.getindex(r::IdOffsetRange, i::$DT)
    	r.parent[Integer(i - r.offset)] + r.offset
    end
    @eval @propagate_inbounds function Base.getindex(r::IdOffsetRange, s::AbstractUnitRange{<:$DT})
        return r.parent[map(Integer, s .- r.offset)] .+ r.offset
    end
end

@propagate_inbounds function Base.getindex(r::IdOffsetRange, s::IdOffsetRange)
    p = r.parent[map(Integer,s .- r.offset)]
    return IdOffsetRange(p, r.offset)
end

# Optimizations
function Base.checkindex(::Type{Bool}, inds::IdOffsetRange, i::HalfInt)
    Base.checkindex(Bool, parent(inds), i - offset(inds)) && isinteger(offset(inds) + i)
end

function Base.isassigned(r::IdOffsetRange, I::Union{HalfInt,Int}...)
    J = to_parentindices(axes(r),I)
    isassigned(parent(r),J...)
end
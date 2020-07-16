struct HalfIntBroadcastStyle{N} <: Broadcast.AbstractArrayStyle{N} end

HalfIntBroadcastStyle{M}(::Val{N}) where {M,N} = HalfIntBroadcastStyle{N}()

function Base.similar(bc::Broadcast.Broadcasted{HalfIntBroadcastStyle{N}}, ::Type{T}) where {T,N}
	HalfIntArray{T,N}(undef, axes(bc)...)
end

Base.eachindex(bc::Broadcast.Broadcasted{<:HalfIntBroadcastStyle}) = CartesianIndicesHalfInt(axes(bc))

Base.BroadcastStyle(::Type{<:AbstractHalfIntegerArray{T,N}}) where {T,N} = HalfIntBroadcastStyle{N}()

@inline function Base.copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted{<:HalfIntBroadcastStyle})
	axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{AbstractArray} # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = Broadcast.preprocess(dest, bc)
    @simd for I in eachindex(bc′)
        @inbounds dest[I] = bc′[I]
    end
    return dest
end

@inline function Base.getindex(bc::Broadcast.Broadcasted, I::CartesianIndexHalfInt)
    @boundscheck checkbounds(bc, I)
    @inbounds Broadcast._broadcast_getindex(bc, I)
end

# Shadow the indexing from Broadcast
Base.@propagate_inbounds _newindex(ax::Tuple, I::Tuple) = (ifelse(Base.unsafe_length(ax[1])==1, ax[1][1], I[1]), _newindex(Base.tail(ax), Base.tail(I))...)
Base.@propagate_inbounds _newindex(ax::Tuple{}, I::Tuple) = ()
Base.@propagate_inbounds _newindex(ax::Tuple, I::Tuple{}) = (ax[1][1], _newindex(Base.tail(ax), ())...)
Base.@propagate_inbounds _newindex(ax::Tuple{}, I::Tuple{}) = ()
@inline _newindex(I, keep, Idefault) =
    (ifelse(keep[1], I[1], Idefault[1]), _newindex(Base.tail(I), Base.tail(keep), Base.tail(Idefault))...)
@inline _newindex(I, keep::Tuple{}, Idefault) = ()  # truncate if keep is shorter than I

@propagate_inbounds Broadcast.newindex(arg, I::HalfInteger) = CartesianIndexHalfInt(_newindex(axes(arg), (I,)))
@propagate_inbounds function Broadcast.newindex(arg, I::CartesianIndexHalfInt)
	CartesianIndexHalfInt(_newindex(axes(arg), Tuple(I)))
end
@inline Broadcast.newindex(i::HalfInteger, keep::Tuple{Bool}, idefault) = ifelse(keep[1], i, idefault[1])
@inline Broadcast.newindex(i::HalfInteger, keep::Tuple{}, idefault) = CartesianIndexHalfInt(())
@inline function Broadcast.newindex(I::CartesianIndexHalfInt, keep, Idefault)
	CartesianIndexHalfInt(_newindex(Tuple(I), keep, Idefault))
end

Base.dataids(A::AbstractHalfIntegerWrapper) = Base.dataids(parent(A))
Broadcast.broadcast_unalias(dest::AbstractHalfIntegerWrapper, src::AbstractHalfIntegerWrapper) = parent(dest) === parent(src) ? src : Broadcast.unalias(dest, src)

@inline Base.checkbounds(bc::Broadcast.Broadcasted, I::CartesianIndexHalfInt) =
    Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,))
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

Base.dataids(A::AbstractHalfIntegerArray) = Base.dataids(parent(A))
Broadcast.broadcast_unalias(dest::AbstractHalfIntegerArray, src::AbstractHalfIntegerArray) = parent(dest) === parent(src) ? src : Broadcast.unalias(dest, src)

function Base.eachindex(bc::Broadcast.Broadcasted{<:Union{Nothing, Broadcast.BroadcastStyle},<:Tuple{IdOffsetRange,Vararg{IdOffsetRange}}})
	CartesianIndicesHalfInt(axes(bc))
end

@inline Base.checkbounds(bc::Broadcast.Broadcasted, I::CartesianIndexHalfInt) =
    Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,))
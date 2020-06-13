for DT in [:Integer, :HalfInteger]
    @eval function Base.print_matrix_row(io::IO,
        X::Union{AbstractHalfIntegerVecOrMat, AdjOrTransAbsHalfIntVecOrMat,
        HalfIntSubArray}, 
        A::Vector,
        i::$DT, cols::AbstractVector, sep::AbstractString)
        
        for (k, j) = enumerate(cols)
            k > length(A) && break
            if isassigned(X,HalfInt(i),HalfInt(j))
                x = X[i,j]
                a = Base.alignment(io, x)
                sx = sprint(show, x, context=io, sizehint=0)
            else
                a = Base.undef_ref_alignment
                sx = Base.undef_ref_str
            end
            l = repeat(" ", A[k][1]-a[1]) # pad on left and right as needed
            r = j == axes(X, 2)[end] ? "" : repeat(" ", A[k][2]-a[2])
            prettysx = Base.replace_in_print_matrix(X,i,j,sx)
            print(io, l, prettysx, r)
            if k < length(A); print(io, sep); end
        end
    end
end

function Base.show(io::IO, X::HIAorSM)
    Y = unwraphalfint(X)
    show(io, Y)
end

function Base.show(io::IO, X::AdjOrTransAbsHalfIntVecOrMat)
    Y = wrapperop(X)(unwraphalfint(X))
    show(io, Y)
end

function Base.show(io::IO, X::Union{LinearIndicesHalfInt,CartesianIndicesHalfInt})
    Y = unwraphalfint(collect(X))
    show(io, Y)    
end

Base.show(io::IO, i::CartesianIndexHalfInt) = (print(io, "CartesianIndexHalfInt"); show(io, Tuple(i)))
Base.show(io::IO, r::OneTo) = print(io, "HalfIntegerArrays.OneTo(", r.stop, ")")
Base.show(io::IO, r::IdOffsetRange) = print(io,first(r), ':', last(r))

function Base.show_nd(io::IO, a::AbstractHalfIntegerArray, print_matrix::Function, label_slices::Bool)
    limit::Bool = get(io, :limit, false)
    if isempty(a)
        return
    end
    tailinds = Base.tail(Base.tail(axes(a)))
    nd = ndims(a)-2
    for I in CartesianIndicesHalfInt(tailinds)
        idxs = Tuple(I)
        if limit
            for i = 1:nd
                ii = idxs[i]
                ind = tailinds[i]
                if length(ind) > 10
                    if ii == ind[firstindex(ind)+3] && all(d->idxs[d]==first(tailinds[d]),1:i-1)
                        for j=i+1:nd
                            szj = length(axes(a, j+2))
                            indj = tailinds[j]
                            if szj>10 && first(indj)+2 < idxs[j] <= last(indj)-3
                                @goto skip
                            end
                        end
                        #println(io, idxs)
                        print(io, "...\n\n")
                        @goto skip
                    end
                    if ind[firstindex(ind)+2] < ii <= ind[end-3]
                        @goto skip
                    end
                end
            end
        end
        if label_slices
            print(io, "[:, :, ")
            for i = 1:(nd-1); print(io, "$(idxs[i]), "); end
            println(io, idxs[end], "] =")
        end
        slice = view(a, axes(a,1), axes(a,2), idxs...)
        print_matrix(io, slice)
        print(io, idxs == map(last,tailinds) ? "" : "\n\n")
        @label skip
    end
end

function Base.replace_in_print_matrix(A::AbstractHalfIntegerMatrix, i::HalfInteger, j::HalfInteger, s::AbstractString)
    J = to_parentindices(axes(A),(i,j))
    Base.replace_in_print_matrix(parent(A),J...,s)
end
function Base.replace_in_print_matrix(A::AbstractHalfIntegerVector, i::HalfInteger, j::HalfIntegerOrInteger, s::AbstractString)
    i′ = parentindex(Base.axes1(A), i)
    ensureInt(j)
    Base.replace_in_print_matrix(parent(A),i′,unsafeInt(j),s)
end

for DT in [:Integer, :HalfIntegerOrInteger]
    @eval function Base.replace_in_print_matrix(A::AdjOrTransAbsHalfIntVecOrMat, i::$DT, j::$DT, s::AbstractString)
        Base.replace_in_print_matrix(A.parent, j, i, s)
    end
end

function Base.showarg(io::IO, a::HalfIntSubArray, toplevel)
    print(io, "HalfIntSubArray(")
    Base.showarg(io, parent(a), false)
    if ndims(a) > 0
        iocompact = IOContext(io, :compact => true)
        Base.showindices(iocompact, UnitRange.(axes(a))...)
    end
    print(io, ')')
    toplevel && print(io, " with eltype ", eltype(a))
end

function Base.showarg(io::IO, a::HalfIntArray, toplevel)
    print(io, "HalfIntArray(")
    Base.showarg(io, parent(a), false)
    if ndims(a) > 0
        iocompact = IOContext(io, :compact => true)
        Base.showindices(iocompact, axes(a)...)
    end
    print(io, ')')
    toplevel && print(io, " with eltype ", eltype(a))
end

function Base.showarg(io::IO, a::SpinMatrix, toplevel)
    print(io, "SpinMatrix(")
    Base.showarg(io, parent(a), false)
    print(io, ", ")
    print(io, a.j)
    print(io, ')')
    toplevel && print(io, " with eltype ", eltype(a))
end

function Base.showarg(io::IO, a::CartesianIndicesHalfIntParent{N}, toplevel) where {N}
    print(io, "CartesianIndicesHalfIntParent{",N,"}")
end

function Base.showarg(io::IO, a::LinearIndicesHalfIntParent{N}, toplevel) where {N}
    print(io, "LinearIndicesHalfIntParent{",N,"}")
end

function Base.showarg(io::IO, r::Iterators.Pairs{<:HalfInteger, <:Any, <:Any, T}, toplevel) where T <: Union{AbstractVector, Tuple}
    print(io, "pairs(::$T)")
end

function Base.showarg(io::IO, r::Iterators.Pairs{<:CartesianIndexHalfInt, <:Any, <:Any, T}, toplevel) where T <: AbstractArray
    print(io, "pairs(::$T)")
end

function Base.showarg(io::IO, r::Iterators.Pairs{<:CartesianIndexHalfInt, <:Any, <:Any, T}, toplevel) where T<:AbstractVector
    print(io, "pairs(IndexCartesian(), ::$T)")
end

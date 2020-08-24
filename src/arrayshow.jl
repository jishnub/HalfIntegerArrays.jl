for DT in [:Integer, :HalfInteger]
    @eval function Base.print_matrix_row(io::IO,
        X::Union{AbstractHalfIntegerVecOrMat, AdjOrTransAbsHalfIntVecOrMat,
        HalfIntSubArray}, 
        A::Vector,
        i::$DT, cols::AbstractVector, sep::AbstractString)
        
        for (k, j) = enumerate(cols)
            k > length(A) && break
            if isassigned(X, i, j)
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

function Base.print_matrix(io::IO, X::Union{AbstractHalfIntegerVecOrMat, AdjOrTransAbsHalfIntVecOrMat,
    HalfIntSubArray},
    pre::AbstractString = " ",  # pre-matrix string
    sep::AbstractString = "  ", # separator between elements
    post::AbstractString = "",  # post-matrix string
    hdots::AbstractString = "  \u2026  ",
    vdots::AbstractString = "\u22ee",
    ddots::AbstractString = "  \u22f1  ",
    hmod::Integer = 5, vmod::Integer = 5)

    hmod, vmod = Int(hmod)::Int, Int(vmod)::Int
    if !(get(io, :limit, false)::Bool)
        screenheight = screenwidth = typemax(Int)
    else
        sz = displaysize(io)::Tuple{Int,Int}
        screenheight, screenwidth = sz[1] - 4, sz[2]
    end
    screenwidth -= length(pre)::Int + length(post)::Int
    presp = repeat(" ", length(pre)::Int)  # indent each row to match pre string
    postsp = ""
    @assert textwidth(hdots) == textwidth(ddots)
    sepsize = length(sep)::Int
    rowsA, colsA = UnitRange(axes(X,1)), UnitRange(axes(X,2))
    m, n = length(rowsA)::Int, length(colsA)::Int
    # To figure out alignments, only need to look at as many rows as could
    # fit down screen. If screen has at least as many rows as A, look at A.
    # If not, then we only need to look at the first and last chunks of A,
    # each half a screen height in size.
    halfheight = div(screenheight,2)
    if m > screenheight
        rowsA = [rowsA[(0:halfheight-1) .+ firstindex(rowsA)]; rowsA[(end-div(screenheight-1,2)+1):end]]
    end
    # Similarly for columns, only necessary to get alignments for as many
    # columns as could conceivably fit across the screen
    maxpossiblecols = div(screenwidth, 1+sepsize)
    if n > maxpossiblecols
        colsA = [colsA[(0:maxpossiblecols-1) .+ firstindex(colsA)]; colsA[(end-maxpossiblecols+1):end]]
    end
    A = Base.alignment(io, X, rowsA, colsA, screenwidth, screenwidth, sepsize)
    # Nine-slicing is accomplished using print_matrix_row repeatedly
    if m <= screenheight # rows fit vertically on screen
        if n <= length(A) # rows and cols fit so just print whole matrix in one piece
            for i in rowsA
                print(io, i == first(rowsA) ? pre : presp)
                Base.print_matrix_row(io, X,A,i,colsA,sep)
                print(io, i == last(rowsA) ? post : postsp)
                if i != last(rowsA); println(io); end
            end
        else # rows fit down screen but cols don't, so need horizontal ellipsis
            c = div(screenwidth-length(hdots)::Int+1,2)+1  # what goes to right of ellipsis
            Ralign = reverse(Base.alignment(io, X, rowsA, reverse(colsA), c, c, sepsize)) # alignments for right
            c = screenwidth - sum(map(sum,Ralign)) - (length(Ralign)-1)*sepsize - length(hdots)::Int
            Lalign = Base.alignment(io, X, rowsA, colsA, c, c, sepsize) # alignments for left of ellipsis
            for i in rowsA
                print(io, i == first(rowsA) ? pre : presp)
                Base.print_matrix_row(io, X,Lalign,i,colsA[1:length(Lalign)],sep)
                print(io, (i - first(rowsA)) % hmod == 0 ? hdots : repeat(" ", length(hdots)::Int))
                Base.print_matrix_row(io, X, Ralign, i, (n - length(Ralign)) .+ colsA, sep)
                print(io, i == last(rowsA) ? post : postsp)
                if i != last(rowsA); println(io); end
            end
        end
    else # rows don't fit so will need vertical ellipsis
        if n <= length(A) # rows don't fit, cols do, so only vertical ellipsis
            for i in rowsA
                print(io, i == first(rowsA) ? pre : presp)
                Base.print_matrix_row(io, X,A,i,colsA,sep)
                print(io, i == last(rowsA) ? post : postsp)
                if i != rowsA[end] || i == rowsA[halfheight]; println(io); end
                if i == rowsA[halfheight]
                    print(io, i == first(rowsA) ? pre : presp)
                    print_matrix_vdots(io, vdots, A, sep, vmod, 1, false)
                    print(io, i == last(rowsA) ? post : postsp * '\n')
                end
            end
        else # neither rows nor cols fit, so use all 3 kinds of dots
            c = div(screenwidth-length(hdots)::Int+1,2)+1
            Ralign = reverse(Base.alignment(io, X, rowsA, reverse(colsA), c, c, sepsize))
            c = screenwidth - sum(map(sum,Ralign)) - (length(Ralign)-1)*sepsize - length(hdots)::Int
            Lalign = Base.alignment(io, X, rowsA, colsA, c, c, sepsize)
            r = mod((length(Ralign)-n+1),vmod) # where to put dots on right half
            for i in rowsA
                print(io, i == first(rowsA) ? pre : presp)
                Base.print_matrix_row(io, X,Lalign,i,colsA[1:length(Lalign)],sep)
                print(io, (i - first(rowsA)) % hmod == 0 ? hdots : repeat(" ", length(hdots)::Int))
                Base.print_matrix_row(io, X,Ralign,i,(n-length(Ralign)).+colsA,sep)
                print(io, i == last(rowsA) ? post : postsp)
                if i != rowsA[end] || i == rowsA[halfheight]; println(io); end
                if i == rowsA[halfheight]
                    print(io, i == first(rowsA) ? pre : presp)
                    print_matrix_vdots(io, vdots, Lalign, sep, vmod, 1, true)
                    print(io, ddots)
                    print_matrix_vdots(io, vdots, Ralign, sep, vmod, r, false)
                    print(io, i == last(rowsA) ? post : postsp * '\n')
                end
            end
        end
        if isempty(rowsA)
            print(io, pre)
            print(io, vdots)
            length(colsA) > 1 && print(io, "    ", ddots)
            print(io, post)
        end
    end
end

function Base.show(io::IO, X::AbstractHalfIntegerWrapper)
    show(io, parent(X))
end

function Base.show(io::IO, X::AdjOrTransAbsHalfIntVecOrMat)
    Y = wrapperop(X)(unwraphalfint(X))
    show(io, Y)
end

Base.show(io::IO, i::CartesianIndexHalfInt) = (print(io, "CartesianIndexHalfInt"); show(io, Tuple(i)))
Base.show(io::IO, r::OneTo) = print(io, "HalfIntegerArrays.OneTo(", r.stop, ")")
Base.show(io::IO, r::IdOffsetRange) = print(io,first(r), ':', last(r))

# In general collect the array before displaying
Base.show(io::IO, a::AbstractHalfIntegerArray) = show(io, collect(a))

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

Base.replace_in_print_matrix(A::AbstractMatrix,i::HalfIntegerOrInteger,j::HalfIntegerOrInteger,s::AbstractString) = s
Base.replace_in_print_matrix(A::AbstractVector,i::HalfIntegerOrInteger,j::HalfIntegerOrInteger,s::AbstractString) = s

for DT in [:Integer, :HalfIntegerOrInteger]
    @eval function Base.replace_in_print_matrix(A::AbstractHalfIntegerWrapper{<:Any,2}, i::$DT, j::$DT, s::AbstractString)
        J = to_parentindices(axes(A),(i,j))
        Base.replace_in_print_matrix(parent(A),J...,s)
    end
    @eval function Base.replace_in_print_matrix(A::AbstractHalfIntegerWrapper{<:Any,1}, i::$DT, j::$DT, s::AbstractString)
        i′ = parentindex(Base.axes1(A), i)
        ensureInt(j)
        Base.replace_in_print_matrix(parent(A),i′,unsafeInt(j),s)
    end
end

for DT in [:Integer, :HalfIntegerOrInteger]
    @eval function Base.replace_in_print_matrix(A::AdjOrTransAbsHalfIntVecOrMat, i::$DT, j::$DT, s::AbstractString)
        Base.replace_in_print_matrix(A.parent, j, i, s)
    end
end

function Base.showarg(io::IO, a::HalfIntSubArray, toplevel)
    print(io, "view(")
    Base.showarg(io, parent(a), false)
    iocompact = IOContext(io, :compact => true)
    Base.showindices(iocompact, a.indices...)
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

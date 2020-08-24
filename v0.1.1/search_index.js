var documenterSearchIndex = {"docs":
[{"location":"#","page":"Reference","title":"Reference","text":"CurrentModule = HalfIntegerArrays","category":"page"},{"location":"#HalfIntegerArrays-1","page":"Reference","title":"HalfIntegerArrays","text":"","category":"section"},{"location":"#","page":"Reference","title":"Reference","text":"Modules = [HalfIntegerArrays]","category":"page"},{"location":"#HalfIntegerArrays.CartesianIndexHalfInt","page":"Reference","title":"HalfIntegerArrays.CartesianIndexHalfInt","text":"CartesianIndexHalfInt(i, j, k...) -> I\nCartesianIndexHalfInt((i, j, k...)) -> I\n\nCreate a multidimensional index I, which can be used for indexing a multidimensional AbstractHalfIntegerArray.   In particular, for an array A, the operation A[I] is equivalent to A[i,j,k...].  One can freely mix integer, half-integer and CartesianIndex indices; for example, A[Ipre, i, Ipost] (where Ipre and Ipost are CartesianIndex indices and i is an integer or a half-integer)  can be a useful expression when writing algorithms that work along a single dimension of an array of arbitrary dimensionality.\n\nExamples\n\njulia> h = HalfIntArray(reshape(1:4, 2, 2), -1//2:1//2, -1//2:1//2)\n2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:\n 1  3\n 2  4\n\njulia> h[CartesianIndexHalfInt(-1//2, 1//2)]\n3\n\njulia> h[CartesianIndexHalfInt(-1//2), 1//2]\n3\n\n\n\n\n\n","category":"type"},{"location":"#HalfIntegerArrays.CartesianIndicesHalfInt","page":"Reference","title":"HalfIntegerArrays.CartesianIndicesHalfInt","text":"CartesianIndicesHalfInt((istart:istop, jstart:jstop, ...)) -> R\n\nDefine a region R spanning a multidimensional rectangular range of integer indices. These are most commonly encountered in the context of iteration, where for I in R ... end will return CartesianIndexHalfInt indices I equivalent to the nested loops\n\nfor j = jstart:jstop\n    for i = istart:istop\n        ...\n    end\nend\n\nConsequently these can be useful for writing algorithms that work in arbitrary dimensions.\n\nA CartesianIndicesHalfInt type is equivalent to a CartesianIndices type  for integer ranges. The difference is that a CartesianIndicesHalfInt allows the  ranges to be half-integer AbstractUnitRanges. They are therefore suitable for  indexing into AbstractHalfIntegerArrays.\n\nCartesianIndicesHalfInt(A::AbstractArray) -> R\n\nAs a convenience, constructing a CartesianIndicesHalfInt from an array makes a range of its indices.\n\nExamples\n\njulia> h = HalfIntArray(reshape(1:4, 2, 2), -1//2:1//2, -1//2:1//2)\n2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:\n 1  3\n 2  4\n\njulia> c = CartesianIndicesHalfInt(h)\n2×2 CartesianIndicesHalfInt{2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}} with indices -1/2:1/2×-1/2:1/2:\n CartesianIndexHalfInt(-1/2, -1/2)  CartesianIndexHalfInt(-1/2, 1/2)\n CartesianIndexHalfInt(1/2, -1/2)   CartesianIndexHalfInt(1/2, 1/2)\n\njulia> c[1/2,1/2]\nCartesianIndexHalfInt(1/2, 1/2)\n\n\n\n\n\n","category":"type"},{"location":"#HalfIntegerArrays.HalfIntArray","page":"Reference","title":"HalfIntegerArrays.HalfIntArray","text":"HalfIntArray(A::AbstractArray, indices::AbstractUnitRange...)\n\nReturn a HalfIntArray that shares that shares element type and size with the first argument,  but used the given indices, which are checked for compatible size.\n\nThe indices may be Integer or HalfInteger ranges with a unit step.  Rational ranges over integer or half-integer values may also be provided.\n\nExamples\n\njulia> h = HalfIntArray(reshape(1:9,3,3), -1:1, -1:1)\n3×3 HalfIntArray(reshape(::UnitRange{Int64}, 3, 3), -1:1, -1:1) with eltype Int64 with indices -1:1×-1:1:\n 1  4  7\n 2  5  8\n 3  6  9\n\njulia> h[-1, -1]\n1\n\njulia> h = HalfIntArray(reshape(1:4,2,2), -1//2:1//2, -1//2:1//2)\n2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:\n 1  3\n 2  4\n\njulia> h[1//2, 1//2]\n4\n\n\n\n\n\n","category":"type"},{"location":"#HalfIntegerArrays.HalfIntArray-Union{Tuple{N}, Tuple{T}, Tuple{Union{Missing, Nothing, UndefInitializer},Tuple{Vararg{AbstractUnitRange,N}}}} where N where T","page":"Reference","title":"HalfIntegerArrays.HalfIntArray","text":"HalfIntArray{T}(init, indices::AbstractUnitRange...)\n\nReturn a HalfIntArray having elements of type T and axes as specified by indices.  The initializer init may be one of undef, missing or nothing.\n\nExamples\n\njulia> HalfIntArray{Union{Int,Missing}}(undef, 0:1, 0:1)\n2×2 HalfIntArray(::Array{Union{Missing, Int64},2}, 0:1, 0:1) with eltype Union{Missing, Int64} with indices 0:1×0:1:\n missing  missing\n missing  missing\n\n\n\n\n\n","category":"method"},{"location":"#HalfIntegerArrays.LinearIndicesHalfInt","page":"Reference","title":"HalfIntegerArrays.LinearIndicesHalfInt","text":"LinearIndicesHalfInt(A::AbstractArray)\n\nReturn a LinearIndicesHalfInt array with the same shape and axes as A, holding the linear index of each entry in A. Indexing this array with cartesian indices allows mapping them to linear indices.\n\nFor arrays with conventional indexing (indices start at 1), or any multidimensional array, linear indices range from 1 to length(A). However, for AbstractVectors linear indices are axes(A, 1), and therefore do not start at 1 for vectors with unconventional indexing.\n\nCalling this function is the \"safe\" way to write algorithms that exploit linear indexing.\n\nLinearIndicesHalfInt are equivalent to LinearIndices for arrays with integer axes, however LinearIndicesHalfInt also support arrays with half-integer axes. They are  therefore the suitable for working with AbstractHalfIntegerArrays.\n\nExamples\n\njulia> h = HalfIntArray(reshape(1:4, 2, 2), -1//2:1//2, -1//2:1//2)\n2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:\n 1  3\n 2  4\n\njulia> linds = LinearIndicesHalfInt(h)\n2×2 LinearIndicesHalfInt{2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}} with indices -1/2:1/2×-1/2:1/2:\n 1  3\n 2  4\n\njulia> linds[1/2, 1/2]\n4\n\njulia> linds[CartesianIndexHalfInt(1/2, 1/2)]\n4\n\njulia> v = HalfIntArray([1,2,3], -1:1)\n3-element HalfIntArray(::Array{Int64,1}, -1:1) with eltype Int64 with indices -1:1:\n 1\n 2\n 3\n\njulia> lindsv = LinearIndicesHalfInt(v)\n3-element LinearIndicesHalfInt{1,Tuple{Base.OneTo{Int64}}} with indices -1:1:\n -1\n  0\n  1\n\njulia> lindsv[0]\n0\n\n\n\n\n\n","category":"type"},{"location":"#HalfIntegerArrays.SpinMatrix","page":"Reference","title":"HalfIntegerArrays.SpinMatrix","text":"SpinMatrix(A::AbstractMatrix, [j::Real])\n\nReturn a SpinMatrix that allows the underlying matrix A to be indexed using  integers or half-integers, depending on the value of the angular momentum j.  The value of j needs to be either an Integer a half-integral Real number. If it is not provided it is automatically inferred from the size of the array A. The parent matrix A needs to use 1-based indexing, and have a size of (2j+1, 2j+1).\n\nThe axes of the SpinMatrix for an angular momentum j will necessarily be (-j:j, -j:j).\n\nExamples\n\njulia> SpinMatrix(reshape(1:4,2,2))\n2×2 SpinMatrix(reshape(::UnitRange{Int64}, 2, 2), 1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:\n 1  3\n 2  4\n\njulia> SpinMatrix(zeros(ComplexF64,3,3), 1)\n3×3 SpinMatrix(::Array{Complex{Float64},2}, 1) with eltype Complex{Float64} with indices -1:1×-1:1:\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n 0.0+0.0im  0.0+0.0im  0.0+0.0im\n\n\n\n\n\n","category":"type"},{"location":"#HalfIntegerArrays.SpinMatrix-Union{Tuple{T}, Tuple{Union{Missing, Nothing, UndefInitializer},Real}} where T","page":"Reference","title":"HalfIntegerArrays.SpinMatrix","text":"SpinMatrix{T}(init, j)\n\nCreate a SpinMatrix of type T for the angular momentum j.  An underlying Matrix of size (2j+1, 2j+1) is allocated in the process, with values  set by the initializer init that may be one of undef, missing or nothing.\n\nExamples\n\njulia> SpinMatrix{Union{Int,Missing}}(undef, 1//2)\n2×2 SpinMatrix(::Array{Union{Missing, Int64},2}, 1/2) with eltype Union{Missing, Int64} with indices -1/2:1/2×-1/2:1/2:\n missing  missing\n missing  missing\n\n\n\n\n\n","category":"method"},{"location":"#HalfIntegerArrays.AbstractHalfIntegerArray","page":"Reference","title":"HalfIntegerArrays.AbstractHalfIntegerArray","text":"AbstractHalfIntegerArray{T,N}\n\nSupertype for N-dimensional arrays (or array-like types) with elements of type T.  Such arrays may be indexed using integers as well as half-integers.\n\n\n\n\n\n","category":"type"},{"location":"#HalfIntegerArrays.AbstractHalfIntegerWrapper","page":"Reference","title":"HalfIntegerArrays.AbstractHalfIntegerWrapper","text":"AbstractHalfIntegerWrapper{T,N} <: AbstractHalfIntegerArray{T,N}\n\nSupertype for N-dimensional arrays (or array-like types) with elements of type T  that wrap a parent array that has integer indices. The wrapper acts as a map between the  indices of the array and those of the parent.\n\nHalfIntArray, SpinMatrix and other types are subtypes of this.\n\n\n\n\n\n","category":"type"},{"location":"#HalfIntegerArrays.HalfIntSubArray","page":"Reference","title":"HalfIntegerArrays.HalfIntSubArray","text":"HalfIntSubArray{T,N,P,I,L}\n\nN-dimensional view into a parent array (of type P <: AbstractHalfIntegerArray)  with an element type T, restricted by a tuple of indices (of type I).  L is true for types that support fast linear indexing, and false otherwise.\n\nTheir behavior is analogous to SubArrays, except they enable indexing with  half-integers.\n\nConstruct HalfIntSubArrays using the view function, or equivalently  the @view macro.\n\nExample\n\njulia> h = HalfIntArray(reshape(1:9, 3, 3), -1:1, -1:1)\n3×3 HalfIntArray(reshape(::UnitRange{Int64}, 3, 3), -1:1, -1:1) with eltype Int64 with indices -1:1×-1:1:\n 1  4  7\n 2  5  8\n 3  6  9\n\njulia> hv = @view h[-1:1, 0]\n3-element HalfIntSubArray(view(reshape(::UnitRange{Int64}, 3, 3), 1:3, 2), 1:3) with eltype Int64 with indices 1:3:\n 4\n 5\n 6\n\njulia> hv[1]\n4\n\njulia> h = HalfIntArray(reshape(collect(1:4), 2, 2), -1//2:1//2, -1//2:1//2)\n2×2 HalfIntArray(::Array{Int64,2}, -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:\n 1  3\n 2  4\n\njulia> hv = @view h[:, 1//2]\n2-element HalfIntSubArray(view(::Array{Int64,2}, :, 2), -1/2:1/2) with eltype Int64 with indices -1/2:1/2:\n 3\n 4\n\njulia> hv[1//2] = 10\n10\n\njulia> h\n2×2 HalfIntArray(::Array{Int64,2}, -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:\n 1   3\n 2  10\n\n\n\n\n\n","category":"type"},{"location":"#HalfIntegerArrays.IdOffsetRange","page":"Reference","title":"HalfIntegerArrays.IdOffsetRange","text":"ro = IdOffsetRange(r::AbstractUnitRange, offset::HalfInteger)\n\nConstruct an \"identity offset range\".  Numerically, collect(ro) == collect(r) .+ offset, with the additional property that  axes(ro,1) = axes(r, 1) .+ offset.  When r starts at 1, then ro[i] == i and even ro[ro] == ro, i.e., it's the \"identity,\" which is the  origin of the \"Id\" in IdOffsetRange. The element type is a HalfInteger.\n\n\n\n\n\n","category":"type"},{"location":"#HalfIntegerArrays.OneTo","page":"Reference","title":"HalfIntegerArrays.OneTo","text":"HalfIntegerArrays.OneTo(n)\n\nDefine an AbstractUnitRange that behaves like 1:n, with the added distinction that the lower limit is guaranteed (by the type system) to be 1. The elements are integers, but of a HalfInteger type.\n\n\n\n\n\n","category":"type"}]
}
# HalfIntegerArrays

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jishnub.github.io/HalfIntegerArrays.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/HalfIntegerArrays.jl/dev)
[![Build Status](https://github.com/jishnub/HalfIntegerArrays.jl/workflows/CI/badge.svg)](https://github.com/jishnub/HalfIntegerArrays.jl/actions)
[![Build Status](https://travis-ci.com/jishnub/HalfIntegerArrays.jl.svg?branch=master)](https://travis-ci.com/jishnub/HalfIntegerArrays.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jishnub/HalfIntegerArrays.jl?svg=true)](https://ci.appveyor.com/project/jishnub/HalfIntegerArrays-jl)
[![Coverage](https://codecov.io/gh/jishnub/HalfIntegerArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jishnub/HalfIntegerArrays.jl)
[![Coverage](https://coveralls.io/repos/github/jishnub/HalfIntegerArrays.jl/badge.svg?branch=master)](https://coveralls.io/github/jishnub/HalfIntegerArrays.jl?branch=master)

Arrays that may have half-integer indices, commonly encountered while working with rotations and spin.

This package is very much a WIP, and bugs are expected. Please open an issue if you encounter a bug.

# Prerequisites

The package is to be used alongside [HalfIntegers.jl](https://github.com/sostock/HalfIntegers.jl). This is installed automatically with this package, and may be imported as shown below.

# Installation

```julia
julia> ]
pkg> add https://github.com/jishnub/HalfIntegerArrays.jl

julia> using HalfIntegerArrays
julia> using HalfIntegerArrays.HalfIntegers
```

# Usage

There are two types exported: `HalfIntArray` and `SpinMatrix`. The former represents an arbitrary array with possibly half-integral axes, whereas the latter represents a square matrix corresponding to an angular momentum `j`. In the second case the array is guaranteed to have a size of `(2j+1, 2j+1)`, and axes `(-j:j, -j:j)`.

`HalfIntArray`s are wrappers around `AbstractArray`s, and may be constructed in two ways: firstly by providing the axes for the parent array, eg:

```julia
julia> h = HalfIntArray(ones(Int,3,3), -1:1, -1:1)
3×3 HalfIntArray(::Array{Int64,2}, -1:1, -1:1) with eltype Int64 with indices -1:1×-1:1:
 1  1  1
 1  1  1
 1  1  1

# We may wrap structured arrays
julia> import LinearAlgebra: Diagonal

julia> h = HalfIntArray(Diagonal([1,2]), -1//2:1//2, -1//2:1//2)
2×2 HalfIntArray(::Diagonal{Int64,Array{Int64,1}}, -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  ⋅
 ⋅  2
```

and secondly by using the typed constructor

```julia
julia> h = HalfIntArray{Float64}(undef,-1//2:1//2, -1//2:1//2)
2×2 HalfIntArray(::Array{Float64,2}, -1/2:1/2, -1/2:1/2) with eltype Float64 with indices -1/2:1/2×-1/2:1/2:
 0.0           0.0
 6.89924e-310  0.0

julia> h = HalfIntArray{Union{Float64,Missing}}(missing,-1:1, -1:1)
3×3 HalfIntArray(::Array{Union{Missing, Float64},2}, -1:1, -1:1) with eltype Union{Missing, Float64} with indices -1:1×-1:1:
 missing  missing  missing
 missing  missing  missing
 missing  missing  missing
```

In the second case an underlying array of an appropriate size is initialized that has the specified element type.

A `SpinMatrix` may be constructed as 

```julia
julia> s = SpinMatrix(zeros(3,3)) # the angular momentum is inferred
3×3 SpinMatrix(::Array{Float64,2}, 1) with eltype Float64 with indices -1:1×-1:1:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

# Specifying the angular momentum will allocate an appropriate array
julia> s = SpinMatrix{ComplexF64}(undef, 1//2)
2×2 SpinMatrix(::Array{Complex{Float64},2}, 1/2) with eltype Complex{Float64} with indices -1/2:1/2×-1/2:1/2:
 6.91635e-310+6.91635e-310im  6.91637e-310+6.91637e-310im
 6.91635e-310+6.91637e-310im  6.91637e-310+6.91637e-310im
```

A `SpinMatrix` requires the parent array to have `1`-based indexing. It does not, however, impose any size restriction on the parent array. The only restriction is on the number of dimensions: the parent array needs to be a `Vector` or a `Matrix`.

## Indexing

The arrays may be indexed with integral or half-integral values. For optimal performance it's preferable to use integral, floating-point or `HalfInt` types as indices. A `HalfInt` may be constructed using the function `half`, such that `half(n) == n/2`. Alternately they may also be constructed as `HalfInt(n)` which is numerically equivalent to `n`. Rational numbers may be used as well, but these might not be as performant.

An example with a half-integral spin:

```julia
julia> s = SpinMatrix(reshape(1:4,2,2), 1//2)
2×2 SpinMatrix(reshape(::UnitRange{Int64}, 2, 2), 1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> s[1//2,1//2]
4

julia> import HalfIntegerArrays: half

julia> s[-half(1),half(1)]
3
```

An example with integral spin:

```julia
julia> s = SpinMatrix(reshape(1:9,3,3))
3×3 SpinMatrix(reshape(::UnitRange{Int64}, 3, 3), 1) with eltype Int64 with indices -1:1×-1:1:
 1  4  7
 2  5  8
 3  6  9

julia> s[1,1]
9
```

### Linear and Cartesian Indexing

Julia's default `CartesianIndex`, `CartesianIndices` and `LinearIndices` types require integer indices, therefore these are not compatible with `HalfIntArray`s. This package exports the equivalent types `CartesianIndexHalfInt`, `CartesianIndicesHalfInt` and `LinearIndicesHalfInt` that support both integer and half-integer indices. These are therefore the safe choices for indexing into `HalfIntArray`s.

```julia
julia> h = HalfIntArray(reshape(1:4,2,2), 0:1, 0:1)
2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), 0:1, 0:1) with eltype Int64 with indices 0:1×0:1:
 1  3
 2  4

julia> eachindex(IndexCartesian(), h)
2×2 CartesianIndicesHalfInt{2,Tuple{Base.OneTo{Int64},Base.OneTo{Int64}}} with indices 0:1×0:1:
 CartesianIndexHalfInt(0, 0)  CartesianIndexHalfInt(0, 1)
 CartesianIndexHalfInt(1, 0)  CartesianIndexHalfInt(1, 1)

julia> h[cinds[1,0]]
2
```

Indexing with `CartesianIndices` work as well for arrays with integer indices, where the axes ranges are specified while creating the range.

## Broadcasting

Broadcasting works, but is a bit slow at the moment. For optimal performance it's better to broadcast on the parent array.

```julia
julia> h = HalfIntArray(reshape(1:4,2,2), -1//2:1//2, -1//2:1//2)
2×2 HalfIntArray(reshape(::UnitRange{Int64}, 2, 2), -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> h .+ h
2×2 HalfIntArray(::Array{Int64,2}, -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 2  6
 4  8

julia> @btime $h .+ $h;
  193.130 ns (2 allocations: 192 bytes)

julia> @btime parent($h) .+ parent($h);
  94.443 ns (1 allocation: 160 bytes)
```

A `SpinMatrix` will get converted to a `HalfIntArray` on broadcasting. This behaviour might change in the future.

```julia
julia> s = SpinMatrix(reshape(1:4,2,2),half(1))
2×2 SpinMatrix(reshape(::UnitRange{Int64}, 2, 2), 1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 1  3
 2  4

julia> s .+ s
2×2 HalfIntArray(::Array{Int64,2}, -1/2:1/2, -1/2:1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 2  6
 4  8

# It's possible to recreate the wrapper
julia> s .+ s |> SpinMatrix
2×2 SpinMatrix(::Array{Int64,2}, 1/2) with eltype Int64 with indices -1/2:1/2×-1/2:1/2:
 2  6
 4  8
```

# Comparison with [OffsetArrays](https://github.com/JuliaArrays/OffsetArrays.jl)

A `HalfIntArray` with integer axes is equivalent to an `OffsetArray`, except that a `HalfIntArray` is somewhat less performant when it comes to indexing.

```julia
julia> using OffsetArrays

julia> oa = OffsetArray(reshape(1:9,3,3), -1:1, -1:1)
3×3 OffsetArray(reshape(::UnitRange{Int64}, 3, 3), -1:1, -1:1) with eltype Int64 with indices -1:1×-1:1:
 1  4  7
 2  5  8
 3  6  9

julia> h = HalfIntArray(parent(oa), axes(oa)...)
3×3 HalfIntArray(reshape(::UnitRange{Int64}, 3, 3), -1:1, -1:1) with eltype Int64 with indices -1:1×-1:1:
 1  4  7
 2  5  8
 3  6  9

julia> h == oa
true
```
using HalfIntegerArrays
using Test
using LinearAlgebra
using HalfIntegers
using OffsetArrays

import HalfIntegerArrays: IdOffsetRange, OneTo, offset, parentindex

@test isempty(Test.detect_ambiguities(HalfIntegerArrays, Base, Core))
@test isempty(Test.detect_ambiguities(HalfIntegerArrays, OffsetArrays))

@testset "Constructor" begin
	@testset "parent index" begin
	    @test HalfIntegerArrays.to_parentindices((1,2),()) == (1,1)
	    @test HalfIntegerArrays.parentindex(IdOffsetRange(1:10,2), 3:2:9) == 1:2:7
	    @test HalfIntegerArrays.parentindex(1:10, 3:2:9) == 3:2:9
	    @test HalfIntegerArrays.parentindex(IdOffsetRange(1:10, 1), 2) == 1
	    @test HalfIntegerArrays.parentindex(OffsetArrays.IdOffsetRange(1:10, 1), 2) == 1
	    @test HalfIntegerArrays.parentindex(IdOffsetRange(1:10, 1), 11) == 10
	    @test HalfIntegerArrays.parentindex(OffsetArrays.IdOffsetRange(1:10, 1), 11) == 10
	end

	@testset "HalfIntArray" begin
		function testHIA(T,N,h,arr,j)
			@test h isa HalfIntArray{T,N,Array{T,N}}
			@test h.offsets == Tuple(-j-1 for i=1:N)
			@test axes(h) == Tuple(-j:j for i=1:N)
			@test axes(h,1) == -j:j
			@test axes(h,N+1) == 1:1
			@test size(h) == Tuple(2j+1 for i=1:N)
			@test size(h,1) == 2j+1
			@test size(h,N+1) == 1
			@test eltype(h) == T
			@test HalfIntegerArrays.unwraphalfint(h) === arr
			@test HalfIntegerArrays.unwraphalfint(h,parent(h)) === arr

			@test HalfIntArray(h,Tuple(0 for i=1:N)) == h
			@test HalfIntArray(h,Tuple(half(0) for i=1:N)) == h
		end
		function testHIAfromArray(j)
			for T in [Float64,ComplexF64]
				arr = zeros(T,Int(2j+1),Int(2j+1))
				h = HalfIntArray(arr, (HalfInt(-j-1), HalfInt(-j-1)))
				testHIA(T,2,h,arr,j)
				h = HalfIntArray(arr, HalfInt(-j-1), HalfInt(-j-1))
				testHIA(T,2,h,arr,j)
				h = HalfIntArray(arr, -j:j, -j:j)
				testHIA(T,2,h,arr,j)
			
				arr = zeros(T,Int(2j+1))
				h = HalfIntArray(arr, (HalfInt(-j-1),))
				testHIA(T,1,h,arr,j)
				h = HalfIntArray(arr, HalfInt(-j-1))
				testHIA(T,1,h,arr,j)
				h = HalfIntArray(arr, -j:j)
				testHIA(T,1,h,arr,j)
			end
		end

		for j in [0,1//2,1]
			testHIAfromArray(j)
		end

		# zero dim arrays
		arr = zeros()
		h = HalfIntArray(arr,())
		@test h == HalfIntArray(arr)
		@test axes(h) == ()
		@test Base.axes1(h) == Base.OneTo(1)
		@test ndims(h) == 0
		@test HalfIntArray(h,()) == h
		@test length(h) == 1
		@test !isempty(h)

		h = HalfIntArray{Float64}(undef)
		@test length(h) == 1
		@test !isempty(h)

		# Empty arrays
		h1 = HalfIntArray{Float64}(undef, 0:-1)
		h2 = HalfIntArray{Float64}(undef, 0:-1, 0:-1)
		for (i,h) in enumerate([h1,h2])
			@test ndims(h) == i
			@test isempty(h)
			@test length(h) == 0
		end

		h = HalfIntArray(rand(2,2),0:1,0:1)
		@test isassigned(h,1,1)
		@test isassigned(h,HalfInt(1),HalfInt(1))
		@test isassigned(h,HalfInt(1),1)
		@test isassigned(h,1,HalfInt(1))

		@testset "Initializers" begin
			function testnothingmissing(T)
				h = HalfIntArray{T}(T(), -1:1)
				@test eltype(h) === T
				@test ndims(h) == 1
				@test axes(h) == (-1:1,)
				h = HalfIntArray{T}(T(), (-1:1,))
				@test eltype(h) === T
				@test ndims(h) == 1
				@test axes(h) == (-1:1,)
				h = HalfIntArray{T,1}(T(), -1:1)
				@test eltype(h) === T
				@test ndims(h) == 1
				@test axes(h) == (-1:1,)
				h = HalfIntArray{T,1}(T(), (-1:1,))
				@test eltype(h) === T
				@test ndims(h) == 1
				@test axes(h) == (-1:1,)
			end

			for T in [Nothing,Missing]
				testnothingmissing(T)
			end

			h = HalfIntArray{Float64}(undef, -1:1)
			@test parent(h) isa Array{Float64,1}
			@test axes(h) == (-1:1,)
			h = HalfIntArray{ComplexF64,2}(undef, -1:1, 0:2)
			@test parent(h) isa Array{ComplexF64,2}
			@test axes(h) == (-1:1, 0:2)
		end

		@testset "OffsetArray" begin
			oa = OffsetArray(reshape(1:4,2,2),2:3,2:3)
			h = HalfIntArray(oa, -half(1):half(1), -half(1):half(1))
			@test parent(h) === oa
			@test h.offsets == (-half(5),-half(5))

			h2 = HalfIntArray(parent(oa),axes(oa))
			@test parent(h2) === parent(oa)
			@test axes(h2) == axes(oa)

			oa2 = OffsetArray(parent(h2),UnitRange{Int}.(axes(h2)))
			@test parent(h2) === parent(oa2) === parent(oa)
			@test axes(h2) == axes(oa2)
		end
	end
	@testset "SpinMatrix" begin
		function testSM(T,N,h,arr,j)
			@test h isa SpinMatrix{T,Array{T,N}}
			@test h.j == j
			@test axes(h) == (-j:j,-j:j)
			@test axes(h,1) == -j:j
			@test axes(h,2) == -j:j
			@test axes(h,3) == 1:1
			@test size(h) == (2j+1,2j+1)
			@test size(h,1) == 2j+1
			@test size(h,2) == 2j+1
			@test size(h,3) == 1
			@test eltype(h) == T
			@test HalfIntegerArrays.unwraphalfint(h) === arr
			@test HalfIntegerArrays.unwraphalfint(h,parent(h)) === arr
			@test SpinMatrix(h) === h
			@test SpinMatrix(parent(h), h.j) == h
		end
		function testSMfromArray(j)
			for T in [Float64,ComplexF64]
				arr = zeros(T,Int(2j+1),Int(2j+1))
				h = SpinMatrix(arr, j)
				testSM(T,2,h,arr,j)

				h = SpinMatrix(arr)
				testSM(T,2,h,arr,j)			
			end
		end

		for j in [0,1//2,1]
			testSMfromArray(j)
		end
		@test_throws Exception testSMfromArray(-1)

		@testset "Initializers" begin
			function testnothingmissing(T)
				h = SpinMatrix{T}(T(), 2)
				@test eltype(h) === T
				@test ndims(h) == 2
				@test axes(h) == (-2:2,-2:2)

				h = SpinMatrix{T}(T(), 1//2)
				@test eltype(h) === T
				@test ndims(h) == 2
				@test axes(h) == (-1//2:1//2,-1//2:1//2)
			end

			for T in [Nothing,Missing]
				testnothingmissing(T)
			end

			h = SpinMatrix{ComplexF64}(undef, 1)
			@test parent(h) isa Array{ComplexF64,2}
			@test axes(h) == (-1:1, -1:1)

			@test_throws ArgumentError SpinMatrix{ComplexF64}(undef, -1)
		end

		@testset "OffsetArray" begin
			oa = OffsetArray(reshape(1:4,2,2),1:2,1:2)
			h = SpinMatrix(oa, half(1))
			@test parent(h) === oa

			oa = OffsetArray(reshape(1:4,2,2),2:3,2:3)
			@test_throws ArgumentError SpinMatrix(oa, half(1))

			s = SpinMatrix(parent(oa))
			@test s.j == half(1)
			@test axes(s) == (-half(1):half(1),-half(1):half(1))

			oa = OffsetArray(reshape(1:9,3,3),-1:1,-1:1)
			s = SpinMatrix(parent(oa))
			oa2 = OffsetArray(parent(s),UnitRange{Int}.(axes(s)))
			@test parent(oa2) === parent(oa) === parent(s)
			@test axes(oa2) == axes(oa) == axes(s)
		end
	end
	@testset "HalfIntArray ⇆ SpinMatrix" begin
		s = SpinMatrix(rand(5,5))
		h = HalfIntArray(parent(s),axes(s))
		@test axes(h) == axes(s)
		@test parent(h) == parent(s)

		s′ = SpinMatrix(h)
		@test s′ === s

		h′ = HalfIntArray(parent(s′),axes(s′))
		@test h′ === h

		# Ignore the axes of a HalfIntArray
		h = HalfIntArray(rand(2,2),4:5,4:5)
		s = SpinMatrix(h)
		@test parent(s) == parent(h)
		@test s.j == half(1)

		# Matrices must be square
		h = HalfIntArray(rand(2,3),1:2,1:3)
		@test_throws ArgumentError SpinMatrix(h)
	end
	@testset "offset" begin
		of = HalfIntegerArrays.offset(1:3,2) 
		@test of === HalfInt(0)
		of = HalfIntegerArrays.offset(2:3,2)
		@test of === HalfInt(-1)
	end
	@testset "unwraphalfint" begin
		h = HalfIntArray{Float64}(undef,-1:1,-1:1)
		a = parent(h)
		@test HalfIntegerArrays.unwraphalfint(h,a) === a
	end
	@testset "IndexStyle and parenttype" begin
		h = HalfIntArray{Float64}(undef,-1:1,-1:1)
		a = parent(h)
		@test Base.IndexStyle(typeof(h)) == IndexLinear()
		@test HalfIntegerArrays.parenttype(h) == typeof(a)
		@test HalfIntegerArrays.parenttype(typeof(h)) == typeof(a)

		s = SpinMatrix{ComplexF64}(undef,2)
		a = parent(s)
		@test Base.IndexStyle(typeof(s)) == IndexLinear()
		@test HalfIntegerArrays.parenttype(s) == typeof(a)
		@test HalfIntegerArrays.parenttype(typeof(s)) == typeof(a)        
	end
	@testset "IdOffsetRange" begin
		function tests(r,p,f,l,o)
			@test parent(r) == first(p):last(p)
			@test first(r) === HalfInt(f)
			@test last(r) === HalfInt(l)
			ax = IdOffsetRange(Base.OneTo(p),HalfInt(o))
			@test Base.axes1(r) === ax
			@test Base.axes(r) === (ax,)
		end
		r = IdOffsetRange(1:3)
		tests(r,1:3,1,3,0)
		@test isassigned(r,1)
		@test isassigned(r,1,1)

		r = IdOffsetRange(1:3, HalfInt(2))
		tests(r,1:3,3,5,2)
		@test eltype(r) == HalfInt

		r = IdOffsetRange{HalfInt,UnitRange{Int}}(1:3, HalfInt(2))
		tests(r,1:3,3,5,2)
		
		r = IdOffsetRange{HalfInt,UnitRange{Int}}(1:3, 2)
		tests(r,1:3,3,5,2)

		r = IdOffsetRange{HalfInt,Base.OneTo{Int}}(1:3, 2)
		tests(r,1:3,3,5,2)

		r = IdOffsetRange{HalfInt}(1:3, HalfInt(2))
		tests(r,1:3,3,5,2)

		r2 = IdOffsetRange(r)
		@test r2 === r

		r2 = typeof(r)(r)
		@test r2 === r

		r = IdOffsetRange{HalfInt,UnitRange{Int}}(1:3, HalfInt(2))
		r2 = typeof(r)(r)
		@test r2 === r

		r2 = IdOffsetRange{HalfInt}(r)
		@test r2 === r

		r = IdOffsetRange(Base.OneTo(3),HalfInt(0))
		r2 = IdOffsetRange{HalfInt,UnitRange{Int}}(r)
		@test parent(r2) isa UnitRange{Int}
		@test parent(r2) == parent(r)
		@test r2.offset == r.offset
	end
	@testset "OneTo" begin
		r = OneTo(3)
		@test first(r) === HalfInt(1)
		@test r.stop === HalfInt(3)
		@test step(r) === HalfInt(1)
		@test length(r) === 3

		r = OneTo(HalfInt(1):HalfInt(2))
		@test first(r) === HalfInt(1)
		@test r.stop === HalfInt(2)
		@test step(r) === HalfInt(1)
		@test length(r) === 2

		r = OneTo(HalfInt(1):half(5))
		@test first(r) === HalfInt(1)
		@test r.stop === HalfInt(2)
		@test step(r) === HalfInt(1)
		@test length(r) === 2

		@test_throws ArgumentError OneTo(HalfInt(3):HalfInt(5))

		@test OneTo(1:3) === OneTo(3)
		@test OneTo(Base.OneTo(3)) === OneTo(3)
		@test OneTo{HalfInt32}(Base.OneTo(3)) === OneTo{HalfInt32}(3)
		@test OneTo(OneTo(3)) === OneTo(3)
		@test OneTo{HalfInt}(OneTo(3)) === OneTo(3)
		@test OneTo{HalfInt32}(OneTo(3)) === OneTo{HalfInt32}(3)

		@test Base.OneTo{Int}(OneTo(3)) === Base.OneTo(3)

		@test convert(AbstractUnitRange{HalfInt32}, OneTo(3)) === OneTo{HalfInt32}(3)
		@test convert(AbstractUnitRange{HalfInt32}, Base.OneTo(3)) === OneTo{HalfInt32}(3)

		@test convert(AbstractUnitRange{Int}, OneTo(3)) === Base.OneTo(3)
	end
end

@testset "indexing" begin
	@testset "LinearIndices" begin
		@testset "Constructor" begin
		    l = LinearIndices((1:2,1:2))
		    lininds = LinearIndicesHalfInt(l,(HalfInt(1),HalfInt(2)))
		    @test lininds.lininds == l
		    @test lininds.offsets == (HalfInt(1),HalfInt(2))
		    @test eachindex(lininds) == eachindex(l)
		    @test eachindex(IndexLinear(),lininds) == eachindex(IndexLinear(),l)
		    @test IndexStyle(typeof(lininds)) == IndexLinear()
		    
		    l = LinearIndices((1:2,1:2))
		    lininds = LinearIndicesHalfInt(l,(1,0))
		    @test lininds.lininds == l
		    @test axes(lininds) == (axes(l,1) .+ 1, axes(l,2))
		    @test lininds.offsets == (HalfInt(1),HalfInt(0))
		    
		    l = LinearIndices((1:2,1:2))
		    lininds = LinearIndicesHalfInt(l)
		    @test lininds.lininds == l
		    @test lininds.offsets == (HalfInt(0),HalfInt(0))
		    @test axes(lininds) == axes(l) 
		end
		h = HalfIntArray(rand(2,2), -half(1):half(1), -half(1):half(1))
		lininds = LinearIndicesHalfInt(h)
		for (i,v) in enumerate(lininds)
			@test lininds[i] == v == i
		end
		cartinds = CartesianIndicesHalfInt(lininds)
		for (i,I) in enumerate(cartinds)
			@test lininds[I] == lininds[Tuple(I)...] == i
		end
		@test first(lininds) == 1
		@test last(lininds) == length(lininds) == length(h)
		
		@test lininds[lininds] == lininds
		@test lininds[cartinds] == lininds
		@test h[lininds] == h

		h = HalfIntArray(rand(2), -half(1):half(1))
		lininds = LinearIndicesHalfInt(h)
		for v in lininds
			@test lininds[v] == v
		end
		cartinds = CartesianIndicesHalfInt(lininds)
		for I in cartinds
			@test lininds[I] == Tuple(I)[1]
		end
		@test first(lininds) == first(axes(h,1))
		@test last(lininds) == last(axes(h,1))

		@test lininds[lininds] == lininds
		@test lininds[cartinds] == lininds
		@test h[lininds] == h

		@test_throws Exception LinearIndices(h)

		h = HalfIntArray(rand(2), 0:1)
		linindsHI = LinearIndicesHalfInt(h)
		linindsHIP = parent(linindsHI)
		lininds = LinearIndices(h)
		@test parent(linindsHIP) == LinearIndices(1:length(linindsHIP))
		@test linindsHI.offsets == linindsHIP.offsets == h.offsets
		@test linindsHI[linindsHI] == linindsHI
		@test h[linindsHI] == h
		@test h[lininds] == h
		@test parent(h)[linindsHI.lininds] == parent(h)
		for (i,LI) in enumerate(lininds)
			@test linindsHI[LI] == lininds[LI] == linindsHIP[i] == LI
		end

		r = axes(linindsHI,1)
		@test linindsHI[r] == r
		r = first(linindsHI):last(linindsHI)
		@test linindsHI[r] == r
		@test linindsHIP[1:length(linindsHIP)] == r

		h = HalfIntArray(rand(2,2), 0:1, 0:1)
		linindsHI = LinearIndicesHalfInt(h)
		linindsHIP = parent(linindsHI)
		lininds = LinearIndices(h)
		@test parent(linindsHIP) == LinearIndices((1:2,1:2))
		@test linindsHI.offsets == linindsHIP.offsets == h.offsets
		@test linindsHI[linindsHI] == linindsHI
		@test h[linindsHI] == h
		@test h[lininds] == h
		@test parent(h)[linindsHI.lininds] == parent(h)
		for (i,LI) in enumerate(linindsHI)
			@test linindsHI[LI] == lininds[LI] == linindsHIP[i] == LI
		end
		for (i,LI) in enumerate(lininds)
			@test linindsHI[LI] == lininds[LI] == linindsHIP[i] == LI
		end

		@test linindsHI[1:length(linindsHI)] == 1:length(linindsHI)

		h = HalfIntArray(rand(2,2), -half(1):half(1), -half(1):half(1))
		linindsHI = LinearIndicesHalfInt(h)
		linindsHIP = parent(linindsHI)
		@test parent(linindsHIP) == LinearIndices((1:2,1:2))
		@test linindsHI.offsets == linindsHIP.offsets == h.offsets
		@test h[linindsHI] == h
		for (i,LI) in enumerate(linindsHI)
			@test linindsHI[LI] == linindsHIP[i] == LI
		end

		@test linindsHI[1:length(linindsHI)] == 1:length(linindsHI)
	end
	@testset "CartesianIndices" begin
		@testset "CartesianIndexHalfInt" begin
			function test(c::CartesianIndexHalfInt{N}, p, of) where {N}
				@test parent(c) == CartesianIndex(p)
				@test c.offsets == of
				@test Tuple(c) == map(+,p,of)
				@test length(c) == N
				for i in 1:N
					@test c[i] == p[i] + of[i]
				end
				@test c == c
				@test hash(c) == hash(c)
				@test isequal(c,c)
				@test -c == CartesianIndexHalfInt(map(-,p),map(-,of))
				@test c + c == CartesianIndexHalfInt(map(+,p,p),map(+,of,of))
				@test c - c == CartesianIndexHalfInt(map(-,p,p),map(-,of,of))
				@test c*2 == CartesianIndexHalfInt(map(x->2x,p),map(x->2x,of))
				@test 2c == CartesianIndexHalfInt(map(x->2x,p),map(x->2x,of))
				@test_throws ErrorException iterate(c)
				@test min(c,c) == c
				@test max(c,c) == c
			end

			c = CartesianIndexHalfInt()
			test(c,(),())
			@test length(c) == 0
			@test eltype(c) == HalfInt

			c = CartesianIndexHalfInt(())
			test(c,(),())

			c = CartesianIndexHalfInt((1,))
			test(c,(1,),(HalfInt(0),))
			@test length(c) == 1
			@test zero(c) == CartesianIndexHalfInt((0,))
			@test oneunit(c) == CartesianIndexHalfInt((1,))
			@test hash(c) != hash(parent(c))

			@test convert(Int,c) == 1
			@test convert(Tuple,c) == (1,)

			c = CartesianIndexHalfInt(1)
			test(c,(1,),(HalfInt(0),))

			c = CartesianIndexHalfInt(0.5)
			test(c,(0,),(half(1),))

			c = CartesianIndexHalfInt{1}(1)
			test(c,(1,),(HalfInt(0),))

			c = CartesianIndexHalfInt{3}(1)
			test(c,(1,1,1),(HalfInt(0),HalfInt(0),HalfInt(0)))

			c = CartesianIndexHalfInt{3}()
			test(c,(1,1,1),(HalfInt(0),HalfInt(0),HalfInt(0)))

			c = CartesianIndexHalfInt{3}(half(1))
			test(c,(0,0,0),(half(1),HalfInt(1),HalfInt(1)))

			c = CartesianIndexHalfInt{3}(0.5)
			test(c,(0,0,0),(half(1),HalfInt(1),HalfInt(1)))
			
			c = CartesianIndexHalfInt{3}((0.5,))
			test(c,(0,0,0),(half(1),HalfInt(1),HalfInt(1)))

			c = CartesianIndexHalfInt((1,),(-half(1),))
			test(c,(1,),(-half(1),))

			c = CartesianIndexHalfInt((1,2),(-half(1),half(0)))
			test(c,(1,2),(-half(1),half(0)))
			@test zero(c) == CartesianIndexHalfInt((0,0))
			@test oneunit(c) == CartesianIndexHalfInt((1,1))

			c = CartesianIndexHalfInt((-half(1),half(0)))
			test(c,(0,0),(-half(1),half(0)))

			c = CartesianIndexHalfInt(-half(1),half(0))
			test(c,(0,0),(-half(1),half(0)))

			c = CartesianIndexHalfInt{2}(-half(1),half(0))
			test(c,(0,0),(-half(1),half(0)))

			c = CartesianIndexHalfInt{2}((-half(1),half(0)))
			test(c,(0,0),(-half(1),half(0)))

			c = CartesianIndexHalfInt{2}(())
			test(c,(1,1),(HalfInt(0),HalfInt(0)))

			@test isless(CartesianIndexHalfInt((1,1)),CartesianIndexHalfInt((2,1)))

			@test CartesianIndexHalfInt(1,CartesianIndexHalfInt(2,3)) == CartesianIndexHalfInt(1,2,3)
			@test CartesianIndexHalfInt((1,CartesianIndexHalfInt(2,3))) == CartesianIndexHalfInt(1,2,3)

			@test HalfIntegerArrays._flatten() == ()
			@test HalfIntegerArrays.flatten(()) == ()
			@test HalfIntegerArrays.flatten((1,)) === (HalfInt(1),)
			@test HalfIntegerArrays.flatten((half(1),)) == (half(1),)
			@test HalfIntegerArrays.flatten((0.5,)) == (0.5,)
			c = CartesianIndexHalfInt(-half(1),half(0))
			@test HalfIntegerArrays.flatten((c,)) == (-half(1),-half(0))

			@test Base.index_ndims(CartesianIndexHalfInt(1,1)) == (true,true)
			@test Base.index_ndims(CartesianIndexHalfInt(1)) == (true,)
			@test Base.index_ndims(CartesianIndexHalfInt(1,1,1)) == (true,true,true)
		end
		@testset "CartesianIndicesHalfInt" begin
			function testpof(c, p, of)
				@test c.cartinds == CartesianIndices(p)
				@test c.offsets == of
			end

			@test HalfIntegerArrays._in(true,(),(),())
			@test !HalfIntegerArrays._in(false,(),(),())

			c = CartesianIndicesHalfInt()
			testpof(c, (), ())
			@test c isa CartesianIndicesHalfInt{0,Tuple{}}
			@test Base.IndexStyle(typeof(c)) == IndexCartesian()
			@test Base.IteratorSize(typeof(c)) == Base.HasShape{0}()
			@test length(c) == 1
			@test size(c) == ()
			@test axes(c) == ()
			@test collect(c) == fill(CartesianIndexHalfInt())
			@test ndims(c) == 0
			@test eltype(c) == CartesianIndexHalfInt{0}
			@test iterate(c) == (CartesianIndexHalfInt(), true)
			@test CartesianIndexHalfInt() in c
			@test first(c) == CartesianIndexHalfInt()
			@test last(c) == CartesianIndexHalfInt()

			c = CartesianIndicesHalfInt(CartesianIndices((1:3,2:3)),(half(0),half(0)))
			testpof(c, (1:3,2:3), (half(0),half(0)))
			@test c isa CartesianIndicesHalfInt{2,Tuple{UnitRange{Int},UnitRange{Int}}}
			@test Base.IndexStyle(typeof(c)) == IndexCartesian()
			@test Base.IteratorSize(typeof(c)) == Base.HasShape{2}()
			@test size(c) == (3,2)
			@test length(c) == 6
			@test axes(c) === (IdOffsetRange(Base.OneTo(3),half(0)),IdOffsetRange(Base.OneTo(2),half(0)))
			@test ndims(c) == 2
			@test eltype(c) == CartesianIndexHalfInt{2}
			@test first(c) == CartesianIndexHalfInt((1,2),(half(0),half(0)))
			@test last(c) == CartesianIndexHalfInt((3,3),(half(0),half(0)))
			@test iterate(c) == (first(c),parent(first(c)))
			for (ind,cind) in enumerate(c)
				@test c[ind] == cind
			end
			cind = CartesianIndex(3,2)
			cindh = CartesianIndexHalfInt(cind,(half(0),half(0)))
			@test iterate(c, CartesianIndex(2,2)) == (cindh,cind)
			@test cindh in c
			@test HalfIntegerArrays._iterate(c, nothing) === nothing

			h = HalfIntArray(rand(2,2),-half(1):half(1),-half(1):half(1))
			c = CartesianIndicesHalfInt(axes(h))
			@test c isa CartesianIndicesHalfInt{2,Tuple{Base.OneTo{Int},Base.OneTo{Int}}}
			@test size(c) == (2,2)
			@test length(c) == 4
			@test axes(c) === axes(h)

			r = (-half(1):half(1),-half(1):half(1))
			c = CartesianIndicesHalfInt(r)
			@test c isa CartesianIndicesHalfInt{2,Tuple{Base.OneTo{Int},Base.OneTo{Int}}}
			@test size(c) == (2,2)
			@test length(c) == 4
			@test axes(c) === (IdOffsetRange(Base.OneTo(2),-half(3)),IdOffsetRange(Base.OneTo(2),-half(3)))

			h = HalfIntArray(rand(2,2),-half(1):half(1),-half(1):half(1))
			lininds = LinearIndicesHalfInt(h)
			cartinds = CartesianIndicesHalfInt(h)
			@test cartinds[lininds] == cartinds
			@test h[cartinds] == h
			@test_throws Exception CartesianIndices(h)

			h = HalfIntArray(rand(2,2),1:2,1:2)
			cindsHI = CartesianIndicesHalfInt(h)
			cinds = CartesianIndices(h)
			cindsp = CartesianIndices(parent(h))
			@test cindsHI[cindsHI] == cindsHI
			@test cindsHI[cinds] == cindsHI
			@test parent(cindsHI)[cindsp] == parent(cindsHI)
			@test parent(parent(cindsHI)) == cindsp
			@test h[cindsHI] == h
			@test h[cinds] == h
			@test parent(h)[cindsHI.cartinds] == parent(h)
			for (i,CI) in enumerate(cindsHI)
				@test cindsHI[i] == cindsHI[CI]
			end
			@test h[CartesianIndexHalfInt(()),1,1] == h[1,1]
			@test h[CartesianIndicesHalfInt(()),1,1][] == h[1,1]
			@test Base.checkbounds_indices(Bool, axes(h), (CartesianIndicesHalfInt(()),1,1))

			h = HalfIntArray(rand(3), -1:1)
			c = CartesianIndicesHalfInt(h)
			@test h[c] == h
			@test h[CartesianIndicesHalfInt(()),-1][] == h[-1]
			@test Base.checkbounds_indices(Bool, axes(h), (CartesianIndicesHalfInt(()),-1))

			h = HalfIntArray(zeros())
			c = CartesianIndicesHalfInt(h)
			@test h[c] == h
			@test h[c,1] == h
			h′ = h[CartesianIndicesHalfInt((1:1,1:1)),1] 
			@test axes(h′) == (1:1,1:1)
			@test h′[1,1] == h[]
			@test Base.checkbounds_indices(Bool, axes(h), (CartesianIndicesHalfInt((1:1,1:1)),1))

			@test Base.index_ndims(CartesianIndicesHalfInt((1:1,1:1))) == (true,true)
			@test Base.index_ndims(CartesianIndicesHalfInt((1:1,))) == (true,)
			@test Base.index_ndims(CartesianIndicesHalfInt((1:2,1:1,1:1))) == (true,true,true)
		end
		@testset "to_indices and bounds" begin
			h = HalfIntArray(rand(3,3),-1:1,-1:1)
			inds = (HalfInt(1),HalfInt(-1))
			@test to_indices(h,(HalfInt(1),CartesianIndexHalfInt(-1))) === inds
			@test to_indices(h,(CartesianIndexHalfInt(1),HalfInt(-1))) === inds
			@test to_indices(h,(HalfInt(1),HalfInt(-1))) === inds
			@test to_indices(h,(CartesianIndexHalfInt(1),CartesianIndexHalfInt(-1))) === inds
			@test to_indices(h,(CartesianIndexHalfInt(1,-1),)) === inds

			cinds = CartesianIndicesHalfInt((1:1,1:1))
			@test to_indices(h,(cinds,)) == (cinds,)
			@test to_indices(h,axes(h),(cinds,)) == (cinds,)

			@test checkbounds(Bool, h, CartesianIndexHalfInt(1,1))
			@test Base.checkbounds_indices(Bool, axes(h), (CartesianIndexHalfInt(1),1))

			h2 = HalfIntArray(rand(3),-1:1)
			@test Base.checkbounds_indices(Bool, axes(h2), (CartesianIndexHalfInt(1),))
			@test Base.checkbounds_indices(Bool, axes(h2), (CartesianIndexHalfInt(1),1))
			
			@test Base.checkbounds_indices(Bool, (), (CartesianIndexHalfInt(1),1))
		end
	end

	@testset "util" begin
		@test HalfIntegerArrays._maybetail(()) == ()
		@test HalfIntegerArrays._maybetail((1,)) == ()
		@test HalfIntegerArrays._maybetail((1,2)) == (2,)
	end
	
	@testset "eachindex" begin
		h = HalfIntArray(rand(2,2),-half(1):half(1),-half(1):half(1))
		@test eachindex(h) === Base.OneTo(4)
		@test eachindex(IndexLinear(),h) === Base.OneTo(4) 
		@test eachindex(IndexCartesian(),h) === CartesianIndicesHalfInt(axes(h))
	end
	
	js = [0,1//2,1]
	function testallequal(h, val)
		@test all(h .== val)
		@test all(parent(h) .== val)
	end
	@testset "HalfIntArray" begin
		@testset "getindex" begin
			@testset "2D" begin
				function testgetindexscalar(j)
					N = Int(twice(j)) + 1
					a = reshape([i for i=1:N^2],N,N)
					h = HalfIntArray(a,-j:j,-j:j)
					for i in eachindex(a,h)
						@test h[i] == a[i]
						@test h[float(i)] == a[i]
						@test_throws ArgumentError h[float(i) + 0.5]
						@test h[Rational(i)] == a[i]
					end
					for (j,aj) in zip(axes(h,2),axes(a,2)), 
						(i,ai) in zip(axes(h,1),axes(a,1))
						@test h[i,j] == a[ai,aj]
						@test h[i,float(j)] == a[ai,aj]
						@test h[float(i),j] == a[ai,aj]
						@test h[float(i),float(j)] == a[ai,aj]
						@test h[Rational(i),j] == a[ai,aj]
						@test h[i,Rational(j)] == a[ai,aj]
						@test h[Rational(i),Rational(j)] == a[ai,aj]
						@test h[Rational(i),float(j)] == a[ai,aj]
						@test h[float(i),Rational(j)] == a[ai,aj]
					end

					inds = (first.(axes(h))...,)
					@test h[inds...,HalfInt(1)] == h[inds...]
					@test h[inds...,1] == h[inds...]
					@test h[inds...,1,1] == h[inds...]
					@test h[inds...,1.0] == h[inds...]
					@test h[inds...,1//1] == h[inds...]
				end
				function testgetindexcolon(j)
					N = Int(twice(j)) + 1
					a = reshape([i for i=1:N^2],N,N)
					h = HalfIntArray(a,-j:j,-j:j)
					h′ = h[:]
					@test axes(h′) == (Base.OneTo(N^2),)
					for i in eachindex(h)
						@test h′[i] == h[i]
					end

					h′ = h[:,:]
					@test typeof(h′) == typeof(h)
					@test axes(h′) == axes(h)
					for i in eachindex(h,h′)
						@test h′[i] == h[i]
					end

					for i2 in axes(h,2)
						h′ = h[:,i2]
						@test axes(h′,1) == axes(h,1)
						for i1 in axes(h,1)
							@test h′[i1] == h[i1,i2]
						end
						h′ = h[:,float(i2)]
						@test axes(h′,1) == axes(h,1)
						for i1 in axes(h,1)
							@test h′[i1] == h[i1,i2]
						end
						h′ = h[:,Rational(i2)]
						@test axes(h′,1) == axes(h,1)
						for i1 in axes(h,1)
							@test h′[i1] == h[i1,i2]
						end
					end

					for i1 in axes(h,1)
						h′ = h[i1,:]
						@test axes(h′,1) == axes(h,2)
						for i2 in axes(h,2)
							@test h′[i2] == h[i1,i2]
						end
						h′ = h[float(i1),:]
						@test axes(h′,1) == axes(h,2)
						for i2 in axes(h,2)
							@test h′[i2] == h[i1,i2]
						end
						h′ = h[Rational(i1),:]
						@test axes(h′,1) == axes(h,2)
						for i2 in axes(h,2)
							@test h′[i2] == h[i1,i2]
						end
					end
				end
				function testgetindexrange(j)
					N = Int(twice(j)) + 1
					a = reshape([i for i=1:N^2],N,N)
					h = HalfIntArray(a,-j:j,-j:j)

					for i2 in axes(h,2)
						h′ = h[-j:j,i2]
						@test axes(h′,1) == Base.OneTo(N)
						for (ind,i1) in enumerate(axes(h,1))
							@test h′[ind] == h[i1,i2]
						end
						if VERSION >= v"1.2"
							h′ = h[Base.IdentityUnitRange(-j:j),i2]
							@test axes(h′,1) == -j:j
							for i1 in axes(h,1)
								@test h′[i1] == h[i1,i2]
							end
						end
						if isinteger(j)
							h′ = h[Base.OneTo(Int(j)),i2]
							for ind in axes(h′,1)
								@test h′[ind] == h[ind,i2]
							end
						end

						h′ = h[-j:j,float(i2)]
						@test axes(h′,1) == Base.OneTo(N)
						for (ind,i1) in enumerate(axes(h,1))
							@test h′[ind] == h[i1,i2]
						end
						h′ = h[-j:j,Rational(i2)]
						@test axes(h′,1) == Base.OneTo(N)
						for (ind,i1) in enumerate(axes(h,1))
							@test h′[ind] == h[i1,i2]
						end

						r = -j:2:j
						h′ = h[r,i2]
						@test axes(h′,1) == axes(r, 1)
						for (ind,i1) in enumerate(axes(h′,1))
							@test h′[ind] == h[r[ind],i2]
						end
					end

					for i1 in axes(h,1)
						h′ = h[i1,-j:j]
						@test axes(h′,1) == Base.OneTo(N)
						for (ind,i2) in enumerate(axes(h,2))
							@test h′[ind] == h[i1,i2]
						end
						if VERSION >= v"1.2"
							h′ = h[i1,Base.IdentityUnitRange(-j:j)]
							@test axes(h′,1) == -j:j
							for i2 in axes(h,2)
								@test h′[i2] == h[i1,i2]
							end
						end
						h′ = h[float(i1),-j:j]
						@test axes(h′,1) == Base.OneTo(N)
						for (ind,i2) in enumerate(axes(h,2))
							@test h′[ind] == h[i1,i2]
						end
						h′ = h[Rational(i1),-j:j]
						@test axes(h′,1) == Base.OneTo(N)
						for (ind,i2) in enumerate(axes(h,2))
							@test h′[ind] == h[i1,i2]
						end

						h′ = h[i1,-j:1:j]
						@test axes(h′,1) == Base.OneTo(N)
						for (ind,i2) in enumerate(axes(h,2))
							@test h′[ind] == h[i1,i2]
						end
					end
				end

				@testset "scalar" begin
					testgetindexscalar.(js)
				end
				@testset "colon" begin
					testgetindexcolon.(js)
				end
				@testset "range" begin
					testgetindexrange.(js)
				end
			end
			@testset "1D" begin
				function testgetindexscalar(j)
					N = Int(twice(j)) + 1
					a = [i for i=1:N]
					h = HalfIntArray(a,-j:j)
					for (i,ai) in zip(axes(h,1),axes(a,1))
						@test h[i] == a[ai]
					end
				end
				function testgetindexcolon(j)
					N = Int(twice(j)) + 1
					a = [i for i=1:N]
					h = HalfIntArray(a,-j:j)
					h′ = h[:]
					for (i,ai) in zip(axes(h′,1),axes(a,1))
						@test h′[i] == a[ai]
					end
				end
				function testgetindexrange(j)
					N = Int(twice(j)) + 1
					a = [i for i=1:N]
					h = HalfIntArray(a,-j:j)
					h′ = h[-j:j]
					for (i,ai) in zip(axes(h′,1),axes(a,1))
						@test h′[i] == a[ai]
					end
				end
				@testset "scalar" begin
					testgetindexscalar.(js)
				end
				@testset "colon" begin
					testgetindexcolon.(js)
				end
				@testset "range" begin
					testgetindexrange.(js)
				end
			end
		end
		@testset "setindex!" begin
			@testset "2D" begin
				function testsetindex!scalar(j)
					N = Int(twice(j)) + 1
					h = HalfIntArray(zeros(N,N),-j:j,-j:j)
					a = parent(h)
					for i in eachindex(a,h)
						val = rand()
						h[i] = val
						@test h[i] == a[i] == val
						val = rand()
						h[float(i)] = val
						@test h[i] == a[i] == val
						val = rand()
						h[Rational(i)] = val
						@test h[i] == a[i] == val
					end
					for (j,aj) in zip(axes(h,2),axes(a,2)), 
						(i,ai) in zip(axes(h,1),axes(a,1))

						val = rand()
						h[i,j] = val
						@test h[i,j] == val

						if isinteger(i) && isinteger(j)
							val = rand()
							h[Int(i),Int(j)] = val
							@test h[i,j] == val
						elseif isinteger(i)
							val = rand()
							h[Int(i),j] = val
							@test h[i,j] == val
						elseif isinteger(j)
							val = rand()
							h[i,Int(j)] = val
							@test h[i,j] == val
						end

						val = rand()
						h[Rational(i),j] = val
						@test h[i,j] == val

						val = rand()
						h[i,Rational(j)] = val
						@test h[i,j] == val
						
						val = rand()
						h[Rational(i),Rational(j)] = val
						@test h[i,j] == val
						
						val = rand()
						h[float(i),j] = val
						@test h[i,j] == val
						
						val = rand()
						h[i,float(j)] = val
						@test h[i,j] == val
						
						val = rand()
						h[float(i),float(j)] = val
						@test h[i,j] == val

						val = rand()
						h[float(i),Rational(j)] = val
						@test h[i,j] == val
						
						val = rand()
						h[Rational(i),float(j)] = val
						@test h[i,j] == val
					end

					inds = (first.(axes(h))...,HalfInt(1))
					h[inds...] = 4
					@test h[inds...] == 4
					
					inds = (first.(axes(h))...,1)
					h[inds...] = 5
					@test h[inds...] == 5
				end
				function testsetindex!colon(j)
					N = Int(twice(j)) + 1
					h = HalfIntArray(zeros(N,N),-j:j,-j:j)

					val = rand()
					h[:] .= val
					testallequal(h, val)

					for inds in ((:,:),(:,:,:),(:,:,:,:))
						val = rand()
						h[inds...] .= val
						testallequal(h, val)

						val = rand()
						h[inds...,1] .= val
						testallequal(h, val)

						val = rand()
						h[inds...,1,1] .= val
						testallequal(h, val)
					end

					for i2 in axes(h,2)
						val = rand()
						h[:,i2] .= val
						@test all(h[:,i2] .== val)
						
						val = rand()
						h[:,float(i2)] .= val
						@test all(h[:,i2] .== val)

						val = rand()
						h[:,Rational(i2)] .= val
						@test all(h[:,i2] .== val)

						val = rand()
						if isinteger(i2)
							h[:,Int(i2)] .= val
							@test all(h[:,i2] .== val)
						end
					end

					for i1 in axes(h,1)
						val = rand()
						h[i1,:] .= val
						@test all(h[i1,:] .== val)

						val = rand()
						h[float(i1),:] .= val
						@test all(h[i1,:] .== val)

						val = rand()
						h[Rational(i1),:] .= val
						@test all(h[i1,:] .== val)

						if isinteger(i1)
							val = rand()
							h[Int(i1),:] .= val
							@test all(h[i1,:] .== val)
						end
					end
				end
				function testsetindex!range(j)
					N = Int(twice(j)) + 1
					h = HalfIntArray(zeros(N,N),-j:j,-j:j)

					for i2 in axes(h,2)
						val = rand()
						h[-j:j,i2] .= val
						@test all(h[-j:j,i2] .== val)

						if VERSION >= v"1.2"
							val = rand()
							h[Base.IdentityUnitRange(-j:j),i2] .= val
							@test all(h[Base.IdentityUnitRange(-j:j),i2] .== val)
						end
						if isinteger(j)
							val = rand()
							h[Base.OneTo(Int(j)),i2] .= val
							@test all(h[Base.OneTo(Int(j)),i2] .== val)
						end

						val = rand()
						h[-j:j,float(i2)] .= val
						@test all(h[-j:j,i2] .== val)
						
						val = rand()
						h[-j:j,Rational(i2)] .= val
						@test all(h[-j:j,i2] .== val)
					end

					for i1 in axes(h,1)
						val = rand()
						h[i1,-j:j] .= val
						@test all(h[i1,-j:j] .== val)
						
						if VERSION >= v"1.2"
							val = rand()
							h[i1,Base.IdentityUnitRange(-j:j)] .= val
							@test all(h[i1,Base.IdentityUnitRange(-j:j)] .== val)
						end

						val = rand()
						h[float(i1),-j:j] .= val
						@test all(h[i1,-j:j] .== val)

						val = rand()
						h[Rational(i1),-j:j] .= val
						@test all(h[i1,-j:j] .== val)
					end
				end
				function testsetindex!mixed(j)
					N = Int(twice(j)) + 1
					h = HalfIntArray(zeros(N,N),-j:j,-j:j)

					val = rand()
					h[-j:j,:] .= val
					@test all(h[-j:j,:] .== val)
					
					val = rand()
					h[-j:j,:,:] .= val
					@test all(h[-j:j,:] .== val)
					
					val = rand()
					h[-j:j,:,1] .= val
					@test all(h[-j:j,:] .== val)
					
					val = rand()
					h[-j:j,:,1:1] .= val
					@test all(h[-j:j,:] .== val)

					val = rand()
					h[:,-j:j] .= val
					@test all(h[:,-j:j] .== val)
					
					val = rand()
					h[:,-j:j,1] .= val
					@test all(h[:,-j:j] .== val)

					val = rand()
					h[:,-j:j,1:1] .= val
					@test all(h[:,-j:j] .== val)
				end

				@testset "scalar" begin
					testsetindex!scalar.(js)
				end
				@testset "colon" begin
					testsetindex!colon.(js)
				end
				@testset "range" begin
					testsetindex!range.(js)
				end
				@testset "mixed" begin
					testsetindex!mixed.(js)
				end
			end
			@testset "1D" begin
				function testsetindex!scalar(j)
					N = Int(twice(j)) + 1
					h = HalfIntArray(zeros(N),-j:j)
					a = parent(h)
					for (i,ai) in zip(axes(h,1),axes(a,1))
						val = rand()
						h[i] = val
						@test h[i] == val
						@test a[ai] == val

						if isinteger(i)
							val = rand()
							h[i] = val
							@test h[Int(i)] == val
						end
					end
				end
				function testsetindex!colon(j)
					N = Int(twice(j)) + 1
					h = HalfIntArray(zeros(N),-j:j)
					a = parent(h)

					val = rand()
					h[:] .= val
					testallequal(h, val)

					for inds in ((:,:),(:,:,:),(:,:,:,:))
						val = rand()
						h[inds...] .= val
						testallequal(h, val)

						val = rand()
						h[inds...,1] .= val
						testallequal(h, val)
						
						val = rand()
						h[inds...,1,1] .= val
						testallequal(h, val)
					end
				end
				function testsetindex!range(j)
					N = Int(twice(j)) + 1
					h = HalfIntArray(zeros(N),-j:j)
					h[-j:j] .= 4
					@test all(h[-j:j] .== 4)

					h[-j:j,1] .= 4
					@test all(h[-j:j] .== 4)
				end
				function testsetindex!mixed(j)
					N = Int(twice(j)) + 1
					h = HalfIntArray(zeros(N),-j:j)

					val = rand()
					h[:,1:1] .= val
					testallequal(h, val)

					val = rand()
					h[-j:j,:] .= val
					testallequal(h, val)
				end
				@testset "scalar" begin
					testsetindex!scalar.(js)
				end
				@testset "colon" begin
					testsetindex!colon.(js)
				end
				@testset "range" begin
					testsetindex!range.(js)
				end
				@testset "mixed" begin
					testsetindex!mixed.(js)
				end
			end
		end
	end
	@testset "SpinMatrix" begin
		@testset "getindex" begin
			function testgetindexscalar(j)
				N = Int(twice(j)) + 1
				a = reshape([i for i=1:N^2],N,N)
				h = SpinMatrix(a,j)
				for i in eachindex(a,h)
					@test h[i] == a[i]
				end
				for (j,aj) in zip(axes(h,2),axes(a,2)), 
					(i,ai) in zip(axes(h,1),axes(a,1))
					@test h[i,j] == a[ai,aj]
					@test h[float(i),j] == a[ai,aj]
					@test h[i,float(j)] == a[ai,aj]
					@test h[float(i),float(j)] == a[ai,aj]
					@test h[Rational(i),j] == a[ai,aj]
					@test h[i,Rational(j)] == a[ai,aj]
					@test h[float(i),Rational(j)] == a[ai,aj]
					@test h[Rational(i),float(j)] == a[ai,aj]
					@test h[Rational(i),Rational(j)] == a[ai,aj]
				end
			end
			function testgetindexcolon(j)
				N = Int(twice(j)) + 1
				a = reshape([i for i=1:N^2],N,N)
				h = SpinMatrix(a,j)
				h′ = h[:]
				@test axes(h′) == (Base.OneTo(N^2),)
				for i in eachindex(h)
					@test h′[i] == h[i]
				end

				h′ = h[:,:]
				@test h′ isa HalfIntArray{eltype(h),2}
				@test axes(h′) == axes(h)
				for i in eachindex(h,h′)
					@test h′[i] == h[i]
				end

				for i2 in axes(h,2)
					h′ = h[:,i2]
					@test axes(h′,1) == axes(h,1)
					for i1 in axes(h,1)
						@test h′[i1] == h[i1,i2]
					end
				end

				for i1 in axes(h,1)
					h′ = h[i1,:]
					@test axes(h′,1) == axes(h,2)
					for i2 in axes(h,2)
						@test h′[i2] == h[i1,i2]
					end
				end
			end
			function testgetindexrange(j)
				N = Int(twice(j)) + 1
				a = reshape([i for i=1:N^2],N,N)
				h = SpinMatrix(a,j)
				for i2 in axes(h,2)
					h′ = h[-j:j,i2]
					@test axes(h′,1) == Base.OneTo(N)
					for (ind,i1) in enumerate(axes(h,1))
						@test h′[ind] == h[i1,i2]
					end
					if VERSION >= v"1.2"
						h′ = h[Base.IdentityUnitRange(-j:j),i2]
						@test axes(h′,1) == -j:j
						for i1 in axes(h,1)
							@test h′[i1] == h[i1,i2]
						end
					end
					if isinteger(j)
						h′ = h[Base.OneTo(Int(j)),i2]
						for ind in axes(h′,1)
							@test h′[ind] == h[ind,i2]
						end
					end
				end

				for i1 in axes(h,1)
					h′ = h[i1,-j:j]
					@test axes(h′,1) == Base.OneTo(N)
					for (ind,i2) in enumerate(axes(h,2))
						@test h′[ind] == h[i1,i2]
					end
					if VERSION >= v"1.2"
						h′ = h[i1,Base.IdentityUnitRange(-j:j)]
						@test axes(h′,1) == -j:j
						for i2 in axes(h,2)
							@test h′[i2] == h[i1,i2]
						end
					end

					h′ = h[float(i1),-j:j]
					@test axes(h′,1) == Base.OneTo(N)
					for (ind,i2) in enumerate(axes(h,2))
						@test h′[ind] == h[i1,i2]
					end
					h′ = h[Rational(i1),-j:j]
					@test axes(h′,1) == Base.OneTo(N)
					for (ind,i2) in enumerate(axes(h,2))
						@test h′[ind] == h[i1,i2]
					end
				end
			end
			function testgetindexrangecolon(j)
			end
			@testset "scalar" begin
				testgetindexscalar.(js)
			end
			@testset "colon" begin
				testgetindexcolon.(js)
			end
			@testset "range" begin
				testgetindexrange.(js)
			end
			@testset "range + colon" begin
				testgetindexrangecolon.(js)
			end
		end
		@testset "setindex!" begin
			function testsetindex!scalar(j)
				N = Int(twice(j)) + 1
				h = SpinMatrix(zeros(N,N),j)
				a = parent(h)
				for i in eachindex(a,h)
					val = rand()
					h[i] = val
					@test h[i] == val
					@test a[i] == val
				end
				for (j,aj) in zip(axes(h,2),axes(a,2)), 
					(i,ai) in zip(axes(h,1),axes(a,1))

					val = rand()
					h[i,j] = val
					@test h[i,j] == val
					@test a[ai,aj] == val
				end
			end
			function testsetindex!colon(j)
				N = Int(twice(j)) + 1
				h = SpinMatrix(zeros(N,N),j)
				a = parent(h)

				val = rand()
				h[:] .= val
				testallequal(h, val)

				for inds in ((:,:),(:,:,:),(:,:,:,:))
					val = rand()
					h[inds...] .= val
					testallequal(h, val)

					val = rand()
					h[inds...,1] .= val
					testallequal(h, val)

					val = rand()
					h[inds...,1,1] .= val
					testallequal(h, val)
				end

				for i2 in axes(h,2)
					val = rand()
					h[:,i2] .= val
					@test all(h[:,i2] .== val)
				end

				for i1 in axes(h,1)
					val = rand()
					h[i1,:] .= val
					@test all(h[i1,:] .== val)
				end
			end
			function testsetindex!range(j)
				N = Int(twice(j)) + 1
				h = SpinMatrix(zeros(N,N),j)

				for i2 in axes(h,2)
					val = rand()
					h[-j:j,i2] .= val
					@test all(h[-j:j,i2] .== val)

					if VERSION >= v"1.2"
						val = rand()
						h[Base.IdentityUnitRange(-j:j),i2] .= val
						@test all(h[Base.IdentityUnitRange(-j:j),i2] .== val)
					end
					if isinteger(j)
						val = rand()
						h[Base.OneTo(Int(j)),i2] .= val
						@test all(h[Base.OneTo(Int(j)),i2] .== val)
					end

					val = rand()
					h[-j:j,float(i2)] .= val
					@test all(h[-j:j,i2] .== val)

					val = rand()
					h[-j:j,Rational(i2)] .= val
					@test all(h[-j:j,i2] .== val)
				end

				for i1 in axes(h,1)
					val = rand()
					h[i1,-j:j] .= val
					@test all(h[i1,-j:j] .== val)

					if VERSION >= v"1.2"
						val = rand()
						h[i1,Base.IdentityUnitRange(-j:j)] .= val
						@test all(h[i1,Base.IdentityUnitRange(-j:j)] .== val)
					end

					val = rand()
					h[float(i1),-j:j] .= val
					@test all(h[i1,-j:j] .== val)
					
					val = rand()
					h[Rational(i1),-j:j] .= val
					@test all(h[i1,-j:j] .== val)
				end
			end
			function testsetindex!mixed(j)
				N = Int(twice(j)) + 1
				h = SpinMatrix(zeros(N,N),j)

				val = rand()
				h[-j:j,:] .= val
				@test all(h[-j:j,:] .== val)
				
				val = rand()
				h[-j:j,:,:] .= val
				@test all(h[-j:j,:] .== val)
				
				val = rand()
				h[-j:j,:,1] .= val
				@test all(h[-j:j,:] .== val)
				
				val = rand()
				h[-j:j,:,1:1] .= val
				@test all(h[-j:j,:] .== val)

				val = rand()
				h[:,-j:j] .= val
				@test all(h[:,-j:j] .== val)
				
				val = rand()
				h[:,-j:j,1] .= val
				@test all(h[:,-j:j] .== val)

				val = rand()
				h[:,-j:j,1:1] .= val
				@test all(h[:,-j:j] .== val)
			end
			@testset "scalar" begin
				testsetindex!scalar.(js)
			end
			@testset "colon" begin
				testsetindex!colon.(js)
			end
			@testset "range" begin
				testsetindex!range.(js)
			end
			@testset "mixed" begin
			   testsetindex!mixed.(js)
			end
		end
	end
	@testset "IdOffsetRange" begin
		r = IdOffsetRange(1:3, HalfInt(2))
		for i in axes(r,1)
			@test r[i] === HalfInt(i)
			@test r[HalfInt(i)] === HalfInt(i)
		end
		@test r[3:5] === HalfInt(3):HalfInt(5)
		@test r[HalfInt(3):HalfInt(5)] === HalfInt(3):HalfInt(5)
		@test r[r] === r
		@test_throws BoundsError r[1]
		@test_throws BoundsError r[6]

		r2 = IdOffsetRange{HalfInt32}(3:4, HalfInt(0))
		@test r[r2] == r2
	end
	@testset "OneTo" begin
		r = OneTo(3)
		for i in axes(r,1)
			@test r[i] === HalfInt(i)
		end
		for i in axes(r,1)
			@test r[Base.OneTo(i)] === OneTo(i)
			@test r[OneTo(i)] === OneTo(i)
		end

		@test_throws BoundsError r[4]
		@test_throws BoundsError r[1:4]
	end
	@testset "cartesian" begin
		arr = reshape([1,2,3,4],2,2)
		h = HalfIntArray(arr,-half(1):half(1),-half(1):half(1))

		@test h[CartesianIndicesHalfInt(h)] == h

		@test h[CartesianIndexHalfInt(half(1),half(1))] == 4
		@test h[half(1),CartesianIndexHalfInt(half(1))] == 4
		@test h[CartesianIndexHalfInt(half(1)),half(-1)] == 2

		h[CartesianIndexHalfInt(half(1),half(1))] = 3
		@test h[CartesianIndexHalfInt(half(1),half(1))] == 3

		h[CartesianIndexHalfInt(half(1)),half(-1)] = 1
		@test h[CartesianIndexHalfInt(half(1)),half(-1)] == 1

		v = HalfIntArray([1,2],-half(1):half(1))

		@test v[CartesianIndicesHalfInt(v)] == v
	end
end

@testset "axes operations" begin
	r = IdOffsetRange(1:3, HalfInt(2))
	@test Base.unsafe_indices(r) == (r,)
	@test Base.reduced_index(r) == IdOffsetRange(3:3, HalfInt(0))

	h = HalfIntArray{Float64}(undef,-1:1)
	@test Base.compute_offset1(h,1,size(h),axes(h),(-1,)) == 0
	@test Base.compute_offset1(h,1,size(h),axes(h),(0,)) == 1
	@test Base.compute_offset1(h,1,size(h),axes(h),(1,)) == 2

	if VERSION >= v"1.2"
		r2 = Base.IdentityUnitRange(IdOffsetRange(1:3, HalfInt(2)))
		@test Base.reduced_index(r2) == IdOffsetRange(3:3, HalfInt(0))
	end
end

@testset "mutate HalfIntVector" begin
	h = HalfIntArray(ones(3),-1:1)
	
	resize!(h,5)
	@test size(h,1) == 5
	
	push!(h,6)
	@test h[end] === 6.0
	@test size(h,1) == 6
	
	a = pop!(h)
	@test a === 6.0
	@test size(h,1) == 5

	empty!(h)
	@test size(h,1) == 0
end

@testset "subarray" begin
	@testset "HalfIntArray" begin
		@testset "0D" begin
			h = HalfIntArray(ones())
			v = @view h[]
			@test ndims(v) == 0
			@test v[] === h[]
			@test v[1] === h[]
			@test axes(v) === axes(h) == ()
			@test v[:] == h[:]

			v1 = @view h[:]
			v2 = @view h[:,1]
			v3 = @view h[1,:]
			for v in [v1,v2,v3]
				@test ndims(v) == 1
				@test axes(v) == (IdOffsetRange(Base.OneTo(1)),)
				@test v[] == h[]
				@test v[1] == h[]
				@test v[1,1] == h[]
				@test ndims(v[:]) == 1
				@test v[:] == h[:]
				@test ndims(v[:,:]) == 2
				@test v[:,:] == h[:,:]
			end

			v = @view h[]
			v[] = 4
			@test h[] == 4
			v .= 5
			@test h[] == 5
		end
		@testset "1D" begin
			h = HalfIntArray(rand(3), 0:2)

			@testset "0D slice" begin
				v = @view h[0]
				@test v[] == h[0]
				v[] = 34
				@test h[0] == 34
			end
			
			@test_throws BoundsError @view h[]

			@testset "entire array" begin
				v1 = @view h[:]
				@test h[CartesianIndicesHalfInt(h)] == v1[CartesianIndicesHalfInt(v1)]
				@test h[LinearIndicesHalfInt(h)] == v1[LinearIndicesHalfInt(v1)]

				for i in eachindex(v1)
					val = rand()
					v1[i] = val
					@test h[i] == val
				end
				v2 = @view h[:,1]
				v3 = @view h[:,1,1]

				for v in [v1, v2, v3]
					@test ndims(v) == 1
					@test axes(v) === axes(h)
					@test all(v .== h)
					val = rand(1:10)
					v .= val
					@test all(h .== val)
					@test all(h[:] .== val)
					@test all(h[:,1] .== val)
					@test all(h[:,:] .== val)
				end
			end

			v = @view h[:,:]
			@test ndims(v) == 2
			@test all(v .== h)
			v .= 3
			@test all(h .== 3)

			v = @view h[1,:]
			@test ndims(v) == 1
			@test v[] == h[1]
			v[] = 6
			@test h[1] == 6

			v = @view h[0:2]
			@test axes(v) == (1:3,)
			@test ndims(v) == 1
			@test all(v[:] .== h[0:2])
			v .= 4
			@test all(h .== 4)

			@test_throws BoundsError @view h[-1]
			@test_throws BoundsError @view h[1:3]
		end
		@testset "2D" begin
			h = HalfIntArray(rand(2,4), 0:1, -half(3):half(3))

			@testset "0D slice" begin
				v = @view h[0,1/2]
				@test ndims(v) == 0
				@test v[] == h[0,1/2]
				@test v[1] == h[0,1/2]
				v[] = 40
				@test h[0,1/2] == 40
			end

			@testset "1D view" begin
				v = @view h[:]
				@test ndims(v) == 1
				@test axes(v) == (Base.OneTo(length(h)),)
				@test all(v .== vec(h))
				v .= 1
				@test all(h .== 1)
				
				v = vec(h)
				@test ndims(v) == 1
				@test axes(v) == (Base.OneTo(length(h)),)
				v .= 2
				@test all(h .== 2)
			end

			@testset "entire array" begin
				v1 = @view h[:,:]
				@test h[CartesianIndicesHalfInt(h)] == v1[CartesianIndicesHalfInt(v1)]
				@test h[LinearIndicesHalfInt(h)] == v1[LinearIndicesHalfInt(v1)]
				@test HalfIntegerArrays.parenttype(typeof(v1)) == typeof(parent(v1))

				v2 = @view h[:,:,1]
				v3 = @view h[:,:,1,1]


				for v in [v1, v2, v3]
					@test ndims(v) == 2
					@test axes(v) === axes(h)
					@test all(v .== h)
					val = rand(1:10)
					v .= val
					@test all(h .== val)
					@test all(h[:] .== val)
					@test all(h[:,:] .== val)
					@test all(h[:,:,1] .== val)
				end
			end

			@testset "2D slices" begin
				v = @view h[0:1,-half(1):half(1)]
				@test all(v .== h[0:1,-half(1):half(1)])
				v .= 3
				@test all(h[0:1,-half(1):half(1)] .== 3)
			end

			@testset "1D slices" begin
				v2 = @view h[0,:]
				@test axes(v2,1) == axes(h,2)
				@test all(v2 .== h[0,:])
				v2 .= 4
				@test all(h[0,:] .== 4)
				
				v1 = @view h[:,1/2]
				@test axes(v1,1) == axes(h,1)
				@test all(v1 .== h[:,1/2])
				v1 .= 5
				@test all(h[:,1/2] .== 5)

				for v in [v1,v2]
					@test ndims(v) == 1
				end
			end

			@testset "add dims" begin
				v1 = @view h[:,:,:]
				v2 = @view h[:,:,:,1]
				v3 = @view h[:,:,1,:]

				for v in [v1, v2, v3]
					@test ndims(v) == 3
					@test all(v .== h)
					val = rand()
					v .= val
					@test all(h .== val)
				end
			end
		end
		@testset "3D" begin
			h = HalfIntArray(rand(2,2,2), 
			0:1, -half(1):half(1), HalfInt(2):HalfInt(3))

			@testset "0D slice" begin
				v = @view h[0,1/2,2]
				@test ndims(v) == 0
				@test v[] == h[0,1/2,2]
				@test v[1] == h[0,1/2,2]
				v[] = 40
				@test h[0,1/2,2] == 40
			end

			@testset "1D view" begin
				v = @view h[:]
				@test ndims(v) == 1
				@test axes(v) == (Base.OneTo(length(h)),)
				@test all(v .== vec(h))
				v .= 1
				@test all(h .== 1)
				
				v = vec(h)
				@test ndims(v) == 1
				@test axes(v) == (Base.OneTo(length(h)),)
				v .= 2
				@test all(h .== 2)
			end

			@testset "entire array" begin
				v1 = @view h[:,:,:]
				@test h[CartesianIndicesHalfInt(h)] == v1[CartesianIndicesHalfInt(v1)]
				@test h[LinearIndicesHalfInt(h)] == v1[LinearIndicesHalfInt(v1)]

				v2 = @view h[:,:,:,1]
				v3 = @view h[:,:,:,1,1]

				for v in [v1, v2, v3]
					@test ndims(v) == 3
					@test axes(v) === axes(h)
					@test all(v .== h)
					val = rand(1:10)
					v .= val
					@test all(h .== val)
					@test all(h[:] .== val)
					@test all(h[:,:,:] .== val)
					@test all(h[:,:,:,1] .== val)
					@test all(h[:,:,:,1,1] .== val)
					@test all(h[:,:,:,:,1] .== val)
				end
			end

			@testset "1D slices" begin
				v1 = @view h[:,1/2,2]
				@test all(v1 .== h[:,1/2,2])
				v1 .= 1
				@test all(h[:,1/2,2] .== 1)

				v2 = @view h[0,:,3]
				@test all(v2 .== h[0,:,3])
				v2 .= 2
				@test all(h[0,:,3] .== 2)

				v3 = @view h[0,1/2,:]
				@test all(v3 .== h[0,1/2,:])
				v3 .= 3
				@test all(h[0,1/2,:] .== 3)

				for v in [v1, v2, v3]
					@test ndims(v) == 1
				end
			end

			@testset "2D slices" begin
				v12 = @view h[:,:,2]
				@test all(v12 .== h[:,:,2])
				v12 .= 1
				@test all(h[:,:,2] .== 1)

				v23 = @view h[0,:,:]
				@test all(v23 .== h[0,:,:])
				v23 .= 2
				@test all(h[0,:,:] .== 2)

				v13 = @view h[:,1/2,:]
				@test all(v13 .== h[:,1/2,:])
				v13 .= 3
				@test all(h[:,1/2,:] .== 3)

				for v in [v12, v23, v13]
					@test ndims(v) == 2
				end
			end

			@testset "add dims" begin
				v1 = @view h[:,:,:,:]
				v2 = @view h[:,:,:,:,1]
				v3 = @view h[:,:,:,1,:]

				for v in [v1, v2, v3]
					@test all(v .== h)
					@test ndims(v) == 4
					val = rand()
					v .= val
					@test all(h .== val)
				end
			end
		end
	end
	@testset "SpinMatrix" begin
		h = SpinMatrix(rand(2,2))
			
		v = @view h[1/2,1/2]
		@test ndims(v) == 0
		@test v[] == h[1/2,1/2]
		@test v[1] == h[1/2,1/2]
		v[] = 40
		@test h[1/2,1/2] == 40

		v = @view h[:]
		@test ndims(v) == 1
		@test axes(v) == (Base.OneTo(length(h)),)
		@test all(v .== vec(h))
		v .= 1
		@test all(h .== 1)

		v1 = @view h[:,:]
		@test h[CartesianIndicesHalfInt(h)] == v1[CartesianIndicesHalfInt(v1)]
		@test h[LinearIndicesHalfInt(h)] == v1[LinearIndicesHalfInt(v1)]
		v2 = @view h[:,:,1]
		v3 = @view h[:,:,1,1]

		for v in [v1, v2, v3]
			@test ndims(v) == 2
			@test axes(v) === axes(h)
			@test all(v .== h)
			val = rand(1:10)
			v .= val
			@test all(h .== val)
			@test all(h[:] .== val)
			@test all(h[:,:] .== val)
			@test all(h[:,:,1] .== val)
		end

		v2 = @view h[-1/2,:]
		@test axes(v2,1) == axes(h,2)
		@test all(v2 .== h[-1/2,:])
		v2 .= 4
		@test all(h[-1/2,:] .== 4)
		
		v1 = @view h[:,1/2]
		@test axes(v1,1) == axes(h,1)
		@test all(v1 .== h[:,1/2])
		v1 .= 5
		@test all(h[:,1/2] .== 5)

		for v in [v1,v2]
			@test ndims(v) == 1
		end

		v1 = @view h[:,:,:]
		v2 = @view h[:,:,:,1]
		v3 = @view h[:,:,1,:]

		for v in [v1, v2, v3]
			@test ndims(v) == 3
			@test all(v .== h)
			val = rand()
			v .= val
			@test all(h .== val)
		end
	end
	@testset "CartesianIndicesHalfInt" begin
		h = HalfIntArray(rand(2,2),1:2,3:4)
		c = CartesianIndicesHalfInt(h)
		v = @view c[:]
		@test all(v .== vec(c))

		v = @view c[1:1,3:3]
		@test v[1,1] == c[1,3]
		@test v[CartesianIndex(1,1)] == c[CartesianIndex(1,3)]
		@test v[CartesianIndexHalfInt(1,1)] == c[CartesianIndexHalfInt(1,3)]

		h = SpinMatrix(rand(2,2))
		c = CartesianIndicesHalfInt(h)
		v = @view c[:,:]
		for I in CartesianIndicesHalfInt(h)
			@test v[I] == c[I]
		end
		for I in LinearIndicesHalfInt(h)
			@test v[I] == c[I]
		end
		@test v[c] == c[v]
	end
	@testset "LinearIndices" begin
		h = HalfIntArray(rand(2,2),1:2,3:4)
		inds = LinearIndicesHalfInt(h)
		v = @view inds[:]
		@test all(v .== vec(inds))

		v = @view inds[1:1,3:3]
		@test v[1,1] == inds[1,3]
		@test v[CartesianIndex(1,1)] == inds[CartesianIndex(1,3)]
		@test v[CartesianIndexHalfInt(1,1)] == inds[CartesianIndexHalfInt(1,3)]

		h = SpinMatrix(rand(2,2))
		inds = LinearIndicesHalfInt(h)
		v = @view inds[:,:]
		for I in CartesianIndicesHalfInt(inds)
			@test v[I] == inds[I]
		end
		for I in LinearIndicesHalfInt(inds)
			@test v[I] == inds[I]
		end
		@test v[inds] == inds[v]
	end
	@testset "subarray" begin
		h = HalfIntArray(rand(2,2),1:2,3:4)
		v = @view h[:,:]
		v′ = @view v[:,:]
		@test axes(v′) == axes(v) == axes(h)
		@test all(v .== v′)

		v = @view h[1:2,3:4]
		@test axes(v) == (1:2,1:2)
		v′ = @view v[2:2,2:2]
		@test ndims(v′) == 2
		@test axes(v′) == (1:1, 1:1)
		@test v′[1,1] == h[2,4]
	end
	@testset "wrappers" begin
		h = HalfIntArray(rand(2,2),1:2,3:4)
		v = @view h'[:,:]
		@test axes(v) == reverse(axes(h))
		@test_broken all(v .== h')

		h = SpinMatrix(rand(2,2));
		v = @view h'[:,:];
		@test axes(v) == reverse(axes(h));
		@test_broken all(v .== h')
	end
end

@testset "similar" begin

	@testset "util" begin
		# These might change
		@test HalfIntegerArrays.indexoffset(Colon()) === HalfInt(0)
		@test HalfIntegerArrays.indexoffset(1) === HalfInt(0)
		@test HalfIntegerArrays.indexlength(Colon()) === Colon()
		@test HalfIntegerArrays.indexlength(1) === 1
	end

	h = HalfIntArray(rand(3,5),-1:1,-2:2)
	
	@testset "offset axes" begin
		h′ = similar(h)
		@test h′ isa typeof(h)
		@test axes(h′) == axes(h)

		h′ = similar(h,axes(h))
		@test h′ isa typeof(h)
		@test axes(h′) == axes(h)

		h′ = similar(h,eltype(h),axes(h))
		@test h′ isa typeof(h)
		@test axes(h′) == axes(h)

		h′ = similar(h,eltype(h))
		@test h′ isa typeof(h)
		@test axes(h′) == axes(h)

		h′ = similar(h, (IdOffsetRange(1:2),))
		@test h′ isa HalfIntArray{eltype(h),1}
		@test axes(h′) == (IdOffsetRange(1:2),)

		h′ = similar(h, (IdOffsetRange(1:2),OneTo(3)))
		@test h′ isa HalfIntArray{eltype(h),2}
		@test axes(h′) == (IdOffsetRange(1:2),IdOffsetRange(1:3))
	end

	@testset "Non-offset axes" begin
		h′ = similar(h,eltype(h),(1,2))
		@test eltype(h′) == eltype(h)
		@test axes(h′) == (Base.OneTo(1),Base.OneTo(2))

		h′ = similar(h,eltype(h),())
		@test eltype(h′) == eltype(h)
		@test ndims(h′) == 0
		@test axes(h′) == ()

		h′ = similar(h,eltype(h),(Base.OneTo(2),3))
		@test eltype(h′) == eltype(h)
		@test axes(h′) == (Base.OneTo(2),Base.OneTo(3))
		
		h′ = similar(h,eltype(h),(OneTo(2),Base.OneTo(2),3))
		@test eltype(h′) == eltype(h)
		@test axes(h′) == (Base.OneTo(2),Base.OneTo(2),Base.OneTo(3))
	end

	@testset "OneTo" begin
		h′ = similar(h, (OneTo(1), OneTo(2)))
		@test h′ isa Array{eltype(h),2}
		@test axes(h′) == (OneTo(1), OneTo(2))
		
		h′ = similar(h, (OneTo(1),))
		@test h′ isa Array{eltype(h),1}
		@test axes(h′) == (OneTo(1),)
	end

	@testset "broadcasting" begin
		h′ = similar(Array{Int}, (IdOffsetRange(1:2),OneTo(3)))
		@test h′ isa HalfIntArray{Int, 2}
		@test axes(h′) == (IdOffsetRange(1:2),IdOffsetRange(1:3))
	end
end

@testset "reshape" begin
	@testset "HalfIntArray" begin
		h = HalfIntArray(rand(3,5),-1:1, -2:2)

		@testset "offset axes" begin
			h′ = HalfIntArray(parent(h),1:3,1:5)
			@test reshape(h,(1:3,1:5)) == h′
			@test reshape(h,1:3,1:5) == h′
			@test reshape(h,(OneTo(3),1:5)) === h′
			@test reshape(h,OneTo(3),1:5) === h′
			@test reshape(h,(1:3,OneTo(5))) === h′
			@test reshape(h,1:3,OneTo(5)) === h′
			@test reshape(h,(IdOffsetRange(1:3),OneTo(5))) === h′
			@test reshape(h,IdOffsetRange(1:3),OneTo(5)) === h′
			@test reshape(h,(IdOffsetRange(1:3),IdOffsetRange(1:5))) === h′
			@test reshape(h,IdOffsetRange(1:3),IdOffsetRange(1:5)) === h′
		end

		@testset "Non-offset axes" begin
			@test reshape(h,:) === h 
			@test reshape(h,size(h)) == parent(h)
			@test reshape(h,Int32.(size(h))) == parent(h)
			@test reshape(h,size(h)...) == parent(h)
			@test reshape(h,size(h,1),:) == parent(h)
			@test reshape(h,(size(h,1),:)) == parent(h)
			@test reshape(h,(size(h,1),Base.OneTo(size(h,2)))) == parent(h)
			@test reshape(h,size(h,1),Base.OneTo(size(h,2))) == parent(h)
			@test_throws DimensionMismatch reshape(h,(:,:))
			@test_throws DimensionMismatch reshape(h,())
		end

		@testset "OneTo" begin
			h′ = reshape(h, OneTo(3), OneTo(5))
			@test h′ isa Array{eltype(h),2}
			@test h′ === parent(h)
			
			h′ = reshape(h, OneTo(15))
			@test h′ isa Array{eltype(h),1}
			@test h′ == vec(parent(h))
		end

		@testset "val" begin
		    @test reshape(h, Val(1)) == vec(parent(h))
		    @test reshape(h, Val(2)) === h
		    @test reshape(h, Val(3)) == HalfIntArray(reshape(parent(h),Val(3)),axes(h)...,1:1)
		end
	end
	@testset "SpinMatrix" begin
		s = SpinMatrix(rand(5,5), 2)

		@testset "offset axes" begin
			h′ = HalfIntArray(parent(s),1:5,1:5)
			@test reshape(s,axes(h′)...) == h′
			@test reshape(s,axes(h′)) == h′
			
			h′ = reshape(s, OneTo(5), IdOffsetRange(-2:2))
			@test parent(h′) === parent(s)
			@test axes(h′,1) === IdOffsetRange(Base.OneTo(5))
			@test axes(h′,2) === IdOffsetRange(Base.OneTo(5),-3)
			
			h′ = reshape(s, (OneTo(5), IdOffsetRange(-2:2)))
			@test parent(h′) === parent(s)
			@test axes(h′,1) === IdOffsetRange(Base.OneTo(5))
			@test axes(h′,2) === IdOffsetRange(Base.OneTo(5),-3)

			h′ = reshape(s, (IdOffsetRange(-1:3), IdOffsetRange(-2:2)))
			@test parent(h′) === parent(s)
			@test axes(h′,1) === IdOffsetRange(Base.OneTo(5),-2)
			@test axes(h′,2) === IdOffsetRange(Base.OneTo(5),-3)
		end

		@testset "Non-offset axes" begin
			@test reshape(s,:) === s 
			@test reshape(s,size(s)) == parent(s)
			@test reshape(s,Int32.(size(s))) == parent(s)
			@test reshape(s,size(s)...) == parent(s)
			@test reshape(s,size(s,1),:) == parent(s)
			@test reshape(s,(size(s,1),:)) == parent(s)
			@test reshape(s,(size(s,1),Base.OneTo(size(s,2)))) == parent(s)
			@test reshape(s,size(s,1),Base.OneTo(size(s,2))) == parent(s)
			@test_throws DimensionMismatch reshape(s,(:,:))
			@test_throws DimensionMismatch reshape(s,())
		end

		@testset "OneTo" begin
			s′ = reshape(s, OneTo(5), OneTo(5))
			@test s′ isa Array{eltype(s),2}
			@test s′ === parent(s)

			s′ = reshape(s, OneTo(25))
			@test s′ isa Array{eltype(s),1}
			@test s′ == vec(parent(s))
		end

		@testset "val" begin
		    @test reshape(s, Val(1)) == vec(parent(s))
		    @test reshape(s, Val(2)) === s
		    @test reshape(s, Val(3)) == HalfIntArray(reshape(parent(s),Val(3)),axes(s)...,1:1)
		end
	end
end

@testset "LinearAlgebra" begin
	hdiag = HalfIntArray(Diagonal([1,1,1]),-1:1,-1:1)

	@testset "indexing" begin
		h1 = HalfIntArray(rand(ComplexF64,3,3),-1:1,-1:1)
		h1by2 = HalfIntArray(rand(ComplexF64,2,2),-1//2:1//2,-1//2:1//2)

		hv = HalfIntArray(rand(ComplexF64,4), -half(3):half(3))
		
		s1 = SpinMatrix(rand(ComplexF64,3,3),1)
		s1by2 = SpinMatrix(rand(ComplexF64,2,2),1//2)

		@test hdiag' == hdiag
		@test transpose(hdiag) == hdiag
		@test keys(hdiag') == keys(hdiag)
		@test keys(pairs(hdiag')) == keys(pairs(hdiag))

		@testset "transpose" begin
			@testset "HalfIntArray" begin
				@testset "CartesianIndicesHalfInt" begin
					for h in [h1, h1by2, hv]
						hT = transpose(h)
						for I in CartesianIndicesHalfInt(hT)
							cinds = Tuple(I)
							cindsp = reverse(cinds)
							@test hT[I] == h[cindsp...]
							@test hT[cinds...] == h[cindsp...]
							val = rand(ComplexF64)
							hT[cinds...] = val
							@test hT[cinds...] == val
							@test h[cindsp...] == val
						end
					end
				end
				@testset "LinearIndicesHalfInt" begin
					for h in [h1, h1by2, hv]
						hT = transpose(h)
						for i in LinearIndicesHalfInt(hT)
							CI = CartesianIndicesHalfInt(hT)
							cinds = Tuple(CI[i])
							cindsp = reverse(cinds)
							@test hT[i] == h[cindsp...]
							@test hT[Tuple(i)...] == h[cindsp...]
							val = rand(ComplexF64)
							hT[i] = val
							@test hT[i] == val
							@test h[cindsp...] == val
						end
					end
				end
			end 
			@testset "SpinMatrix" begin
				@testset "CartesianIndices" begin
					for s in [s1, s1by2]
						sT = transpose(s)
						for i in CartesianIndicesHalfInt(sT)
							@test sT[i] == s[reverse(Tuple(i))...]
							@test sT[Tuple(i)...] == s[reverse(Tuple(i))...]
							val = rand(ComplexF64)
							sT[Tuple(i)...] = val
							@test sT[Tuple(i)...] == val
							@test s[reverse(Tuple(i))...] == val
						end
					end
				end
				@testset "LinearIndicesHalfInt" begin
					for s in [s1, s1by2]
						sT = transpose(s)
						Cinds = CartesianIndicesHalfInt(sT)
						for i in LinearIndicesHalfInt(sT)
							cinds = Tuple(Cinds[i])
							cindsp = reverse(cinds)
							@test sT[i] == s[cindsp...]
							val = rand(ComplexF64)
							sT[i] = val
							@test sT[i] == val
							@test s[cindsp...] == val
						end
					end
				end
			end
		end
		@testset "adjoint" begin
			@testset "HalfIntArray" begin
				@testset "CartesianIndicesHalfInt" begin
					for h in [h1, h1by2, hv]
						hA = adjoint(h)
						for I in CartesianIndicesHalfInt(hA)
							cinds = Tuple(I)
							cindsp = reverse(cinds)
							@test hA[I] == adjoint(h[cindsp...])
							@test hA[cinds...] == adjoint(h[cindsp...])
							val = rand(ComplexF64)
							hA[cinds...] = val
							@test hA[cinds...] == val
							@test h[cindsp...] == adjoint(val)
						end
					end
				end
				@testset "LinearIndicesHalfInt" begin
					for h in [h1, h1by2, hv]
						hT = adjoint(h)
						for i in LinearIndicesHalfInt(hT)
							CI = CartesianIndicesHalfInt(hT)
							cinds = Tuple(CI[i])
							cindsp = reverse(cinds)
							@test hT[i] == adjoint(h[cindsp...])
							@test hT[Tuple(i)...] == adjoint(h[cindsp...])
							val = rand(ComplexF64)
							hT[i] = val
							@test hT[i] == val
							@test h[cindsp...] == adjoint(val)
						end
					end
				end
			end 
			@testset "SpinMatrix" begin
				@testset "CartesianIndices" begin
					for s in [s1, s1by2]
						sA = adjoint(s)
						for I in CartesianIndicesHalfInt(sA)
							cinds = Tuple(I)
							cindsp = reverse(Tuple(I))
							@test sA[I] == adjoint(s[cindsp...])
							@test sA[cinds...] == adjoint(s[cindsp...])
							val = rand(ComplexF64)
							sA[cinds...] = val
							@test sA[cinds...] == val
							@test s[cindsp...] == adjoint(val)
						end
					end
				end
				@testset "LinearIndicesHalfInt" begin
					for s in [s1, s1by2]
						sA = adjoint(s)
						Cinds = CartesianIndicesHalfInt(sA)
						for i in LinearIndicesHalfInt(sA)
							cinds = Tuple(Cinds[i])
							cindsp = reverse(cinds)
							@test sA[i] == adjoint(s[cindsp...])
							val = rand(ComplexF64)
							sA[i] = val
							@test sA[i] == val
							@test s[cindsp...] == adjoint(val)
						end
					end
				end
			end
		end
	end

	@testset "reshape" begin
	    @test reshape(hdiag',3,3) == reshape(hdiag,3,3)
	    @test reshape(hdiag',0:2,0:2) == reshape(hdiag,0:2,0:2)
	end

	@testset "inv" begin

		# invertible matrices
		arr3by3 = [1 2 4; 1 3 5; 6 7 8]
		arr2by2 = [1 2; 5 7]
		h1 = HalfIntArray(arr3by3,-1:1,-1:1)
		h1by2 = HalfIntArray(arr2by2,-1//2:1//2,-1//2:1//2)
		s1 = SpinMatrix(h1)
		s1by2 = SpinMatrix(h1by2)

		@test inv(hdiag) == hdiag

		@testset "HalfIntArray" begin
			for h in [h1, h1by2]
				hinv = inv(h)
				@test hinv isa HalfIntArray
				@test hinv.offsets == h.offsets
				@test parent(hinv) == inv(parent(h))
			end
		end
		@testset "SpinMatrix" begin
			for s in [s1, s1by2]
				sinv = inv(s)
				@test sinv isa SpinMatrix
				@test sinv.j == s.j
				@test parent(sinv) == inv(parent(s)) 
			end
		end
	end

	@testset "matrix multiplication" begin

		@test hdiag*hdiag == hdiag

		h1 = HalfIntArray(rand(3,3),-1:1,-1:1)
		h1by2 = HalfIntArray(rand(2,2),-1//2:1//2, -1//2:1//2)
		s1 = SpinMatrix(rand(3,3),1)
		s1by2 = SpinMatrix(rand(2,2),1//2)

		@testset "HalfIntArray" begin
			for h in [h1, h1by2]
				htimesh = h*h
				@test htimesh isa HalfIntArray
				@test parent(htimesh) == parent(h)*parent(h)
				@test htimesh.offsets == h.offsets
			end
		end
		@testset "SpinMatrix" begin
			for h in [s1, s1by2]
				htimesh = h*h
				@test htimesh isa SpinMatrix
				@test parent(htimesh) == parent(h)*parent(h)
				@test htimesh.j == h.j
			end 
		end
	end

	@testset "is.." begin
		h1 = HalfIntArray(reshape(1:9,3,3),-1:1,-1:1)
		h1by2 = HalfIntArray(reshape(1:4,2,2),-1//2:1//2, -1//2:1//2)
		s1 = SpinMatrix(reshape(1:9,3,3),1)
		s1by2 = SpinMatrix(reshape(1:4,2,2),1//2)

		for h in [h1, h1by2, s1, s1by2]
			@test !isdiag(h1)
		end
		@test isdiag(hdiag)

		@test istriu(hdiag)
		@test istril(hdiag)
		@test !istriu(h1)
		@test !istril(h1)
	end

	@testset "diaginds and diag" begin
	    @test diagind(hdiag) == diagind(parent(hdiag))
	    @test diag(hdiag) == hdiag.parent.diag
	end
end

@testset "Broadcasting" begin
	@testset "Broadcast" begin
		h = HalfIntArray(rand(3,5),-1:1, -2:2)
		b = Broadcast.broadcasted(+,h,1) |> Broadcast.instantiate
		@test Base.dataids(h) == Base.dataids(parent(h))
		@test Broadcast.broadcast_unalias(h,h) === h
		@test eachindex(b) == eachindex(IndexCartesian(),h)
		for I in eachindex(b)
			@test b[I] == h[I] + 1
		end

		@test Broadcast.newindex(h, HalfInteger(1)) == CartesianIndexHalfInt(HalfInteger(1),HalfInteger(1))
		@test HalfIntegerArrays._newindex((half(1),),(true,),(half(0),)) == (half(1),)
		@test HalfIntegerArrays._newindex((half(1),),(false,),(half(0),)) == (half(0),)
	end
	@testset "HalfIntArray" begin
		@testset "A + A" begin
			@testset "2D" begin
				h = HalfIntArray(rand(3,5),-1:1, -2:2)
				h′ = h .+ h
				@test parent(h′) == parent(h) .+ parent(h)
				@test axes(h′) == axes(h)
				@test typeof(h′) == typeof(h)
			end

			@testset "1D" begin
				v = HalfIntArray(rand(2),-half(1):half(1))
				v′ = v .+ v
				@test parent(v′) == parent(v) .+ parent(v)
				@test axes(v′) == axes(v)
				@test typeof(v′) == typeof(v)
			end

			@testset "0D" begin
				z = HalfIntArray(ones())
				z′ = z .+ z
				@test z′[] == z[] + z[]
			end    
		end

		@testset "A + 1" begin
			@testset "2D" begin
				h = HalfIntArray(rand(3,5),-1:1, -2:2)
				h′ = h .+ 1
				@test parent(h′) == parent(h) .+ 1
				@test axes(h′) == axes(h)
				@test typeof(h′) == typeof(h)
			end

			@testset "1D" begin
				v = HalfIntArray(rand(2),-half(1):half(1))
				v′ = v .+ 1
				@test parent(v′) == parent(v) .+ 1
				@test axes(v′) == axes(v)
				@test typeof(v′) == typeof(v)
			end

			@testset "0D" begin
				z = HalfIntArray(ones())
				z′ = z .+ 1
				@test z′[] == z[] + 1
			end
		end
		
		@testset "inplace" begin
			@testset "2D" begin
				h = HalfIntArray(rand(3,5),-1:1, -2:2)
				h′ = copy(h)
				h′ .+= 1
				@test parent(h′) == parent(h) .+ 1
				@test axes(h′) == axes(h)
				@test typeof(h′) == typeof(h)
			end

			@testset "1D" begin
				v = HalfIntArray(rand(2),-half(1):half(1))
				v′ = copy(v)
				v′ .+= 1
				@test parent(v′) == parent(v) .+ 1
				@test axes(v′) == axes(v)
				@test typeof(v′) == typeof(v)
			end

			@testset "0D" begin
				z = HalfIntArray(zeros())
				z′ = HalfIntArray(zeros())
				z′ .+= 1
				@test z′[] == z[] + 1
			end   
		end
	end
	@testset "SpinMatrix" begin
		@testset "A + A" begin
			@testset "spin 1" begin
				h = SpinMatrix(rand(3,3),1)
				h′ = h .+ h
				@test parent(h′) == parent(h) .+ parent(h)
				@test axes(h′) == axes(h)
			end
			@testset "spin 1/2" begin
				h = SpinMatrix(rand(2,2),half(1))
				h′ = h .+ h
				@test parent(h′) == parent(h) .+ parent(h)
				@test axes(h′) == axes(h)
			end
			@testset "spin 0" begin
				h = SpinMatrix(rand(1,1),0)
				h′ = h .+ h
				@test parent(h′) == parent(h) .+ parent(h)
				@test axes(h′) == axes(h)
			end
		end

		@testset "A + 1" begin
			@testset "spin 1" begin
				h = SpinMatrix(rand(3,3),1)
				h′ = h .+ 1
				@test parent(h′) == parent(h) .+ 1
				@test axes(h′) == axes(h)
			end
			@testset "spin 1/2" begin
				h = SpinMatrix(rand(2,2),half(1))
				h′ = h .+ 1
				@test parent(h′) == parent(h) .+ 1
				@test axes(h′) == axes(h)
			end
			@testset "spin 0" begin
				h = SpinMatrix(rand(1,1),0)
				h′ = h .+ 1
				@test parent(h′) == parent(h) .+ 1
				@test axes(h′) == axes(h)
			end
		end
		
		@testset "inplace" begin
			@testset "spin 1" begin
				h = SpinMatrix(rand(3,3),1)
				h′ = similar(h)
				h′ .= h
				h′ .+= h 
				@test parent(h′) == parent(h) .+ parent(h)
				@test axes(h′) == axes(h)
			end
			@testset "spin 1/2" begin
				h = SpinMatrix(rand(2,2),half(1))
				h′ = similar(h)
				h′ .= h
				h′ .+= h 
				@test parent(h′) == parent(h) .+ parent(h)
				@test axes(h′) == axes(h)
			end
			@testset "spin 0" begin
				h = SpinMatrix(rand(1,1),0)
				h′ = similar(h)
				h′ .= h
				h′ .+= h 
				@test parent(h′) == parent(h) .+ parent(h)
				@test axes(h′) == axes(h)
			end
		end
	end
end

@testset "hashing" begin
	@testset "HalfIntArray" begin
		A = HalfIntArray(zeros(2,2), 0:1, 0:1)
		h = hash(:HalfIntArray, hash(parent(A), hash(A.offsets)))
		@test hash(A) == h
	end
	@testset "SpinMatrix" begin
		A = SpinMatrix(zeros(2,2))
		h = hash(:SpinMatrix, hash(parent(A), hash(A.j)))
		@test hash(A) == h 
	end
	@testset "HalfIntSubArray" begin
	    A = HalfIntArray(zeros(2,2), 0:1, 0:1)
	    AV = @view A[:,:] 
		h = hash(:HalfIntSubArray, hash(parent(AV), hash(AV.offsets)))
		@test hash(AV) == h
	end
	@testset "CartesianIndexHalfInt" begin
		C = CartesianIndexHalfInt(1,1)
		h0 = HalfIntegerArrays.cartindexhash_seed
		h = hash(:CartesianIndexHalfInt, hash(parent(C), hash(C.offsets, h0)))
		@test hash(C) == h
	end
	@testset "CartesianIndicesHalfInt" begin
		A = HalfIntArray(zeros(2,2), 0:1, 0:1)
		C = CartesianIndicesHalfInt(A)
		h = hash(:CartesianIndicesHalfInt, hash(C.cartinds, hash(C.offsets)))
		@test hash(C) == h
	end
	@testset "LinearIndicesHalfInt" begin
		A = HalfIntArray(zeros(2,2), 0:1, 0:1)
		L = LinearIndicesHalfInt(A)
		h = hash(:LinearIndicesHalfInt, hash(L.lininds, hash(L.offsets)))
		@test hash(L) == h
	end
end

@testset "keys and pairs" begin
	@testset "keys" begin
	    h = HalfIntArray(rand(2,2), 0:1, 0:1)
	    @test keys(h) == CartesianIndicesHalfInt(h)

	    h = HalfIntArray(rand(2), 0:1)
	    @test keys(h) == LinearIndicesHalfInt(h)

	    h = HalfIntArray(rand(2,2), 0:1, 0:1)
	    @test keys(IndexLinear(), h) == eachindex(IndexLinear(),h)
	end
	@testset "pairs" begin
	    h = HalfIntArray(rand(2,2), 0:1, 0:1)
	    p = pairs(IndexLinear(), h)
	    @test axes(p) == axes(h)
	    @test keys(p) == LinearIndicesHalfInt(h)
	    p = pairs(IndexCartesian(), h)
	    @test axes(p) == axes(h)
	    @test keys(p) == CartesianIndicesHalfInt(h)
	end
end

@testset "show" begin
	io = IOBuffer()

	function testshow(io, x)
		show(io, x)
		take!(io)
		show(io, MIME"text/plain"(), x)
		take!(io)
	end

	h = HalfIntArray{Float64,2}(undef,-1:1,-1:1)
	testshow(io,h)
	
	h = HalfIntArray{Float64}(undef,-1:1)
	testshow(io,h)

	s = SpinMatrix{Float64}(undef,1)
	testshow(io,s)

	r = HalfIntegerArrays.IdOffsetRange(1:3, 2)
	testshow(io,r)

	r = HalfIntegerArrays.OneTo(half(2))
	testshow(io,r)

	c = CartesianIndexHalfInt(1,1)
	testshow(io,c)

	h0D = HalfIntArray(zeros())
	hempty_1D = HalfIntArray{Float64}(undef,0:-1)
	hempty_2D = HalfIntArray{Float64}(undef,0:-1,0:-1)
	hempty_3D = HalfIntArray{Float64}(undef,0:-1,0:-1,0:-1)
	Base.show_nd(stdout,hempty_3D,Base.print_matrix,true)
	v = HalfIntArray(rand(2),0:1)

	h0 = HalfIntArray(rand(1,1),0:0,0:0)
	s0 = SpinMatrix(rand(1,1),0)
	h1 = HalfIntArray(rand(3,3),-1:1,-1:1)
	s1 = SpinMatrix(rand(3,3), 1)
	h1by2 = HalfIntArray(rand(2,2), -half(1):half(1), -half(1):half(1))
	s1by2 = SpinMatrix(rand(2,2), 1//2)
	hundef = HalfIntArray{String}(undef,1:1)

	for arr in [h1, s1, h1by2, s1by2, h0, s0, v, 
		h0D, hempty_1D, hempty_2D, hempty_3D, hundef]

		c = CartesianIndicesHalfInt(arr)
		testshow(io, c)
		testshow(io, parent(c))
		l = LinearIndicesHalfInt(arr)
		testshow(io, l)
		testshow(io, parent(l))

		if arr isa AbstractVecOrMat
			testshow(io, adjoint(arr))
			testshow(io, transpose(arr))
			testshow(io, adjoint(LinearIndicesHalfInt(arr)))
			testshow(io, transpose(LinearIndicesHalfInt(arr)))
		end
	end
	
	h3D = HalfIntArray(rand(3,3,20),-1:1,-1:1, 1:20)
	testshow(io, h3D)
	show(IOContext(io, :limit => true), MIME"text/plain"(), h3D)
	show(IOContext(io, :limit => false), MIME"text/plain"(), h3D)

	vh3D = @view h3D[:,:,:]
	testshow(io, vh3D)

	c = CartesianIndicesHalfInt(h3D)
	testshow(io, c)
	testshow(io, parent(c))
	Base.showarg(io, parent(c), true)
	l = LinearIndicesHalfInt(h3D)
	testshow(io, l)
	testshow(io, parent(l))
	Base.showarg(io, parent(l), true)

	Base.showarg(io, pairs(c), true)
	Base.showarg(io, pairs(IndexCartesian(), v), true)
	Base.showarg(io, pairs(IndexLinear(), v), true)
end
using HalfIntegerArrays
using Documenter

DocMeta.setdocmeta!(HalfIntegerArrays, :DocTestSetup, :(using HalfIntegerArrays); recursive=true)

makedocs(;
    modules=[HalfIntegerArrays],
    authors="Jishnu Bhattacharya",
    repo="https://github.com/jishnub/HalfIntegerArrays.jl/blob/{commit}{path}#L{line}",
    sitename="HalfIntegerArrays",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jishnub.github.io/HalfIntegerArrays.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jishnub/HalfIntegerArrays.jl",
)

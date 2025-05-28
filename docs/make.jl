using MetidaNLM
using Documenter

DocMeta.setdocmeta!(MetidaNLM, :DocTestSetup, :(using MetidaNLM); recursive=true)

makedocs(;
    modules=[MetidaNLM],
    authors="Vladimir Arnautov",
    sitename="MetidaNLM.jl",
    format=Documenter.HTML(;
        canonical="https://PharmCat.github.io/MetidaNLM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/PharmCat/MetidaNLM.jl",
    devbranch="main",
)

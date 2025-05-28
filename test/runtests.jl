using MetidaNLM
using Test, MetidaNCA, CSV, DataFrames, Distributions, LabelledArrays

pkdata2  = CSV.File(joinpath(dirname(pathof(MetidaNCA)), "..", "test", "csv", "pkdata2.csv")) |> DataFrame

@testset "MetidaNLM.jl" begin

    model = MetidaNLM.@nlmodel begin
        @fixed begin
            kₐ ∈ Domain(Real, 0.4, eps(), Inf)
            kₑ ∈ Domain(Real, 0.1, eps(), Inf)
            V ∈ Domain(Real, 10.0, eps(), Inf)
            sigma ∈ Domain(Real, 10.0, eps(), Inf)
        end

        @model begin
            c = 100.0 * kₐ * (exp(-kₑ * t) - exp(-kₐ * t)) / (V * (kₐ - kₑ))
        end

        @errormodel begin 
            obs ~ Normal(c, sigma)
        end
    end


    pki = pkimport(pkdata2, :Time, :Concentration, :Subject)

    @test_nowarn dfres = MetidaNLM.fit(pki, model)

end








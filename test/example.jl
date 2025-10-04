
using MetidaNCA # use MetidaNCA to load PK data
using CSV, DataFrames, Distributions, LabelledArrays, Random, LinearAlgebra
using OrderedCollections

# load PK data from MetidaNCA (https://raw.githubusercontent.com/PharmCat/MetidaNCA.jl/refs/heads/main/test/csv/pkdata2.csv)
pkdata2  = CSV.File(joinpath(dirname(pathof(MetidaNCA)), "..", "test", "csv", "pkdata2.csv")) |> DataFrame
pki = pkimport(pkdata2, :Time, :Concentration, :Subject)
# Data example:
#=
Subject	Formulation	Time	Concentration
1	T	0	0
1	T	0.5	178.949
1	T	1	190.869
1	T	1.5	164.927
2	R	0	0
2	R	0.5	62.222
2	R	1	261.177
2	R	1.5	234.063
=#

# Example of model macro
# One compartment PK model with absorbtion? 2 random effects
# Define model
model = MetidaNLM.@nlmodel begin
    # Parameters for oprimization
    @parameters begin
        kₐ    ∈ Domain(Real, 0.4, eps(), Inf)
        kₑ    ∈ Domain(Real, 0.1, eps(), Inf)
        V     ∈ Domain(Real, 10.0, eps(), Inf)
        sigma ∈ Domain(Real, 10.0, eps(), Inf)
        omega ∈ Domain(Matrix, [0.1 0; 0 0.1], [eps() -Inf; -Inf eps()], [Inf Inf; Inf Inf])
    end
    # Random effect parameters
    @random begin
        η ~ MvNormal(omega)
    end
    # model equations
    # define V_, kₑ_, D as additional parameters and then calculate final equation c   =  kₐ_ * (exp(-kₑ_ * t) - exp(-kₐ_ * t)) / (V_ * (kₐ_ - kₑ_))  wherer D = 100
    @model begin
        V_  = V *  exp(η[1])
        kₑ_ = kₑ * exp(η[2])
        kₐ_ = kₐ
        D   = 100.0
        c   = kₐ_ * (exp(-kₑ_ * t) - exp(-kₐ_ * t)) / (V_ * (kₐ_ - kₑ_))
    end
    # differential equations foy dynamic part
    # Not used yet only for test purpose
    @diffeq begin
        ∂C = -C*kₐ
    end
    # initial condition for differential equations 
    # Not used yet only for test purpose
    @initial begin 
        C = 4.0
    end
    # output parameters - only c (V_, kₑ_, D  - temororary)
    @outparams begin 
        c 
    end
    # residual error model (simple additime Normal error)
    @errormodel begin 
        obs ~ Normal(c, sigma)
    end
end



# define fixed effects
params = (kₐ = 0.4, kₑ = 0.1, V = 10.0, sigma = 10., omega = [0.1 0; 0 0.1])
# define random effects
rand_params  = (η = [0.3, 0.2],)
# model execution
# funtion made with modelfunc function in @nlmodel macros 
model.modelefunc(0.6, params, rand_params)
# calculate model parameters for subject 1 in pki subject dataset
map(pki[1].time) do t  model.modelefunc(t, params, rand_params).c end

# Distribution for time point 0.6
model.distfunc(0.6, params, rand_params)



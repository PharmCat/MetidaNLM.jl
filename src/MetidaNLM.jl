module MetidaNLM

using LinearAlgebra, LabelledArrays, OrderedCollections, Distributions, DataFrames

using DifferentialEquations
using MetidaNCA, Optim

import Base: length, size, show
import MetidaNCA: AbstractIdData, PKSubject, DataSet, getdata, getid

export length, size, show
const substr = ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"]

abstract type AbstractDomain end

struct TwoStep end
struct AllPooled end
struct AllPooledWeighted end

struct PKSubjectParameters <: AbstractIdData
    parameters
    id
end

struct NLModelResult
    result
    ds
end

# Domain structure - describe domeains - Real, Vector, Matrix, minimal and maximum values of parameters
struct Domain{Type, T} <: AbstractDomain
    init::T
    lower::T 
    upper::T
    function Domain(dt::Type, init::T, lower::T, upper::T) where T 
        if !(dt in (Real, Vector, Matrix)) error("Unsupported Domain type") end

        if dt == Real
        elseif dt == Vector
            (length(init) == length(lower) == length(upper)) || error("Different length")
        elseif dt == Matrix
            (size(init) == size(lower) == size(upper)) || error("Different size")
        end
        new{dt, T}(init, lower, upper)
    end
end

function idkeys(data::DataSet)
    d = getdata(data)
    s = Set(keys(getid(first(d))))
    if length(d) > 1
        for i = 2:length(d)
            for k in keys(getid(d[i]))
                push!(s, k)
            end
        end
    end
    s
end

function replacedigits(s)
    replace(s, "0" => "₀", "1" => "₁", "2" =>"₂", "3" =>"₃", "4" =>"₄", "5" =>"₅", "6" =>"₆", "7" =>"₇", "8" =>"₈", "9" =>"₉")
end
function tupletosubstr(t::Tuple{Vararg{Int}})
    str = replacedigits(string(t[1]))
    if length(t) > 1
        for i = 2:length(t)
            str *= ","*replacedigits(string(t[i]))
        end
    end
    str
end
#function ntconv(nt::NT, s)  where NT <: NamedTuple{<: Tuple{N, Symbol}, NTuple{N, V}} where N where V
#        Tuple(NamedTuple{s}(nt)) 
#end
function ntconv(nt::NT, s)  where NT <: NamedTuple{S, V} where V where S
        Tuple(NamedTuple{s}(nt)) 
end
function ntconv(nt::NT, s)  where NT <:  LArray 
        Tuple(nt[collect(s)]) 
end
function Base.length(d::D)  where D <: Domain
    length(d.init)
end
function Base.size(d::D) where D <: Domain
    size(d.init)
end


struct NLModel
    params       # parameters for optimization
    domains      # domains list
    randdepparams # parameterst in params that used in random part
    randparams   # random parameters
    randexpr     # expressions for random parameters
    modelparams  # parameters for model equations
    modelvars
    modelexpr    # model expressions
    errorexpr    # expreaaion for residual error model
    modelefunc   # generated model function
    diffeqvars
    outparams    # output parameters from model equation
    diffeqexpr   # differential equation expressions
    initsvals    # initial values
    diffeqfunc   # solution for differential equations
    distfunc     # function to make distributiions
end
# This function make model function from expressions and lists of parameters
# Make function f(t, model_parameters, indivdual_random_parasmeters)
function modelfunc(modelexpr, postexpr, modelparams, randparams, modelvars)
    modelvars_tuple = tuple(modelvars...)
    x = quote
        function generated_function(t, fixed_params, random_params)
            $([:(local $v = fixed_params.$v)  for v in modelparams]...)
            $([:(local $v = random_params.$v) for v in randparams]...)
            $(modelexpr)
            $(postexpr)
            return NamedTuple{$modelvars_tuple}($(Expr(:tuple, modelvars...)))
        end
    end
    eval(x)
end

# calculate model values for subject s and model m with parameters p, and individual random effect r (if r == Nothing random parameters will be generated)
# 
function subjectmodelvals(m::NLModel, s, p, r = Nothing)
    # <- CODE HERE to get individual random values, if r is nothing make zero values
    map(s.time) do t m.modelefunc(t, p, r) end
end

# generate tuple of random parameters equals zero  
function genrandpzeros(m::NLModel)
    # <- CODE HERE !!! 
end

# make distribution for random effects model m with parameters p
function randdistfunc(m::NLModel, p)
    # <- CODE HERE !!! 
end

# generate function to evaluate model  and get distributions for observation model (residuals) 
function modeldistfunc(modelexpr, postexpr, params, randparams, modelvars, errorexpr)
    modelvars_tuple = tuple(modelvars...)
    x = quote
        function (t, params, random_params)
            $([:(local $v = params.$v)  for v in params]...)
            $([:(local $v = random_params.$v) for v in randparams]...)
            $(modelexpr)
            $(postexpr)
            $(errorexpr)
            # return only last distribution (should be corrected)
        end
    end
    eval(x)
end

# calculate model values for subject s and model m with parameters p, and individual random effect r (if r == Nothing random parameters will be generated) adnd make distributions
# 
function subjectmodeldistvals(m::NLModel, s, p, r = Nothing)
    # <- CODE HERE to get individual random values, if r is nothing make zero values
    map(s.time) do t m.distfunc(t, p, r) end
end



# TBD 
function diffeqfunk(diffeqexpr, diffeqvars, params)
    x = quote
        function (du, u, p, t)

            $(Expr(:block, [:($(v) = p.$v) for v in params]...))

            $(Expr(:block, [:($(v) = u.$v) for v in diffeqvars]...))

            $(Expr(:block, [:(du[$(QuoteNode(k))] = $(v)) for (k,v) in diffeqexpr]...))
        end
    end
    eval(x)
end



# show functions:
function Base.show(io::IO, m::NLModel) 
    println("    Non-linear model")
    print("Parameters: $(first(m.params))")
    if length(m.params) > 1
        for i  = 2:length(m.params)
             print(", ", m.params[i])
        end
    end
    println()
    for p in m.params
        println("    ", p, " ∈ ", m.domains[p])
    end
    println("Variables: $(string(m.modelvars))")
end
function Base.show(io::IO, d::Domain{Real, T}) where T
    print("Real Domain")
end
function Base.show(io::IO, d::Domain{Vector, T}) where T
    print("Vector Domain")
end
function Base.show(io::IO, d::Domain{Matrix, T}) where T
    print("Matrix Domain")
end
function Base.show(io::IO, m::NLModelResult) 
    show(io, m.ds) 
end

# include file with macro code
    include("macros.jl")

end
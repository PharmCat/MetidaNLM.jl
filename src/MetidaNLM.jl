module MetidaNLM

using LinearAlgebra, LabelledArrays, OrderedCollections, Distributions, DataFrames
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
    fixedparams
    fixeddomain
    modelparams
    modelvars
    modelexpr
    modelefunc
    distfunc
end

function modelfunc(mexpr, pexpr, args, params, vars)
    pex = Expr(:tuple, params...)
    aex = Expr(:tuple, args...)
    x = quote
        function ($aex...)
            $(mexpr)
            $(pexpr)
            $pex
        end
    end
    eval(x)
end

function modeldistfunc(mexpr, pexpr, dexpr, args, params, vars)
    pex = Expr(:tuple, params...)
    aex = Expr(:tuple, args...)
    x = quote
        function (time, $aex...)
            map(time) do t
                $(mexpr)
                $(pexpr)
                $(dexpr)
            end
        end
    end
    eval(x)
end


# Optimization vector length and initials
function optvec(model::NLModel)
    i = 0
    s = Symbol[]
    m = Dict()
    for (k,d) in model.fixeddomain
        i += length(d)
        if size(d) == ()
            push!(s, k)
            m[k] = k => 0
        else
            for j = 1:length(d)
                sind = Symbol(string(k)*"_"*replacedigits(string(j)))
                push!(s, sind)
                m[sind] = k => j
            end
        end
    end
    i,s,m
end


function subjlogpdfsum(subject, model, params)
    dists = model.distfunc(subject.time, ntconv(params, tuple(model.fixedparams...))...)
    obs = MetidaNCA.getobs(subject)
    res = 0.0
    for i = 1:length(dists)
        res += logpdf(dists[i], obs[i])
    end
    res
end

function fit(subject::S, model) where S <: PKSubject
    n, names, m = optvec(model)
    inits       = @LArray zeros(n) tuple(names...)
    lb = similar(inits)
    ub = similar(inits)
    for name in names
        if size(model.fixeddomain[name]) == ()
            inits[name] = model.fixeddomain[name].init
            lb[name] = model.fixeddomain[name].lower
            ub[name] = model.fixeddomain[name].upper
        else
            inits[name] = model.fixeddomain[m[name][1]].init[m[name][2]]
            lb[name] = model.fixeddomain[m[name][1]].lower[m[name][2]]
            ub[name] = model.fixeddomain[m[name][1]].upper[m[name][2]]
        end
    end

    optfunnk(x) = -subjlogpdfsum(subject, model, x)

    inner_optimizer = NelderMead()
    
    #return optfunnk, inits, lb, ub
    results = optimize(optfunnk, lb, ub, inits, Fminbox(inner_optimizer))

    PKSubjectParameters(Optim.minimizer(results), subject.id)
end

function fit(data::D, model) where D <: DataSet
    df = DataFrame([name => Float64[] for name in model.fixedparams])
    ds = DataSet(PKSubjectParameters[])
    for s in getdata(data)
        res = fit(s, model) 
        push!(df, res.parameters)
        push!(ds, res)
    end
    idk = idkeys(data)
    for k in idk
        df[!, k] = getid(data, :, k)
    end

    NLModelResult(ds, df)
end


function Base.show(io::IO, m::NLModel) 
    println("    Non-linear model")
    print("Parameters: $(first(m.fixedparams))")
    if length(m.fixedparams) > 1
        for i  = 2:length(m.fixedparams)
             print(", ", m.fixedparams[i])
        end
    end
    println()
    for p in m.fixedparams
        println("    ", p, " ∈ ", m.fixeddomain[p])
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

    include("macros.jl")

end
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
    dists = model.distfunc(subject.time, ntconv(params, tuple(model.params...))...)
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
        if size(model.domains[name]) == ()
            inits[name] = model.domains[name].init
            lb[name] = model.domains[name].lower
            ub[name] = model.domains[name].upper
        else
            inits[name] = model.domains[m[name][1]].init[m[name][2]]
            lb[name] = model.domains[m[name][1]].lower[m[name][2]]
            ub[name] = model.domains[m[name][1]].upper[m[name][2]]
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


using MetidaNCA, CSV, DataFrames, Distributions, LabelledArrays

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


pkdata2  = CSV.File(joinpath(dirname(pathof(MetidaNCA)), "..", "test", "csv", "pkdata2.csv")) |> DataFrame
pki = pkimport(pkdata2, :Time, :Concentration, :Subject)

#params = (kₐ = 0.4, kₑ = 0.1, V = 10.0, sigma = 10.)

params = LVector(kₐ = 0.4, kₑ = 0.1, V = 10.0, sigma = 10.)

MetidaNLM.subjlogpdfsum(pki[1], model, params)

params = LVector(kₐ = 0.4, kₑ = 0.1, V = 10.0, sigma = 10.)

MetidaNLM.optvec(model)


#optf, init, lb, ub 

results = MetidaNLM.fit(pki[1], model)




kₐ = 0.4
kₑ = 0.1
V = 10.0
sigma = 10.
cf(t) = 100.0 * kₐ * (exp(-kₑ * t) - exp(-kₐ * t)) / (V * (kₐ - kₑ))

cf2(t, kₐ, kₑ, V, sigma) = 100.0 * kₐ * (exp(-kₑ * t) - exp(-kₐ * t)) / (V * (kₐ - kₑ))

dists = @. Normal(cf(pki[1].time ), sigma)
sum(@. logpdf(dists, pki[1].obs))

optfunk(x) = begin 
    dists = @. Normal(cf2(pki[1].time, x... ), x[4])
    -sum(@. logpdf(dists, pki[1].obs))
end

optfunk([0.4,0.1,10.0,10.])

inner_optimizer = NelderMead()
    
results = optimize(optfunk, [eps(),eps(),eps(),eps()], [Inf,Inf,Inf,Inf], [0.4,0.1,10.0,10.], Fminbox(inner_optimizer))


results = optimize(optfunk, [eps(),eps(),eps(),eps()], [Inf,Inf,Inf,Inf], params, Fminbox(inner_optimizer))


results = optimize(optf,lb, ub, init, Fminbox(inner_optimizer))


Optim.minimizer(results)


optfunk2(x) = begin 
    -MetidaNLM.subjlogpdfsum(pki[1], model, x)
end

results = optimize(optfunk2, [eps(),eps(),eps(),eps()], [Inf,Inf,Inf,Inf], params, Fminbox(inner_optimizer))

dfres = MetidaNLM.fit(pki, model)



d = MetidaNLM.Domain(Real, 1.0 , 1.0 , 3.0)
length(d)

MetidaNLM.Domain(Vector, [1.0] , [1.0] , [3.0])


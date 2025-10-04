

using MetidaNCA
using CSV, DataFrames, Distributions, LabelledArrays, Random, LinearAlgebra
using OrderedCollections

Pkg.add(name="MetidaNCA", version="0.7.1")

pkdata2  = CSV.File(joinpath(dirname(pathof(MetidaNCA)), "..", "test", "csv", "pkdata2.csv")) |> DataFrame
pki = pkimport(pkdata2, :Time, :Concentration, :Subject)

# Usage example
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
        η ~ MvNormal(rp)
    end
    # model equations
    @model begin
        #c = 100.0 * kₐ * exp(η) * (exp(-kₑ * t) - exp(-kₐ * exp(η) * t)) / (V * (kₐ * exp(η) - kₑ)) 
        V_  = V *  exp(η[1])
        kₑ_ = kₑ * exp(η[2])
        c   =  D / V_ * exp(-kₑ_ * t)
    end
    # differential equations foy dynamic part
    @diffeq begin
        ∂C = -C*kₐ
    end
    # initial condition for differential equations 
    @initial begin 
        C = 4.0
    end
    @outparams begin 
        c 
    end

    
    # residual error model
    @errormodel begin 
        obs ~ Normal(c, sigma)
    end
end



# define fixed effects
params = (kₑ = 0.1, D = 1.0, V = 10.0, sigma = 10., omega = [0.1 0; 0 0.1])
# define random effects
rand_params  = (η = [0.3, 0.2],)
# model execution
model.modelefunc(0.6, params, rand_params)
map(pki[1].time) do t  model.modelefunc(t , params, rand_params).c end


#=
function log_posterior(time, y, params, sigma, omega)
    residuals      = y - map(time) do t  model.modelefunc(t , params).c end
    #return residuals
    log_likelihood = -0.5 / sigma * sum(residuals.^2) - 0.5 * length(time) * log(2 * pi * sigma)
    #return log_likelihood
    log_prior      = logpdf(MvNormal(omega), params.η)
    return log_likelihood + log_prior
end
res = log_posterior(pki[1].time, pki[1].obs, params, 10.0, [0.1 0; 0 0.1])
=#

function log_posterior(time, y, params, rand_params)
    dists          = map(time) do t  Normal(model.modelefunc(t, params, rand_params).c, sqrt(fixed_params.sigma)) end
    #residuals      = y - map(time) do t  model.modelefunc(t , params).c end
    log_likelihood = sum(@. logpdf(dists, y)) 
    #return log_likelihood
    #log_likelihood = -0.5 / sigma * sum(residuals.^2) - 0.5 * length(time) * log(2 * pi * sigma)
    log_prior      = logpdf(MvNormal(params.omega), rand_params.η)
    #return log_prior
    return log_likelihood + log_prior
end
log_posterior(pki[1].time, pki[1].obs, params, rand_params)

params = (kₑ = 0.1, D = 1.0, V = 10.0, sigma = 10., η = [0.37639575296588584, 0.6476912933952113])
log_posterior(pki[1].time, pki[1].obs, params, 10.0, [0.1 0; 0 0.1])

# Генерирует выборку из p(η | y, theta) с помощью Метрополиса-Гастингса.
function metropolis_hastings(time, y, params, sigma, omega, n_samples, proposal_cov, rng)
    η_current      = rand(rng, MvNormal(omega))
    params_current = (kₑ = 0.1, D = 1.0, V = 10.0, sigma = 10., η = η_current)
    println(η_current)
    log_p_current  = log_posterior(time, y, params_current, sigma, omega)
    #return log_p_current
    samples = Vector{Vector{Float64}}()
    accept_count = 0
    firstprint = true
    firstprob = true
    for _ in 1:n_samples + 200
        η_proposed = rand(rng, MvNormal(η_current, proposal_cov))
        params_proposed = (kₑ = 0.1, D = 1.0, V = 10.0, sigma = 10., η = η_proposed)
        if firstprint println(η_proposed) end
        log_p_proposed = log_posterior(time, y, params_proposed, sigma, omega)
        if firstprint println(log_p_proposed) end
        log_p_current  = log_posterior(time, y, params_current, sigma, omega)
        if firstprint println(log_p_current) end
        if log_p_proposed == -Inf || log_p_current == -Inf
            push!(samples, η_current)
            continue
        end
        diff = log_p_proposed - log_p_current
        if firstprint println(diff) end
        accept_prob = diff > 0 ? 1.0 : exp(diff)
        if firstprint println("prob:", accept_prob) end
        accept_lim = rand(rng)
        if firstprint println("lim:", accept_lim) end
        if accept_lim < accept_prob
            η_current = η_proposed
            params_current = (kₑ = 0.1, D = 1.0, V = 10.0, sigma = 10., η = η_current)
            accept_count += 1
            if firstprob println(accept_count) end
            firstprob = false 
        end
        push!(samples, η_current)
        firstprint = false
    end
    @info "Acceptance rate: $(accept_count / n_samples)"
    return mean(samples[201:end])  # Burn-in 200 итераций
end
res = metropolis_hastings(pki[1].time, pki[1].obs, params, 10., [0.1 0; 0 0.1], 500, 0.1 * [0.1 0; 0 0.1], MersenneTwister(123))


#metropolis_hastings(time, y, params, sigma, omega, n_samples, proposal_cov, rng)
function saem(data, init, max_iter::Int=200, K_1::Int=50, epsilon::Float64=1e-4, rng::MersenneTwister=MersenneTwister(42))
    N = length(data)
    total_observations = sum(x-> length(x), data.ds)
    theta = init
    S_1 = zeros(2)
    S_2 = zeros(2, 2)
    S_3 = 0.0
    theta_history = [deepcopy(theta)]
    z_history = Vector{Matrix{Float64}}()
    omega = [0.1 0; 0 0.1]
    sigma = 10.
    for k in 1:max_iter
        # S-шаг
        z_k = map(data.ds) do subj metropolis_hastings(subj.time, subj.obs, theta, sigma, omega, 500, 0.1 * omega, rng)  end
        z_k = hcat(z_k...)'
        push!(z_history, copy(z_k))
        
        # A-шаг
        gamma_k = k < K_1 ? 1.0 : 1.0 / (k - K_1 + 1)
        S_1 = (1 - gamma_k) * S_1 + gamma_k * sum(z_k, dims=1)[:]
        S_2 = (1 - gamma_k) * S_2 + gamma_k * sum(z_k[i, :] * z_k[i, :]' for i in 1:N)

        
        S_3 = (1 - gamma_k) * S_3 + gamma_k * sum(map(pki.ds) do subj sum((subj.obs .- map(time) do t model.modelefunc(t , params).c end).^2) end)
        
        # M-шаг
        old_theta = deepcopy(theta)
        new_mu     = S_1 / N
        new_Omega  = S_2 / N - new_mu * new_mu'
        new_sigma2 = S_3 / total_observations

        @info "Raw Omega before ensure_positive_definite: $new_Omega"
        new_Omega = ensure_positive_definite(new_Omega)


        theta = (V = new_mu[1], kₑ = new_mu[2], D = 1.0, sigma = sqrt(new_sigma2), η = η_current)
        omega = new_Omega
        sigma = sqrt(new_sigma2)
        push!(theta_history, deepcopy(theta))
        
        # Проверка сходимости
        theta_diff = norm(theta.mu - old_theta.mu)^2 + norm(theta.Omega - old_theta.Omega)^2 + (theta.sigma2 - old_theta.sigma2)^2
        if theta_diff < epsilon
            @info "Сходимость достигнута на итерации $k"
            break
        end
    end
    
    return theta, theta_history, z_history
end






struct ModelParameters
    mu::Vector{Float64}      # Средние значения (mu_V, mu_k)
    Omega::Matrix{Float64}   # Ковариационная матрица
    sigma2::Float64         # Дисперсия ошибки
end
function modelx(t, z_i::Vector{Float64}, theta, D_i::Float64)
    V_i = theta.mu[1] * exp(z_i[1])
    k_i = theta.mu[2] * exp(z_i[2])
    return D_i / V_i * exp.(-k_i * t)
end
function log_posterior(z_i::Vector{Float64}, y_i::Vector{Float64}, theta::ModelParameters, t::Vector{Float64}, D_i::Float64)
    residuals = y_i - modelx(t, z_i, theta, D_i)
    #return residuals
    log_likelihood = -0.5 / theta.sigma2 * sum(residuals.^2) - 0.5 * length(t) * log(2 * pi * theta.sigma2)
    #return log_likelihood
    try
        log_prior = logpdf(MvNormal(theta.Omega), z_i)
        #return log_prior
        return log_likelihood + log_prior
    catch e
        @warn "Ковариационная матрица Omega неположительно определённая: $e"
        return -Inf
    end
end
theta = ModelParameters([10., 0.1], [0.1 0; 0 0.1], 10.)
log_posterior([0.3, 0.2], pki[1].obs, theta, pki[1].time, 1.0)
log_posterior([0.37639575296588584, 0.6476912933952113], pki[1].obs, theta, pki[1].time, 1.0)


function metropolis_hastings(y_i, theta::ModelParameters, t, D_i, n_samples=500, proposal_cov::Matrix{Float64}=0.1 * theta.Omega, rng=MersenneTwister())
    z_current = rand(rng, MvNormal(theta.Omega))
    println(z_current)
    log_p_current = log_posterior(z_current, y_i, theta, t, D_i)
    #return log_p_current
    samples = Vector{Vector{Float64}}()
    accept_count = 0
    firstprint = true
    firstprob = true 
    for _ in 1:n_samples + 200
        z_proposed = rand(rng, MvNormal(z_current, proposal_cov))
        if firstprint println(z_proposed) end
        log_p_proposed = log_posterior(z_proposed, y_i, theta, t, D_i)
        if firstprint println(log_p_proposed) end
        log_p_current = log_posterior(z_current, y_i, theta, t, D_i)
        if firstprint println(log_p_current) end
        if log_p_proposed == -Inf || log_p_current == -Inf
            push!(samples, z_current)
            continue
        end
        diff = log_p_proposed - log_p_current
        if firstprint println(diff) end
        accept_prob = diff > 0 ? 1.0 : exp(diff)
        if firstprint println(accept_prob) end
        accept_lim = rand(rng)
        if firstprint println(accept_lim) end
        if accept_lim < accept_prob
            z_current = z_proposed
            accept_count += 1
            if firstprob println(accept_count) end
            firstprob = false 
        end
        push!(samples, z_current)
        firstprint = false
    end
    @info "Acceptance rate: $(accept_count / n_samples)"
    return mean(samples[201:end])  # Burn-in 200 итераций
end
res2 = metropolis_hastings(pki[1].obs, theta, pki[1].time, 1., 500, 0.1 * theta.Omega, MersenneTwister(123))

function ensure_positive_definite(matrix::Matrix{Float64}, min_eigenvalue::Float64=1e-6)
    eig = eigen(matrix)
    eigenvalues = max.(eig.values, min_eigenvalue)
    return eig.vectors * Diagonal(eigenvalues) * eig.vectors'
end

function saem(y::Matrix, t::Vector, D_i, max_iter::Int=200, K_1::Int=50, epsilon::Float64=1e-4, rng::MersenneTwister=MersenneTwister(42))
    N, n_i = size(y)
    theta = ModelParameters([10., 0.1], [0.1 0; 0 0.1], 10.)
    S_1 = zeros(2)
    S_2 = zeros(2, 2)
    S_3 = 0.0
    theta_history = [deepcopy(theta)]
    z_history = Vector{Matrix{Float64}}()
    
    for k in 1:max_iter
        # S-шаг
        z_k = [metropolis_hastings(y[i, :], theta, t, D_i, 500, 0.1 * theta.Omega, rng) for i in 1:N]
        z_k = hcat(z_k...)'
        push!(z_history, copy(z_k))
        
        # A-шаг
        gamma_k = k < K_1 ? 1.0 : 1.0 / (k - K_1 + 1)
        S_1 = (1 - gamma_k) * S_1 + gamma_k * sum(z_k, dims=1)[:]
        S_2 = (1 - gamma_k) * S_2 + gamma_k * sum(z_k[i, :] * z_k[i, :]' for i in 1:N)
        S_3 = (1 - gamma_k) * S_3 + gamma_k * sum(sum((y[i, :] - modelx(t, z_k[i, :], theta, D_i)).^2) for i in 1:N)
        
        # M-шаг
        old_theta = deepcopy(theta)
        new_mu = S_1 / N
        new_Omega = S_2 / N - new_mu * new_mu'
        @info "Raw Omega before ensure_positive_definite: $new_Omega"
        new_Omega = ensure_positive_definite(new_Omega)
        new_sigma2 = S_3 / (N * n_i)
        theta = ModelParameters(new_mu, new_Omega, new_sigma2)
        push!(theta_history, deepcopy(theta))
        
        # Проверка сходимости
        theta_diff = norm(theta.mu - old_theta.mu)^2 + norm(theta.Omega - old_theta.Omega)^2 + (theta.sigma2 - old_theta.sigma2)^2
        if theta_diff < epsilon
            @info "Сходимость достигнута на итерации $k"
            break
        end
    end
    
    return theta, theta_history, z_history
end

saem( permutedims(pki[1].obs), pki[1].time, 1., 500, 50, 1e-4, MersenneTwister(42))




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



using DifferentialEquations,LabelledArrays 


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



model.diffeqfunc

u0 = LVector(C = 0.4,)
tspan = (0.0, 100.0)
p = LVector(kₐ = 0.4, kₑ = 0.1, V = 10.0, sigma = 10.)
ts = 0:10:100

prob = ODEProblem(model.diffeqfunc, u0, tspan, p)
sol = solve(prob, saveat = ts)





function model(t, z_i::Vector{Float64}, theta, D_i::Float64)
    V_i = theta.mu[1] * exp(z_i[1]) # Объем распределения
    k_i = theta.mu[2] * exp(z_i[2]) # Константа элиминации
    Kabs = theta.mu[2]  # Константа абсорбции
    return D_i * Kabs  * (exp(-k_i * t) - exp(-Kabs * t)) / (V_i * (Kabs - k_i)) 
end




using Metida,  CSV, DataFrames, StatsModels, CategoricalArrays, Distributions
df0.varbin = map(df0.var) do x if x > 1 return 1 else return 0 end end 
lmm = MetidaGLMM.GLMM(MetidaGLMM.@glmmformula(varbin ~ sequence + period + formulation), df0;
    random=Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    dist=MetidaGLMM.BinomialDist, link=MetidaGLMM.LogitLink)
MetidaGLMM.fit_pql!(lmm)
println(StatsBase.coef(lmm))



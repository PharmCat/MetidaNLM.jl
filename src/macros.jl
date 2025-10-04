
# Make vector of model parameters from expressions
# Use only variabels
# Not include :t 
# used to get all tokens from expression 
function addexprtok!(v, ex::Expr)
    if ex.head == :call
        if length(ex.args) > 1
            for i = 2:length(ex.args)
                addexprtok!(v, ex.args[i])
            end
        end
    elseif ex.head == :ref
        for a in ex.args
            addexprtok!(v, a)
        end
    end
    v
end
function addexprtok!(v, ex::Symbol)
    # add Symbol only if it is not equal :t - time
    if ex != :t 
        push!(v, ex) 
    end 
end
function addexprtok!(v, ex)
    nothing
end



"""
model construction
"""
macro nlmodel(expression)
    # Parameters for optimization
    params      = OrderedSet{Symbol}()
    # Covariate list
    covariates  = OrderedSet{Symbol}()
    # Domains for parameters
    domains     = OrderedDict{Symbol, Domain}()
    # parameters that used to calculate individal random values (distribution parameters)
    randdepparams = OrderedSet{Symbol}()
    # Random parameters
    randparams  = OrderedSet{Symbol}()
    # Model parameters (fixed parameters)
    modelparams = OrderedSet{Symbol}()
    # model output variables
    modelvars   = OrderedSet{Symbol}()
    # Observed variable list (real data)
    obsvars     = OrderedSet{Symbol}()
    # derivatives list
    diffeqvars  = OrderedSet{Symbol}()
    # output parameters
    outparams   = OrderedSet{Symbol}()
    # Expression for @model part - mode equations
    local modelexpr  = Expr(:block)

    local postexpr   = Expr(:block)
    # Expressinos for differential equations
    local diffeqexpr = OrderedDict{Symbol, Expr}()
    # Expression for error-model (Distribution)
    local errorexpr  = Expr(:block)
    # Expression for random-effect-model (Distribution)
    local randexpr   = Expr(:block)

    local initsvals  = nothing

    # Parsing sections 1 (step 1)
    # First step - find @parameters and make parameter list - params
    for ex ∈ expression.args
        (ex isa LineNumberNode) && continue
        # All section should be basen on macrocall
        if ex.head != :macrocall 
             @warn "Expression \"$(string(ex))\" found in the model description. It will be dropped."
             continue
        end
        println("Macro: ", ex.args[1])
        # parameters section, parse parameters names and domains (type and limitation for each parameter)
        # from each line symbols from left side collected in params
        # expression fron right side of ∈ evaluated and collected in domains
        # result of right side evaluating should be Domain
        if ex.args[1] == Symbol("@parameters")
            println("Macro @parameters:")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end

            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue # not parse LineNumberNode
                # if we have ∈ function - we can parse paremeters
                if ex_.head == :call && ex_.args[1] in (:∈, :in)
                    # not add t as parameter - this name is reserved for time
                    ex_.args[2] == :t && continue
                    println("Param: ", ex_.args[2])
                    # Push in parameter list
                    push!(params, ex_.args[2])
                    println("Definition: ", ex_.args[3])
                    # evaluate expression after ∈ and store result in Dict domains
                    domains[ex_.args[2]] = eval(ex_.args[3])
                else
                    println("wrong parameter definition...")
                end
            end
            break
        end
    end

    # Parsing sections 2 (step 2)
    for ex ∈ expression.args
        (ex isa LineNumberNode) && continue
        # All section should be basen on macrocall
        if ex.head != :macrocall 
             @warn "Expression \"$(string(ex))\" found in the model description. It will be dropped."
             continue
        end

        println("Macro: ", ex.args[1])
        if ex.args[1] == Symbol("@parameters")
            continue
        # random-model section - each line is random effect and corresponding Distribution
        # right side collected in randparams
        # left side (expression) after ~ collected in randexpr
        elseif ex.args[1] == Symbol("@random")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end
            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                if ex_.head == :call && ex_.args[1] == :~
                    lsv   = ex_.args[2] # left side variable (random parameter)
                    rse   = ex_.args[3] # right side expression (Distribution)
                    if lsv == :t 
                        error("Time variable cann't be redefined.")
                    end
                    println("Random param: ", lsv)
                    # add left side token to randvars
                    push!(randparams, lsv)
                    println("Definition: ", rse)
                    # add right side expression to randexpr
                    push!(randexpr.args, :($lsv = $(rse)))  
                else
                    println("wrong @random definition...")
                end
            end
        # Covariates parsing (list of covariates)
        # all covariates collected in covariates
        elseif ex.args[1] == Symbol("@covariates")
            println(ex)
            if isa(ex.args[3], Expr)
                println(ex.args[3].head)
                if ex.args[3].head == :block
                    expr_ = ex.args[3].args
                elseif ex.args[3].head == :tuple
                    expr_ = ex.args[3].args
                else
                    expr_ = ex.args[3]
                end
            else
                expr_ = ex.args
            end
            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                # CHECK, SHOULD BE SYMBOL
                push!(covariates, ex_)
            end
        # Model equations, parse equations for models
        # Contains simple equations (differential equation not included, DE should be in @diffeq section)
        # each lenr shoul be equation
        # left side before = - output values
        # right side after = - expressions
        # all symbols in right side expressoin should be declared before or should be in params list
        elseif ex.args[1] == Symbol("@model")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end

            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                if ex_.head == :(=)
                    if ex_.args[1] == :t 
                        error("Time variable cann't be redefined.") # time variable can't be changed
                    end
                    lsv   = ex_.args[1] # left side variable
                    rse   = ex_.args[2] # right side expression
                    println("Param: ", lsv)
                    println("Definition: ", rse)
                    addexprtok!(modelparams, rse)
                    push!(modelvars, lsv)
                    push!(modelexpr.args, :($lsv = $(rse)))  
                    println("Model parameters modelparams $modelparams")
                else
                    println("wrong parameter definition...")
                end
            end
        # Differential equations
        # ∂ should be used to declare differential equation
        # Example: ∂C = -C*kₐ
        elseif ex.args[1] == Symbol("@diffeq")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end

            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                if ex_.head == :(=)

                    lsv = ex_.args[1] # left side - derivative token
                    sp = string(lsv)  # make string
                    if sp[1] != '∂' || length(sp) < 2 # first symbol should be ∂ then follow variable name
                        error("wrong diffeq parameter definition...")
                    end
                    lsv = Symbol(chop(sp, head=1, tail=0))
                    # Push in diffeq vars
                    push!(diffeqvars, lsv)
                    # add diff eq expression 
                    diffeqexpr[lsv] = :($(ex_.args[2]))
                else
                    println("wrong diffeq parameter definition...")
                end
            end
        # Initial conditions for differnential equations
        elseif ex.args[1] == Symbol("@initial")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end

            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                inits_n = Symbol[]
                inits_v = Float64[]
                if ex_.head == :(=)
                    lsv   = ex_.args[1] # left side variable
                    rse   = ex_.args[2] # right side expression
                    if lsv == :t 
                        error("Time variable cann't be redefined.")
                    end
                    push!(inits_n, lsv)
                    push!(inits_v, rse)
                    if length(inits_n) > 1
                        initsvals = @LArray inits_v (inits_n...)
                    else
                        initsvals = @LArray inits_v ((inits_n[1], ))
                    end
                else
                    println("wrong initial parameter definition...")
                end
            end

        # Define output parameters
        # collect parameters for output 
        elseif ex.args[1] == Symbol("@outparams")
            println(ex)
            if isa(ex.args[3], Expr)
                println(ex.args[3].head)
                if ex.args[3].head == :block
                    expr_ = ex.args[3].args
                elseif ex.args[3].head == :tuple
                    expr_ = ex.args[3].args
                else
                    expr_ = ex.args[3]
                end
            else
                expr_ = ex.args
            end
            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                # CHECK, SHOULD BE SYMBOL
                push!(outparams, ex_)
            end
        # Additional calculations, post clculation step
        elseif ex.args[1] == Symbol("@post")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end
            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                if ex_.head == :(=)
                    lsv   = ex_.args[1] # left side variable
                    rse   = ex_.args[2] # right side expression
                    if lsv == :t 
                        error("Time variable cann't be redefined.")
                    end
                    # SOME CODE
                else
                    println("wrong parameter definition...")
                end
            end
        # Parse Statistical error model
        # Each line is output some observed data
        # left side of ~ - data in pk subject
        # right side of ~ expression to evaluate Distribution
        # all symbols in right side should be listed in params or modelvars (or outparams)
        elseif ex.args[1] == Symbol("@errormodel")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end
            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                if ex_.head == :call && ex_.args[1] == :~ # only ~ symbol used to define residual errors
                    lsv = ex_.args[2] # left side observed variable corresponding to real data
                    rse = ex_.args[3] # Distribution for residual error should be distribution
                    println("Param: ", lsv)
                    push!(obsvars, lsv) # add to observed variable list
                    println("Definition: ", rse)
                    push!(errorexpr.args, :($lsv = $(rse)))  # add to error model distributions expression list
                else
                    println("wrong @errormodel definition...")
                end
            end
        # Additional parameters for calculation
        elseif ex.args[1] == Symbol("@carry")
            # TBD
        # If unknown macrocall
        else # if we find uncnown macrocall
            @warn "Unknown expression... $(ex.args[1])!!!"
            continue
        end
    end

    println("set diff")
    # REMOVE ALL TOKENS FROM MODEL PARAMETER LIST THAT IN RANDOM PARAMETER LIST
    setdiff!(modelparams, randparams)
    # REMOVE ALL TOKENS FROM MODEL PARAMETER LIST THAT IN THE LEFT SIDE OF MODEL EQUATIONS
    setdiff!(modelparams, modelvars)


    for ex in randexpr.args  # add all tokens from random parameters defenition to randdepparams list
        addexprtok!(randdepparams, ex.args[2])
    end
    # all parameters in randdepparams in randdepparams should be in modelparams
    if any(x-> x ∉ params, randdepparams) 
        error("Some parameters in randdepparams ($randdepparams) not in params ($modelparams)" ) 
    end

    if length(outparams) > 0 # if output parameters defined used only parameter in outparams list
        all(x-> x in modelvars, outparams) || error("parameters")
    else # else all parameters in modelvars used as output parameters
        outparams = modelvars
    end

    # if no initsvals - all initvals will be equal zero
    if isnothing(initsvals) 
        initsvals  = LVector(; [d => zero(Float64) for d in diffeqvars]...)
    end

    println("makemodel functiom")
    # make model function
    mf = modelfunc(modelexpr, postexpr, modelparams, randparams,  outparams)

    println("modeldistfunc functiom")
    # make distribution function 
    mdf = modeldistfunc(modelexpr, postexpr, params, randparams,  outparams, errorexpr)

    println("diffeqfunk functiom")
    # make differential eqation solving function
    def = diffeqfunk(diffeqexpr, diffeqvars, params)
    # make model object
    return NLModel(params, domains, randdepparams, randparams, randexpr, modelparams, modelvars, modelexpr, errorexpr, mf, diffeqvars, outparams, diffeqexpr, initsvals, def, mdf)
end
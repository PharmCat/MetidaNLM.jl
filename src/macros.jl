

macro nlmodel(expression)

    fixedparams = OrderedSet{Symbol}()
    covariates  = OrderedSet{Symbol}()
    fixeddomain = OrderedDict{Symbol, Domain}()
    modelparams = OrderedSet{Symbol}()
    modelvars   = OrderedSet{Symbol}()
    obsvars     = OrderedSet{Symbol}()

    local modelexpr = Expr(:block)
    local postexpr  = Expr(:block)
    local errorexpr = Expr(:block)

    for ex ∈ expression.args
        (ex isa LineNumberNode) && continue
        if ex.head != :macrocall 
             @warn "Expression \"$(string(ex))\" found in the model description. It will be dropped."
             continue
        end

        println("Macro: ", ex.args[1])
        
            # Fixed effects
        if ex.args[1] == Symbol("@fixed")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end

            for ex_ ∈ expr_

                (ex_ isa LineNumberNode) && continue

                if ex_.head == :call && ex_.args[1] in (:∈, :in)
                    ex_.args[2] == :t && continue
                    println("Param: ", ex_.args[2])
                    push!(fixedparams, ex_.args[2])
                    println("Definition: ", ex_.args[3])
                    fixeddomain[ex_.args[2]] = eval(ex_.args[3])
  
                else
                    println("wrong parameter definition...")
                end
            end

        elseif ex.args[1] == Symbol("@random")

            # Covariates
        elseif ex.args[1] == Symbol("@covariates")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end

            # Model equations
        elseif ex.args[1] == Symbol("@model")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end

            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                if ex_.head == :(=)
                    ex_.args[2] == :t && continue
                    p = ex_.args[1]
                    println("Param: ", p)
                    println("Definition: ", ex_.args[2])
                    
                    push!(modelvars, p)
                    push!(modelexpr.args, :($p = $(ex_.args[2])))  

                else
                    println("wrong parameter definition...")
                end
            end
        
            # Differential equations
        elseif ex.args[1] == Symbol("@diffeq")

        elseif ex.args[1] == Symbol("@initial")

            # Additional calculations
        elseif ex.args[1] == Symbol("@post")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end

            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                if ex_.head == :(=)
                    ex_.args[2] == :t && continue
                    p = ex_.args[1]

                    println("Param: ", p)
                    println("Definition: ", ex_.args[2])
                    
                    push!(modelparams, p)
                    push!(postexpr.args, :($p = $(ex_.args[2])))  

                else
                    println("wrong parameter definition...")
                end
            end
            
            # Ststistical error model
        elseif ex.args[1] == Symbol("@errormodel")
            if ex.args[3].head == :block
                expr_ = ex.args[3].args
            else
                expr_ = (ex.args[3],)
            end
            for ex_ ∈ expr_
                (ex_ isa LineNumberNode) && continue
                if ex_.head == :call && ex_.args[1] == :~
                    p = ex_.args[2]
                    println("Param: ", ex_.args[2])
                    push!(obsvars, ex_.args[2])
                    println("Definition: ", ex_.args[3])
                    push!(errorexpr.args, :($p = $(ex_.args[3])))  
  
                else
                    println("wrong @errormodel definition...")
                end
            end

        elseif ex.args[1] == Symbol("@carry")

        else
            @warn "Unknown expression..."
            continue
        end
    end

    mf = modelfunc(modelexpr, postexpr, tuple(fixedparams...), tuple(modelparams...), tuple(modelvars...))

    mdf = modeldistfunc(modelexpr, postexpr, errorexpr, tuple(fixedparams...), tuple(modelparams...), tuple(modelvars...))

    return NLModel(fixedparams, fixeddomain, modelparams, modelvars, modelexpr, mf, mdf)
end
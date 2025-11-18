using StatsBase

# TODO: Consider transformation with distance weighting to deal with multimodal
# case => not relevant for the application after Laplace approximation

function transform!(_::Val{1},tsamples,samples,weights)
    θbar = mean(samples,dims=2)
    θbarw = mean(samples,weights,dims=2)

    for (i,c) in enumerate(eachcol(samples))
        tsamples[:,i] .=  c .- θbar .+ θbarw
    end
end

function transform!(_::Val{2},tsamples,samples,weights)
    θbar,v = mean_and_var(samples,2)
    θbarw, vw = mean_and_var(samples,weights,2)

    for (i,c) in enumerate(eachcol(samples))
        tsamples[:,i] .=  (sqrt.(vw) ./ sqrt.(v)) .* (c .- θbar) .+ θbarw
    end
end

function transform!(_::Val{3},tsamples,samples,weights)
    θbar,Σ = mean_and_cov(samples,2)
    θbarw, Σw = mean_and_cov(samples,weights,2)

    F = svd!(Σ)
    Fw = svd!(Σw)
    M = (Fw.U * Diagonal(sqrt.(Fw.S))) / (F.U * Diagonal(sqrt.(F.S)))
    # C = cholesky!(Σ)
    # Cw = cholesky!(Σw)
    # M = Cw.L / C.L

    for (i,c) in enumerate(eachcol(samples))
        tsamples[:,i] .=  M * (c .- θbar) .+ θbarw
    end
end

function mmis(target_logden,proposal_logden,samples::AbstractMatrix,fun=x->one(eltype(x));
              k_threshold=min(1-1/log10(size(samples,2)),0.7),)
    specific_weights(x) = target_logden(x) - proposal_logden(x) + log(fun(x))


    log_weights = map(specific_weights,eachcol(samples))

    k = psis!(log_weights) 
    tsamples = similar(samples)
    log_tweights = similar(log_weights)

    w = similar(log_tweights)
    w .= exp.(log_weights .- logsumexp(log_weights))
    

    while k > k_threshold
        for j in 1:3
            transform!(Val(j),tsamples,samples,Weights(w))
            for (i,c) in enumerate(eachcol(tsamples))
                log_tweights[i] = specific_weights(c)
            end
            tk = psis!(log_tweights)

            if tk < k 
                k = tk
                copyto!(samples,tsamples)
                copyto!(log_weights,log_tweights)
                w .= exp.(log_weights .- logsumexp(log_weights))
                break
            end

            if j == 3
                @warn "Transformations could not reduce k below the threshold. The results are likely innacurrate"
                return  (; ξ = k, estimate = logsumexp(log_weights), log_iw = log_weights, samples = samples)
            end
        end
    end

    return  (; ξ = k, estimate = logsumexp(log_weights), log_iw = log_weights, samples = samples)
end

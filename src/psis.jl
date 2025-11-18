# Simplified implamentation from ParetoSmooth.jl to
# avoid dependency
#
using LinearAlgebra
using LogExpFunctions
using Statistics


@static if VERSION >= v"1.8"
    @inline exp_inline(x) = @inline exp(x)
else
    const exp_inline = exp
end

"""
    gpd_fit(
        sample::AbstractVector{T<:Real},
        wip::Bool=true, 
        min_grid_pts::Integer=30, 
        sort_sample::Bool=false
    ) -> (ξ::T, σ::T)

Return a named list of estimates for the parameters ξ (shape) and σ (scale) of the
generalized Pareto distribution (GPD), assuming the location parameter is 0.

# Arguments

  - `sample::AbstractVector`: A numeric vector. The sample from which to estimate
    the parameters.
  - `wip::Bool = true`: Logical indicating whether to adjust ξ based on a weakly informative
    Gaussian prior centered on 0.5. Defaults to `true`.
  - `min_grid_pts::Integer = 30`: The minimum number of grid points used in the fitting
    algorithm. The actual number used is `min_grid_pts + ⌊sqrt(length(sample))⌋`.

# Note

Estimation method taken from Zhang, J. and Stephens, M.A. (2009). The parameter ξ is the
negative of k.
"""
function gpd_fit(
    sample::AbstractVector{T};
    wip::Bool=true,
    min_grid_pts::Integer=30,
    sort_sample::Bool=false,
) where T<:Real

    len = length(sample)
    # sample must be sorted, but we can skip if sample is already sorted
    if sort_sample
        sample = sort(sample; alg=QuickSort)
    end


    grid_size = min_grid_pts + isqrt(len)  # isqrt = floor sqrt
    n_0 = 10  # determines how strongly to nudge ξ towards .5
    x_star = inv(3 * sample[(len + 2) ÷ 4])  # magic number. ¯\_(ツ)_/¯
    invmax = inv(sample[len])

    # build pointwise estimates of ξ and θ at each grid point
    θ_hats = similar(sample, grid_size)
    @fastmath @. θ_hats = invmax + (1 - sqrt((grid_size + 1) / $(1:grid_size))) * x_star
    ξ_hats = similar(θ_hats)
    for i = eachindex(ξ_hats, θ_hats)
        ξ_hat = zero(eltype(ξ_hats))
        for j = eachindex(sample)
            ξ_hat += log1p(-θ_hats[i] * sample[j])
        end
        ξ_hats[i] = ξ_hat/len
    end

    log_like = ξ_hats  # Reuse preallocated array
    # Calculate profile log-likelihood at each estimate:
    for i = eachindex(ξ_hats, θ_hats)
        ξ_hats[i] = len * (log(-θ_hats[i] / ξ_hats[i]) - ξ_hats[i] - 1)
    end
    # Calculate weights from log-likelihood:
    weights = log_like  # Reuse preallocated array
    log_norm = logsumexp(log_like)
    for i = eachindex(log_like)
        log_like[i] = exp_inline(log_like[i] - log_norm)
    end
    # Take weighted mean:
    θ_hat = zero(Base.promote_eltype(θ_hats, weights))
    @simd for i = eachindex(θ_hats, weights)
        θ_hat += θ_hats[i] * weights[i]
    end
    ξ = zero(θ_hat)
    @simd for i = eachindex(sample)
        ξ += log1p(-θ_hat * sample[i])
    end
    ξ /= len
    σ::T = -ξ / θ_hat

    # Drag towards .5 to reduce variance for small len
    if wip
        @fastmath ξ = (ξ * len + 0.5 * n_0) / (len + n_0)
    end

    return ξ, σ

end


"""
    gpd_quantile(p::T, k::T, sigma::T) where {T<:Real} -> T

Compute the `p` quantile of the Generalized Pareto Distribution (GPD).

# Arguments

  - `p`: A scalar between 0 and 1.
  - `ξ`: A scalar shape parameter.
  - `σ`: A scalar scale parameter.

# Returns

A quantile of the Generalized Pareto Distribution.
"""
function gpd_quantile(p, ξ::T, sigma::T) where {T <: Real}
    return sigma * expm1(-ξ * log1p(-p)) / ξ
end


function psis(
    log_ratios::AbstractVector{T};
) where T <: Real

    sample_size = length(log_ratios)
    tail_length = floor(Int,min(0.2*sample_size,3sqrt(sample_size)))
    tail_start = sample_size - tail_length + 1

    log_ratio_index = collect(zip(log_ratios,Base.OneTo(length(log_ratios))))
    partialsort!(log_ratio_index,(tail_start-1):sample_size;by=first)
    log_is_ratios = first.(log_ratio_index)
    @views tail = log_is_ratios[tail_start:sample_size]
    biggest = tail[end]
    @. tail = exp(tail - biggest)


    cutoff = exp(log_is_ratios[tail_start - 1]-biggest)
    ξ = if any(isinf.(tail))
        Inf 
    else
        @. tail = tail - cutoff

        # save time not sorting since tail is already sorted
        ξraw, σ = gpd_fit(tail)
        # regularization for small tail_length
        ξ = (tail_length*ξraw + 10*0.5)/(tail_length+10)

        @. tail = gpd_quantile(($(1:tail_length) - 0.5) / tail_length, ξ, σ) + cutoff
        ξ
    end

    @. tail = log(tail) + biggest

    # unsort the ratios to their original position:
    invpermute!(log_is_ratios, last.(log_ratio_index))

    return ξ, log_is_ratios
end

function psis!(
    log_ratios::AbstractVector{T};
) where T <: Real

    sample_size = length(log_ratios)
    tail_length = floor(Int,min(0.2*sample_size,3sqrt(sample_size)))
    tail_start = sample_size - tail_length + 1

    log_ratio_index = collect(zip(log_ratios,Base.OneTo(length(log_ratios))))
    partialsort!(log_ratio_index,(tail_start-1):sample_size;by=first)
    log_ratios .= first.(log_ratio_index)
    biggest = log_ratios[end]
    threshold = log(eps(zero(T)))
    if log_ratios[tail_start] - biggest < threshold
        tail_start += findfirst(
            >(biggest + threshold),
            view(log_ratios,tail_start:sample_size)
        ) - 1
        tail_length = sample_size-tail_start+1
    end
    tail = view(log_ratios,tail_start:sample_size)
    @. tail = exp(tail - biggest)


    cutoff = exp(log_ratios[tail_start - 1]-biggest)
    ξ = if any(isinf.(tail))
        Inf 
    else
        @. tail = tail - cutoff

        # save time not sorting since tail is already sorted
        ξraw, σ = gpd_fit(tail)

        # regularization for small tail_length
        ξ = (tail_length*ξraw + 10*0.5)/(tail_length+10)
        @. tail = gpd_quantile(($(1:tail_length) - 0.5) / tail_length, ξ, σ) + cutoff
        ξ
    end

    @. tail = log(tail) + biggest

    # unsort the ratios to their original position:
    invpermute!(log_ratios, last.(log_ratio_index))

    return ξ
end

function psis(target_logden,proposal_logden,samples::AbstractMatrix,
              fun=x->one(eltype(x)))
    specific_weights(x) = target_logden(x) - proposal_logden(x) + log(fun(x))

    # box[] = (target_logden,proposal_logden,samples,fun,k_threshold,specific_weights)

    log_weights = specific_weights.(eachcol(samples))

    k = psis!(log_weights) 

    return  (; ξ = k, estimate = logsumexp(log_weights), log_iw = log_weights, samples = samples)
end

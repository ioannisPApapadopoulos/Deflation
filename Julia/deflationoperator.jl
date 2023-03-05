function deflation_step_adjustment(x::AbstractVector{T}, update::AbstractVector{T}, known_roots::AbstractVector, M::AbstractMatrix{T}, p::Int=2, α::T=one(T)) where T
        dMy = getdMy(x, update, known_roots, M, p, α)
        minv = one(T) / deflation_evaluate(x, known_roots, M, p, α)
        one(T)/(one(T) - minv*dMy)
end

function getdMy(x::AbstractVector, update::AbstractVector, known_roots::AbstractVector, M::AbstractMatrix, p::Int=2, α::T=one(T)) where T
    deriv = deflation_derivative(x, known_roots, M, p, α)
    # defcon has a minus sign here, but that's because PETSc
    # calculates the update so that state = state - update rather
    # than state = state + update
    deriv*M*update
end

function norm_squared(y::AbstractVector, root::AbstractVector, M::AbstractMatrix)
    # Inner product matrix is necessary to compute correct norm 
    # induced by function space
    (y - root)' *  M * (y - root)
end

function derivative_norm_squared(y::AbstractVector, root::AbstractVector)
    # Derivative is in the dual space, hence should be a row vector
    2 * (y - root)'
end

function deflation_evaluate(y::AbstractVector{T}, known_roots::AbstractVector, M::AbstractMatrix{T}, p::Int=2, α::T=one(T)) where T
    m = one(T)
    for iter = 1:lastindex(known_roots)
        normsq = norm_squared(y, known_roots[iter], M)
        factor = normsq^(-p/2) + α
        m = m * factor
    end
    m
end

function deflation_derivative(y::AbstractVector{T}, known_roots::AbstractVector, M::AbstractMatrix, p::Int=2, α::T=one(T)) where T

    N = lastindex(known_roots)
    
    normsqs = [norm_squared(y, known_roots[i], M) for i in 1:N]
    dnormsqs = [derivative_norm_squared(y, known_roots[i]) for i in 1:N]

    factors = [normsqs[i]^(-p/2) + α  for i in 1:N]
    dfactors = [(-p/2) * normsqs[i]^((-p/2) - one(T)) for i in 1:N]
   
    sum((prod(factors) ./ factors) .* dfactors .* dnormsqs)
end
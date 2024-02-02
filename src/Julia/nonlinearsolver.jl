"""
Basic Newton
"""
function newtonls(x::AbstractVector{T}, residual, jacobian, known_roots::AbstractVector=[], M::AbstractMatrix=Diagonal(ones(T,length(x))), tol::T=1e-9, max_iter::Int=1000, p::Int=2, α::T=one(T)) where T
    iter = 0
    
    r = residual(x)
    @assert r isa AbstractVector
    J = jacobian(x)
    @assert J isa AbstractMatrix

    norm_residual = norm(r,2)
    print("Iteration 0, residual norm = $norm_residual\n")
    while norm_residual > 1e-9  && iter < max_iter
        update = -J \ r   
        if ~isempty(known_roots)
            τ = deflation_step_adjustment(x, update, known_roots, M, p, α)
            update = τ * update
        end

        # update = nls.LineSearch.adjust(x, update, nls.damping)
        x = x + update
        r = residual(x)
        J = jacobian(x)
        norm_residual = norm(r,2)
        iter += 1
        print("Iteration $iter, residual norm = $norm_residual\n")
    end
    if iter == max_iter
        print("Iteration max reached")
    end
    return x
end

"""
Benson-Munson RSLS
"""
function reduced_residual(r::AbstractVector{T}, x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}) where T
    rr = r[:]
    rr[x .<= lb] = min.(rr[x .<= lb], zero(T))
    rr[x .>= ub] = max.(rr[x .>= ub], zero(T))
    return rr
end

function project!(x, lb, ub)
    b = x;
    b[x .< lb] = lb[x .< lb]
    b[x .> ub] = ub[x .> ub]
    return b
end

# The reduced space Benson & Munson active-set strategy. 
# A Newton method that enforces box constraints. 
function bensonmunson(x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}, residual, 
    jacobian, known_roots::AbstractVector=[], M::AbstractMatrix=Diagonal(ones(T,length(x))), 
    tol::T=1e-9, max_iter::Int=1000, p::Int=2, α::T=one(T)) where T

    index = Vector(1:lastindex(x))
    iter = 0
    inactive = index[:]

    project!(x, lb, ub)

    r = residual(x); @assert r isa AbstractVector
    J = jacobian(x); @assert J isa AbstractMatrix

    norm_residual_Ω = norm(reduced_residual(r, x, lb, ub))
    print("Iteration 0, residual norm = $norm_residual_Ω\n")

    n = length(x)

    while norm_residual_Ω > tol && iter < max_iter
        update = zeros(T, n)
        update[inactive] = -J[inactive, inactive] \ r[inactive]
        project!(update,lb-x,ub-x)

        if norm(update) < 1e-10
            update = -r
        end

        if ~isempty(known_roots)
            τ = deflation_step_adjustment(x, update, known_roots, M, p, α)
            update =  τ * update
        end

        # update = nls.LineSearch.adjust(x, update, nls.damping)
        x = x + update
        project!(x,lb,ub)

        r = residual(x)
        J = jacobian(x)
        norm_residual_Ω = norm(reduced_residual(r, x, lb, ub))

        active_lb = index[x .<= lb]
        active_lb = intersect(active_lb, index[r .> zero(T)])
        active_ub = index[x .>=ub]
        active_ub = intersect(active_ub, index[r .< zero(T)])
        active = vcat(active_lb, active_ub)
        index2 = index[:]
        index2[active] .= 0
        inactive  = findall(x->x!=0, index2)

        iter += 1
        print("Iteration $iter, residual norm = $norm_residual_Ω\n")
    end
    
    if iter == max_iter
        print("Iteration max reached")
        # disp(normResidualOmega)
    end
    return x
end

"""
Benson-Munson SSLS
"""
function Φ(a::AbstractVector{T}, b::AbstractVector{T}) where T
    @assert length(a) == length(b)
    a + b - sqrt.(a.^2 + b.^2)     
end

function dΦ(a::AbstractVector{T}, b::AbstractVector{T}) where T
    @assert length(a) == length(b)
    if any(abs.(a) .> 1e-6) || any(abs.(b) .> 1e-6)
        return one(T) .- a./sqrt.(a.^2 + b.^2)
    else
        return ones(T, length(a)) ./ 2
    end
end

function FB(x, r, lb, ub, wherenoconstraint, wherelbconstraint, whereubconstraint, whereequalconstraint, wherebothconstraint)
    T = eltype(x)
    out = zeros(T,length(x))
    #  FIXME add a check for all indices here
    out[wherenoconstraint] = r[wherenoconstraint]
    
    idx = whereubconstraint
    out[idx] = Φ(ub[idx] - x[idx], -r[idx])
    
    idx = wherelbconstraint
    out[idx] = Φ(x[idx]-lb[idx],r[idx])
    
    idx = wherebothconstraint
    out[idx] = Φ(x[idx]-lb[idx],Φ(ub[idx]-x[idx],-r[idx]))
    
    idx = whereequalconstraint
    out[idx] = lb[idx] - x[idx]
    return out
end

function computeScaleAndShift(x, r, lb, ub, wherenoconstraint, wherelbconstraint, whereubconstraint, whereequalconstraint, wherebothconstraint)
    T = eltype(x)
    n = length(x)
    dshift = ones(T, n)
    dscale = ones(T, n)
    
    dshift[wherenoconstraint] .= zero(T)
    dscale[wherenoconstraint] .= one(T)
    
    idx = whereubconstraint
    dshift[idx] = dΦ(ub[idx] - x[idx], -r[idx])
    dscale[idx] = dΦ(-r[idx],ub[idx]-x[idx])
    
    idx = wherelbconstraint
    dshift[idx] = dΦ(x[idx] - lb[idx], r[idx])
    dscale[idx] = dΦ(r[idx], x[idx] - lb[idx])
   
    
    idx = wherebothconstraint;
    dshift1 = dΦ(x[idx] - lb[idx], -Φ(ub[idx] - x[idx], -r[idx]))
    dscale1 = dΦ(-Φ(ub[idx]-x[idx],-r[idx]), x[idx] - lb[idx])
    dshift2 = dΦ(ub[idx]-x[idx],-r[idx])
    dscale2 = dΦ(-r[idx],ub[idx] - x[idx])
    dshift[idx] = dshift1 + dscale1.*dshift2
    dscale[idx] = dscale1.*dscale2
    
    idx = whereequalconstraint
    dshift[idx] .= one(T)
    dscale[idx] .= zero(T)
    return (dshift, dscale)
end

function ssls(x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}, residual, 
    jacobian, known_roots::AbstractVector=[], M::AbstractMatrix=Diagonal(ones(T,length(x))), 
    tol::T=1e-9, max_iter::Int=1000, p::Int=2, α::T=one(T)) where T

    iter = 0
    
    project!(x,lb,ub)
    
    bound_tol = 1e10
    wherenoconstraint = intersect(findall(x->x<-bound_tol, lb), findall(x->x>bound_tol, ub))
    wherelbconstraint = intersect(findall(x->x>=-bound_tol, lb), findall(x->x>bound_tol, ub))
    whereubconstraint = intersect(findall(x->x<-bound_tol,lb), findall(x->x<=bound_tol, ub))
    whereequalconstraint = findall(x->x==ub, lb)
    wherebothconstraint = intersect(findall(x->x>=-bound_tol,lb), findall(x->x<=bound_tol,ub))
    
    J = jacobian(x); @assert J isa AbstractMatrix
    r = residual(x); @assert r isa AbstractVector
             
    fb = FB(x, r, lb, ub, wherenoconstraint,wherelbconstraint,whereubconstraint,whereequalconstraint,wherebothconstraint)
    normFB = norm(fb)
    print("Iteration 0, residual norm = $normFB\n")
    
    while normFB > tol && iter < max_iter
        
        (dshift, dscale) = computeScaleAndShift(x, r, lb, ub, wherenoconstraint,wherelbconstraint,whereubconstraint,whereequalconstraint,wherebothconstraint)
        shiftedJacobian = Diagonal(dshift) + Diagonal(dscale) * J

        update = -shiftedJacobian \ fb
        
        if ~isempty(known_roots)
            τ = deflation_step_adjustment(x, update, known_roots, M, p, α)
            update =  τ * update
        end
        
        # update = nls.LineSearch.adjust(x, update, nls.damping)
        x = x + update
        
        r = residual(x)
        J = jacobian(x)                
      
        fb = FB(x, r, lb, ub, wherenoconstraint,wherelbconstraint,whereubconstraint,whereequalconstraint,wherebothconstraint)
        normFB = norm(fb)               
        iter += 1
        print("Iteration $iter, residual norm = $normFB\n")
    end
    
    if iter == max_iter
        print("Iteration max reached")
    end
    return x
end
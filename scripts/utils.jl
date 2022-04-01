module Utils 
    using LinearAlgebra
    
    """
    Return skew symmetric matrix of a vector. Equivalent to a cross product
    operation
    """
    function skew(v)
        return [0 -v[3] v[2];
                v[3] 0 -v[1];
                -v[2] v[1] 0]
    end

    const H = [zeros(1,3); I];

    function L(Q)
        [Q[1] -Q[2:4]'; Q[2:4] Q[1]*I + skew(Q[2:4])]
    end

    function R(Q)
        [Q[1] -Q[2:4]'; Q[2:4] Q[1]*I - skew(Q[2:4])]
    end

    function quaternion_differential(quat)
        return L(quat) * H
    end 

    function ricatti(A, B, Q, R)
        P = copy(Q)
        P_prev = zero(P)
        K = zeros(size(B'))
        res = norm(P-P_prev)
        while res > 1e-7
            K = (R + B'*P_prev*B)\ B'*P*A
            P = Q + A'*P_prev*(A-B*K)
            res = norm(P-P_prev)
            P_prev = copy(P)
        end
        return P, K
    end 

    function ρ(ϕ)
        q = 1/sqrt(1+norm(ϕ)) * [1;ϕ]
    end
    
    function q_inv(q)
        return [q[1];-q[2:end]]
    end


end 
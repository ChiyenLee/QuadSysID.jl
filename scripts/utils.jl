
"""
Return skew symmetric matrix of a vector. Equivalent to a cross product
operation
"""
function skew(v)
    return [0 -v[3] v[2];
            v[3] 0 -v[1];
            -v[2] v[1] 0]
end
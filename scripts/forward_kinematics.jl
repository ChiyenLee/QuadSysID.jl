
using ForwardDiff
const x_hip = 0.183 # hip dispalcement from the trunk frame 
const y_hip = 0.047 
const Δy_thigh = 0.08505 # thigh displacement from the hip frame 
const l_limb = 0.2 

"""
    fk(q::AbstractVector{T})
        Given joint angles compute the position of feet relative to obody rigin  
        p = [x_FR, y_FR, z_FR, x_FL, y_FL, z_FL, ...]
        feet order: FR, FL, RR, RL 
"""
function fk(q::Matrix)
    q_FL = q[:,1]
    q_FR = q[:,2]
    q_RL = q[:,3]
    q_RR = q[:,4]

    foot_pos = zeros(3,4)
    foot_pos[:,1] .= fk_leg( 1,  1, q_FL)
    foot_pos[:,2] .= fk_leg( 1, -1, q_FR)
    foot_pos[:,3] .= fk_leg(-1,  1, q_RL)
    foot_pos[:,4] .= fk_leg(-1, -1, q_RR)
    return foot_pos
end

function fk_leg(x_mir, y_mir, θ)
    return [-l_limb * sin(θ[2] + θ[3]) - l_limb * sin(θ[2]) + x_hip * x_mir; 
            y_hip * y_mir + Δy_thigh* y_mir * cos(θ[1]) + l_limb*sin(θ[1])*cos(θ[2]+θ[3]) + l_limb*sin(θ[1])*cos(θ[2]); 
            Δy_thigh * y_mir *  sin(θ[1]) - l_limb * cos(θ[1]) * cos(θ[2] + θ[3]) - l_limb * cos(θ[1])*cos(θ[2]) ] 
end 

function dfk_leg(x_mir, y_mir, θ)
    return ForwardDiff.jacobian(t->fk_leg(x_mir, y_mir, t), θ)
end 

function dfk(q)
    q_FL = q[:,1]
    q_FR = q[:,2]
    q_RL = q[:,3]
    q_RR = q[:,4]
    J = zeros(4, 3, 3)
    J[1,:,:] = dfk_leg(1, 1, q_FL)
    J[2,:,:] = dfk_leg(1, -1, q_FR)
    J[3,:,:] = dfk_leg(-1, 1, q_RL)
    J[4,:,:] = dfk_leg(-1, -1, q_RR)
    return J
end

# function fk_leg(q::Matrix, in2, in3)
#     t2 = cos(q[1])
#     t3 = cos(q[2])
#     t4 = cos(q[3])
#     t5 = sin(q[1])
#     t6 = sin(q[2])
#     t7 = sin(q[3])
#     t8 = q[2] + q[3]
#     t9 = sin(t8)

#     p = zeros(3)
#     p[1] = (((in3[1] + in2[3] * t9) - in3[5] * t9) - t6 * in3[4]) + in2[1] * cos(t8)
#     p[2] = ((((((((in3[2] + in2[2] * t2) + in3[2] * t2) + t3 * t5 * in3[4]) +
#                     in2[1] * t3 * t5 * t7) + in2[1] * t4 * t5 * t6) - in2[3] * t3 *
#                                                       t4 * t5) + in2[3] * t5 * t6 * t7) + in3[5] * t3 * t4 * t5) - in3[5] * t5 * t6 * t7;
#     t8 = in2[1] * t2;
#     t9 = in2[3] * t2;
#     p_bf_tmp = in3[5] * t2;
#     p_bf[3] = (((((((in2[2] * t5 + in3[3] * t5) - t2 * t3 * in3[4]) - t8 * t3 * t7)
#                   - t8 * t4 * t6) + t9 * t3 * t4) - t9 * t6 * t7) - p_bf_tmp * t3 *
#                                                                     t4) + p_bf_tmp * t6 * t7;
# end 
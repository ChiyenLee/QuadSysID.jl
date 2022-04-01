include("utils.jl")
using Main.Utils
struct CentroidalModel 
    J::Matrix
    m::Real 
    p1::Vector # FR 
    p2::Vector # FL 
    p3::Vector # RR 
    p4::Vector # RL
end 

function dynamics(model::CentroidalModel, x, u)
    g = [0.0, 0.0, -9.81]
    p = x[1:3]
    ṗ = x[4:6]
    q = x[7:10]
    ω = x[11:13]

    Q = Utils.H'Utils.L(q)*Utils.R(q)'Utils.H #Rotation matrix
    r1 = (model.p1 - p)
    r2 = (model.p2 - p)
    r3 = (model.p3 - p)
    r4 = (model.p4 - p)
    M = [I(3) I(3) I(3) I(3);
         Utils.skew(r1) Utils.skew(r2) Utils.skew(r3) Utils.skew(r4)]
    u_out = M * u
    f = u_out[1:3]
    τ = u_out[4:6]

    # linear dynamics
    p̈ = 1/model.m * f + g
    
    # attitude dynamics
    damping = 0.5
    stiction = 0.
    q̇ = 0.5*Utils.L(q)*Utils.H*ω
    ω_dot =  model.J \ (Q'*τ - damping *ω - Utils.skew(ω) * model.J * ω - stiction * sign.(ω))

    return [ṗ; p̈; q̇; ω_dot]

end

function linearize(model, x, u)
    A = ForwardDiff.jacobian(t->RobotDynamics.dynamics(model, t, u), x)
    B = ForwardDiff.jacobian(t->RobotDynamics.dynamics(model, x, t), u)
    return A, B
end 

function dynamics_rk4(model, x,u,h)
    #RK4 integration with zero-order hold on u
    f1 = dynamics(model, x, u)
    f2 = dynamics(model, x + 0.5*h*f1, u)
    f3 = dynamics(model, x + 0.5*h*f2, u)
    f4 = dynamics(model, x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end
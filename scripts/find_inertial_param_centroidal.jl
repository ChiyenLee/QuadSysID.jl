using Pkg; Pkg.activate(".")
using Rotations
using SparseArrays
using LinearAlgebra
using ForwardDiff
using PyPlot
using ApproxFun
using DataStructures

pygui(true)
include("centroidal_model.jl")
include("utils.jl")
include("data_processing.jl")


axis = 3

## Initialize a model 
m = 12.454 # kg 
p_FR = [0.1308, -0.1308, 0.0]
p_FL = [0.1308,  0.1308, 0.0]
p_RR = [-0.1308, -0.1308, 0.0]
p_RL = [-0.1308,  0.1308, 0.0]
# I_inertia = Diagonal([0.1,0.25,0.3])
I_inertia = [0.1 0.0 0.0; 
             0.0 0.25 0.;
             0.0 0. 0.3]

model = CentroidalModel(I_inertia, m, p_FR, p_FL, p_RR, p_RL)

## Come up with a simple LQR controller to control headings 
x_eq = zeros(13); x_eq[7] = 1.0; 
u_eq = zeros(12); u_eq[3:3:12] .= m * 9.81 / 4

h = 0.001 # discretization step
A = ForwardDiff.jacobian(t->dynamics(model,t, u_eq), x_eq)
B = ForwardDiff.jacobian(t->dynamics(model, x_eq, t), u_eq)
Ad = A*h + I # super coarse approximate discretizatoin step 
Bd = B*h 
G = blockdiag(sparse(1.0I(6)), 
              sparse(Utils.quaternion_differential(x_eq[7:10])), 
              sparse(1.0I(3)))
Ad = G' * Ad * G 
Bd = G' * Bd

Q = Matrix(Diagonal([20, 20, 10000., 1.0, 1.0, 10., 1600, 1800, 1500., 10., 5, 1.])) *1e-2
R = Matrix(Diagonal(kron(ones(4), [0.5, 0.5, .5]))) * 1e-2
P, K = Utils.ricatti(Ad, Bd, Q, R)

## Simulattion variabRobotDynamics.
dt = 0.001
tf = 3.0
times = 0:dt:tf 
xs = zeros(length(times), 13); xs[1,7] = 1.0
ω₀ = 2 * π * 5.0
ang_des_list = sin.(times * ω₀) * 0.05
x_des = zeros(13); x_des[7] = 1.0
δx = zeros(12)

# roll, pitch, yaw velocity data to be collected 
ωs = zeros(length(times), 3) 
us = zeros(length(times), 12)
ts = copy(times)

# control delay 
control_delay = Queue{Vector{Float64}}()
delay_time_step = 1

## Oscillation for yaw 
xs = zeros(length(times), 13); xs[1,7] = 1.0
ang_des = zeros(3)
for i in 1:length(times)-1
    quat = xs[i,7:10]
    ang_des[axis] = ang_des_list[i]
    quat_des = Utils.ρ(ang_des)
    quat_err = Utils.L(quat) * quat_des
    ang_err = quat_err[2:end]

    δx[1:6] = xs[i,1:6] - x_des[1:6]
    δx[7:9] = ang_err 
    δx[10:12] = xs[i,11:13] - x_des[11:13]
    u = -K * δx 

    if length(control_delay) < delay_time_step
        enqueue!(control_delay, u)
        u = zeros(12)
    else 
        enqueue!(control_delay, u)
        u = dequeue!(control_delay)
    end
    us[i,:] = u

    xs[i+1, :] = dynamics_rk4(model, xs[i,:], u, dt)
end 
ωs = xs[:,10+axis]

function grf_to_torques(model, x, u)
    quat = x[7:10]
    p = x[1:3]
    Q = UnitQuaternion(quat)
    r1 = (model.p1 - p)
    r2 = (model.p2 - p)
    r3 = (model.p3 - p)
    r4 = (model.p4 - p)
    M = [I(3) I(3) I(3) I(3);
         Utils.skew(r1) Utils.skew(r2) Utils.skew(r3) Utils.skew(r4)]
    u_out = M * u 

    τ_body = Q' * u_out[4:6]
    return τ_body
end 

τs = zeros(length(times), 3)
for i in 1:size(us,1)
    τs[i,:] = grf_to_torques(model, xs[i,:], us[i,:])
end 

ind_cutoff = 2000
t_cutoff = times[ind_cutoff:end] .- times[ind_cutoff] 

## Fitting a fourier series
S_ω = Fourier(Interval(t_cutoff[1], t_cutoff[end]))
f_ω = Fun(S_ω, ApproxFun.transform(S_ω, ωs[ind_cutoff:end]))
f_ω = chop(f_ω, 0.001)

D = Derivative(S_ω)
df_ω = D * f_ω

S_τ = Fourier(Interval(t_cutoff[1], t_cutoff[end]))
f_τ = Fun(S_τ,ApproxFun.transform(S_τ, τs[ind_cutoff:end,3]))
f_τ = chop(f_τ, 0.01)

h = 0.005
t_all = 0:h:3
θ_dot = f_ω.(t_all)
θ_ddot = df_ω.(t_all)
θ_dot_abs = sign.(θ_dot)
τ = f_τ.(t_all) 

n = length(t_all)
A = zeros(n, 3);
for i in 1:n
    A[i,1] = real(θ_ddot[i])
    A[i,2] = real(θ_dot[i])
    A[i,3] = real(θ_dot_abs[i])
end
result = A \ τ

subplot(3,1,1)
plot(times, ωs)
plot(t_all, θ_dot)
xlim([t_all[1], t_all[end]])

subplot(3,1,2)
plot(t_all, θ_ddot)

subplot(3,1,3)
plot(times, τs[:,3])
plot(t_all, τ)
xlim([t_all[1], t_all[end]])

println(result)


##
mag_control, phase_control, f_control, _ = fit_cosine(τ, 1/h)
mag_imu, phase_imu, f_imu, _ = fit_cosine(θ_ddot, 1/h)

println(" ")
println("Interpolated result")
println("imu phase : ", phase_imu)
println("imu freq : ", f_imu )
println("control phase : ", phase_control)
println("control freq : ", f_control)



##
function dynamic_fit_cost(times, sinusoid_params, inertial_params)
    a_τ = sinusoid_params[1]
    f_τ = sinusoid_params[2]
    ϕ_τ = sinusoid_params[3]

    a_i = sinusoid_params[4]
    f_i = sinusoid_params[5]
    ϕ_i = sinusoid_params[6]

    J = inertial_params[1]
    C = inertial_params[2]
    ϕ_off = inertial_params[3]

    ω_i = cos_wave(a_i, ϕ_i, f_i, 0, times)
    ω_dot_i = sin_wave(-a_i * 2*π*f_i, ϕ_i, f_i, 0.0, times)
    τ = cos_wave(a_τ, ϕ_τ + ϕ_off, f_τ, 0, times)

    return sum(((J * ω_dot_i .+ C * ω_i) .- τ).^2)

end


##
mag_control, phase_control, f_control, _ = fit_cosine(τ, 1/h)
mag_angular, phase_angular, f_angular, _ = fit_cosine(θ_dot, 1/h)
sine_params = [mag_control; f_control; phase_control; mag_angular; f_angular; phase_angular]

inertial_params = [0., 0., 0.919329]

res = dynamic_fit_cost(times, sine_params, inertial_params)
i_params = copy(inertial_params)
counter = 0
while counter < 50
    J = ForwardDiff.gradient(t->dynamic_fit_cost(times, sine_params, t), i_params)
    res = dynamic_fit_cost(times, sine_params, i_params)
    δi = -res / J
    i_params = i_params .+ δi'
    counter += 1

end

# println(i_params)

## 
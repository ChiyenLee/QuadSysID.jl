using Pkg
Pkg.activate(".")
using DelimitedFiles
using PyPlot
using LinearAlgebra
using Statistics
using ApproxFun
using DSP

include("forward_kinematics.jl")
include("utils.jl")
include("data_processing.jl")
pygui(true)

## data_folder_path = "sys_id/x_wiggle_01_05"
data_folder_path = "data2/z_wiggle"
JOINT_MSG = "hardware_a1-joint_foot.csv"
IMU_MSG = "hardware_a1-imu.csv"
CONTROL_MSG = "a1_debug-control_output.csv"

joint, joint_header = readdlm(joinpath(data_folder_path, JOINT_MSG), ',', header=true)
control, control_header= readdlm(joinpath(data_folder_path, CONTROL_MSG), ',', header=true)
imu, imu_header = readdlm(joinpath(data_folder_path, IMU_MSG), ',', header=true)
axis_name = split(data_folder_path, '/')[2][1]
axis = axis_name == 'x' ? 1 :
       axis_name == 'y' ? 2 :
       axis_name == 'z' ? 3 : 1
t_start = 2.0
t_end = 2.5

t_joint = joint[:,3] + joint[:,4] * 1e-9
t_control = control[:,3] + control[:,4] * 1e-9
t_imu = imu[:,3] + imu[:,4] * 1e-9
t_init = t_imu[1]


## Interpolate the torque data 
wrench_control, _ = grf2Torque(joint, control)
τ_control = wrench_control[:,3+axis]
τ_c0, t_c0, dt_c = get_data_window(t_start, t_end, τ_control, t_control, t_init)

S_τ = Fourier(Interval(t_c0[1], t_c0[end]))
f_τ = Fun(S_τ,ApproxFun.transform(S_τ, τ_c0))
f_τ = chop(f_τ, 0.999)

# Extract and interpolate the rotational velocity  
ω = convert.(Float64, imu[:,18+axis])
ω_0, t_i0, dt_i = get_data_window(t_start, t_end, ω, t_imu, t_init)
S_ω = Fourier(Interval(t_i0[1], t_i0[end]))
f_ω = Fun(S_ω, ApproxFun.transform(S_ω, ω_0))
f_ω = chop(f_ω, 0.999)

D = Derivative(S_ω)
df_ω = D * f_ω

## Inertial param fitting 
h = 0.001
t_all = t_start:h:t_end
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

##
figure()
subplot(3,1,1)
plot(t_all, θ_dot, label="̇θ " * axis_name )
plot(t_i0, ω_0, label="measured ω" * axis_name)
xlim([t_all[1], t_all[end]])
title(axis_name * "excitation")
ylabel("rad/s")
xlabel("seconds (s)")
legend()

subplot(3,1,2)
plot(t_all, θ_ddot, label="̈θ " * axis_name)
xlim([t_all[1], t_all[end]])
ylabel("rad/s^2")
xlabel("seconds (s)")
legend()

subplot(3,1,3)
plot(t_all, τ, label="τ " * axis_name)
plot(t_c0, τ_c0, label="measured τ" * axis_name)
xlim([t_all[1], t_all[end]])
ylabel("Nm")
xlabel("seconds (s)")
legend()

println("Result ")
println(round.(result, digits=2))

##
mag_control, phase_control, f_control, _ = fit_cosine(τ_c0, 1/dt_c)
mag_imu, phase_imu, f_imu, _ = fit_cosine(ω_0, 1/dt_i)

println("Raw result")
println("imu phase : ", phase_imu)
println("imu freq : ", f_imu )
println("control phase : ", phase_control)
println("control freq : ", f_control)

mag_control, phase_control, f_control, _ = fit_cosine(τ, 1/h)
mag_imu, phase_imu, f_imu, _ = fit_cosine(θ_dot, 1/h)

println(" ")
println("Interpolated result")
println("imu phase : ", phase_imu)
println("imu freq : ", f_imu )
println("control phase : ", phase_control)
println("control freq : ", f_control)

##
t_imu = imu[:,3] + imu[:,4] * 1e-9
t_control = control[:,3] + control[:,4] * 1e-9
dt = diff(t_imu[1:2000])

temp = digitalfilter(Lowpass(0.01), Butterworth(20))
ω_y = convert.(Float64, imu[1:end,19])
ω_y = filtfilt(temp, ω_y)

wrench_y = wrench_control[ind_cutoff:end,4]
θ_ddot_diff = diff(ω_y[1:2000]) ./ dt
ind_closest = searchsortedlast.(Ref(t_control), t_imu) 
τ = zeros(1999)
for i in 1:1999
    ind_control = ind_closest[i]
    if ind_control == 0
        τ[i] = wrench_y[1]
    else
        τ[i] = wrench_y[ind_control]
    end
end 

n = 1999
A = zeros(1999, 3);
for i in 1:n
    A[i,1] = real(θ_ddot_diff[i])
    A[i,2] = real(ω[i])
    A[i,3] = sign(ω[i])
end
τ = filtfilt(temp, τ)
result = A \ τ 
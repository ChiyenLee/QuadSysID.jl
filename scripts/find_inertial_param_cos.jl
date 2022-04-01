using Pkg
Pkg.activate(".")
using DelimitedFiles
using PyPlot
using LinearAlgebra
using Statistics
using ApproxFun

include("forward_kinematics.jl")
include("utils.jl")
include("data_processing.jl")
pygui(true)


## Loading Data
# data_folder_path = "sys_id/z_wiggle_01_05"
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
t_end = 4.5


## Extract times for all sensors 
t_joint = joint[:,3] + joint[:,4] * 1e-9
t_control = control[:,3] + control[:,4] * 1e-9
t_imu = imu[:,3] + imu[:,4] * 1e-9
t_init = t_imu[1]

## Extract τ from control
wrench_control, _ = grf2Torque(joint, control)
τ_control = wrench_control[:,3+axis]
τ_c0, t_c0, dt_c = get_data_window(t_start, t_end, τ_control, t_control, t_init)
fs_c = 1/dt_c
τ_c0 = τ_c0 .- mean(τ_c0)
a_c, ϕ_c, f_c, off_c = fit_cosine(τ_c0, fs_c; zero_pad_num=10000)
τ_c0_in = cos_wave(a_c, ϕ_c, f_c, off_c, t_c0 .- t_start)

# plot(t_c0, τ_c0_in )
# plot(t_c0, τ_c0 )

# Extract grf from joint torques 
wrench_joints, grf_joints = jtorque2Btorque(joint)
τ_joint = wrench_joints[:, 3+axis]
τ_j0, t_j0, dt_j = get_data_window(t_start, t_end, τ_joint, t_joint, t_init)
fs_j = 1/dt_j
off_j = mean(τ_j0)
τ_j0 = τ_j0 .- off_j
a_j, ϕ_j, f_j, off_j = fit_cosine(τ_j0, fs_j; zero_pad_num=15000)
τ_j0_in = cos_wave(a_j, ϕ_j, f_j, off_j, t_j0 .- t_start)

# plot(t_j0, -τ_j0_in)
# plot(t_j0, -τ_j0)

# plot(t_control[1:500], τ_control[1:500])
# plot(t_joint[1:500], -τ_joint[1:500])

# Extract angular velocities
ω = convert.(Float64, imu[:,18+axis])
ω_0, t_i0, dt_i = get_data_window(t_start, t_end, ω, t_imu, t_init)
fs_ω = 1/dt_i
a_i, ϕ_i, f_i, off_i = fit_cosine(ω_0, fs_ω; zero_pad_num=10000)
ω_0_in = cos_wave(a_i, ϕ_i, f_i, off_i, t_i0 .- t_start)
# plot(t_i0, ω_0_in)
# plot(t_i0, ω_0)


## Take the analytical derivative 
h = 0.001
t_i = t_start:h:t_end
ω_i = cos_wave(a_i, ϕ_i, f_i, off_i, t_i .- t_start)
ω_i = ω_i .- mean(ω_i)
ω_dot_i = sin_wave(-a_i * 2*π*f_i, ϕ_i, f_i, 0.0, t_i .- t_start)
τ_j_in = cos_wave(a_j, ϕ_j, f_j, off_j, t_i  .- t_start)
τ_j_in = τ_j_in .- mean(τ_j_in)
τ_c_in = cos_wave(a_c, ϕ_c, f_c, off_c, t_i .- t_start)
τ_c_in = τ_c_in .- mean(τ_c_in)

n = length(t_i)
A = zeros(n, 2);
for i in 1:n
    A[i,1] = real(ω_dot_i[i])
    A[i,2] = real(ω_i[i])
end
result = A \ τ_c_in

##
println("Raw result")
println("imu phase : ", ϕ_i)
println("imu freq : ", f_i )
println("control phase : ",ϕ_c)
println("control freq : ", f_c)

mag_control, phase_control, f_control, _ = fit_cosine(τ_c_in, 1/h;zero_pad_num=10000)
mag_imu, phase_imu, f_imu, _ = fit_cosine(ω_i, 1/h;zero_pad_num=10000)

println(" ")
println("Interpolated result")
println("imu phase : ", phase_imu)
println("imu freq : ", f_imu )
println("control phase : ", phase_control)
println("control freq : ", f_control)

##
sine_params = [a_c; f_c; ϕ_c; a_i; f_i; ϕ_i]
inertial_params = [0.25, 0., 0.0]

res = dynamic_fit_cost(t_i, sine_params, inertial_params)
i_params = copy(inertial_params)
counter = 0
α = 1
β = 0.1
while counter < 1300
    J = ForwardDiff.gradient(t->dynamic_fit_cost(t_i, sine_params, t), i_params)
    res = dynamic_fit_cost(t_i, sine_params, i_params)
    δi = -res / J
    while dynamic_fit_cost(t_i, sine_params, i_params + α*δi') > res + β * α * J' * δi'
        α = 0.5 * α
    end
    i_params = i_params .+ δi' * α
    counter += 1

    println(res)
end

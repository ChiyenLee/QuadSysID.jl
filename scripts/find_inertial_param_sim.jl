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

##
data_folder_path = "wiggle_sim/y_wiggle_sim"
JOINT_MSG = "a1_gazebo-joint_states.csv"
IMU_MSG = "trunk_imu.csv"
CONTROL_MSG = "a1_debug-control_output.csv"
joint, joint_header = readdlm(joinpath(data_folder_path, JOINT_MSG), ',', header=true)
control, control_header= readdlm(joinpath(data_folder_path, CONTROL_MSG), ',', header=true)
imu, imu_header = readdlm(joinpath(data_folder_path, IMU_MSG), ',', header=true)
axis_name = split(data_folder_path, '/')[2][1]
axis = axis_name == 'x' ? 1 :
       axis_name == 'y' ? 2 :
       axis_name == 'z' ? 3 : 1
t_start = 0.7
t_end = 2.26

## Extract time
t_joint = joint[:,3] + joint[:,4] * 1e-9
t_control = control[:,3] + control[:,4] * 1e-9
t_imu = imu[:,3] + imu[:,4] * 1e-9
t_init = t_imu[1]

## Extract control 
wrench_control, _ = grf2Torque(joint, control; sim=true)
τ_control = wrench_control[:,3+axis]
τ_c0, t_c0, dt_c = get_data_window(t_start, t_end, τ_control, t_control, t_init)
fs_c = 1/dt_c
off_c = mean(τ_c0)
τ_c0 = τ_c0 .- off_c
a_c, ϕ_c, f_c, _ = fit_cosine(τ_c0, fs_c; zero_pad_num=10000)
τ_c0_in = cos_wave(a_c, ϕ_c, f_c, off_c, t_c0 .- t_start)

# plot(t_c0, τ_c0_in .- mean(τ_c0_in))
# plot(t_c0, τ_c0 .- mean(τ_c0))

## Extract grf from joint torques 
wrench_joints, grf_joints = jtorque2Btorque(joint; sim=true)
τ_joint = wrench_joints[:, 3+axis]
τ_j0, t_j0, dt_j = get_data_window(t_start, t_end, τ_joint, t_joint, t_init)
fs_j = 1/dt_j
off_j = mean(τ_j0)
τ_j0 = τ_j0 .- off_j
a_j, ϕ_j, f_j, _ = fit_cosine(τ_j0, fs_j; zero_pad_num=10000)
τ_j0_in = cos_wave(a_j, ϕ_j, f_j, off_j, t_j0 .- t_start)

# plot(t_j0, -(τ_j0_in .- mean(τ_j0_in)))
# plot(t_j0, -(τ_j0 .- mean(τ_j0)))

## Extract angular velocities
ω = convert.(Float64, imu[:,18+axis])
ω_0, t_i0, dt_i = get_data_window(t_start, t_end, ω, t_imu, t_init)
fs_ω = 1/dt_i
a_i, ϕ_i, f_i, off_i = fit_cosine(ω_0, fs_ω; zero_pad_num=15000)
ω_0_in = cos_wave(a_i, ϕ_i, f_i, off_i, t_i0 .- t_start)
# plot(t_i0, ω_0_in)
# plot(t_i0, ω_0)

##
h=0.005
t_i = t_start:h:t_end
# t_i = t_start:0.001:t_start+3
ω_i = cos_wave(a_i, ϕ_i, f_i, off_i, t_i .- t_start)
ω_i = ω_i .- mean(ω_i)
ω_dot_i = sin_wave(-a_i * 2*π*f_i, ϕ_i, f_i, 0.0, t_i .- t_start)
τ_j_in = cos_wave(a_j, ϕ_j, f_j, off_j, t_i .- t_start)
τ_j_in = τ_j_in .- mean(τ_j_in)
τ_c_in = cos_wave(a_c, ϕ_c, f_c, off_c, t_i .- t_start)
τ_c_in = τ_c_in .- mean(τ_c_in)


n = length(t_i)
A = zeros(n, 2);
for i in 1:n
    A[i,1] = real(ω_dot_i[i])
    A[i,2] = real(ω_i[i])
end
result = A \ τ_j_in

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
subplot(3,1,1)
plot(t_i, ω_i)
plot(t_i0, ω_0)

subplot(3,1,2)
plot(t_i,ω_dot_i)

subplot(3,1,3)
plot(t_i, τ_c_in)
plot(t_c0, τ_c0_in)

println(result)
## fft analysis
let
    # t = 0:dt_c:2.0
    # wave = sin.(2*pi*2.1*t)*0.01
    wave = τ_c0 .- mean(τ_c0)
    No = length(wave)
    y = [wave ; zeros(eltype(wave), nextpow(2,8000))]
    # y = τ_c0_in
    N = length(y)
    Fy = rfft(y)
    ak = abs.(Fy) * 2/No
    bk = angle.(Fy) * 2/N
    freqs = collect((0:N÷2)/N .* fs_c )

    # println(freqs[1:15])
    # println( (N÷2)/N )

    # figure(1)
    # subplot(2,1,1)
    # stem(freqs[1:500], ak[1:500])
    # plot(irfft(Fy,N))
    # plot(y)

    # subplot(2,1,2)
    a, ϕ, f, off = fit_cosine(wave, fs_c; zero_pad_num=28000)
    y_out = cos_wave(a, ϕ, f, off, t_c0 .- t_start)
    plot(t_c0, y_out)
    # plot(wave)

end 



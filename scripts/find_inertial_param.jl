using Pkg
Pkg.activate(".")
using DelimitedFiles
using PyPlot
using LinearAlgebra
using Statistics
using ApproxFun

include("forward_kinematics.jl")
include("utils.jl")
pygui(true)

data_folder_path = "sys_id/x_wiggle_01_10"
# data_folder_path = "data2/z_wiggle"
JOINT_MSG = "hardware_a1-joint_foot.csv"
IMU_MSG = "hardware_a1-imu.csv"
CONTROL_MSG = "a1_debug-control_output.csv"

joint, joint_header = readdlm(joinpath(data_folder_path, JOINT_MSG), ',', header=true)
control, control_header= readdlm(joinpath(data_folder_path, CONTROL_MSG), ',', header=true)
imu, imu_header = readdlm(joinpath(data_folder_path, IMU_MSG), ',', header=true)
"""
Convert ground reaction forces to body torque
"""
function grf2Torque(joint_readings, control_readings)
    # Find the last closest point in time
    t_joints = joint_readings[:,3] + joint_readings[:,4] * 1e-9
    t_control = control_readings[:,3] + control_readings[:,4] * 1e-9 
    ind_closest = searchsortedlast.(Ref(t_joints), t_control) 
    wrenches = zeros(length(ind_closest), 6)
    # Loop through the lists
    for ind_control in 1:length(ind_closest)
        ind_joint = ind_closest[ind_control]
        joint_positions = reshape(joint_readings[ind_joint,7:18], (3,4))
        grfs = convert.(Float64,control_readings[ind_control, 7:18])

        # Find foot positions 
        foot_pos = fk(joint_positions)
        # Use cross product to find the torque on the body
        M = [I(3) I(3) I(3) I(3);
             skew(foot_pos[:,1]) skew(foot_pos[:,2]) skew(foot_pos[:,3]) skew(foot_pos[:,4])]
        total_wrench = M * grfs
        wrenches[ind_control, :] = total_wrench
    end 
    return wrenches, t_control
end

"""
Convert joint torques to body torques  
"""
function jtorque2Btorque(joint_readings)
    t_joints = joint_readings[:,3] + joint_readings[:,4]*1e-9 
    grfs = zeros(length(t_joints), 12)
    wrenches = zeros(length(t_joints), 6)
    for ind_joint in 1:length(t_joints)
        # convert joint torques to grf 
        joint_pos = reshape(joint_readings[ind_joint,7:18], (3,4))
        joint_pos = convert.(Float64, joint_pos)
        joint_torques = reshape(joint_readings[ind_joint, 39:50], (3,4))
        joint_torques = convert.(Float64, joint_torques)
        J = dfk(joint_pos)

        grf = zeros(3,4)
        for i in 1:4
            grf[:,i] = J[i,:,:]' \ joint_torques[:,i]
        end
        grf = reshape(grf, (12))

        # Convert grf to body torques 
        foot_pos = fk(joint_pos)
        M = [I(3) I(3) I(3) I(3);
            skew(foot_pos[:,1]) skew(foot_pos[:,2]) skew(foot_pos[:,3]) skew(foot_pos[:,4])]
        total_wrench = M * grf

        wrenches[ind_joint, :] = total_wrench
        grfs[ind_joint, :] = grf
    end 
    return wrenches, grfs
end

# ind cutoff 
ind_cutoff = 10

# Interpolate the torque data 
wrench_control, t_control = grf2Torque(joint, control)
t_control = control[:,3] + control[:,4] * 1e-9
t_init = t_control[ind_cutoff]
t_control = t_control[ind_cutoff:end] .- t_init
wrench_x = wrench_control[ind_cutoff:end,4]
wrench_y = wrench_control[ind_cutoff:end,5]
wrench_z = wrench_control[ind_cutoff:end,6]
wrench = wrench_x

S_τ = Fourier(Interval(t_control[1], t_control[end]))
f_τ = Fun(S_τ,ApproxFun.transform(S_τ,wrench))
f_τ = chop(f_τ, 0.01)

# Extract and interpolate the rotational velocity  
ω_x = convert.(Float64, imu[ind_cutoff:end,19])
ω_y = convert.(Float64, imu[ind_cutoff:end,20])
ω_z = convert.(Float64, imu[ind_cutoff:end,21])
ω = ω_x
t_imu = imu[:,3] + imu[:,4] * 1e-9
t_imu = t_imu[ind_cutoff:end] .- t_init 

S_ω = Fourier(Interval(t_imu[1], t_imu[end]))
f_ω = Fun(S_ω, ApproxFun.transform(S_ω, ω))
f_ω = chop(f_ω, 0.01)

D = Derivative(S_ω)
df_ω = D * f_ω

# Inertial param fitting 
t_all = 0:0.01:10
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
plot(τ)
plot(θ_ddot)

subplot(3,1,1)
plot(t_all, θ_dot)
plot(t_imu, ω)
xlim([t_all[1], t_all[end]])

subplot(3,1,2)
plot(t_all, θ_ddot)
xlim([t_all[1], t_all[end]])

subplot(3,1,3)
plot(t_all, τ)
plot(t_control, wrench)
xlim([t_all[1], t_all[end]])


joint_body_wrench, joint_grf = jtorque2Btorque(joint)
t_joint = joint[:,3] + joint[:,4] * 1e-9 .- t_init 
plot(t_joint, joint_body_wrench[:,5])
plot(t_control, wrench_y)
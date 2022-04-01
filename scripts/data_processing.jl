using FFTW

"""
Given a set of data and a sample frequency, returns the magnitude, phases,
frequency, and offset for the best fitting cosine 
"""
function fit_cosine(data, f_sample; zero_pad_num=0)
    if zero_pad_num > 0
        y = [data;zeros(eltype(data), nextpow(2,zero_pad_num))]
    else
        y = data 
    end
    No = size(data,1)
    N = size(y,1)

    fs = collect(((0: (N÷2)-1))/N .* f_sample) # get the list of frequency
    as = fft(y,1)[1:N÷2,:] # fourier coefficient 
    a_off1 = as[2:end, :]
    inds_max = argmax(abs.(a_off1), dims=1) # find the max 
    magnitudes = abs.(a_off1[inds_max]) * 2/No
    phases = angle.(a_off1[inds_max])
    f = fs[[i[1] + 1 for i in inds_max ]]
    offset = abs.(as[1,:]) * 2/No .* sign.(angle.(as[1,:]))

    return magnitudes, phases, f, offset 
end 

"""
Convert ground reaction forces to body torque
"""
function grf2Torque(joint_readings, control_readings ; sim=false)
    # Find the last closest point in time
    t_joints = joint_readings[:,3] + joint_readings[:,4] * 1e-9
    t_control = control_readings[:,3] + control_readings[:,4] * 1e-9 
    ind_closest = searchsortedlast.(Ref(t_joints), t_control) 
    wrenches = zeros(length(ind_closest), 6)
    # Loop through the lists
    for ind_control in 1:length(ind_closest)
        ind_joint = ind_closest[ind_control]
        joint_positions = reshape(joint_readings[ind_joint,7:18], (3,4))
        if sim 
            for i in 1:4
                joint_positions[:,i] = joint_positions[[2,3,1], i]
            end
        end
        grfs = convert.(Float64,control_readings[ind_control, 7:18])

        # Find foot positions 
        foot_pos = fk(joint_positions)
        # Use cross product to find the torque on the body
        M = [I(3) I(3) I(3) I(3);
             Utils.skew(foot_pos[:,1]) Utils.skew(foot_pos[:,2]) Utils.skew(foot_pos[:,3]) Utils.skew(foot_pos[:,4])]
        total_wrench = M * grfs
        wrenches[ind_control, :] = total_wrench
    end 
    return wrenches, t_control #t_joints[ind_closest]
end

"""
Convert joint torques to body torques  
"""
function jtorque2Btorque(joint_readings; sim=false)
    t_joints = joint_readings[:,3] + joint_readings[:,4]*1e-9 
    grfs = zeros(length(t_joints), 12)
    wrenches = zeros(length(t_joints), 6)
    for ind_joint in 1:length(t_joints)
        # convert joint torques to grf 
        if sim == false
            joint_pos = reshape(joint_readings[ind_joint,7:18], (3,4))
            joint_pos = convert.(Float64, joint_pos)
            joint_torques = reshape(joint_readings[ind_joint, 39:50], (3,4))
            joint_torques = convert.(Float64, joint_torques)
        else
            ## Extract grf from joint torques 
            joint_pos = reshape(joint_readings[ind_joint,7:18], (3,4))
            joint_pos = convert.(Float64, joint_pos)
            joint_torques = reshape(joint_readings[ind_joint, 31:42], (3,4))
            joint_torques = convert.(Float64, joint_torques)
            for i in 1:4
                joint_pos[:,i] = joint_pos[[2,3,1], i]
                joint_torques[:,i] = joint_torques[[2,3,1],i]
            end
        end 
        J = dfk(joint_pos)

        grf = zeros(3,4)
        for i in 1:4
            grf[:,i] = J[i,:,:]' \ joint_torques[:,i]
        end
        grf = reshape(grf, (12))

        # Convert grf to body torques 
        foot_pos = fk(joint_pos)
        M = [I(3) I(3) I(3) I(3);
            Utils.skew(foot_pos[:,1]) Utils.skew(foot_pos[:,2]) Utils.skew(foot_pos[:,3]) Utils.skew(foot_pos[:,4])]
        total_wrench = M * grf

        wrenches[ind_joint, :] = total_wrench
        grfs[ind_joint, :] = grf
    end 
    return wrenches, grfs
end

"""
Get a window from start_time to end_time from a data. t0 is the referenced start time
"""
function get_data_window(start_time, end_time, data, t_data, t0)
    dt = mean(diff(t_data))
    t_data = t_data .- t0
    start_index = searchsortedlast(t_data, start_time)
    end_index = searchsortedlast(t_data, end_time)
    
    data_window = data[start_index:end_index]
    t_window = t_data[start_index:end_index]
    return data_window, t_window, dt
end


cos_wave(a, ϕ, f, off, t) = a .* cos.(2*π*f .* t .+ ϕ) .+ off'
sin_wave(a, ϕ, f, off, t) = a .* sin.(2*π*f .* t .+ ϕ) .+ off'



def run_parallel_generate_ruptures(strike_path, dip_path, fault_path, mod_path, slab_path,
                                   load_distances, UTM_zone, tMw, hurst, Ldip, Lstrike,
                                   num_modes, Nrealizations, rake, rise_time, rise_time_depths0, rise_time_depths1,
                                   max_slip, lognormal, slip_standard_deviation, scaling_law,
                                   force_magnitude,
                                   force_area, mean_slip_name, hypocenter, slip_tol, force_hypocenter,
                                   no_random, shypo, use_hypo_fraction, shear_wave_fraction_shallow,
                                   shear_wave_fraction_deep,
                                   max_slip_rule, zvals, stochastic_rake):
    '''
    Depending on user selected flags parse the work out to different functions
    '''
    import numpy as np
    from numpy import load, save, genfromtxt, log10, cos, sin, deg2rad, savetxt, zeros, where
    from mudpy import fakequakes
    from obspy.taup import TauPyModel
    import warnings

    #Packages not used in the current version, but may be necessary in future
    # from time import gmtime, strftime
    # from numpy.random import shuffle
    # from obspy import UTCDateTime
    # import geopy.distance

    # I don't condone it but this cleans up the warnings
    warnings.filterwarnings("ignore")

    # Get structure model
    vel_mod_file = mod_path

    # Fix input formats
    rise_time_depths = [rise_time_depths0, rise_time_depths1]
    # hypocenter=[hypocenter_lon,hypocenter_lat,hypocenter_dep]
    target_Mw = zeros(len(tMw))
    for rMw in range(len(tMw)):
        target_Mw[rMw] = float(tMw[rMw])

    # Should I calculate or load the distances?
    # LOOK Could replace this with one direct path....
    if load_distances == 1:
        Dstrike = load(strike_path)
        Ddip = load(dip_path)
    else:
        Dstrike, Ddip = fakequakes.subfault_distances_3D(fault_path, slab_path, UTM_zone)
        save(strike_path, Dstrike)
        save(dip_path, Ddip)

    # Read fault and prepare output variable
    whole_fault = genfromtxt(fault_path)


    # Get TauPyModel
    #TODO fix path
    velmod = TauPyModel(model=vel_mod_file.split('.')[0])

    # Now loop over the number of realizations
    realization = 0
    for kmag in range(len(target_Mw)):
        for kfault in range(Nrealizations):

            # Prepare output
            fault_out = zeros((len(whole_fault), 15))
            fault_out[:, 0:8] = whole_fault[:, 0:8]
            fault_out[:, 10:12] = whole_fault[:, 8:]

            # Sucess criterion
            success = False
            while success == False:
                # Select only a subset of the faults based on magnitude scaling
                current_target_Mw = target_Mw[kmag]
                ifaults, hypo_fault, Lmax, Wmax, Leff, Weff, option, Lmean, Wmean = fakequakes.select_faults(
                    whole_fault, Dstrike, Ddip, current_target_Mw, num_modes, scaling_law,
                    force_area, no_shallow_epi=False, no_random=no_random, subfault_hypocenter=shypo,
                    use_hypo_fraction=use_hypo_fraction)
                print(option)
                fault_array = whole_fault[ifaults, :]
                Dstrike_selected = Dstrike[ifaults, :][:, ifaults]
                Ddip_selected = Ddip[ifaults, :][:, ifaults]

                # Determine correlation lengths from effective length.width Leff and Weff
                if Lstrike == 'MB2002':  # Use scaling
                    # Ls=10**(-2.43+0.49*target_Mw)
                    Ls = 2.0 + (1. / 3) * Leff
                elif Lstrike == 'auto':
                    Ls = 17.7 + 0.34 * Leff
                else:
                    Ls = Lstrike
                if Ldip == 'MB2002':  # Use scaling
                    # Ld=10**(-1.79+0.38*target_Mw)
                    Ld = 1.0 + (1. / 3) * Weff
                elif Ldip == 'auto':
                    Ld = 6.8 + 0.4 * Weff
                else:
                    Ld = Ldip

                # Get the mean uniform slip for the target magnitude
                if mean_slip_name == None:
                    mean_slip, mu = fakequakes.get_mean_slip(target_Mw[kmag], fault_array, vel_mod_file)
                else:
                    foo, mu = fakequakes.get_mean_slip(target_Mw[kmag], fault_array, vel_mod_file)
                    mean_fault = genfromtxt(mean_slip_name)
                    mean_slip = (mean_fault[:, 8] ** 2 + mean_fault[:, 9] ** 2) ** 0.5

                    # keep onlt faults that have man slip inside the fault_array seelcted faults
                    mean_slip = mean_slip[ifaults]

                    # get the area in those selected faults
                    area = fault_array[:, -2] * fault_array[:, -1]

                    # get the moment in those selected faults
                    moment_on_selected = (area * mu * mean_slip).sum()

                    # target moment
                    target_moment = 10 ** (1.5 * target_Mw[kmag] + 9.1)

                    # How much do I need to upscale?
                    scale_factor = target_moment / moment_on_selected

                    # rescale the slip
                    mean_slip = mean_slip * scale_factor

                    # Make sure mean_slip has no zero slip faults
                    izero = where(mean_slip == 0)[0]
                    mean_slip[izero] = slip_tol

                # Get correlation matrix
                C = fakequakes.vonKarman_correlation(Dstrike_selected, Ddip_selected, Ls, Ld, hurst)

                # Lognormal or not?
                if lognormal == False:
                    # Get covariance matrix
                    C_nonlog = fakequakes.get_covariance(mean_slip, C, target_Mw[kmag], fault_array, vel_mod_file,
                                                         slip_standard_deviation)
                    # Get eigen values and eigenvectors
                    eigenvals, V = fakequakes.get_eigen(C_nonlog)
                    # Generate fake slip pattern
                    rejected = True
                    while rejected == True:
                        #                        slip_unrectified,success=make_KL_slip(fault_array,num_modes,eigenvals,V,mean_slip,max_slip,lognormal=False,seed=kfault)
                        slip_unrectified, success = fakequakes.make_KL_slip(fault_array, num_modes, eigenvals, V,
                                                                            mean_slip, max_slip, zvals, lognormal=False,
                                                                            seed=None)
                        slip, rejected, percent_negative = fakequakes.rectify_slip(slip_unrectified, percent_reject=13)
                        if rejected == True:
                            print(
                                '... ... ... negative slip threshold exceeeded with %d%% negative slip. Recomputing...' % (
                                    percent_negative))
                else:
                    # Get lognormal values
                    C_log, mean_slip_log = fakequakes.get_lognormal(mean_slip, C, target_Mw[kmag], fault_array,
                                                                    vel_mod_file, slip_standard_deviation)
                    # Get eigen values and eigenvectors
                    eigenvals, V = fakequakes.get_eigen(C_log)
                    # Generate fake slip pattern
                    #                    slip,success=make_KL_slip(fault_array,num_modes,eigenvals,V,mean_slip_log,max_slip,lognormal=True,seed=kfault)
                    slip, success = fakequakes.make_KL_slip(fault_array, num_modes, eigenvals, V, mean_slip_log,
                                                            max_slip, zvals, lognormal=True, seed=None)

                # Slip pattern sucessfully made, moving on.
                # Rigidities
                foo, mu = fakequakes.get_mean_slip(target_Mw[kmag], whole_fault, vel_mod_file)
                fault_out[:, 13] = mu

                # Calculate moment and magnitude of fake slip pattern
                M0 = sum(slip * fault_out[ifaults, 10] * fault_out[ifaults, 11] * mu[ifaults])
                Mw = (2. / 3) * (log10(M0) - 9.1)

                # Check max_slip_rule
                if max_slip_rule == True:

                    max_slip_from_rule = 10 ** (-4.94 + 0.71 * Mw)  # From Allen & Hayes, 2017
                    max_slip_tolerance = 3

                    if slip.max() > max_slip_tolerance * max_slip_from_rule:
                        success = False
                        print('... ... ... max slip condition violated max_slip_rule, recalculating...')

                # Force to target magnitude
                if force_magnitude == True:
                    M0_target = 10 ** (1.5 * target_Mw[kmag] + 9.1)
                    M0_ratio = M0_target / M0
                    # Multiply slip by ratio
                    slip = slip * M0_ratio
                    # Recalculate
                    M0 = sum(slip * fault_out[ifaults, 10] * fault_out[ifaults, 11] * mu[ifaults])
                    Mw = (2. / 3) * (log10(M0) - 9.1)

                # check max_slip again
                if slip.max() > max_slip:
                    success = False
                    print('... ... ... max slip condition violated due to force_magnitude=True, recalculating...')

            # Get stochastic rake vector
            if stochastic_rake == True:
                stoc_rake = fakequakes.get_stochastic_rake(rake, len(slip))
            elif stochastic_rake == False:
                stoc_rake = rake * np.ones(len(slip))

            # Place slip values in output variable
            fault_out[ifaults, 8] = slip * cos(deg2rad(stoc_rake))
            fault_out[ifaults, 9] = slip * sin(deg2rad(stoc_rake))

            # Move hypocenter to somewhere with a susbtantial fraction of peak slip
            #            slip_fraction=0.25
            #            islip=where(slip>slip.max()*slip_fraction)[0]
            #            shuffle(islip) #randomize
            #            hypo_fault=ifaults[islip[0]] #select first from randomized vector

            # Calculate and scale rise times
            rise_times = fakequakes.get_rise_times(M0, slip, fault_array, rise_time_depths, stoc_rake, rise_time,
                                                   option=option)

            # Place rise_times in output variable
            fault_out[:, 7] = 0
            fault_out[ifaults, 7] = rise_times

            # Calculate rupture onset times
            if force_hypocenter == False:  # Use random hypo, otehrwise force hypo to user specified
                hypocenter = whole_fault[hypo_fault, 1:4]
            # else:
            #    hypocenter=whole_fault[shypo,1:4]

            # edit ...
            # if rise_time==2:
            #     shear_wave_fraction_shallow=1/60/60/24*2
            #     shear_wave_fraction_deep=   1/60/60/24*2
            #     print("c")
            # else:
            #     L_rescale=(Lmax/1500)+2.5
            #     shear_wave_fraction_shallow=1/60/60/24*2*(3.5/L_rescale)**2
            #     shear_wave_fraction_deep=   1/60/60/24*2*(3.5/L_rescale)**2
            #     print("m")
            if rise_time == 'SSE':  # For moedling SSes
                shear_wave_fraction_shallow = 1 / 60 / 60 / 24 * 2
                shear_wave_fraction_deep = 1 / 60 / 60 / 24 * 2
            else:  # regular EQs, do nothing
                pass

            t_onset, length2fault = fakequakes.get_rupture_onset(slip, fault_array, mod_path,
                                                                 hypocenter, rise_time_depths,
                                                                 M0, velmod,
                                                                 shear_wave_fraction_shallow=shear_wave_fraction_shallow,
                                                                 shear_wave_fraction_deep=shear_wave_fraction_deep)
            fault_out[:, 12] = 0
            fault_out[ifaults, 12] = t_onset

            fault_out[:, 14] = 0
            fault_out[ifaults, 14] = length2fault / t_onset

            #The below functions were not necessary for making the fault_out

            # # Calculate location of moment centroid
            # centroid_lon, centroid_lat, centroid_z = fakequakes.get_centroid(fault_out)
            #
            # # Calculate average risetime
            # rise = fault_out[:, 7]
            # avg_rise = np.mean(rise)
            #
            # # Calculate average rupture velocity
            # lon_array = fault_out[:, 1]
            # lat_array = fault_out[:, 2]
            # vrupt = []
            #
            # for i in range(len(fault_array)):
            #     if t_onset[i] > 0:
            #         # r = geopy.distance.geodesic((hypocenter[1], hypocenter[0]), (lat_array[i], lon_array[i])).km
            #         # vrupt.append(r/t_onset[i])
            #         vrupt.append(length2fault[i] / t_onset[i])
            #
            # avg_vrupt = np.mean(vrupt)
            #
            return fault_out
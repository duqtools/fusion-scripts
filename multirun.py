from prepare_im_runs import *

instructions_list = ['setup base', 'create base', 'run base']
json_file_name = '/afs/eufus.eu/user/g/g2mmarin/public/scripts/template_prepare_input.json'
json_input_raw = open(json_file_name)
json_input = json.load(json_input_raw)

# Averages profiles over a given interval
#json_input['instructions']['average'] = True
# Rebases the equilibrium with the same time base as core profile to avoid large files
json_input['instructions']['rebase'] = True
# Avoids Zeff larger than 4.5, for which everything would be carbon at low te
json_input['instructions']['correct zeff'] = True
# Pushes te boundary up or down when very low or high (allows for impurities and corrects for mistakes in the fit)
#json_input['instructions']['correct boundaries'] = True
# Allows to set the boundaries for Te with a few different recipes
json_input['instructions']['set boundaries'] = True
# Adds the profiles very early when they are not available with assumptions and extrapolation
json_input['instructions']['add early profiles'] = True
# Peaks the electron temperature. Boundary is kept fixed
json_input['instructions']['peak temperature'] = 1.2
# Multiplies the whole ion temperature profile
json_input['instructions']['multiply ion temperature'] = 0.8
# Changes the sign of the current. Should not be needed anymore...
json_input['instructions']['flipping ip'] = True
# Multiplies the ion temperature for the specified value when Ti/Te is too large
json_input['instructions']['correct ion temperature'] = 3.0
# Multiplies the q profile for a certain value. For a parabolic q profile, a value smaller than 1 will make it more hollow (axis closer to 1)
# Default q profile when adding early profiles should be parabolic
json_input['instructions']['multiply q profile'] = 0.7

# Specifies the evolution for the Zeff profile. 'peaked zeff evolved' is flat at the beginning and then goes peaked. Original does not do anything (so it will prolly be flat)
json_input['zeff profile'] = 'peaked zeff'
# Specifies the evolution for the average zeff. 'hyperbole starts large and quickly decreases'
json_input['zeff evolution'] = 'hyperbole'

# Specifies how peaked Zeff will be
json_input['zeff profile parameter'] = 0.0
# Specifies how long it takes for Zeff to be reduced to the measured one
json_input['zeff evolution parameter'] = 0.3
# Specifies the starting value for Zeff
json_input['zeff max evolution'] = 4.0

# Specifies the method to choose the te boundaries. Available:
# 'add' adds a certain value to the boundary te, axis is kept constant
# 'add no start' adds but not at the start. The regions are joined continuosly
# 'add early' starts from 20 eV and then goes linearly to the value at 0.05s, on top of adding the given value
# 'add early high' starts from 100 eV and then goes linearly to the value at 0.05s on top of adding the given value
# 'add on te' is available as method ti and will copy the te boundary and add a value on that
# 'add on te profile' is available as method ti and will copy the te profile and add a value on that. Might need more testing
json_input['boundary instructions']['method te'] = 'add early'
json_input['boundary instructions']['method ti'] = 'add on te'
# Sets a limit for the maximum te at the boundary
json_input['boundary instructions']['te bound up'] = 120
# Sets the value to be added at te
json_input['boundary instructions']['te sep'] = 0
# Sets the value to be added at ti. false will do nothing. Te will set the same value as te. A number will add the value to ti
json_input['boundary instructions']['ti sep'] = 20
# For the option 'add early' it sets the starting temperature and the time when to start using the experimentla data
json_input['boundary instructions']['time continuity'] = 0.1
# Sets the electron temperature at the very beginning of the simulation. The value will be linearly interpolated until time continuity
json_input['boundary instructions']['temp start'] = 20

# Sets a flat q profile at the beginning
json_input['extra early options']['flat q profile'] = False
# Sets a central peaking to the initial te, ti. The number represents the value at the axis divided by the value at the edge
json_input['extra early options']['te peaking 0'] = 3
json_input['extra early options']['ti peaking 0'] = 3
# Same but for density
json_input['extra early options']['ne peaking 0'] = 2
# Sets options for the first profile. 'first profile' sets the first measured profiles as the early profile.
# Parabolic sets a parabolic profile, linear a linear one. The boundaries are not changed.
json_input['extra early options']['electron density option'] = 'parabolic'
json_input['extra early options']['ion temperature option'] = 'parabolic'
json_input['extra early options']['electron temperature option'] = 'parabolic'


#Single run

run_test = IntegratedModellingRuns(64862, instructions_list, 'ohmic', 'run100test', run_input = 5, run_start = 2000, run_output = 3000, json_input = json_input, esco_timesteps = 100, output_timesteps = 100, time_start = 0.03, time_end = 0.33, density_feedback = True, force_run = True, force_input_overwrite = True)
run_test.setup_create_compare()


# Single scan

shot_numbers = [64965, 64954, 64958, 64862, 56653]
runs_input = [5, 5, 5, 5, 5]
runs_name, first_number = [], 675
runs_start = [1080, 1080, 1080, 1080, 1080]
runs_output = [1806, 1806, 1806, 1806, 1806]
times = [[0.6, 0.8], [0.3, 0.5], [0225, 0.425], [0.45, 0.65], [0.425, 0.625]]

runs_name, first_number = [], 300

for shot_number, run_name, run_input, run_start, run_output, time in zip(shot_numbers, runs_name, runs_input, runs_start, runs_output, times):

    run_name = 'run' + str(run_number) + '_' + str(shot_number) + '_' + 'run_iden_dummy'
    run_number += 1

    run_test = IntegratedModellingRuns(shot_number, instructions_list, 'equilibrium_ser_saw', run_name, run_input = run_input, run_start = run_start, run_output = run_output, json_input = json_input, esco_timesteps = 200, output_timesteps = 100, time_start = time[0], time_end = time[1], force_run = True)
    #run_test.setup_create_compare()
    run_start += 10
    run_output += 1



# Multi scan

multiply_q_profiles = [0.5, 0.3, 0.1]
te_peaking_0s = [10, 20, 30]
te_seps = [0, 30]
shot_number = 64958

run_number = 110
run_start, run_output = 1620, 2025

for multiply_q_profile in multiply_q_profiles:
    for te_peaking_0 in te_peaking_0s:
        for te_sep in te_seps:
            json_input['instructions']['multiply q profile'] = multiply_q_profile
            json_input['extra early options']['te peaking 0'] = te_peaking_0
            json_input['boundary instructions']['te sep'] = te_sep

            run_name = 'run' + str(run_number) + '_' + str(shot_number) + '_' + 'test'
            run_number += 1

            run_test = IntegratedModellingRuns(64958, instructions_list, 'qlk_fb_test2', run_name, run_input = 5, run_start = run_start, run_output = run_output, json_input = json_input, esco_timesteps = 100, output_timesteps = 100, time_start = 0.03, time_end = 0.33, density_feedback = True, force_run = True, force_input_overwrite = True)
            #run_test.setup_create_compare()
            run_start += 10
            run_output += 1


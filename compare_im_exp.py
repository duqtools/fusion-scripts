import os
import sys
import re
import copy
import numpy as np
import math
import getpass
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
#import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython import display
import argparse

from compare_im_runs import *
from prepare_im_input import open_and_get_core_profiles

from packaging import version

min_imas_version_str = "3.28.0"
min_imasal_version_str = "4.7.2"

try:
    import imas
except ImportError:
    warnings.warn("IMAS Python module not found or not configured properly, tools need IDS to work!", UserWarning)
if imas is not None:
    from imas import imasdef
    vsplit = imas.names[0].split("_")
    imas_version = version.parse(".".join(vsplit[1:4]))
    ual_version = version.parse(".".join(vsplit[5:]))
    if imas_version < version.parse(min_imas_version_str):
        raise ImportError("IMAS version must be >= %s! Aborting!" % (min_imas_version_str))
    if ual_version < version.parse(min_imasal_version_str):
        raise ImportError("IMAS AL version must be >= %s! Aborting!" % (min_imasal_version_str))





'''

Compare the model or the fit to the experimental data, considering errorbars.

Example of usage:

plot_exp_vs_model('tcv', 64965, 5, 517, 0.05, 0.15, signals = ['ti', 'ni'], verbose = 0)

'''

def plot_exp_vs_model(db, shot, run_exp, run_model, time_begin, time_end, signals = ['te', 'ne', 'ti', 'ni'], label = None, verbose = False):

    variable_names = {}
    if 'te' in signals:
        variable_names['electron temperature'] = [
        'core_profiles.profiles_1d[].electrons.temperature_fit.measured',
        'core_profiles.profiles_1d[].electrons.temperature_fit.measured_error_upper',
        'core_profiles.profiles_1d[].electrons.temperature'
    ]
    if 'ne' in signals:
        variable_names['electron density'] = [
        'core_profiles.profiles_1d[].electrons.density_fit.measured',
        'core_profiles.profiles_1d[].electrons.density_fit.measured_error_upper',
        'core_profiles.profiles_1d[].electrons.density'
    ]
    if 'ti' in signals:
        variable_names['ion temperature'] = [
        'core_profiles.profiles_1d[].t_i_average_fit.measured',
        'core_profiles.profiles_1d[].t_i_average_fit.measured_error_upper',
        'core_profiles.profiles_1d[].t_i_average'
    ]
    if 'ni' in signals:
        variable_names['impurity density'] = [
        'core_profiles.profiles_1d[].ion[1].density_fit.measured',
        'core_profiles.profiles_1d[].ion[1].density_fit.measured_error_upper',
        'core_profiles.profiles_1d[].ion[1].density'
    ]

    core_profiles_exp = open_and_get_core_profiles(db, shot, run_exp)
    #legacy
    #core_profiles_model = open_and_get_core_profiles(db, shot, run_model)

    core_profiles_model = open_and_get_core_profiles_from_run(db, shot, run_model)

    t_cxrs = []

    for profile in core_profiles_exp.profiles_1d:
        t_cxrs.append(profile.t_i_average_fit.time_measurement)

    t_cxrs = np.asarray(t_cxrs).flatten()
    t_cxrs = t_cxrs[np.where(t_cxrs < time_begin, False, True)]
    t_cxrs = t_cxrs[np.where(t_cxrs > time_end, False, True)]

    errors = []

    for variable in variable_names:
        exp_data = get_onesig(core_profiles_exp,variable_names[variable][0],time_begin,time_end=time_end)
        errorbar = get_onesig(core_profiles_exp,variable_names[variable][1],time_begin,time_end=time_end)

        for time in exp_data:

            # Clean data that are not in the core
            exp_data[time]['y'] = exp_data[time]['y'][np.where(exp_data[time]['x'] > 1, False, True)]
            errorbar[time]['y'] = errorbar[time]['y'][np.where(exp_data[time]['x'] > 1, False, True)]
            errorbar[time]['x'] = errorbar[time]['x'][np.where(exp_data[time]['x'] > 1, False, True)]
            exp_data[time]['x'] = exp_data[time]['x'][np.where(exp_data[time]['x'] > 1, False, True)]

            # Clean experimental data that were not filled properly
            exp_data[time]['x'] = exp_data[time]['x'][np.where(exp_data[time]['y'] < -0, False, True)]
            errorbar[time]['y'] = errorbar[time]['y'][np.where(exp_data[time]['y'] < -0, False, True)]
            errorbar[time]['x'] = errorbar[time]['x'][np.where(exp_data[time]['y'] < -0, False, True)]
            exp_data[time]['y'] = exp_data[time]['y'][np.where(exp_data[time]['y'] < -0, False, True)]

            # Clean experimental data that are filled with nans
            exp_data[time]['x'] = exp_data[time]['x'][np.where(np.isnan(exp_data[time]['y']), False, True)]
            errorbar[time]['y'] = errorbar[time]['y'][np.where(np.isnan(exp_data[time]['y']), False, True)]
            errorbar[time]['x'] = errorbar[time]['x'][np.where(np.isnan(exp_data[time]['y']), False, True)]
            exp_data[time]['y'] = exp_data[time]['y'][np.where(np.isnan(exp_data[time]['y']), False, True)]

        # Hacky. But well...
        time_vector_exp = np.asarray(list(exp_data.keys()))

        if variable == 'ion temperature' or variable == 'impurity density':
            exp_data_new, errorbar_new = {}, {}
            time_vector_exp = np.asarray(list(exp_data.keys()))
            for time_cxrs in t_cxrs:
                time_closest = time_vector_exp[np.abs(time_vector_exp - time_cxrs).argmin(0)]
                exp_data_new[time_cxrs] = exp_data[time_closest]
                errorbar_new[time_cxrs] = errorbar[time_closest]

            exp_data = exp_data_new
            errorbar = errorbar_new
            time_vector_exp = np.unique(t_cxrs)

            # Errors are stored differently. I think this is the right way and errors should be like this also for te and ne...
            for time in time_vector_exp:
                errorbar[time]['y'] = errorbar[time]['y'] - exp_data[time]['y']


        fit = get_onesig(core_profiles_model,variable_names[variable][2],time_begin,time_end=time_end)
        time_vector_fit = np.asarray(list(fit.keys()))

        # For every timelisce in experiments, remap all the fits on that x and then interpolate. It is necessary if the x coordinate changes in time
        ytable_final = []
        for time_exp in time_vector_exp:
            ytable_temp = None
            for time_fit in time_vector_fit:
                y_new = fit_and_substitute(fit[time_fit]["x"], exp_data[time_exp]["x"], fit[time_fit]["y"])
                ytable_temp = np.vstack((ytable_temp, y_new)) if ytable_temp is not None else np.atleast_2d(y_new)

            ytable_temp.reshape(time_vector_fit.shape[0], exp_data[time_exp]["x"].shape[0])

            ytable_new = None
            for ii in range(exp_data[time_exp]['x'].shape[0]):
                y_new = fit_and_substitute(time_vector_fit, time_exp, ytable_temp[:, ii])
                ytable_new = np.vstack((ytable_new, y_new)) if ytable_new is not None else np.atleast_2d(y_new)

            ytable_final.append(ytable_new)

        #fit_values, exp_values = [], []
        #for itime, time_exp in enumerate(time_vector_exp):
        #    fit_values.append(ytable_final[itime][5])
        #    exp_values.append(exp_data[time_exp]['y'][5])


        #fit_values = np.asarray(fit_values).flatten()
        #exp_values = np.asarray(exp_values).flatten()

        if verbose == 2:
            for i, time in enumerate(exp_data):

                plt.errorbar(exp_data[time]['x'], exp_data[time]['y'], yerr=errorbar[time]['y'], linestyle = ' ', label = 'Experiment')
                plt.plot(exp_data[time]['x'], ytable_final[i], label = 'Fit/Model')
                plt.title(str(time))
                plt.legend()
                plt.show()

        # Calculate error
        error_time = []
        for i, time in enumerate(exp_data):
            error_time_space = []
            for y_fit, y_exp, error_point in zip(ytable_final[i], exp_data[time]['y'], errorbar[time]['y']):
                error_time_space.append(abs(y_fit[0] - y_exp)/error_point)

            if verbose == 2:
                if label:
                    plt.plot(exp_data[time]['x'], error_time_space, 'bo', label = label)
                else:
                    plt.plot(exp_data[time]['x'], error_time_space, 'bo')
                plt.title(str(time))
                plt.xlabel(r'\rho [-]')
                plt.ylabel('Error')
                plt.legend()
                plt.show()

            error_time.append(sum(error_time_space)/len(exp_data[time]['y']))

        if verbose == 1 or verbose == 2:
            if label:
                plt.plot(time_vector_exp, error_time, label = label)
            else:
                plt.plot(time_vector_exp, error_time)
            plt.title(variable)
            plt.xlabel(r't [s]')
            plt.ylabel('Error')
            plt.legend()
            plt.show()

        error_variable = sum(error_time)/len(exp_data)
        errors.append(error_variable)
        print('The error for ' + variable + ' is ' + str(error_variable))

    return(errors)


def open_and_get_core_profiles_from_run(db, shot, run_name, username=None, backend='mdsplus'):

    if not username:
        username=getpass.getuser()

    username = '/pfs/work/' + username + '/jetto/runs/' + run_name + '/imasdb/'

    print(username)

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    data_entry = imas.DBEntry(imas_backend, db, shot, 2, user_name=username)

    op = data_entry.open()

    if op[0]<0:
        cp=data_entry.create()
        print(cp[0])
        if cp[0]==0:
            print("data entry created")
    elif op[0]==0:
        print("data entry opened")

    core_profiles = data_entry.get('core_profiles')
    data_entry.close()

    return(core_profiles)




if __name__ == "__main__":

    #plot_exp_vs_model('tcv', 64965, 5, 517, 0.05, 0.15, signals = ['ti', 'ne'], verbose = 2)
    #plot_exp_vs_model('tcv', 64862, 5, 1903, 0.05, 0.15, signals = ['te', 'ne'], verbose = 2)
    plot_exp_vs_model('tcv', 64862, 5, 'run254test', 0.05, 0.15, signals = ['te', 'ne'], verbose = 2)
    print('plot and compares experimental data with fits or model')


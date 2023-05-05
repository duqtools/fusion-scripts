import os
import sys
import re
import copy
import numpy as np
import math
import getpass
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display
import argparse
from compare_im_runs import *
from packaging import version

# Minimum versions of IMAS and IMAS-AL required
MIN_IMAS_VERSION = version.parse("3.28.0")
MIN_IMASAL_VERSION = version.parse("4.7.2")


"""
Check if the installed version of IMAS and IMAS-AL satisfies the minimum version requirements.
"""
try:
    import imas
except ImportError:
    warnings.warn("IMAS Python module not found or not configured properly, tools need IDS to work!", UserWarning)
if imas is not None:
    from imas import imasdef
    vsplit = imas.names[0].split("_")
    imas_version = version.parse(".".join(vsplit[1:4]))
    ual_version = version.parse(".".join(vsplit[5:]))
    imas_backend = imasdef.MDSPLUS_BACKEND
    if imas_version < MIN_IMAS_VERSION:
        raise ImportError("IMAS version must be >= %s! Aborting!" % (MIN_IMAS_VERSION))
    if ual_version < MIN_IMASAL_VERSION:
        raise ImportError("IMAS AL version must be >= %s! Aborting!" % (MIN_IMASAL_VERSION))


def get_title_variable(variable):
    """
    Get the title variable string for a given variable.
    """
    if variable == 'electron temperature':
        return r'$T_e$'
    elif variable == 'electron density':
        return r'$n_e$'
    elif variable == 'ion temperature':
        return r'$T_i$'
    elif variable == 'impurity density':
        return r'$n_C$'
    else:
        return ''


def get_label_variable(variable):
    """
    Get the label variable string for a given variable.
    """
    if variable == 'electron temperature':
        return r'$T_e$ [KeV]'
    elif variable == 'electron density':
        return r'$n_e$ $[10^{19} m^{-3}]$'
    elif variable == 'ion temperature':
        return r'$T_i$ [KeV]'
    elif variable == 'impurity density':
        return r'$n_C$ $[10^{19} m^{-3}]$'
    else:
        return ''


def get_variable_names(signals):
    """
    Get the variable names for a list of signals.
    """
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

    return variable_names


def clean_exp_data(exp_data, errorbar):
    time_keys_to_remove = []
    for time in exp_data:

        # Clean data that are not in the core
        exp_data[time]['y'] = exp_data[time]['y'][np.where(exp_data[time]['x'] > 1, False, True)]
        errorbar[time]['y'] = errorbar[time]['y'][np.where(exp_data[time]['x'] > 1, False, True)]
        errorbar[time]['x'] = errorbar[time]['x'][np.where(exp_data[time]['x'] > 1, False, True)]
        exp_data[time]['x'] = exp_data[time]['x'][np.where(exp_data[time]['x'] > 1, False, True)]

        # Clean experimental data that were not filled properly
        mask = exp_data[time]['y'] < -0
        exp_data[time]['y'] = exp_data[time]['y'][~mask]
        exp_data[time]['x'] = exp_data[time]['x'][~mask]
        errorbar[time]['y'] = errorbar[time]['y'][~mask]
        errorbar[time]['x'] = errorbar[time]['x'][~mask]

        # Clean experimental data that are filled with nans
        mask = np.isnan(exp_data[time]['y'])
        exp_data[time]['y'] = exp_data[time]['y'][~mask]
        exp_data[time]['x'] = exp_data[time]['x'][~mask]
        errorbar[time]['y'] = errorbar[time]['y'][~mask]
        errorbar[time]['x'] = errorbar[time]['x'][~mask]

        # Clean experimental data when errobars are too small (polluted data)
        mask = errorbar[time]['y'] < 1.0e-10
        exp_data[time]['y'] = exp_data[time]['y'][~mask]
        exp_data[time]['x'] = exp_data[time]['x'][~mask]
        errorbar[time]['y'] = errorbar[time]['y'][~mask]
        errorbar[time]['x'] = errorbar[time]['x'][~mask]

        if len(exp_data[time]['y']) == 0 or len(errorbar[time]['y']) == 0:
            time_keys_to_remove.append(time)

    #Remove times that end up being empty
    for time in time_keys_to_remove:
        exp_data.pop(time)
        errorbar.pop(time)

    return exp_data, errorbar

def scale_exp_data(exp_data, var):
    if var == 'electron temperature' or var == 'ion temperature':
        for time in exp_data:
            exp_data[time]['y'] *= 1.0e-3
    elif var == 'electron density' or var == 'impurity density':
        for time in exp_data:
            exp_data[time]['y'] *= 1.0e-19

    return exp_data

def scale_model_data(ytable_final, var):
    for ii, ytable_slice in enumerate(ytable_final):
        if var == 'electron temperature' or var == 'ion temperature':
            try:
                ytable_final[ii] = ytable_slice*1.0e-3
            except TypeError:
                print('No experimental data available for slice number ' + str(ii) + '. Aborting')
                exit()
        elif var == 'electron density' or var == 'impurity density':
            try:
                ytable_final[ii] = ytable_slice*1.0e-19
            except TypeError:
                print('No experimental data available for slice number ' + str(ii) + '. Aborting')
                exit()

    return ytable_final

def get_exp_data(db, shot, run, time_begin, time_end, signals):
    exp_data = {}
    variable_names = get_variable_names(signals)
    core_profiles_exp = open_and_get_ids(db, shot, 'core_profiles', run)
    for variable in variable_names:
        data = get_onesig(core_profiles_exp, variable_names[variable][0], time_begin, time_end=time_end)
        #errorbar = get_onesig(core_profiles_exp, variable_names[variable][1], time_begin, time_end=time_end)
        try:
            errorbar = get_onesig(core_profiles_exp, variable_names[variable][1], time_begin, time_end=time_end)
        except OSError:
            errorbar = copy.deepcopy(data)
            for time in data:
                errorbar[time]['x'] = data[time]['x']
                errorbar[time]['y'] = data[time]['y']/5
            print('Careful! Generating dummy errors for ' + variable + ' since experimetal errors are not available')

        data, errorbar = clean_exp_data(data, errorbar)

        data = scale_exp_data(data, variable)
        errorbar = scale_exp_data(errorbar, variable)

        exp_data[variable] = {'data': data, 'errorbar': errorbar}

    return exp_data


def filter_time_range(t_cxrs, time_begin, time_end):
    t_cxrs = np.asarray(t_cxrs).flatten()
    mask = np.logical_and(t_cxrs >= time_begin, t_cxrs <= time_end)
    return t_cxrs[mask]


def get_closest_times(exp_data, t_cxrs):
    time_vector_exp = np.asarray(list(exp_data['data'].keys()))
    exp_data_new = {'data': {}, 'errorbar': {}}
    for time_cxrs in t_cxrs:
        time_closest = time_vector_exp[np.abs(time_vector_exp - time_cxrs).argmin(0)]

        exp_data_new['data'][time_cxrs] = exp_data['data'][time_closest]
        exp_data_new['errorbar'][time_cxrs] = exp_data['errorbar'][time_closest]

    exp_data = exp_data_new
    time_vector_exp = np.unique(t_cxrs)
    #for time in time_vector_exp:
    #    exp_data['data'][time]['y'] = exp_data['errorbar'][time]['y'] - exp_data['data'][time]['y']

    return exp_data['data'], exp_data['errorbar'], time_vector_exp

def get_t_cxrs(core_profiles_exp):
    """
    Extracts time values from the cxrs data in core_profiles_exp.

    :param core_profiles_exp: core profiles object for experimental data
    :return: list of time values
    """
    t_cxrs = []
    for profile in core_profiles_exp.profiles_1d:
        t_cxrs.append(profile.t_i_average_fit.time_measurement)
    return t_cxrs


def get_exp_data_and_errorbar(all_exp_data, t_cxrs, variable):
    """
    Extracts experimental data and error bars for a given variable.

    :param all_exp_data: dictionary of all experimental data
    :param t_cxrs: list of time values
    :param variable: signal name
    :return: exp_data, errorbar, time_vector_exp
    """
    exp_data = all_exp_data[variable]

    if variable in ['ion temperature', 'impurity density']:
        exp_data, errorbar, time_vector_exp = get_closest_times(exp_data, t_cxrs)
    else:
        exp_data, errorbar = exp_data['data'], exp_data['errorbar']
        time_vector_exp = np.asarray(list(exp_data.keys()))

    return exp_data, errorbar, time_vector_exp


def generate_ytable(time_vector_exp, time_vector_fit, fit, exp_data, fit_and_substitute, variable):
    ytable_final = []
    for time_exp in time_vector_exp:
        ytable_temp = None
        for time_fit in time_vector_fit:
            y_new = fit_and_substitute(fit[time_fit]["x"], exp_data[time_exp]["x"], fit[time_fit]["y"])
            ytable_temp = np.vstack((ytable_temp, y_new)) if ytable_temp is not None else np.atleast_2d(y_new)

        ytable_temp = ytable_temp.reshape(time_vector_fit.shape[0], exp_data[time_exp]["x"].shape[0])

        ytable_new = None
        for ii in range(exp_data[time_exp]['x'].shape[0]):
            y_new = fit_and_substitute(time_vector_fit, time_exp, ytable_temp[:, ii])
            ytable_new = np.vstack((ytable_new, y_new)) if ytable_new is not None else np.atleast_2d(y_new)

        ytable_final.append(ytable_new)

    return ytable_final

def get_legend_label(fit_or_model):
    if fit_or_model == 'Model':
        legend_label = 'Integrated modelling'
    elif fit_or_model == 'Fit':
        legend_label = 'Fitted data'
    else:
        legend_label = 'Fit/Model'

    return legend_label

def plot_data_and_model(exp_data, ytable_final, errorbar, variable, fit_or_model, legend_fontsize,
                        title_fontsize, label_fontsize):
    title_variable = get_title_variable(variable)

    for i, time in enumerate(exp_data):
        plt.errorbar(exp_data[time]['x'], exp_data[time]['y'], yerr=errorbar[time]['y'], linestyle=' ',
                     label='HRTS Experimental data')
        legend_label = get_legend_label(fit_or_model)
        plt.plot(exp_data[time]['x'], ytable_final[i], label=legend_label)

        plt.legend(fontsize=legend_fontsize)
        title = title_variable + ' at t = {time:.3f}'.format(time=time)
        plt.title(title, fontsize=title_fontsize)
        plt.xlabel(r'$\rho$ [-]', fontsize=label_fontsize)
        ylabel_variable = get_label_variable(variable)
        plt.ylabel(ylabel_variable, fontsize=label_fontsize)
        plt.show()


def plot_error(time_vector_exp, exp_data, ytable_final, errorbar, variable, label, fit_or_model, verbose):
    error_time = []
    error_time_space_all = []
    for i, time in enumerate(exp_data):
        error_time_space = []
        for y_fit, y_exp, error_point in zip(ytable_final[i], exp_data[time]['y'], errorbar[time]['y']):
            error_time_space.append(abs(y_fit[0] - y_exp) / error_point)

        if verbose == 2:
            legend_label = get_legend_label(fit_or_model)
            plt.plot(exp_data[time]['x'], error_time_space, 'bo', label=legend_label)
            title_variable = get_title_variable(variable)
            title = title_variable + 'error at t =  {time:.3f}'.format(time=time)
            plt.title(str(time))
            plt.xlabel(r'$\rho$ [-]')
            plt.ylabel('Error')
            plt.legend()
            plt.show()

        error_time.append(sum(error_time_space) / len(exp_data[time]['y']))
        error_time_space_all.append(error_time_space)

    if verbose == 1 or verbose == 2:
        title_variable = get_title_variable(variable)
        if not label:
            label = 'Time dependent error'
        plt.plot(time_vector_exp, error_time, label=label)
        plt.title(title_variable)
        plt.xlabel(r't [s]')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

    return error_time, error_time_space_all

def plot_exp_vs_model(db, shot, run_exp, run_model, time_begin, time_end, signals = ['te', 'ne', 'ti', 'ni'], label = None, verbose = False, fit_or_model = None):

    """
    Plots experimental data and model predictions, and calculates errors.

    :param db: machine considered
    :param shot: shot number
    :param run_exp: experimental run name
    :param run_model: model run name
    :param time_begin: start time
    :param time_end: end time
    :param signals: list of signal names to plot
    :param label: label for the plot
    :param verbose: verbosity level
    :param fit_or_model: whether to plot fitted data or model data
    :return: list of errors
    """

    title_fontsize = 17
    label_fontsize = 15
    legend_fontsize = 13

    variable_names = get_variable_names(signals)

    core_profiles_exp = open_and_get_ids(db, shot, 'core_profiles', run_exp)
    core_profiles_model = open_and_get_ids(db, shot, 'core_profiles', run_model)

    t_cxrs = get_t_cxrs(core_profiles_exp)
    t_cxrs = filter_time_range(t_cxrs, time_begin, time_end)

    all_exp_data = get_exp_data(db, shot, run_exp, time_begin, time_end, signals)

    errors, errors_time = {}, {}

    for variable in all_exp_data:

        exp_data, errorbar, time_vector_exp = get_exp_data_and_errorbar(all_exp_data, t_cxrs, variable)

        fit = get_onesig(core_profiles_model,variable_names[variable][2],time_begin,time_end=time_end)
        time_vector_fit = np.asarray(list(fit.keys()))

        # For every timelisce in experiments, remap all the fits on that x and then interpolate. It is necessary if the x coordinate changes in time
        ytable_final = generate_ytable(time_vector_exp, time_vector_fit, fit, exp_data, fit_and_substitute, variable)
        ytable_final = scale_model_data(ytable_final, variable)

        # Plotting routines
        if verbose:
            plot_data_and_model(exp_data, ytable_final, errorbar, variable, fit_or_model, legend_fontsize,
                                title_fontsize, label_fontsize)
        error_time, error_time_space = plot_error(time_vector_exp, exp_data, ytable_final, errorbar, variable, label, fit_or_model, verbose)

        error_variable = sum(error_time)/len(exp_data)
        errors[variable] = error_variable
        errors_time[variable] = error_time

        print('The error for ' + variable + ' is ' + str(error_variable))

    return(errors, errors_time)


def open_and_get_ids(db, shot, ids_name, run=None, username=None, backend='mdsplus'):
    if not username:
        username = getpass.getuser()

    if type(run) is str:
        user_name = f"/pfs/work/{username}/jetto/runs/{run}/imasdb/"
    else:
        user_name = username

    imas_backend = imasdef.MDSPLUS_BACKEND if backend == 'mdsplus' else imasdef.HDF5_BACKEND
    data_entry = imas.DBEntry(imas_backend, db, shot, 2 if type(run) is str else run, user_name=user_name)

    op = data_entry.open()

    if op[0] < 0:
        cp = data_entry.create()
        print(cp[0])
        if cp[0] == 0:
            print("data entry created")
    elif op[0] == 0:
        print("data entry opened")

    ids = data_entry.get(ids_name)
    data_entry.close()

    return ids


if __name__ == "__main__":
    #plot_exp_vs_model('tcv', 64965, 5, 517, 0.05, 0.15, signals = ['ti', 'ne'], verbose = 2)
    #plot_exp_vs_model('tcv', 64862, 5, 1903, 0.05, 0.15, signals = ['te', 'ne'], verbose = 2)
    #plot_exp_vs_model('tcv', 64770, 1, 1, 0.7, 0.9, signals = ['ne'], verbose = 2, fit_or_model = 'fit')
    plot_exp_vs_model('tcv', 64770, 2, 'run_64770_zeff', 0.7, 0.8, signals = ['ti'], verbose = 2, fit_or_model = 'Model')
    #plot_exp_vs_model('tcv', 64965, 5, 'run460_ohmic_predictive2', 0.05, 0.15, signals = ['te', 'ti', 'ne', 'ni'], verbose = 2)
    #plot_exp_vs_model('tcv', 64965, 5, 'run460_ohmic_predictive2', 0.05, 0.15, signals = ['ne'], verbose = 2, fit_or_model = 'Model')
    #plot_exp_vs_model('tcv', 64965, 5, 'run465_64965_ohmic_predictive6', 0.05, 0.15, signals = ['ne'], verbose = 2, fit_or_model = 'Model')
    print('plot and compares experimental data with fits or model')





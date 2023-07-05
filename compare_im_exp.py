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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist


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
    elif variable == 'vloop':
        return r'$V_{loop}$'
    elif variable == 'li3':
        return r'$li_{3}$'
    elif variable == 'ip':
        return r'$I_{p}$'

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
    elif variable == 'vloop':
        return r'$V_{loop}$ $ [V] $'
    elif variable == 'li3':
        return r'$li_{3}$ $ [-] $'
    elif variable == 'ip':
        return r'$I_{p}$ $ [A] $'

    else:
        return ''


def get_variable_names_exp(signals):
    """
    Get the variable names for a list of signals for experimental quantities
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


def get_variable_names(signals):
    """
    Get the variable names for a list of signals for other time traces without the error explicitly
    """
    variable_names = {}
    if 'vloop' in signals:
        variable_names['vloop'] = [
            'summary.global_quantities.v_loop.value'
    ]
    if 'li3' in signals:
        variable_names['li3'] = [
            #'equilibrium.time_slice[].global_quantities.li_3'
            'summary.global_quantities.li.value'
    ]
    if 'ip' in signals:
        variable_names['ip'] = [
            'summary.global_quantities.ip.value'
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
        exp_data, errorbar = apply_mask(exp_data, errorbar, time, mask)

        # Clean experimental data that are filled with nans
        mask = np.isnan(exp_data[time]['y'])
        exp_data, errorbar = apply_mask(exp_data, errorbar, time, mask)

        # Clean experimental data when errobars are too small (polluted data)
        mask = errorbar[time]['y'] < 1.0e-10
        exp_data, errorbar = apply_mask(exp_data, errorbar, time, mask)

        if len(exp_data[time]['y']) == 0 or len(errorbar[time]['y']) == 0:
            time_keys_to_remove.append(time)

    #Remove times that end up being empty
    for time in time_keys_to_remove:
        exp_data.pop(time)
        errorbar.pop(time)

    return exp_data, errorbar


def find_distances_double(errorbars, datas, min_errorbar, min_data):

    distances = []
    for errorbar, data in zip(errorbars, datas):
        distances.append(np.sqrt(np.square(errorbar - min_errorbar) + np.square(data - min_data)))

    return np.asarray(distances)


def find_distances(errorbars, min_errorbar):

    distances = np.abs(errorbars - min_errorbar)

    return distances

def extract_data_from_ydict(data):

    all_data = np.asarray([])
    for inner_dict in data.values():
        all_data = np.hstack((all_data, inner_dict['y']))

    return all_data


def build_normalized_data(data):

    # Exctract all the data
    all_data = extract_data_from_ydict(data)

    # delete repeated data for CX. Careful, this also orders the data
    all_data =  np.unique(all_data)

    #Normalize data
    ave_data = np.average(all_data)
    all_data = all_data/ave_data

    return all_data, ave_data


def smooth_array(arr, window_size):
    # Pad the array to handle edges
    padded_arr = np.pad(arr, (window_size // 2, window_size // 2), mode='edge')

    # Calculate the moving average using convolution
    smoothed_arr = np.convolve(padded_arr, np.ones(window_size) / window_size, mode='valid')

    return smoothed_arr


def find_max_distance(distances):

    # Deletes the largest distances that could skew the average
    distances = distances[:-10]

    ddist = np.diff(distances)
    ddist = smooth_array(ddist, 10)

    ddist_ave = np.average(ddist)

    index = np.argmax(ddist > ddist_ave)
    max_distance = distances[index]

    return max_distance


def find_ave_minimum(data):

    lower_chunk_length = np.size(data)//10
    #Finds the minimum but averages over the first few minimas
    sorted_arr = np.sort(data)  # Sort the array in ascending order
    lowest_chunk = sorted_arr[:lower_chunk_length]  # Get the lowest elements averaging over a tenth of the data
    avg = np.mean(lowest_chunk)  # Calculate the average

    return avg


def filter_large_errorbars(exp_data, errorbar, var):

    # Filtering data when errorbars are too large. Numbers are arbitrary and extracted from a Yann Camenen script.
    # Might be changed or could find a way to calculate 'too large' automatically

    if var == 'electron temperature' or var == 'ion temperature':
        max_errorbars = 100
    elif var == 'electron density':
        max_errorbars = 2e19
    elif var == 'impurity density':
        max_errorbars = 0.7e17
        #max_errorbars = 100e17

    for time in exp_data:

        # Finds the mask
        mask = errorbar[time]['y'] > max_errorbars
        exp_data, errorbar = apply_mask(exp_data, errorbar, time, mask)

    return exp_data, errorbar

def special_filter_errorbars(exp_data, errorbar, var):

    # Special parameter, quite arbitrary, to try not to filter too much data
    mult = 1.0

    # Apply only to impurity density
    all_errorbars, ave_errorbars = build_normalized_data(errorbar)
    min_errorbar = find_ave_minimum(all_errorbars)
    distances = find_distances(all_errorbars, min_errorbar)
    max_distance = mult*find_max_distance(distances)

    for time in exp_data:
        # Finds the mask
        normalized_errorbar = errorbar[time]['y']/ave_errorbars
        distances = find_distances(normalized_errorbar, min_errorbar)
        mask = distances > max_distance
       
        exp_data, errorbar = apply_mask(exp_data, errorbar, time, mask)

    return exp_data, errorbar


def apply_mask(exp_data, errorbar, time, mask):

    exp_data[time]['y'] = exp_data[time]['y'][~mask]
    exp_data[time]['x'] = exp_data[time]['x'][~mask]
    errorbar[time]['y'] = errorbar[time]['y'][~mask]
    errorbar[time]['x'] = errorbar[time]['x'][~mask]

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
                #ytable_final[ii] = ytable_slice*1.0
            except TypeError:
                print('No experimental data available for slice number ' + str(ii) + '. Aborting')
                exit()

    return ytable_final

def get_exp_data(db, shot, run, time_begin, time_end, signals, apply_special_filter = False):
    exp_data = {}
    variable_names = get_variable_names_exp(signals)
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
        if apply_special_filter and variable == 'impurity density':
            data, errorbar = special_filter_errorbars(data, errorbar, variable)
        else:
            data, errorbar = filter_large_errorbars(data, errorbar, variable)

        data = scale_exp_data(data, variable)
        errorbar = scale_exp_data(errorbar, variable)

        exp_data[variable] = {'data': data, 'errorbar': errorbar}

    return exp_data


def filter_time_range(t_cxrs, time_begin, time_end):
    t_cxrs = np.asarray(t_cxrs).flatten()
    mask = np.logical_and(t_cxrs >= time_begin, t_cxrs <= time_end)
    return t_cxrs[mask]


def delete_meaningless_data_cxrs(data, errorbar, time_cxrs):

    mask = errorbar[time_cxrs]['y'] < data[time_cxrs]['y']
    data[time_cxrs]['x'] = data[time_cxrs]['x'][mask]
    data[time_cxrs]['y'] = data[time_cxrs]['y'][mask]
    errorbar[time_cxrs]['x'] = errorbar[time_cxrs]['x'][mask]
    errorbar[time_cxrs]['y'] = errorbar[time_cxrs]['y'][mask]

    return data, errorbar

def get_closest_times(exp_data, t_cxrs):
    time_vector_exp = np.asarray(list(exp_data['data'].keys()))
    exp_data_new = {'data': {}, 'errorbar': {}}
    for time_cxrs in t_cxrs:
        time_closest = time_vector_exp[np.abs(time_vector_exp - time_cxrs).argmin(0)]

        exp_data_new['data'][time_cxrs] = exp_data['data'][time_closest]
        exp_data_new['errorbar'][time_cxrs] = exp_data['errorbar'][time_closest]

        exp_data_new['data'][time_cxrs], exp_data_new['errorbar'][time_cxrs] 
        mask = exp_data_new['errorbar'][time_cxrs]['y']<exp_data_new['data'][time_cxrs]['y']
        exp_data_new['data'][time_cxrs]['x'] = exp_data_new['data'][time_cxrs]['x'][mask]
        exp_data_new['data'][time_cxrs]['y'] = exp_data_new['data'][time_cxrs]['y'][mask]
        exp_data_new['errorbar'][time_cxrs]['x'] = exp_data_new['errorbar'][time_cxrs]['x'][mask]
        exp_data_new['errorbar'][time_cxrs]['y'] = exp_data_new['errorbar'][time_cxrs]['y'][mask]

    exp_data = exp_data_new
    time_vector_exp = np.unique(t_cxrs)

    # Backward compatibility since Ti was stored sligtly differently in old IDSs
    old = False
    if old:
        for time in time_vector_exp:
            exp_data['errorbar'][time]['y'] = exp_data['errorbar'][time]['y'] - exp_data['data'][time]['y']

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


def get_legend_label_experimental(variable):

    if variable == 'electron temperature' or variable == 'electron density':
        legend_label = 'HRTS Experimental data'
    elif variable == 'ion temperature' or variable == 'impurity density':
        legend_label = 'CXRS Experimental data'

    return legend_label


#Show_fit does not do anazthing tight now. Will decide if I want to implement the fit as well later.\
#It is already working in compare bundle and maybe it belongs there
def plot_data_and_model(exp_data, ytable_final, errorbar, variable, fit_or_model, legend_fontsize,
                        title_fontsize, label_fontsize, show_fit = False):
    title_variable = get_title_variable(variable)

    for i, time in enumerate(exp_data):
        legend_label = get_legend_label_experimental(variable)
        plt.errorbar(exp_data[time]['x'], exp_data[time]['y'], yerr=errorbar[time]['y'], fmt='.', linestyle=' ',
                     label=legend_label)
        legend_label = get_legend_label(fit_or_model)
        plt.plot(exp_data[time]['x'], ytable_final[i], label=legend_label)

        plt.legend(fontsize=legend_fontsize)
        title = title_variable + ' at t = {time:.3f}'.format(time=time)
        plt.title(title, fontsize=title_fontsize)
        plt.xlabel(r'$\rho$ [-]', fontsize=label_fontsize)
        ylabel_variable = get_label_variable(variable)
        plt.ylabel(ylabel_variable, fontsize=label_fontsize)
        plt.xlim((0,1))
        plt.show()


def plot_error(time_vector_exp, exp_data, ytable_final, errorbar, variable, label, fit_or_model, verbose, show_fit = False):
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
            plt.ylabel(r'$ \sigma_{\rho} $ [-]')
            plt.xlim((0,1))
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
        plt.ylabel(r'$ \sigma_{t} $ [-]')
        plt.legend()
        plt.show()

    return error_time, error_time_space_all

def plot_exp_vs_model(db, shot, run_exp, run_model, time_begin, time_end, signals = ['te', 'ne', 'ti', 'ni'], label = None, verbose = False, fit_or_model = None, show_fit = False, apply_special_filter = False):

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
    :param apply_special_filter: applies a special filter to the impurity density that tries to avoid clusters with large errorbars
    :return: list of errors
    """

    title_fontsize = 17
    label_fontsize = 15
    legend_fontsize = 13

    variable_names = get_variable_names_exp(signals)

    core_profiles_exp = open_and_get_ids(db, shot, 'core_profiles', run_exp)
    core_profiles_model = open_and_get_ids(db, shot, 'core_profiles', run_model, show_fit = show_fit)

    t_cxrs = get_t_cxrs(core_profiles_exp)
    t_cxrs = filter_time_range(t_cxrs, time_begin, time_end)

    all_exp_data = get_exp_data(db, shot, run_exp, time_begin, time_end, signals, apply_special_filter = apply_special_filter)

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
                                title_fontsize, label_fontsize, show_fit = show_fit)
        error_time, error_time_space = plot_error(time_vector_exp, exp_data, ytable_final, errorbar, variable, label, fit_or_model, verbose, show_fit = show_fit)

        error_variable = sum(error_time)/len(exp_data)
        errors[variable] = error_variable
        errors_time[variable] = error_time

        print('The error for ' + variable + ' is ' + str(error_variable))

    return(errors, errors_time)


def open_and_get_ids(db, shot, ids_name, run=None, username=None, backend='mdsplus', show_fit = False):
    if not username:
        username = getpass.getuser()

    if type(run) is str:
        user_name = f"/pfs/work/{username}/jetto/runs/{run}/imasdb/"
    else:
        user_name = username

    imas_backend = imasdef.MDSPLUS_BACKEND if backend == 'mdsplus' else imasdef.HDF5_BACKEND
    if show_fit:
        data_entry = imas.DBEntry(imas_backend, db, shot, 1 if type(run) is str else run, user_name=user_name)
    else:
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
    #plot_exp_vs_model('tcv', 64965, 3, 'run010_64965_ohmic_predictive', 0.04, 0.33, signals = ['ne','te','ti','ni'], verbose = 1, fit_or_model = 'Model')

    print('plot and compares experimental data with fits or model')





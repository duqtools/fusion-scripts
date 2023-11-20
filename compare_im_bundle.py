import os
import getpass
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
#sys.path.insert(0, '/afs/eufus.eu/user/g/g2mmarin/python_tools/jetto-pythontools')
#import jetto_tools
#import duqtools
import argparse

import compare_im_runs
import prepare_im_input
import compare_im_exp

from sawteeth import plot_inversion_radius

'''

Shows the comparison for multiple signals, multiple plots and multiple settings

Example of usage:

plot_errors('/afs/eufus.eu/user/g/user/mylist.txt', ['summary.global_quantities.v_loop.value', 'core_profiles.profiles_1d[].q', 'summary.global_quantities.li.value', 'te'], plot_type = 1, time_begin = 0.02, time_end = 0.1)
plot_all_traces('runlist_paper1.txt', ['ne', 'te', 'ti', 'ni'], correct_sign=True, time_begin = 0.04, time_end = 0.33)

'''

def input():

    parser = argparse.ArgumentParser(
        description=
    """Compare validation metrics from HFPS input / output IDSs, for multiple quantities and runs. Preliminary version, using scripts from D. Yadykin and M. Marin.\n
    ---
    Examples:\n
    python compare_im_bundle.py --filename runlist.txt --signals te ne --time_begin 0.1 --time_end 0.3 --correct_sign --show_fit --error_type 'absolute' --plot_traces
    python compare_im_bundle.py --filename runlist.txt --signals te ne --time_begin 0.1 --time_end 0.3 --error_type 'absolute'
    ---
    """,
    epilog="",
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--filename",   "-f",               type=str,   default=None,                                 help="Name of the file with the run information")
    parser.add_argument("--signals",     "-s",   nargs='+', type=str,   default=None,                                 help="List of signals to be compared")
    parser.add_argument("--time_begin",                     type=float, default=None,                                 help="Slice shot file beginning at time (s)")
    parser.add_argument("--time_end",                       type=float, default=None,                                 help="Slice shot file ending at time (s)")
    parser.add_argument("--plot_type",                      type=int,   default=2,                                    help="Plot time. 1 groups sensitivities, 2 groups shots")
    parser.add_argument("--username",                       type=str,   default=None,                                 help="Username of the owner of input IDSs and runs")
    parser.add_argument("--correct_sign",                               default=False, action='store_true',           help="Flag to correct sign if opposite for exp and model")
    parser.add_argument("--signal_operations",              type=str,   default=None,                                 help="Performs operations between signals")
    parser.add_argument("--show_fit",                                   default=False, action='store_true',           help="Shows fit instead of modelling results")
    parser.add_argument("--error_type",                     type=str,   default='absolute',                           help="Decides which metric to use for the errors")
    parser.add_argument("--plot_traces",                                default=False, action='store_true',           help="Plot only the errors or the full trace")

    args=parser.parse_args()

    return args


fontsize_title = 20
fontsize_xlabel = 15
fontsize_ylabel = 15
fontsize_legend = 11
fontsize_ticks = 14

exp_signal_list = ['te', 'ne', 'ti', 'ni']

def get_label_variable(variable):
    """
    Get the label variable string for a given variable.
    """
    if variable == 'te':
        return r'$T_e$'
    elif variable == 'ne':
        return r'$n_e$'
    elif variable == 'ti':
        return r'$T_i$'
    elif variable == 'ni':
        return r'$n_C$'
    elif variable == 'summary.global_quantities.v_loop.value':
        return r'$V_{loop}$'
    elif variable == 'summary.global_quantities.li.value':
        return r'$li_{3}$'

    elif variable == 'core_profiles.profiles_1d[].electrons.density':
        return r'$n_e$'
    elif variable == 'core_profiles.profiles_1d[].electrons.temperature':
        return r'$T_e$'
    elif variable == 'core_profiles.profiles_1d[].ion[0].density':
        return r'$n_C$'
    elif variable == 'core_profiles.profiles_1d[].ion[0].temperature':
        return r'$T_i$'

    else:
        return variable


def get_measure_variable(variable, error_type = 'absolute'):
    """
    Get the label variable string for a given variable.
    """
    if error_type == 'relative':
        return r'[-]'

    if variable == 'te':
        return r'[-]'
    elif variable == 'ne':
        return r'[-]'
    elif variable == 'ti':
        return r'[-]'
    elif variable == 'ni':
        return r'[-]'
    elif variable == 'summary.global_quantities.v_loop.value':
        return r'[V]'
    elif variable == 'summary.global_quantities.li.value':
        return r'[-]'

    elif variable == 'core_profiles.profiles_1d[].electrons.density':
        return r'[$m^{-3}$]'
    elif variable == 'core_profiles.profiles_1d[].electrons.temperature':
        return r'[eV]'
    elif variable == 'core_profiles.profiles_1d[].ion[0].density':
        return r'[$m^{-3}$]'
    elif variable == 'core_profiles.profiles_1d[].ion[0].temperature':
        return r'[eV]'

    else:
        return variable


def get_error(shot, run_input, run_output, signal, time_begin = None, time_end = None, db = 'tcv', error_type = 'absolute'):

    time_begin, time_end = compare_im_exp.get_time_begin_and_end(db, shot, run_output, time_begin, time_end)

    username = os.getenv("USER")

    idslist = generate_ids_list(username, db, shot, run_input, [run_output], show_fit = False)
    signals = [signal]
    show_plot = False

    if time_begin > time_end:
        print('Warning! One run did not finish properly. Substituting 0 as error')
        errors = [0.0]
    else:
        time_averages, time_error_averages, profile_error_averages = compare_im_runs.compare_runs(signals, idslist, time_begin, time_end=time_end, plot=False, analyze=True, correct_sign=True, signal_operations=None, error_type = error_type)

        errors = []
        for signal in time_error_averages:
            for run_tag in time_error_averages[signal]:
                errors.append(time_error_averages[signal][run_tag])

        for signal in profile_error_averages:
            for run_tag in profile_error_averages[signal]:
                errors.append(profile_error_averages[signal][run_tag])

    return errors


def get_exp_error(shot, run_input, run_output, signal, time_begin = None, time_end = None, db = 'tcv', show_fit = False, error_type = 'absolute'):

    time_begin, time_end = compare_im_exp.get_time_begin_and_end(db, shot, run_output, time_begin, time_end)

    errors_dict, errors_time_dict = compare_im_exp.plot_exp_vs_model(db, shot, run_input, run_output, time_begin, time_end, signals = [signal], verbose = 0, show_fit = show_fit, error_type = error_type)

    errors = []

    # revert to list format
    for error in errors_dict.values():
        errors.append(error)

    return errors


def get_exp_error_time(shot, run_input, run_output, signal, time_begin = None, time_end = None, db = 'tcv', show_fit = False, error_type = 'absolute'):

    summary = compare_im_exp.open_and_get_ids(db, shot, 'summary', run_output)
    if not time_begin:
        time_begin = min(summary.time) + 0.01
    if not time_end:
        time_end = max(summary.time)

    errors_dict, errors_time_dict = compare_im_exp.plot_exp_vs_model(db, shot, run_input, run_output, time_begin, time_end, signals = [signal], verbose = 0, show_fit = show_fit, error_type = error_type)

    errors_time = []
    #revert to list format
    for error_time in errors_time_dict.values():
        errors_time.append(error_time)

    return errors_time


def get_sawteeth_error(shot, run_output, saw_file_path = None, time_begin = None, time_end = None, db = 'tcv'):

    if not saw_file_path:
        print('File for sawteeth not provided. This should not happen. Terminating')
        exit()

    saw_file = Path(saw_file_path)

    if not saw_file.is_file():
        print('The path provided for the sawteeth file of shot ' + str(shot) + ' is not valid')
        print('Returning 0. Careful! Does not mean perfect agreement!')
        return(0)

    return plot_inversion_radius(db, shot, run_output, saw_file_path, time_start = time_begin, time_end = time_end, error_type = error_type)


def plot_errors_time(filename, signals, time_begin = None, time_end = None, error_type = 'absolute'):

    '''

    filename is the name of the file where the runs are stored

    '''

    db, shot, run_exp, labels, runs_output = read_file_time_dep(filename)

    core_profiles_exp = compare_im_exp.open_and_get_ids(db, shot, 'core_profiles', run_exp)

    t_cxrs = compare_im_exp.get_t_cxrs(core_profiles_exp)
    t_cxrs = compare_im_exp.filter_time_range(t_cxrs, time_begin, time_end)

    fig, ax, num_columns = create_subplots(signals)

    for isignal, signal in enumerate(signals):

        icolumns = int(isignal/num_columns)
        iraws = isignal % num_columns

        #labels = shot_list

        all_exp_data = compare_im_exp.get_exp_data(db, shot, run_exp, time_begin, time_end, signal)
        exp_data, errorbar, time_vector_exp = compare_im_exp.get_exp_data_and_errorbar(all_exp_data, t_cxrs, list(all_exp_data.keys())[0])

        errors = []
        for run_output in runs_output:
            errors.append(get_exp_error_time(shot, run_exp, run_output, signal, time_begin = time_begin, time_end = time_end, error_type = error_type)[0])

        for error, legend_label in zip(errors, labels):
            ax[icolumns][iraws].plot(time_vector_exp, error, label=legend_label)
            #plt.plot(time_vector_exp, error, label=legend_label)

        variable = compare_im_exp.get_variable_names_exp(signal)
        title_variable = compare_im_exp.get_title_variable(list(variable.keys())[0])
        title = 'Normalized error for ' + title_variable

        ax[icolumns][iraws].set_title(title, fontsize = fontsize_title)
        ax[icolumns][iraws].legend(fontsize = fontsize_legend)
        ax[icolumns][iraws].set_xlabel(r'time [s]', fontsize = fontsize_xlabel)
        ax[icolumns][iraws].set_ylabel(r'$\sigma$ [-]', fontsize = fontsize_ylabel)
        fig.tight_layout()

    plt.show()

def separate_signal_lists(signals):

    exp_signal_list_all = ['te', 'ti', 'ne', 'ni']

    exp_signals, mod_signals = [], []
    for signal in signals:
        if signal in exp_signal_list_all:
            exp_signals.append(signal)
        else:
            mod_signals.append(signal)

    return exp_signals, mod_signals


def plot_all_traces(filename, signals, time_begin = None, time_end = None, username = None, correct_sign=False, signal_operations=None, show_fit = False, error_type = 'absolute', error_type_exp = None):

    # Ideally want to use the other functions, but I cannot figure out how to use the axes that way...
    if not username: username=getpass.getuser()
    if not error_type_exp: error_type_exp = error_type
    exp_signals, mod_signals = separate_signal_lists(signals)
    fig, ax, num_columns = create_subplots(signals)

    db, shot, run_exp, labels, runs_output = read_file_time_dep(filename, show_fit = show_fit)

    if db:
        core_profiles_exp = compare_im_exp.open_and_get_ids(db, shot, 'core_profiles', run_exp)
        t_cxrs = compare_im_exp.get_t_cxrs(core_profiles_exp)
        t_cxrs = compare_im_exp.filter_time_range(t_cxrs, time_begin, time_end)

    variables = compare_im_exp.get_variable_names(mod_signals)

    ylabels, titles = {}, {}
    if not signals: signals = []
    for key in variables:
        mod_signals.remove(key)
        mod_signals.append(variables[key][0])
        signals.remove(key)
        signals.append(variables[key][0])
        ylabels[variables[key][0]] = compare_im_exp.get_label_variable(key)
        titles[variables[key][0]] = compare_im_exp.get_title_variable(key)

    # For now not showing the fits in compare runs, where thez should be 0
    idslist = generate_ids_list(username, db, shot, run_exp, runs_output, show_fit = False)
    data_dict, ref_tag = compare_im_runs.generate_data_tables(idslist, mod_signals, time_begin, time_end=time_end, signal_operations=signal_operations, correct_sign=correct_sign, standardize=True)

    options_trace, options_profile = {'error': True, 'error_type' : error_type}, {'average_error': True, 'error_type' : error_type}
    time_error_dict = compare_im_runs.perform_time_trace_analysis(data_dict, **options_trace)
    profile_error_dict = compare_im_runs.perform_profile_analysis(data_dict, **options_profile)

    for isignal, signame in enumerate(signals):
        icolumns = int(isignal/num_columns)
        iraws = isignal % num_columns

        if signame in mod_signals:
            #value_when_true if condition else value_when_false
            title = titles[signame] if signame in titles else signame
            ylabel = ylabels[signame] if signame in ylabels else signame

            if signame+'.'+error_type+'_error.t' in time_error_dict:
                for run, label in zip(time_error_dict[signame+'.'+error_type+'_error'], labels):
                    ax[icolumns][iraws].plot(time_error_dict[signame+'.'+error_type+'_error.t'], time_error_dict[signame+'.'+error_type+'_error'][run].flatten(), label=label)
            elif signame+'.average_'+error_type+'_error.t' in profile_error_dict:
                for run, label in zip(profile_error_dict[signame+'.average_'+error_type+'_error'], labels):
                    ax[icolumns][iraws].plot(profile_error_dict[signame+'.average_'+error_type+'_error.t'], profile_error_dict[signame+'.average_'+error_type+'_error'][run].flatten(), label=label)
            else:
                print('signal ' + signame + ' not recognized, aborting')

            ax[icolumns][iraws].set_title(title)
            ax[icolumns][iraws].legend(loc='best')
            ax[icolumns][iraws].set_xlabel(r'time [s]', fontsize = fontsize_xlabel)
            ax[icolumns][iraws].set_ylabel(r'$\sigma$ ' + ylabel, fontsize = fontsize_ylabel)

        elif signame in exp_signals:

            all_exp_data = compare_im_exp.get_exp_data(db, shot, run_exp, time_begin, time_end, signame)
            exp_data, errorbar, time_vector_exp = compare_im_exp.get_exp_data_and_errorbar(all_exp_data, t_cxrs, list(all_exp_data.keys())[0])

            errors = []
            for run_output in runs_output:
                errors.append(get_exp_error_time(shot, run_exp, run_output, signame, time_begin = time_begin, time_end = time_end, error_type = error_type)[0])
                if show_fit:
                    errors.append(get_exp_error_time(shot, run_exp, run_output, signame, time_begin = time_begin, time_end = time_end, show_fit = show_fit, error_type = error_type_exp)[0])

            for error, legend_label in zip(errors, labels):
                ax[icolumns][iraws].plot(time_vector_exp, error, label=legend_label)

            variable = compare_im_exp.get_variable_names_exp(signame)
            title_variable = compare_im_exp.get_title_variable(list(variable.keys())[0])
            title = 'Normalized error for ' + title_variable

            ax[icolumns][iraws].set_title(title, fontsize = fontsize_title)
            ax[icolumns][iraws].legend(fontsize = fontsize_legend)
            ax[icolumns][iraws].set_xlabel(r'time [s]', fontsize = fontsize_xlabel)
            ax[icolumns][iraws].set_ylabel(r'$\sigma$ [-]', fontsize = fontsize_ylabel)
        else:
            print('signal not recognized. Aborting')
            exit()

    plt.show()


def create_subplots(signal_list):
    if len(signal_list) == 1:
        fig, ax = plt.subplots(1,1)
        ax, num_columns = [[ax]], 1
    elif len(signal_list) == 2:
        fig, ax = plt.subplots(1,2)
        ax, num_columns = [ax], 2
    elif len(signal_list) == 3:
        fig, ax = plt.subplots(1,3)
        ax, num_columns = [ax], 3
    elif len(signal_list) == 4:
        fig, ax = plt.subplots(2,2)
        num_columns = 2
    elif len(signal_list) == 5:
        fig, ax = plt.subplots(1,5)
        num_columns = 1
    elif len(signal_list) == 6:
        fig, ax = plt.subplots(2,3)
        num_columns = 2


    return fig, ax, num_columns


def read_file_time_dep(filename, show_fit = False):

    file_runs = open(filename, 'r')
    lines = file_runs.readlines()

    if len(lines[0].split(' ')) == 3:
        db, shot, run_exp = lines[0].split(' ')
        shot, run_exp = int(shot), int(run_exp)
    else:
        db, shot, run_exp = None, None, None


    labels, runs_output = [], []
    for line in lines[1:]:
        line = line.replace('\n','')
        if '|' in lines[1]:
            labels.append(line.split('|')[-1])
            line = line.split('|')[0]
        else:
            labels.append(line)

        runs_output.append(line)

        if show_fit:
            labels.append('fit')

    return db, shot, run_exp, labels, runs_output


def generate_ids_list(username, db, shot, run_exp, runs_output, show_fit = False):

    #The reference ids, which should be experimental data, goes first
    idslist = [f'{username}/{db}/{shot}/{run_exp}']
    for run in runs_output:
        if type(run) is int or isinstance(run, np.integer):
            ids = f'{username}/{db}/{shot}/{run}'
        elif type(run) is str or isinstance(run, np.str_):
            ids = f'/pfs/work/{username}/jetto/runs/{run}/imasdb/{db}/{shot}/2'
        else:
            print('run of type ' + str(type(run)) + ' is not supported. Aborting')

        idslist.append(ids)

        # Might be useful to also show the agreement of the fit.
        if show_fit and (type(run) is str or isinstance(run, np.str_)):
            ids = f'/pfs/work/{username}/jetto/runs/{run}/imasdb/{db}/{shot}/1'
            idslist.append(ids)

    return idslist


def plot_errors_traces(filename, signals, username = None, time_begin = None, time_end = None, correct_sign=False, steady_state=False, uniform=False, signal_operations=None, show_fit = False, error_type = 'absolute'):

    db, shot, run_exp, labels, runs_output = read_file_time_dep(filename, show_fit = show_fit)
    if not username: username=getpass.getuser()

    variables = compare_im_exp.get_variable_names(signals)

    ylabels, titles = {}, {}
    if not signals: signals = []
    for key in variables:
        signals.remove(key)
        signals.append(variables[key][0])
        ylabels[variables[key][0]] = compare_im_exp.get_label_variable(key)
        titles[variables[key][0]] = compare_im_exp.get_title_variable(key)

    fig, ax, num_columns = create_subplots(signals)
    idslist = generate_ids_list(username, db, shot, run_exp, runs_output, show_fit = show_fit)
    data_dict, ref_tag = compare_im_runs.generate_data_tables(idslist, signals, time_begin, time_end=time_end, signal_operations=signal_operations, correct_sign=correct_sign, standardize=True)

    #options_trace, options_profile = {'absolute_error': True}, {'average_absolute_error': True}
    options_trace, options_profile = {'error': True, 'error_type' : error_type}, {'average_error': True, 'error_type' : error_type}
    time_error_dict = compare_im_runs.perform_time_trace_analysis(data_dict, **options_trace)
    profile_error_dict = compare_im_runs.perform_profile_analysis(data_dict, **options_profile)

    for isignal, signame in enumerate(signals):
        icolumns = int(isignal/num_columns)
        iraws = isignal % num_columns

        title = titles[signame] if signame in titles else signame
        ylabel = ylabels[signame] if signame in ylabels else signame

        if signame+'.'+error_type+'_error.t' in time_error_dict:
            for run, label in zip(time_error_dict[signame+'.'+error_type+'_error'], labels):
                ax[icolumns][iraws].plot(time_error_dict[signame+'.'+error_type+'_error.t'], time_error_dict[signame+'.'+error_type+'_error'][run].flatten(), label=label)
        elif signame+'.average_'+error_type+'_error.t' in profile_error_dict:
            for run, label in zip(profile_error_dict[signame+'.average_'+error_type+'_error'], labels):
                ax[icolumns][iraws].plot(profile_error_dict[signame+'.average_'+error_type+'_error.t'], profile_error_dict[signame+'.average_'+error_type+'_error'][run].flatten(), label=label)
        else:
            print('signal ' + signame + ' not recognized, aborting')

        ax[icolumns][iraws].set_title(title, fontsize = fontsize_title)
        ax[icolumns][iraws].legend(loc='best', fontsize = fontsize_legend)
        ax[icolumns][iraws].set_xlabel(r'time [s]', fontsize = fontsize_xlabel)
        ax[icolumns][iraws].set_ylabel(r'$\sigma$ ' + ylabel, fontsize = fontsize_ylabel)
        #fig.tight_layout()

    plt.show()


def get_input_run_lists(filename):

    file_runs = open(filename, 'r')
    lines = file_runs.readlines()
    shot_list, run_input_list, run_output_list, saw_file_paths = [], [], [], []
    labels_plot = lines[0][:-1].split('|')

    for line in lines[1:]:
        line = line.rstrip()
        shot, run_input, *runs_output = line.split(' ')
        shot_list.append(int(shot))
        run_input_list.append(int(run_input))
        # Want to compare multiple series of runs to check which one is better
        # If the last element of the split is a path, that will be the path for the math file of the sawteeth analysis
        # Being the last value the string has a /n as last character. It is removed by [:-1]
        candidate_saw_path = Path(runs_output[-1])
        if not candidate_saw_path.is_file():
            num_run_series = len(runs_output)
            for run_output in runs_output:
                if run_output.isdigit():
                    run_output_list.append(int(run_output))
                else:
                    run_output = run_output.replace('\n','')
                    run_output_list.append(run_output)
            # Maybe not a nice way to do it but keeps the structure later
            saw_file_paths.append(None)
        else:
            num_run_series = len(runs_output) - 1
            for run_output in runs_output[:-1]:
                if run_output.isdigit():
                    run_output_list.append(int(run_output))
                else:
                    run_output_list.append(run_output)
            saw_file_paths.append(runs_output[-1][:-1])

    run_output_list = np.asarray(run_output_list).reshape(len(lines)-1, num_run_series)

    return num_run_series, shot_list, run_input_list, run_output_list, saw_file_paths, labels_plot


def plot_errors(filename, signal_list, time_begin = None, time_end = None, plot_type = 1, error_type = 'absolute', error_type_exp = None):

    if not error_type_exp: error_type_exp = error_type
    num_run_series, shot_list, run_input_list, run_output_list, saw_file_paths, labels_plot = get_input_run_lists(filename)

    # Pre deciding the structure of the plots with the various possibilities of len(signal_list)
    fig, ax, num_columns = create_subplots(signal_list)
    label_signals, measure_signals = {}, {}
    for signal in signal_list:
        label_signals[signal] = get_label_variable(signal)
        measure_signals[signal] = get_measure_variable(signal, error_type = error_type)

    if plot_type == 1:
        for isignal, signal in enumerate(signal_list):
            # Needed for aoutoformatting
            icolumns = int(isignal/num_columns)
            iraws = isignal % num_columns

            labels = shot_list
            x = np.arange(len(labels))  # the label locations
            width = 1/(num_run_series+1)  # the width of the bars

            for run_serie in range(num_run_series):
                errors = []
                #if signal not in exp_signal_list:
                for shot, run_input, run_output, saw_file_path in zip(shot_list, run_input_list, run_output_list[:,run_serie], saw_file_paths):
                    if signal in exp_signal_list:
                        errors.append(get_exp_error(shot, run_input, run_output, signal, time_begin = time_begin, time_end = time_end, error_type = error_type_exp)[0])
                    elif signal == 'sawteeth':
                        errors.append(get_sawteeth_error(shot, run_output, saw_file_path, time_begin = time_begin, time_end = time_end))
                    else:
                        errors.append(get_error(shot, run_input, run_output, signal, time_begin = time_begin, time_end = time_end, error_type = error_type)[0])

                rects = ax[icolumns][iraws].bar(x - width + width/num_run_series + run_serie*width, errors, width, label=labels_plot[run_serie])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            if signal in ['te', 'ne', 'ti', 'ni', 'sawteeth']:
                ax[icolumns][iraws].set_ylabel('Error [-]', fontsize = fontsize_ylabel)
            else:
                ax[icolumns][iraws].set_ylabel('Distance ' + measure_signals[signal], fontsize = fontsize_ylabel)
            ax[icolumns][iraws].set_title(label_signals[signal], fontsize = fontsize_title)
            ax[icolumns][iraws].set_xticks(x, labels, rotation='vertical', fontsize = fontsize_ticks)
            ax[icolumns][iraws].xaxis.set_tick_params(labelsize = fontsize_ticks)
            ax[icolumns][iraws].legend(fontsize = fontsize_legend)

            fig.tight_layout()

        plt.show()

    elif plot_type == 2:
        for isignal, signal in enumerate(signal_list):
            # Needed for aoutoformatting
            icolumns = int(isignal/num_columns)
            iraws = isignal % num_columns

            x = np.arange(num_run_series)  # the label locations
            width = 1/(len(shot_list)+1)  # the width of the bars

            for shot in range(len(shot_list)):
                errors = []
                for run_output in run_output_list[shot,:]:
                    if signal == 'sawteeth':
                        errors.append(get_sawteeth_error(shot_list[shot], run_output, saw_file_paths[shot], time_begin = time_begin, time_end = time_end))
                    elif signal not in exp_signal_list:
                        errors.append(get_error(shot_list[shot], run_input_list[shot], run_output, signal, time_begin = time_begin, time_end = time_end, error_type = error_type)[0])
                    else:
                        errors.append(get_exp_error(shot_list[shot], run_input_list[shot], run_output, signal, time_begin = time_begin, time_end = time_end, error_type = error_type_exp)[0])

                rects = ax[icolumns][iraws].bar(x - width + width/len(shot_list) + shot*width, errors, width, label=shot_list[shot])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            if signal in ['te', 'ne', 'ti', 'ni', 'sawteeth']:
                ax[icolumns][iraws].set_ylabel('Error [-]', fontsize = fontsize_ylabel)
            else:
                ax[icolumns][iraws].set_ylabel('Distance ' + measure_signals[signal], fontsize = fontsize_ylabel)
            ax[icolumns][iraws].set_title(label_signals[signal], fontsize = fontsize_title)
            ax[icolumns][iraws].set_xticks(x, labels_plot, rotation='vertical', fontsize = fontsize_ticks)
            ax[icolumns][iraws].xaxis.set_tick_params(labelsize = fontsize_ticks)
            if len(shot_list) < 5:
                ax[icolumns][iraws].legend(fontsize = fontsize_legend)
            else:
                ax[icolumns][iraws].legend().set_visible(False)

            fig.tight_layout()

        plt.show()


def main():

    args = input()

    if not args.plot_traces:
        plot_errors(
            filename=args.filename,
            signal_list=args.signals,
            time_begin=args.time_begin,
            time_end=args.time_end,
            plot_type=args.plot_type,
            error_type=args.error_type
        )
    else:
        plot_all_traces(
            filename=args.filename,
            signals=args.signals,
            time_begin=args.time_begin,
            time_end=args.time_end,
            username=args.username,
            show_fit=args.show_fit,
            error_type=args.error_type,
            correct_sign=args.correct_sign,
            signal_operations=args.signal_operations
        )


if __name__ == "__main__":

    main()



import os
import getpass
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, '/afs/eufus.eu/user/g/g2mmarin/python_tools/jetto-pythontools')
import jetto_tools
import duqtools

import compare_im_runs
import prepare_im_input
import compare_im_exp

from sawteeth import plot_inversion_radius

'''

Shows the comparison for multiple signals, multiple plots and multiple settings

Example of usage:

plot_errors('/afs/eufus.eu/user/g/user/mylist.txt', ['summary.global_quantities.v_loop.value', 'core_profiles.profiles_1d[].q', 'summary.global_quantities.li.value', 'te'], plot_type = 1, time_begin = 0.02, time_end = 0.1)


'''

exp_signal_list = ['te', 'ne', 'ti', 'ni']

def get_error(shot, run_input, run_output, signal, time_begin = None, time_end = None, db = 'tcv'):

    # The time compared is the time of the simulation. Still on the experiment time array
    summary = compare_im_exp.open_and_get_ids(db, shot, 'summary', run_output)

    username = os.getenv("USER")
    userlist = [username]
    dblist = [db]
    shotlist = [shot]
    runlist = [run_input,run_output]
    # Exclude very fast equilibrium readjusting on the first few timesteps
    if not time_begin:
        time_begin = min(summary.time) + 0.01
    if not time_end:
        time_end = max(summary.time)
    signal = [signal]
    show_plot = False

    time_averages, time_error_averages, profile_error_averages = compare_im_runs.compare_runs(signal, dblist, shotlist, runlist, time_begin, userlist = userlist, time_end=time_end, plot=False, analyze=True, correct_sign=True, signal_operations=None)

    errors = []
    for signal in time_error_averages:
        for run_tag in time_error_averages[signal]:
            errors.append(time_error_averages[signal][run_tag])

    for signal in profile_error_averages:
        for run_tag in profile_error_averages[signal]:
            errors.append(profile_error_averages[signal][run_tag])

    return errors


def get_exp_error(shot, run_input, run_output, signal, time_begin = None, time_end = None, db = 'tcv', show_fit = False):

    summary = compare_im_exp.open_and_get_ids(db, shot, 'summary', run_output)
    if not time_begin:
        time_begin = min(summary.time) + 0.01
    if not time_end:
        time_end = max(summary.time)

    errors_dict, errors_time_dict = compare_im_exp.plot_exp_vs_model(db, shot, run_input, run_output, time_begin, time_end, signals = [signal], verbose = 0, show_fit = show_fit)

    errors = []

    # revert to list format
    for error in errors_dict.values():
        errors.append(error)

    return errors


def get_exp_error_time(shot, run_input, run_output, signal, time_begin = None, time_end = None, db = 'tcv', show_fit = False):

    summary = compare_im_exp.open_and_get_ids(db, shot, 'summary', run_output)
    if not time_begin:
        time_begin = min(summary.time) + 0.01
    if not time_end:
        time_end = max(summary.time)

    errors_dict, errors_time_dict = compare_im_exp.plot_exp_vs_model(db, shot, run_input, run_output, time_begin, time_end, signals = [signal], verbose = 0, show_fit = show_fit)

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

    return plot_inversion_radius(db, shot, run_output, saw_file_path, time_start = time_begin, time_end = time_end)


def plot_errors_time(filename, signals, time_begin = None, time_end = None):

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
            errors.append(get_exp_error_time(shot, run_exp, run_output, signal, time_begin = time_begin, time_end = time_end)[0])

        for error, legend_label in zip(errors, labels):
            ax[icolumns][iraws].plot(time_vector_exp, error, label=legend_label)
            #plt.plot(time_vector_exp, error, label=legend_label)

        variable = compare_im_exp.get_variable_names_exp(signal)
        title_variable = compare_im_exp.get_title_variable(list(variable.keys())[0])
        title = 'Normalized error for ' + title_variable

        ax[icolumns][iraws].set_title(title)
        ax[icolumns][iraws].legend()
        ax[icolumns][iraws].set_xlabel(r'time [s]')
        ax[icolumns][iraws].set_ylabel(r'$\sigma$ [-]')
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


def plot_all_traces(filename, signals, time_begin = None, time_end = None, username = None, correct_sign=False, signal_operations=None, show_fit = False):

    # Ideally want to use the other functions, but I cannot figure out how to use the axes that way...
    if not username: username=getpass.getuser()
    exp_signals, mod_signals = separate_signal_lists(signals)
    fig, ax, num_columns = create_subplots(signals)

    db, shot, run_exp, labels, runs_output = read_file_time_dep(filename, show_fit = show_fit)

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

    options_trace, options_profile = {'absolute_error': True}, {'average_absolute_error': True}
    time_error_dict = compare_im_runs.perform_time_trace_analysis(data_dict, **options_trace)
    profile_error_dict = compare_im_runs.perform_profile_analysis(data_dict, **options_profile)

    for isignal, signame in enumerate(signals):
        icolumns = int(isignal/num_columns)
        iraws = isignal % num_columns

        if signame in mod_signals:
            #value_when_true if condition else value_when_false
            title = titles[signame] if signame in titles else signame
            ylabel = ylabels[signame] if signame in ylabels else signame

            if signame+'.absolute_error.t' in time_error_dict:
                for run, label in zip(time_error_dict[signame+'.absolute_error'], labels):
                    ax[icolumns][iraws].plot(time_error_dict[signame+'.absolute_error.t'], time_error_dict[signame+'.absolute_error'][run].flatten(), label=label)
            elif signame+'.average_absolute_error.t' in profile_error_dict:
                for run, label in zip(profile_error_dict[signame+'.average_absolute_error'], labels):
                    ax[icolumns][iraws].plot(profile_error_dict[signame+'.average_absolute_error.t'], profile_error_dict[signame+'.average_absolute_error'][run].flatten(), label=label)
            else:
                print('signal ' + signame + ' not recognized, aborting')

            ax[icolumns][iraws].set_title(title)
            ax[icolumns][iraws].legend(loc='best')
            ax[icolumns][iraws].set_xlabel(r'time [s]')
            ax[icolumns][iraws].set_ylabel(r'$\sigma$ ' + ylabel)

        elif signame in exp_signals:

            all_exp_data = compare_im_exp.get_exp_data(db, shot, run_exp, time_begin, time_end, signame)
            exp_data, errorbar, time_vector_exp = compare_im_exp.get_exp_data_and_errorbar(all_exp_data, t_cxrs, list(all_exp_data.keys())[0])

            errors = []
            for run_output in runs_output:
                errors.append(get_exp_error_time(shot, run_exp, run_output, signame, time_begin = time_begin, time_end = time_end)[0])
                if show_fit:
                    errors.append(get_exp_error_time(shot, run_exp, run_output, signame, time_begin = time_begin, time_end = time_end, show_fit = show_fit)[0])

            for error, legend_label in zip(errors, labels):
                ax[icolumns][iraws].plot(time_vector_exp, error, label=legend_label)

            variable = compare_im_exp.get_variable_names_exp(signame)
            title_variable = compare_im_exp.get_title_variable(list(variable.keys())[0])
            title = 'Normalized error for ' + title_variable

            ax[icolumns][iraws].set_title(title)
            ax[icolumns][iraws].legend()
            ax[icolumns][iraws].set_xlabel(r'time [s]')
            ax[icolumns][iraws].set_ylabel(r'$\sigma$ [-]')
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
    db, shot, run_exp = lines[0].split(' ')
    shot, run_exp = int(shot), int(run_exp)

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
        if type(run) is int:
            ids = f'{username}/{db}/{shot}/{run}'
        elif type(run) is str:
            ids = f'/pfs/work/{username}/jetto/runs/{run}/imasdb/{db}/{shot}/2'
        else:
            print('run of type ' + str(type(run)) + ' is not supported. Aborting')

        idslist.append(ids)

        # Might be useful to also show the agreement of the fit.
        if show_fit and type(run) is str:
            ids = f'/pfs/work/{username}/jetto/runs/{run}/imasdb/{db}/{shot}/1'
            idslist.append(ids)

    return idslist


def plot_errors_traces(filename, signals, username = None, time_begin = None, time_end = None, correct_sign=False, steady_state=False, uniform=False, signal_operations=None, show_fit = False):

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

    options_trace, options_profile = {'absolute_error': True}, {'average_absolute_error': True}
    time_error_dict = compare_im_runs.perform_time_trace_analysis(data_dict, **options_trace)
    profile_error_dict = compare_im_runs.perform_profile_analysis(data_dict, **options_profile)

    for isignal, signame in enumerate(signals):
        icolumns = int(isignal/num_columns)
        iraws = isignal % num_columns

        title = titles[signame] if signame in titles else signame
        ylabel = ylabels[signame] if signame in ylabels else signame

        if signame+'.absolute_error.t' in time_error_dict:
            for run, label in zip(time_error_dict[signame+'.absolute_error'], labels):
                ax[icolumns][iraws].plot(time_error_dict[signame+'.absolute_error.t'], time_error_dict[signame+'.absolute_error'][run].flatten(), label=label)
        elif signame+'.average_absolute_error.t' in profile_error_dict:
            for run, label in zip(profile_error_dict[signame+'.average_absolute_error'], labels):
                ax[icolumns][iraws].plot(profile_error_dict[signame+'.average_absolute_error.t'], profile_error_dict[signame+'.average_absolute_error'][run].flatten(), label=label)
        else:
            print('signal ' + signame + ' not recognized, aborting')

        ax[icolumns][iraws].set_title(title)
        ax[icolumns][iraws].legend(loc='best')
        ax[icolumns][iraws].set_xlabel(r'time [s]')
        ax[icolumns][iraws].set_ylabel(r'$\sigma$ ' + ylabel)
        #fig.tight_layout()

    plt.show()

def plot_errors(filename, signal_list, time_begin = None, time_end = None, plot_type = 1):

    file_runs = open(filename, 'r')
    lines = file_runs.readlines()
    shot_list, run_input_list, run_output_list, saw_file_paths = [], [], [], []
    labels_plot = lines[0].split(' ')
    for line in lines[1:]:
        shot, run_input, *runs_output = line.split(' ')
        shot_list.append(int(shot))
        run_input_list.append(int(run_input))
        # Want to compare multiple series of runs to check which one is better
        # If the last element of the split is a path, that will be the path for the math file of the sawteeth analysis
        # Being the last value the string has a /n as last character. It is removed by [:-1]
        candidate_saw_path = Path(runs_output[-1][:-1])
        if not candidate_saw_path.is_file():
            num_run_series = len(runs_output)
            for run_output in runs_output:
                run_output_list.append(int(run_output))
            # Maybe not a nice way to do it but keeps the structure later
            saw_file_paths.append(None)
        else:
            num_run_series = len(runs_output) - 1
            for run_output in runs_output[:-1]:
                run_output_list.append(int(run_output))
            saw_file_paths.append(runs_output[-1][:-1])

    run_output_list = np.asarray(run_output_list).reshape(len(lines)-1, num_run_series)

    # Pre deciding the structure of the plots with the various possibilities of len(signal_list)
    fig, ax, num_columns = create_subplots(signal_list)

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
                        errors.append(get_exp_error(shot, run_input, run_output, signal, time_begin = time_begin, time_end = time_end)[0])
                    elif signal == 'sawteeth':
                         errors.append(get_sawteeth_error(shot, run_output, saw_file_path, time_begin = time_begin, time_end = time_end))
                    else:
                         errors.append(get_error(shot, run_input, run_output, signal, time_begin = time_begin, time_end = time_end)[0])

                rects = ax[icolumns][iraws].bar(x - width + width/num_run_series + run_serie*width, errors, width, label=labels_plot[run_serie])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            if signal in ['te', 'ne', 'ti', 'ni', 'sawteeth']:
                ax[icolumns][iraws].set_ylabel('Error [-]')
            else:
                ax[icolumns][iraws].set_ylabel('Distance')
            ax[icolumns][iraws].set_title(signal)
            ax[icolumns][iraws].set_xticks(x, labels)
            ax[icolumns][iraws].legend()

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
                    if signal not in exp_signal_list:
                        errors.append(get_error(shot_list[shot], run_input_list[shot], run_output, signal, time_begin = time_begin, time_end = time_end)[0])
                    else:
                        errors.append(get_exp_error(shot_list[shot], run_input_list[shot], run_output, signal, time_begin = time_begin, time_end = time_end)[0])

                rects = ax[icolumns][iraws].bar(x - width + width/len(shot_list) + shot*width, errors, width, label=shot_list[shot])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            if signal in ['te', 'ne', 'ti', 'ni', 'sawteeth']:
                ax[icolumns][iraws].set_ylabel('Error [-]')
            else:
                ax[icolumns][iraws].set_ylabel('Distance')
            ax[icolumns][iraws].set_title(signal)
            ax[icolumns][iraws].set_xticks(x, labels_plot)
            ax[icolumns][iraws].legend()

            fig.tight_layout()

        plt.show()


if __name__ == "__main__":
    #plot_errors('/afs/eufus.eu/user/g/g2mmarin/public/scripts/runs_list_show.txt', ['summary.global_quantities.v_loop.value', 'ne', 'summary.global_quantities.li.value', 'te'], time_begin = 0.09, plot_type = 1)
    #plot_errors_time('runlist_time_errors.txt', ['ne', 'te'], time_begin = 0.04, time_end = 0.33)
    #plot_errors_time('runlist_test.txt', ['ne', 'te'], time_begin = 0.04, time_end = 0.33)
    #plot_errors_traces('runlist_test.txt', ['summary.global_quantities.v_loop.value', 'summary.global_quantities.ip.value'], correct_sign=True, time_begin = 0.1, time_end = 0.3)
    #plot_errors_traces('runlist_paper1.txt', ['vloop', 'li3', 'core_profiles.profiles_1d[].electrons.temperature'], correct_sign=True, time_begin = 0.04, time_end = 0.3, show_fit = True)
    #print('for scripting directly')
    #plot_all_traces('runlist_test.txt', ['ne', 'vloop', 'li3', 'core_profiles.profiles_1d[].electrons.temperature'], correct_sign=True, time_begin = 0.1, time_end = 0.3)

    #plot_all_traces('runlist_paper1.txt', ['ne', 'vloop', 'li3', 'core_profiles.profiles_1d[].electrons.temperature'], correct_sign=True, time_begin = 0.1, time_end = 0.3, show_fit = False)
    #plot_all_traces('runlist_paper1.txt', ['ne', 'te', 'ti', 'ni'], correct_sign=True, time_begin = 0.04, time_end = 0.33, show_fit = True)

    #plot_all_traces('runlist_paper1.txt', ['ne'], correct_sign=True, time_begin = 0.04, time_end = 0.33, show_fit = True)
    #plot_all_traces('runlist_paper1.txt', ['te'], correct_sign=True, time_begin = 0.04, time_end = 0.33, show_fit = True)
    #plot_all_traces('runlist_paper1.txt', ['ti'], correct_sign=True, time_begin = 0.04, time_end = 0.33, show_fit = True)
    plot_all_traces('runlist_paper1.txt', ['ni'], correct_sign=True, time_begin = 0.04, time_end = 0.33, show_fit = True)







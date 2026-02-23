#!/usr/bin/env python
import os
import sys
import re
import copy
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate   import cumtrapz

os.environ['MPLBACKEND'] = 'Qt5Agg'

import matplotlib
matplotlib.use('Qt5Agg', force=True)
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display
import argparse
import getpass

import imas

##### CONFIGURATION LISTS
# Feel free to add more options here as they are needed.

allowed_ids_list = [
    'summary',
    'equilibrium',
    'core_profiles',
    'core_sources',
    'core_transport'
]

keys_list = {
    'time_trace': [],
    'profiles_1d': [],
    'profiles_2d': []
}

keys_list['profiles_1d'] = [
    'core_profiles.profiles_1d[].q', 
    'core_profiles.profiles_1d[].electrons.density_thermal',
    'core_profiles.profiles_1d[].electrons.density',
    'core_profiles.profiles_1d[].electrons.density_fit.measured',
    'core_profiles.profiles_1d[].electrons.density_fit.measured_error_upper',
    'core_profiles.profiles_1d[].electrons.temperature',
    'core_profiles.profiles_1d[].electrons.temperature_fit.measured',
    'core_profiles.profiles_1d[].electrons.temperature_fit.measured_error_upper',
    'core_profiles.profiles_1d[].t_i_average',
    'core_profiles.profiles_1d[].t_i_average_fit.measured',
    'core_profiles.profiles_1d[].t_i_average_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[0].temperature',
    'core_profiles.profiles_1d[].ion[0].temperature_fit.measured',
    'core_profiles.profiles_1d[].ion[0].temperature_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[0].density',
    'core_profiles.profiles_1d[].ion[0].density_fit.measured',
    'core_profiles.profiles_1d[].ion[0].density_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[1].temperature',
    'core_profiles.profiles_1d[].ion[1].temperature_fit.measured',
    'core_profiles.profiles_1d[].ion[1].temperature_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[1].density',
    'core_profiles.profiles_1d[].ion[1].density_fit.measured',
    'core_profiles.profiles_1d[].ion[1].density_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[1].state[0].density_thermal',
    'core_profiles.profiles_1d[].ion[1].state[1].density_thermal',
    'core_profiles.profiles_1d[].ion[1].state[2].density_thermal',
    'core_profiles.profiles_1d[].ion[1].state[3].density_thermal',
    'core_profiles.profiles_1d[].ion[1].state[4].density_thermal',
    'core_profiles.profiles_1d[].ion[1].state[5].density_thermal',
    'core_profiles.profiles_1d[].ion[2].temperature',
    'core_profiles.profiles_1d[].ion[2].temperature_fit.measured',
    'core_profiles.profiles_1d[].ion[2].temperature_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[2].density',
    'core_profiles.profiles_1d[].ion[2].density_fit.measured',
    'core_profiles.profiles_1d[].ion[2].density_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[3].temperature',
    'core_profiles.profiles_1d[].ion[3].temperature_fit.measured',
    'core_profiles.profiles_1d[].ion[3].temperature_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[3].density',
    'core_profiles.profiles_1d[].ion[3].density_fit.measured',
    'core_profiles.profiles_1d[].ion[3].density_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[4].temperature',
    'core_profiles.profiles_1d[].ion[4].temperature_fit.measured',
    'core_profiles.profiles_1d[].ion[4].temperature_fit.measured_error_upper',
    'core_profiles.profiles_1d[].ion[4].density',
    'core_profiles.profiles_1d[].ion[4].density_fit.measured',
    'core_profiles.profiles_1d[].ion[4].density_fit.measured_error_upper',
    'core_profiles.profiles_1d[].t_i_average',
    'core_profiles.profiles_1d[].rotation_frequency_tor_sonic',
    'core_profiles.profiles_1d[].zeff',
    'core_profiles.profiles_1d[].grid.rho_tor_norm',
    'core_profiles.profiles_1d[].grid.volume',
    'equilibrium.time_slice[].profiles_1d.psi',
    'equilibrium.time_slice[].profiles_1d.f', 
    'equilibrium.time_slice[].profiles_1d.q', 
    'equilibrium.time_slice[].profiles_1d.pressure', 
    'equilibrium.time_slice[].profiles_1d.rho_tor_norm',
    'equilibrium.time_slice[].profiles_1d.volume',
    'equilibrium.time_slice[].boundary.outline.r', 
    'equilibrium.time_slice[].boundary.outline.z', 
    'equilibrium.time_slice[].profiles_2d[].grid.dim1', 
    'equilibrium.time_slice[].profiles_2d[].grid.dim2',
    'core_sources.source[].profiles_1d[].electrons.energy', 
    'core_sources.source[].profiles_1d[].total_ion_energy', 
    'core_sources.source[].profiles_1d[].j_parallel', 
    'core_sources.source[].profiles_1d[].momentum_tor', 
    'core_sources.source[].profiles_1d[].ion[].particles', 
    'core_sources.source[].profiles_1d[].grid.rho_tor_norm',
]

keys_list['time_trace'] = [
    'core_profiles.global_quantities.ip', 
    'core_profiles.global_quantities.v_loop', 
    'core_profiles.global_quantities.li_3', 
    'core_profiles.global_quantities.energy_diamagnetic',
    'summary.global_quantities.ip.value', 
    'summary.heating_current_drive.power_nbi.value', 
    'summary.heating_current_drive.power_ic.value', 
    'summary.heating_current_drive.power_ec.value', 
    'summary.heating_current_drive.power_lh.value', 
    'summary.stationary_phase_flag.value',
    'summary.global_quantities.v_loop.value', 
    'summary.global_quantities.li.value', 
    'summary.global_quantities.li_mhd.value', 
    'summary.global_quantities.energy_diamagnetic.value', 
    'summary.global_quantities.energy_mhd.value', 
    'summary.global_quantities.energy_thermal.value', 
    'summary.global_quantities.beta_pol.value', 
    'summary.global_quantities.beta_pol_mhd.value', 
    'summary.global_quantities.beta_tor_norm.value', 
    'summary.global_quantities.beta_tor.value', 
    'summary.global_quantities.power_radiated.value', 
    'summary.fusion.neutron_fluxes.total.value',
    'summary.fusion.neutron_fluxes.thermal.value',
    'summary.fusion.neutron_rates.total.value',
    'summary.fusion.neutron_rates.thermal.value',
    'summary.fusion.neutron_rates.dd.total.value',
    'summary.fusion.neutron_rates.dd.thermal.value',
    'summary.fusion.neutron_rates.dt.total.value',
    'summary.fusion.neutron_rates.dt.thermal.value',
    'equilibrium.time_slice[].global_quantities.ip', 
    'equilibrium.time_slice[].global_quantities.li_3', 
    'equilibrium.time_slice[].global_quantities.beta_pol', 
    'equilibrium.time_slice[].global_quantities.beta_tor',
    'equilibrium.time_slice[].global_quantities.psi_axis',
    'equilibrium.time_slice[].global_quantities.psi_boundary',
    'equilibrium.time_slice[].global_quantities.q_axis',
    'equilibrium.time_slice[].global_quantities.q_95',
    'equilibrium.time_slice[].global_quantities.magnetic_axis.r', 
    'equilibrium.time_slice[].global_quantities.magnetic_axis.z'
]

keys_list['profiles_2d'] = [
    'equilibrium.time_slice[].profiles_2d[].psi'
]

keys_list['errors'] = {
    'time_trace': [],
    'profiles_1d': [],
    'profiles_2d': []
}

keys_list['errors']['time_trace'] = [
    'absolute_error'
]

keys_list['errors']['profiles_1d'] = [
    'average_absolute_error'
]

operations = [
    '*2',
    '/2',
    '+',
    '-',
    '*',
    '/'
]

#y_limit_bottom, y_limit_top = None, None

##### FUNCTIONS

def expand_error_keys(category=None):
    error_keys = []
    if category in keys_list and category in keys_list['errors']:
        for var in keys_list[category]:
            for err in keys_list['errors'][category]:
                error_keys.append(var+'.'+key)
    return error_keys

choices_error = ["absolute", "relative", "difference", "squared", "absolute volume", "relative volume", "difference volume", "squared volume"]


def input():

    parser = argparse.ArgumentParser(
        description=
"""Compare validation metrics from HFPS input / output IDSs. Preliminary version, adapted from scripts from D. Yadykin and M. Marin.\n
---
Examples:\n
python compare_im_runs.py --ids 'g2aho/jet/94875/1' 'g2aho/jet/94875/102' --time_begin 48 --time_end 49 --steady_state -sig 'core_profiles.profiles_1d[].q' 'summary.global_quantities.li.value'\n
---
""", 
    epilog="", 
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--backend",  "-b",              type=str,   default="mdsplus", choices=["mdsplus", "hdf5"],       help="Backend with which to access data")
    parser.add_argument("--ids",      "-i",   nargs='+', type=str,   default=None,                                         help="IDS identifiers in which data is stored")
    parser.add_argument("--version",  "-v",              type=str,   default="3",                                          help="UAL version")
    parser.add_argument("--time_begin",                  type=float, default=None,                                         help="Slice shot file beginning at time (s)")
    parser.add_argument("--time_end",                    type=float, default=None,                                         help="Slice shot file ending at time (s)")
    parser.add_argument("--time_out",         nargs='*', type=float, default=None,                                         help="Slice output interpolated to times (s), automatically toggles uniform")
    parser.add_argument("--signal",   "-sig", nargs='+', type=str,   default=None,                                         help="Full IDS signal names to be compared")
#    parser.add_argument("--source",           nargs='*', type=str,   default=['total'],                                    help="sourceid to be plotted(nbi, ec,etc as given in dd description), make sence if core_source is given as target ids, default is total")
#    parser.add_argument("--transport",        nargs='*', type=str,   default=['transport_solver'],                         help="transpid to be plotted(neoclassical, anomalous, ets, cherck dd for more entires), make sence if core_transport is given as target ids, default is transport_solver")
    parser.add_argument("--steady_state",                            default=False, action='store_true',                   help="Flag to identify that the input is a single point")
    parser.add_argument("--save_plot",                               default=False, action='store_true',                   help="Toggle saving of plot into default file names")
    parser.add_argument("--uniform",                                 default=False, action='store_true',                   help="Toggle interpolation to uniform time and radial basis, uses first run as basis unless steady state flag is on")
#    parser.add_argument("--analyze_traces",   nargs='*', type=str,   default=None, choices=["absolute_error"],             help="Define which analyses to perform after time trace comparison plots")
#    parser.add_argument("--analyze_profiles", nargs='*', type=str,   default=None, choices=["average_absolute_error"],     help="Define which analyses to perform after profile comparison plots")
    parser.add_argument("--analyze",                                 default=False, action='store_true',                   help="Toggle extra analysis routines, automatically toggles uniform")
    parser.add_argument("--error_type",                   type=str,  default="absolute", choices=choices_error,            help="Switches between absolute and relative error")
    parser.add_argument("--correct_sign",                            default=None, action='store_true',                    help="Allows to change the sign of the output if it is not identical to reference run")
    parser.add_argument("--function", "-func", nargs='*', type=str,  default=None,                                         help="Specify functions of multiple variables")
    parser.add_argument("--calc_only",                               default=False, action='store_true',                   help="Toggle off all plotting")
    parser.add_argument("--keep_op_signals",                         default=False, action='store_true',                   help="Keeps the signals used for the operation")
    parser.add_argument("--integrate",                               default=False, action='store_true',                   help="Integrates the requested signal")
    parser.add_argument("--integrate_errors",                        default=False, action='store_true',                   help="Integrates the errors")
    parser.add_argument("--use_regular_rho_grid",                    default=False, action='store_true',                   help="Uses a regular grid in rho instead of the grid of the first run")
    parser.add_argument("--labels",            nargs='+', type=str,  default=None,                                         help="Labels to be used for the runs")
    parser.add_argument("--y_limits",          nargs='+', type=float,default=[None, None],                                 help="Fix limits for y in the plots")
    parser.add_argument("--verbose",                                 default=False, action='store_true',                   help="Toggle the generation of plots at every timestep")
    args=parser.parse_args()

    return args


def getShotFile(ids_name, shot, runid, user, database, backend=None):
    if not backend: backend = get_backend(database, shot, runid, username=user)
    print('to be opened', user, database, shot, runid, backend)
    data = None
    ids = imas.DBEntry(backend, database, shot, runid, user_name=user)
    ids.open()
    if ids_name in allowed_ids_list:
#        data = ids.get_slice(ids_name, time, imas.imasdef.CLOSEST_SAMPLE, occurrence=0)
        data = ids.get(ids_name)
    else:
        raise TypeError('IDS given is not implemented yet')
    ids.close()

    return data


def get_backend(db, shot, run, username=None):

    if not username: username = getpass.getuser()

    imas_backend = imas.imasdef.HDF5_BACKEND
    data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()
    if op[0]<0:
        imas_backend = imas.imasdef.MDSPLUS_BACKEND

    data_entry.close()

    data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)
    op = data_entry.open()
    if op[0]<0:
        print('Input does not exist. Aborting generation')

    data_entry.close()

    return imas_backend


def extend_list(inlist, nmax, default=None):
    outlist = inlist
    while len(outlist) < nmax:
        value = inlist[-1] if len(inlist) > 0 else default
        outlist.append(value)
    return outlist

def get_sourceid(ids, sid):
    nsour = len(ids.source)
    nosid = True
    for isour in range(nsour):
        if ids.source[isour].identifier.name == sid:
            sourceid = isour
            nosid = False
            break
    if nosid:
        raise IOError('no sid with name %s found, check ids used' % (sid))
    return sourceid

def get_transpid(ids, tid):
    nmod = len(ids.model)
    notid = True
    for imod in range(nmod):
        if ids.model[imod].identifier.name == tid:
            transpid = imod
            notid = False
            break
    if notid:
        raise IOError('no tid with name %s found, check ids used' % (tid))
    return transpid

def fit_and_substitute(x_old, x_new, y_old):
    ifunc = interp1d(x_old, y_old, fill_value='extrapolate', bounds_error=False)
    y_new = np.array(ifunc(x_new)).flatten()
    y_new[y_new > 1.0e25] = 0.0    # This is just in case
    return y_new

def get_label_variable(variable, show_units=True):

    """
    Get the label variable string for a given variable.
    """

    ylabel = None

    if 'summary.global_quantities.li.value' in variable:
        ylabel = [r'$l_{i3}$ ','[-]']
    elif 'equilibrium.time_slice[].global_quantities.li_3' in variable:
        ylabel = [r'$l_{i3}$ ','[-]']
    elif 'summary.global_quantities.energy_diamagnetic.value' in variable:
        ylabel = [r'$W_{dia}$ ','[J]']
    elif 'summary.global_quantities.energy_thermal.value' in variable:
        ylabel = [r'$W_{thermal}$ ','[J]']
    elif 'summary.global_quantities.v_loop.value' in variable:
        ylabel = [r'$V_{loop}$ ','[V]']

    elif 'core_profiles.profiles_1d[].electrons.density' in variable:
        ylabel = [r'$n_{e}$ ','[$m^{-3}$]']
    elif 'core_profiles.profiles_1d[].q' in variable:
        ylabel = [r'$q$ ','[-]']
    elif 'core_profiles.profiles_1d[].electrons.temperature' in variable:
        ylabel = [r'$T_{e}$ ','[eV]']
    elif 'core_profiles.profiles_1d[].ion[0].temperature' in variable:
        ylabel = [r'$T_{i}$ ','[eV]']
    elif 'core_profiles.profiles_1d[].t_i_average' in variable:
        ylabel = [r'$T_{i}$ ','[eV]']
    elif 'core_profiles.profiles_1d[].ion[1].state[5]' in variable:
        ylabel = [r'$n_{C}^{+6}$ ','[$m^{-3}$]']

    if ylabel:
        if 'absolute_error' in variable:
            ylabel[0] = ylabel[0] + 'Absolute error '

        elif 'relative_error' in variable:
            ylabel[0] = ylabel[0] + 'Relative error '
            ylabel[1] = '[-]'

        elif 'relative_error_volume' in variable:
            ylabel[0] = ylabel[0] + 'Relative error '
            ylabel[1] = '[-]'

        if show_units:
            return ylabel[0] + ylabel[1]
        else:
            return ylabel[0]
    else:
        return variable


fontsize_labels = 17
fontsize_legend = 12
fontsize_ticks = 12
fontsize_title = 17
legend_location_gifs = 'upper right' #Keeping the legend in the same place for the gifs


def get_onesig(ids, signame, time_begin, time_end=None, sid=None, tid=None):
    data_dict = {}
    sigcomp = signame.split('.')
    # IDS name must be the first part of the signal name
    idsname = sigcomp[0]
    if idsname not in allowed_ids_list:
        raise IOError('IDS %s not supported by this tool.' % (idsname))

    # Time vector discovery section here
    tstring = 'ids.time'
    tvec = None
    try:
       tvec = eval(tstring)
       if not isinstance(tvec, np.ndarray):
           tvec = np.array([tvec]).flatten()
    except:
       raise IOError('Time vector not present in IDS')

    # Define indices within the time vector for the user-defined begin and end times
    tb_ind = np.abs(tvec - time_begin).argmin(0)
    te_ind = tb_ind
    if time_end is not None:
         te_ind = np.abs(tvec - time_end).argmin(0)
    if te_ind == tb_ind:
        te_ind = tb_ind + 1

    for tt in np.arange(tb_ind, te_ind):
        xstring = 'None'
        ystring = 'None'
        datatype = 'None'
        sid_ind = -1
        tid_ind = -1
        # Auto-detect 0D, 1D, 2D signals based purely on the name, category lists at top of file
        for key in keys_list:
            if signame in keys_list[key]:
                for ii in range(len(sigcomp)):
                    tstr = sigcomp[ii]
                    mm = re.match(r'^(.+)\[\]$', tstr)
                    # Signal names passed in have empty [] as placeholder for indices, eval function needs a defined index
                    # This section fills in the variable name which holds the index value to be used in eval
                    if mm and mm.group(1):
                        if mm.group(1) == 'time_slice':
                            tstr = 'time_slice[tt]'
                        elif mm.group(1) == 'profiles_1d':
                            tstr = 'profiles_1d[tt]'
                        elif mm.group(1) == 'source':
                            sid_ind = get_sourceid(ids, sid)
#                            print('sourceid', sid, sid_ind)
                            tstr = 'source[sid_ind]'
                        elif mm.group(1) == 'ion':
                            tstr = 'ion[0]'
                        elif mm.group(1) == 'profiles_2d':
                            tstr = 'profiles_2d[0]'
                        else:
                            tid_ind = get_transpid(ids,tid)
#                            print('transpid', tid, tid_ind)
                    sigcomp[ii] = tstr
                ystring = 'ids.' + '.'.join(sigcomp[1:])
                datatype = key
                if datatype == 'time_trace':
                    # Equilibrium IDS is strange
                    if idsname not in ['equilibrium']:
                        ystring += '[tt]'
        # Determine corresponding vectors related to signal data (could be time or radius depending on quantity)
        if idsname == 'summary':
            if datatype == 'time_trace':
                xstring = 'ids.time[tt]'
        if idsname == 'equilibrium':
            if datatype == 'profiles_1d':
                xstring = 'ids.time_slice[tt].profiles_1d.rho_tor_norm'
            if datatype == 'time_trace':
                xstring = 'ids.time[tt]'
            if datatype == 'profiles_2d':
                raise TypeError("No.")
        if idsname == 'core_profiles':
            if datatype == 'profiles_1d':
                xstring = 'ids.profiles_1d[tt].grid.rho_tor_norm'
            if datatype == 'time_trace':
                xstring = 'ids.time[tt]'
        if idsname == 'core_sources':
            if datatype == 'profiles_1d':
                xstring = 'ids.source[sid_ind].profiles_1d[tt].grid.rho_tor_norm'
            if datatype == 'time_trace':
                xstring = 'ids.time[tt]'
        if idsname == 'core_transport':
            if datatype == 'profiles_1d':
                xstring='ids.model[tid_ind].profiles_1d[tt].grid_d.rho_tor_norm'
        # Defining a different x vector for the experimental data
        if idsname == 'core_profiles' and signame == 'core_profiles.profiles_1d[].electrons.temperature_fit.measured':
            xstring = 'ids.profiles_1d[tt].electrons.temperature_fit.rho_tor_norm'
        if idsname == 'core_profiles' and signame == 'core_profiles.profiles_1d[].electrons.temperature_fit.measured_error_upper':
            xstring = 'ids.profiles_1d[tt].electrons.temperature_fit.rho_tor_norm'

        if idsname == 'core_profiles' and signame == 'core_profiles.profiles_1d[].electrons.density_fit.measured':
            xstring = 'ids.profiles_1d[tt].electrons.density_fit.rho_tor_norm'
        if idsname == 'core_profiles' and signame == 'core_profiles.profiles_1d[].electrons.density_fit.measured_error_upper':
            xstring = 'ids.profiles_1d[tt].electrons.density_fit.rho_tor_norm'

        if idsname == 'core_profiles' and signame == 'core_profiles.profiles_1d[].t_i_average_fit.measured':
            xstring = 'ids.profiles_1d[tt].t_i_average_fit.rho_tor_norm'
        if idsname == 'core_profiles' and signame == 'core_profiles.profiles_1d[].t_i_average_fit.measured_error_upper':
            xstring = 'ids.profiles_1d[tt].t_i_average_fit.rho_tor_norm'

        if idsname == 'core_profiles' and signame == 'core_profiles.profiles_1d[].ion[1].density_fit.measured':
            xstring = 'ids.profiles_1d[tt].ion[1].density_fit.rho_tor_norm'
        if idsname == 'core_profiles' and signame == 'core_profiles.profiles_1d[].ion[1].density_fit.measured_error_upper':
            xstring = 'ids.profiles_1d[tt].ion[1].density_fit.rho_tor_norm'

        # Could be generalized to all the states
        if idsname == 'core_profiles' and signame == 'core_profiles.profiles_1d[].ion[1].state[5].density_thermal':
            xstring = 'ids.profiles_1d[tt].grid.rho_tor_norm'

        if xstring == 'None':
            raise IOError('Signal %s not present in IDS.' % (signame))

        # Define x vector (could be time or radius)
        if xstring == 'ids.time[tt]':
            xvec = np.array([tvec[tt]]).flatten()
        else:
            try:
                xvec=eval(xstring)
                if not isinstance(xvec, np.ndarray):
                    xvec = np.array([xvec]).flatten()
            except:
                raise IOError('Radial vector %s not present in IDS.' % (xstring))
        # Define y vector(value of signal)

        try:
            yvec=eval(ystring)
            if not isinstance(yvec, np.ndarray):
                yvec = np.array([yvec]).flatten()
        except:
            raise IOError('Value vector %s not present in IDS.' % (ystring))

        # Check dimension consistency
        if len(yvec) != len(xvec):
            print('dimensions of x,y are not consistent for signal ', ystring,' set y to the same size as x and fill with zeros')
            yvec=np.zeros(len(xvec))

        data_dict[tvec[tt]] = {"x": xvec, "y": yvec}

    return data_dict

def get_onedict(sigvec, user, db, shot, runid, time_begin, time_end=None, sid=None, tid=None, interpolate=False, backend = 'hdf5'):

    out_data_dict = {}
    ids_dict = {}

    # Split signals based on which IDS they come from, allows efficient IDS reading
    for sig in sigvec:
        sigcomp = sig.split('.')
        if sigcomp[0] not in ids_dict:
            ids_dict[sigcomp[0]] = [sig]
        else:
            ids_dict[sigcomp[0]].append(sig)

    t_fields = []
    xt_fields = []
    # Loop over IDSs, extracted all requested signals from each
    for idsname, siglist in ids_dict.items():
        ids = getShotFile(idsname, shot, runid, user, db, backend = backend)
        for signame in siglist:
            raw_data_dict = get_onesig(ids, signame, time_begin, time_end, sid, tid)
            mask_time_keys = []
            for time_key in raw_data_dict.keys():
                if float(time_key) < time_begin:
                    mask_time_keys.append(time_key)
            for time_key in mask_time_keys:
                del raw_data_dict[time_key]
            ytable = None
            new_x = None
            new_t = np.array([])
            for key, val in raw_data_dict.items():
                if val["x"].size > 1:
                    # Allow interpolation to standardize radial vectors within an IDS signal (not usually necessary but could be needed if radial vector changes in time)
                    # Default call of function does not interpolate
                    if interpolate:
                        if new_x is None:
                            new_x = val["x"]
                        new_y = fit_and_substitute(val["x"], new_x, val["y"])
                        ytable = np.vstack((ytable, new_y)) if ytable is not None else np.atleast_2d(new_y)
                    else:
                        if ytable is None:
                            ytable = []
                        ytable.append(val)
                        new_x = val["x"]
                else:
                    ytable = np.hstack((ytable, val["y"])) if ytable is not None else np.array([val["y"]]).flatten()
                # Time of time slice stored as key of raw_data_dict, extract and stack into an actual time vector
                new_t = np.hstack((new_t, key))
            # Store data into container
            out_data_dict[signame] = ytable
            out_data_dict[signame+".t"] = new_t
            if new_x is None:
                t_fields.append(signame)
            else:
                out_data_dict[signame+".x"] = new_x
                xt_fields.append(signame)

    if out_data_dict:
        out_data_dict["time_signals"] = t_fields
        out_data_dict["profile_signals"] = xt_fields

    return out_data_dict

def plot_traces(plot_data, plot_vars=None, single_time_reference=False, labels=None):
    signal_list = plot_data["time_signals"] if "time_signals" in plot_data else keys_list['time_trace']
    if isinstance(plot_vars, list):
        signal_list = plot_vars
    for signame in signal_list:
        pdata = {}
        t_basis = None
        first_run = None
        for run in plot_data:
            if signame in plot_data[run]:
                if run != 'time_signals' and run != 'profile_signals':
                    pdata[run] = {"time": plot_data[run][signame+".t"], "data": plot_data[run][signame]}
                if first_run is not None and t_basis is None:
                    t_basis = pdata[run]["time"]
                if first_run is None:
                    first_run = run
                    if not single_time_reference:
                        t_basis = pdata[run]["time"]
        if pdata:
            print("Plotting %s" % (signame))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for run in pdata:
                xdata = pdata[run]["time"]
                ydata = pdata[run]["data"]
                linestyle = '-'
                linecolor = None
                if run == first_run and single_time_reference:
                    xdata = np.array([np.nanmin(t_basis), np.nanmax(t_basis)])
                    ydata = np.full(xdata.shape, pdata[run]["data"][0])
                    linestyle = '--'
                    linecolor = 'k'

                run_label = create_run_label(pdata, run, first_run=first_run, labels=labels)
                ax.plot(xdata, ydata, label=run_label, c=linecolor, ls=linestyle)
            ax.set_xlabel("time [s]", fontsize = fontsize_labels)
            ax.set_ylabel(get_label_variable(signame), fontsize = fontsize_labels)
            ax.legend(loc='best', fontsize = fontsize_legend)
#            fig.savefig(signame+".png", bbox_inches="tight")
            plt.show()
            plt.close(fig)

def create_run_label(pdata, run, first_run = False, labels=None):

    run_label=None

    if labels:
        for label in labels:
            if isolate_run_name(run) in label:
                run_label = label.replace(isolate_run_name(run), '')

    if first_run == run:
        run_label = 'Experimental measurement'

    if not run_label:
        if len(pdata) == 2 and (first_run != run):
            run_label = 'Integrated modelling prediction'

        if ':' in list(pdata.keys())[0]:
            run_label = isolate_run_name(run)

        if len(pdata) != 2:
            run_label = isolate_run_name(run)

    return run_label

def isolate_run_name(run):

    run_label = run
    pattern = r'(run.*?)'
    # Search for the pattern in the input string
    run_segments, run_contains = run.split('/'), []

    for run_segment in run_segments:
        run_with_runs = re.findall(pattern, run_segment)
        if run_with_runs:
            run_contains.append(run_segment)

    #print(run_contains)

    if len(run_contains) == 4:
        run_label = run_contains[-3]
    elif len(run_contains) == 0 or len(run_contains) == 1:
        run_label = run
    else:
        run_label = run_contains[-1]

    return run_label


def plot_interpolated_traces(interpolated_data, plot_vars=None, labels=None, y_limits=[None, None]):
    signal_list = interpolated_data["time_signals"] if "time_signals" in interpolated_data else keys_list['time_trace']
    if isinstance(plot_vars, list):
        signal_list = plot_vars
    for signame in signal_list:
        if signame in interpolated_data:
            print("Plotting %s" % (signame))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if 'error' in signame:
                #Chatgpt advises against this, but I do not want to set a custom cycler
                ax._get_lines.get_next_color()
                #next(ax._get_lines.prop_cycler)

            for run in interpolated_data[signame]:
                #This is used to set the run labels. Should work most of the times with the new jintrac version
                first_run = None
                if 'run' not in run: first_run = run
                run_label = create_run_label(interpolated_data[signame], run, first_run=first_run, labels=labels)
                ax.plot(interpolated_data[signame+".t"], interpolated_data[signame][run].flatten(), label=run_label)

            ax.set_xlabel("time [s]", fontsize = fontsize_labels)
            ax.set_ylabel(get_label_variable(signame), fontsize = fontsize_labels)
            if y_limits[0] is not None:
                ax.set_ylim(bottom = y_limits[0])
            if y_limits[1] is not None:
                ax.set_ylim(top = y_limits[1])

            ax.legend(loc='best', fontsize = fontsize_legend)
#            fig.savefig(signame+".png", bbox_inches="tight")
            plt.show()
            plt.close(fig)

def plot_gif_profiles(plot_data, plot_vars=None, single_time_reference=False, labels = None):
    signal_list = plot_data["profile_signals"] if "profile_signals" in plot_data else keys_list['profiles_1d']
    if isinstance(plot_vars, list):
        signal_list = plot_vars
    for signame in signal_list:
        pdata = {}
        first_run = None
        tvec = None
        for run in plot_data:
            if run != 'time_signals' and run != 'profile_signals':
                if signame in plot_data[run]:
                    pdata[run] = []
                    if first_run is None:
                        first_run = run
                        if not single_time_reference:
                            for tidx in range(len(plot_data[run][signame])):
                                pdata[run].append({"time": plot_data[run][signame+".t"][tidx], "rho": plot_data[run][signame][tidx]["x"], "data": plot_data[run][signame][tidx]["y"]})
                                tvec = np.hstack((tvec, plot_data[run][signame+".t"][tidx])) if tvec is not None else np.array([plot_data[run][signame+".t"][tidx]])
                    else:
                        tvec_new = np.array([])
                        tvec_final = np.array([])
                        tidxvec = []
                        for tidx in range(len(plot_data[run][signame])):
                            tvec_new = np.hstack((tvec_new, plot_data[run][signame+".t"][tidx]))
                        if tvec is None:
                            tvec = tvec_new.copy()
                        for tidx_orig in range(len(tvec)):
                            tidx = np.abs(tvec[tidx_orig] - tvec_new).argmin(0)
                            pdata[run].append({"time": plot_data[run][signame+".t"][tidx], "rho": plot_data[run][signame][tidx]["x"], "data": plot_data[run][signame][tidx]["y"]})
                            tvec_final = np.hstack((tvec_final, plot_data[run][signame+".t"][tidx])) if tvec_final is not None else np.array([plot_data[run][signame+".t"][tidx]])
                else:
                    tvec_new = np.array([])
                    tvec_final = np.array([])
                    tidxvec = []
                    for tidx in range(len(plot_data[run][signame])):
                        tvec_new = np.hstack((tvec_new, plot_data[run][signame+".t"][tidx]))
                    if tvec is None:
                        tvec = tvec_new.copy()
                    for tidx_orig in range(len(tvec)):
                        tidx = np.abs(tvec[tidx_orig] - tvec_new).argmin(0)
                        pdata[run].append({"time": plot_data[run][signame+".t"][tidx], "rho": plot_data[run][signame][tidx]["x"], "data": plot_data[run][signame][tidx]["y"]})
                        tvec_final = np.hstack((tvec_final, plot_data[run][signame+".t"][tidx]))
        if pdata and single_time_reference:
            if tvec is None:
                for tidx in range(len(plot_data[first_run][signame])):
                    tvec = np.hstack((tvec, plot_data[first_run][signame+".t"][tidx])) if tvec is not None else np.array([plot_data[first_run][signame+".t"][tidx]])
            for tidx in range(len(tvec)):
                pdata[first_run].append({"time": tvec[tidx], "rho": plot_data[first_run][signame][0]["x"], "data": plot_data[first_run][signame][0]["y"]})

        if pdata:
            print("Plotting %s" % (signame))
            Figure = plt.figure()

            # creating a plot
            # lines_plotted = plt.plot([])

            ax = Figure.add_subplot(1, 1, 1)
            ax.set_xlabel(r'$\rho_{tor,norm}$', fontsize = fontsize_labels)
            ax.set_ylabel(get_label_variable(signame), fontsize = fontsize_labels)
            #ax.set_xlabel(r'$\rho$ [-]')
            #ax.set_ylabel(ylabel + ' ' + units)
            ymin = None
            ymax = None
            plot_list = {}
            for run in pdata:
                linestyle = '-'
                linecolor = None
                if run == first_run and single_time_reference:
                    linestyle = '--'
                    linecolor = 'k'
                pp = ax.plot(pdata[run][0]["rho"], pdata[run][0]["data"], label=create_run_label(pdata, run, first_run=first_run, labels=labels), c=linecolor, ls=linestyle)
                for tidx in range(len(pdata[run])):
                    ymin = np.nanmin([ymin, np.nanmin(pdata[run][tidx]["data"])]) if ymin is not None else np.nanmin(pdata[run][tidx]["data"])
                    ymax = np.nanmax([ymax, np.nanmax(pdata[run][tidx]["data"])]) if ymax is not None else np.nanmax(pdata[run][tidx]["data"])
                plot_list[run] = pp[0]

            ax.legend(loc=legend_location_gifs, fontsize = fontsize_legend)

            # putting limits on x axis since it is a trigonometry function (0,2)

            ax.set_xlim([0,1])

            # putting limits on y since it is a cosine function
            ax.set_ylim([ymin,ymax])

            # function takes frame as an input
            def AnimationFunction(frame):

                # line is set with new values of x and y
                for run in pdata:
                    plot_list[run].set_data((pdata[run][frame]["rho"], pdata[run][frame]["data"]))

            # creating the animation and saving it with a name that does not include spaces

            anim_created = FuncAnimation(Figure, AnimationFunction, frames=len(tvec), interval=200)
            #ylabel = ylabel.replace(' ', '_')
            #f = r'animation_' + ylabel + r'.gif'
            #anim_created.save(f, writer='writergif')

            # displaying the video

            video = anim_created.to_html5_video()
            html = display.HTML(video)
            display.display(html)

            plt.show()

            # good practice to close the plt object.
            plt.close()


def plot_interpolated_profiles(interpolated_data, plot_vars=None, labels=None):
    signal_list = interpolated_data["profile_signals"] if "profile_signals" in interpolated_data else keys_list['profiles_1d']
    if isinstance(plot_vars, list):
        signal_list = plot_vars
    for signame in signal_list:
        first_run = None

        if signame in interpolated_data:

            print("Plotting %s" % (signame))

            for tidx in range(len(interpolated_data[signame+".t"])):

                Figure = plt.figure()
                ax = Figure.add_subplot(1, 1, 1)
                ax.set_xlabel(r'$\rho_{tor,norm}$', fontsize = fontsize_labels)
                ax.set_ylabel(get_label_variable(signame), fontsize = fontsize_labels)

                title = get_label_variable(signame, show_units=False)

                ax.set_title(title + ' at t = {time:.3f}'.format(time=interpolated_data[signame+".t"][tidx]), fontsize = fontsize_title)
                # putting limits on x axis since it is a trigonometry function (0,2)
                ax.set_xlim([0,1])

                ax.tick_params(axis='x', labelsize=fontsize_ticks)
                ax.tick_params(axis='y', labelsize=fontsize_ticks)

                # putting limits on y since it is a cosine function
                #ax.set_ylim([ymin,ymax])

                for run in interpolated_data[signame]:
                    #This is used to set the run labels. Should work most of the times with the new jintrac version
                    if 'run' not in run: first_run = run

                    #print("Plotting profiles at index %s" % (tidx))
                    ax.plot(interpolated_data[signame+".x"], interpolated_data[signame][run][tidx], label=create_run_label(interpolated_data[signame], run, first_run=first_run, labels=labels))

                ax.legend(loc=legend_location_gifs, fontsize = fontsize_legend)

                plt.show()


def plot_gif_interpolated_profiles(interpolated_data, plot_vars=None, labels=None):
    signal_list = interpolated_data["profile_signals"] if "profile_signals" in interpolated_data else keys_list['profiles_1d']
    if isinstance(plot_vars, list):
        signal_list = plot_vars
    for signame in signal_list:
        first_run = None
        tvec = None

        if signame in interpolated_data:

            print("Plotting %s" % (signame))
            Figure = plt.figure()

            # creating a plot
            #    lines_plotted = plt.plot([])

            ax = Figure.add_subplot(1, 1, 1)
            ax.set_xlabel(r'$\rho_{tor,norm}$', fontsize = fontsize_labels)
            ax.set_ylabel(get_label_variable(signame), fontsize = fontsize_labels)
            #ax.set_xlabel(r'$\rho$ [-]')
            #ax.set_ylabel(ylabel + ' ' + units)
            ymin = None
            ymax = None
            plot_list = {}

            for run in interpolated_data[signame]:
                #This is used to set the run labels. Should work most of the times with the new jintrac version
                if 'run' not in run: first_run = run

                pp = ax.plot(interpolated_data[signame+".x"], interpolated_data[signame][run][0], label=create_run_label(interpolated_data[signame], run, first_run=first_run, labels=labels))
                for tidx in range(len(interpolated_data[signame+".t"])):
                    ymin = np.nanmin([ymin, np.nanmin(interpolated_data[signame][run][tidx])]) if ymin is not None else np.nanmin(interpolated_data[signame][run][tidx])
                    ymax = np.nanmax([ymax, np.nanmax(interpolated_data[signame][run][tidx])]) if ymax is not None else np.nanmax(interpolated_data[signame][run][tidx])
                plot_list[run] = pp[0]

            ax.legend(loc=legend_location_gifs, fontsize = fontsize_legend)

            # putting limits on x axis since it is a trigonometry function (0,2)
            ax.set_xlim([0,1])

            # putting limits on y since it is a cosine function
            ax.set_ylim([ymin,ymax])

            # function takes frame as an input
            def AnimationFunction(frame):

                # line is set with new values of x and y
                for run in interpolated_data[signame]:
                    plot_list[run].set_data((interpolated_data[signame+".x"], interpolated_data[signame][run][frame]))

            # creating the animation and saving it with a name that does not include spaces

            anim_created = FuncAnimation(Figure, AnimationFunction, frames=len(interpolated_data[signame+".t"]), interval=200)
            #ylabel = ylabel.replace(' ', '_')
            #f = r'animation_' + ylabel + r'.gif'
            #anim_created.save(f, writer='writergif')

            # displaying the video

            video = anim_created.to_html5_video()
            html = display.HTML(video)
            display.display(html)

            plt.show()

            iden_animation = get_label_variable(signame).replace('$','')
            f = r'animation_' + iden_animation.replace(' ','_') + r'.gif'
            anim_created.save(f, writer='writergif')

            # good practice to close the plt object.
            plt.close()


def print_time_traces(data_dict, data_vars=None, inverted_layout=False):
    out_dict = {}
    signal_list = data_dict["time_signals"] if "time_signals" in data_dict else keys_list['time_trace']
    if isinstance(data_vars, list):
        signal_list = data_vars
    for signame in signal_list:
        if inverted_layout:
            if signame in data_dict:
                for run in data_dict[signame]:
                    if len(data_dict[signame][run]) > 0:
                        if run not in out_dict:
                            out_dict[run] = {}
                        #val = np.mean(data_dict[signame][run])
                        val = np.mean(data_dict[signame][run][np.where(np.isnan(data_dict[signame][run]), False, True)])
                        print("%s %s average: %10.6e" % (run, signame, float(val)))
                        out_dict[run][signame] = val
        else:
            for run in data_dict:
                if isinstance(data_dict[run], dict) and signame in data_dict[run] and len(data_dict[run][signame]) > 0:
                    if run not in out_dict:
                        out_dict[run] = {}
                    val = np.mean(data_dict[run][signame][np.where(np.isnan(data_dict[run][signame]), False, True)])
                    #val = np.mean(data_dict[run][signame])
                    print("%s %s average: %10.6e" % (run, signame, float(val)))
                    out_dict[run][signame] = val

    return out_dict


#here for the integrate should not be average but should be last value normalized on the time interval

def print_time_trace_errors(time_error_dict, custom_vars=None, integrate_errors = False):

    out_dict = {}
    signal_list = time_error_dict["time_signals"] if "time_signals" in time_error_dict else expand_error_keys('time_trace')
    if isinstance(custom_vars, list):
        signal_list = custom_vars
    for signame in signal_list:
        if signame in time_error_dict:
            for run in time_error_dict[signame]:
                if signame not in out_dict:
                    out_dict[signame] = {}
                if not integrate_errors:
                    val = np.mean(time_error_dict[signame][run][np.where(np.isnan(time_error_dict[signame][run]), False, True)])
                else:
                    val = time_error_dict[signame][run][-1]/(time_error_dict[signame+'.t'][-1]-time_error_dict[signame+'.t'][0])
                print("%s %s average error: %10.6e" % (run, signame, float(val)))
                out_dict[signame][run] = val
    return out_dict

def print_profile_errors(profile_error_dict, custom_vars=None, integrate_errors = False):
    out_dict = {}
    signal_list = profile_error_dict["time_signals"] if "time_signals" in profile_error_dict else expand_error_keys('time_trace')
    if isinstance(custom_vars, list):
        signal_list = custom_vars
    for signame in signal_list:
        if signame in profile_error_dict:
            for run in profile_error_dict[signame]:
                if signame not in out_dict:
                    out_dict[signame] = {}
                #val = np.mean(profile_error_dict[signame][run][np.where(np.isnan(profile_error_dict[signame][run]), False, True)])
                if not integrate_errors:
                    val = np.mean(profile_error_dict[signame][run][np.where(np.isnan(profile_error_dict[signame][run]), False, True)])
                else:
                    val = profile_error_dict[signame][run][-1]/(profile_error_dict[signame+'.t'][-1]-profile_error_dict[signame+'.t'][0])

                print("%s %s average error: %10.6e" % (run, signame, float(val)))
                out_dict[signame][run] = val
    return out_dict

def absolute_error(data1, data2):
    return np.abs(data1 - data2)

#Switch for relative errors
def relative_error(data1, data2):
    return np.abs(2*(data1 - data2)/(data1 + data2))

def squared_error(data1, data2):
    return abs(4*(data1 - data2)*(data1 - data2)/((data1 + data2)*(data1 + data2)))

def difference_error(data1, data2):
    return 2*(np.abs(data1) - np.abs(data2))/np.abs(data1 + data2)

def calculate_volume_layers_single(volumes):
    volume_layers = np.asarray([])
    for volume in volumes:
        volume_layer = [1.0e-6]
        for volume_pre, volume_post in zip(volume[:], volume[1:]):
            volume_layer.append(volume_post - volume_pre)
        volume_layers = np.hstack((volume_layers, np.asarray(volume_layer)))

    volume_layers = volume_layers.reshape(np.shape(volumes))

    return volume_layers

def calculate_volume_layers(volumes1, volumes2):

    volume_layers1 = calculate_volume_layers_single(volumes1)
    volume_layers2 = calculate_volume_layers_single(volumes2)

    volume_layers = (volume_layers1+volume_layers2)/2

    return volume_layers


#Testing with errors weighted on the volume
def absolute_error_volume(data1, volume1, data2, volume2):
    #volumes = (volume1+volume2)/2
    #volumes_normalization = (volume1[:,-1]+volume2[:,-1])/2
    #for dat1, dat2, volume, volume_normalization in zip(data1, data2, volumes, volumes_normalization):
    #    distances = np.hstack((distances, np.abs(2*(dat1 - dat2)*volume/volume_normalization)))

    volume_layers = calculate_volume_layers(volume1, volume2)
    distances = np.asarray([])

    for dat1, dat2, volume_layer in zip(data1, data2, volume_layers):
        distances = np.hstack((distances, np.abs((dat1 - dat2)*volume_layer)))

    distances = distances.reshape(np.shape(volume1))

    return distances

def relative_error_volume(data1, volume1, data2, volume2):

    volume_layers = calculate_volume_layers(volume1, volume2)
    distances = np.asarray([])
    for dat1, dat2, volume_layer in zip(data1, data2, volume_layers):
        distances = np.hstack((distances, np.abs(2*(dat1 - dat2)/(dat1 + dat2)*volume_layer)))

    distances = distances.reshape(np.shape(volume1))

    return distances

def squared_error_volume(data1, volume1, data2, volume2):

    volume_layers = calculate_volume_layers(volume1, volume2)
    distances = np.asarray([])
    for dat1, dat2, volume_layer in zip(data1, data2, volume_layers):
        distances = np.hstack((distances, np.abs(4*(dat1 - dat2)*(dat1 - dat2)/((dat1 + dat2)*(dat1 + dat2))*volume_layer)))

    distances = distances.reshape(np.shape(volume1))

    return distances

def difference_error_volume(data1, volume1, data2, volume2):

    volume_layers = calculate_volume_layers(volume1, volume2)
    distances = np.asarray([])
    for dat1, dat2, volume, volume_layer in zip(data1, data2, volume_layers):
        distances = np.hstack((distances, (2*(np.abs(dat1) - np.abs(dat2))/(np.abs(dat1) + np.abs(dat2))*volume_layer)))

    distances = distances.reshape(np.shape(volume1))

    return distances


def compute_error_for_all_traces(analysis_dict, error_type = 'absolute', integrate_errors = False):
    out_dict = {}
    out_signal_list = []
    signal_list = analysis_dict["time_signals"] if "time_signals" in analysis_dict else keys_list['time_trace']
    for signame in signal_list:
        if signame in analysis_dict and len(analysis_dict[signame]) > 1:
            first_run = None
            first_data = None
            for run in analysis_dict[signame]:
                if first_data is None:
                    first_run = run
                    first_data = analysis_dict[signame][first_run]
                else:
                    if error_type == 'absolute':
                        var = signame+".absolute_error"
                    elif error_type == 'relative':
                        var = signame+".relative_error"
                    elif error_type == 'squared':
                        var = signame+".squared_error"
                    elif error_type == 'difference':
                        var = signame+".difference_error"

                    elif error_type == 'absolute volume':
                        var = signame+".absolute_error_volume"
                    elif error_type == 'relative volume':
                        var = signame+".relative_error_volume"
                    elif error_type == 'squared volume':
                        var = signame+".squared_error_volume"
                    elif error_type == 'difference volume':
                        var = signame+".difference_error_volume"
                    else:
                        print('Option for the error not recognized') #Should raise exception
                        exit()
                    if var not in out_dict:
                        out_dict[var] = {}
                    if error_type == 'absolute':
                        comp_data = absolute_error(analysis_dict[signame][run].flatten(), first_data.flatten())
                    elif error_type == 'relative':
                        comp_data = relative_error(analysis_dict[signame][run].flatten(), first_data.flatten())
                    elif error_type == 'squared':
                        comp_data = squared_error(analysis_dict[signame][run].flatten(), first_data.flatten())
                    elif error_type == 'difference':
                        comp_data = difference_error(analysis_dict[signame][run].flatten(), first_data.flatten())

                    elif error_type == 'absolute volume':
                        comp_data = absolute_error(analysis_dict[signame][run].flatten(), first_data.flatten())
                    elif error_type == 'relative volume':
                        comp_data = relative_error(analysis_dict[signame][run].flatten(), first_data.flatten())
                    elif error_type == 'squared volume':
                        comp_data = squared_error(analysis_dict[signame][run].flatten(), first_data.flatten())
                    elif error_type == 'difference volume':
                        comp_data = difference_error(analysis_dict[signame][run].flatten(), first_data.flatten())

                    else:
                        print('Option for the error not recognized') #Should raise exception
                        exit()
                    out_dict[var][run+":"+first_run] = comp_data.copy()

                    if integrate_errors:
                        if np.isnan(out_dict[var][run+":"+first_run][0]):
                            out_dict[var][run+":"+first_run][0] = 0 #First time needs to be there for integration to start at the right time
                        # Filter nans
                        analysis_dict[signame+".t"] = analysis_dict[signame+".t"][~np.isnan(out_dict[var][run+":"+first_run])]
                        out_dict[var][run+":"+first_run] = out_dict[var][run+":"+first_run][~np.isnan(out_dict[var][run+":"+first_run])]

                        out_dict[var][run+":"+first_run] = cumtrapz(out_dict[var][run+":"+first_run], analysis_dict[signame+".t"])

                    out_dict[var+".t"] = analysis_dict[signame+".t"]
                    if integrate_errors:
                        out_dict[var+".t"] = (out_dict[var+".t"][:-1] + out_dict[var+".t"][1:]) / 2.0

                    out_signal_list.append(var)

    out_dict["time_signals"] = out_signal_list
    return out_dict

def compute_average_error_for_all_profiles(analysis_dict, error_type = 'absolute', integrate_errors = False):
    out_dict = {}
    out_signal_list = []
    signal_list = analysis_dict["profile_signals"] if "profile_signals" in analysis_dict else keys_list['profiles_1d']
    for signame in signal_list:
        if signame in analysis_dict and len(analysis_dict[signame]) > 1:
            first_run = None
            first_data = None
            for run in analysis_dict[signame]:
                if first_data is None:
                    first_run = run
                    first_data = analysis_dict[signame][first_run]
                    if 'volume' in error_type:
                        signame_volume = 'core_profiles.profiles_1d[].grid.volume'
                        first_data_volume = analysis_dict[signame_volume][first_run]
                else:
                    if error_type == 'absolute':
                        var = signame+".average_absolute_error"
                    elif error_type == 'relative':
                        var = signame+".average_relative_error"
                    elif error_type == 'difference':
                        var = signame+".average_difference_error"
                    elif error_type == 'squared':
                        var = signame+".average_squared_error"

                    elif error_type == 'absolute volume':
                        var = signame+".average_absolute_error_volume"
                    elif error_type == 'relative volume':
                        var = signame+".average_relative_error_volume"
                    elif error_type == 'difference volume':
                        var = signame+".average_difference_error_volume"
                    elif error_type == 'squared volume':
                        var = signame+".average_squared_error_volume"

                    else:
                        print('Option for the error not recognized') #Should raise exception
                        exit()
                    if var not in out_dict:
                        out_dict[var] = {}
                    if error_type == 'absolute':
                        comp_data = absolute_error(analysis_dict[signame][run], first_data)
                    elif error_type == 'relative':
                        comp_data = relative_error(analysis_dict[signame][run], first_data)
                    elif error_type == 'difference':
                        comp_data = difference_error(analysis_dict[signame][run], first_data)
                    elif error_type == 'squared':
                        comp_data = squared_error(analysis_dict[signame][run], first_data)

                    elif error_type == 'absolute volume':
                        comp_data = absolute_error_volume(analysis_dict[signame][run], analysis_dict[signame_volume][run], first_data, first_data_volume)
                    elif error_type == 'relative volume':
                        comp_data = relative_error_volume(analysis_dict[signame][run], analysis_dict[signame_volume][run], first_data, first_data_volume)
                    elif error_type == 'difference volume':
                        comp_data = difference_error_volume(analysis_dict[signame][run], analysis_dict[signame_volume][run], first_data, first_data_volume)
                    elif error_type == 'squared volume':
                        comp_data = squared_error_volume(analysis_dict[signame][run], analysis_dict[signame_volume][run], first_data, first_data_volume)

                    else:
                        print('Option for the error not recognized') #Should raise exception

                    out_dict[var][run+":"+first_run] = np.average(comp_data, axis=1)   #Averaging the errors

                    if 'squared' in error_type:
                        out_dict[var][run+":"+first_run] = np.sqrt(out_dict[var][run+":"+first_run])

                    if integrate_errors:
                        out_dict[var][run+":"+first_run] = cumtrapz(out_dict[var][run+":"+first_run], analysis_dict[signame+".t"])

                    out_dict[var+".t"] = analysis_dict[signame+".t"]
                    if integrate_errors:
                        out_dict[var+".t"] = (out_dict[var+".t"][:-1] + out_dict[var+".t"][1:]) / 2.0

                    out_signal_list.append(var)
    out_dict["profile_signals"] = out_signal_list
    return out_dict

def perform_time_trace_analysis(analysis_dict, **kwargs):
    out_dict = {}
    out_signal_list = []
    #absolute_error_flag = kwargs.pop("absolute_error", False)
    error_flag = kwargs.pop("error", False)
    error_type = kwargs.pop("error_type", 'absolute')
    integrate_errors = kwargs.pop("integrate_errors", False)
    if error_flag:
        abs_err_dict = compute_error_for_all_traces(analysis_dict, error_type = error_type, integrate_errors = integrate_errors)
        signal_list = abs_err_dict.pop("time_signals")
        out_dict.update(abs_err_dict)
        for signal in signal_list:
            if signal not in out_signal_list:
                out_signal_list.append(signal)
        #out_signal_list.extend(signal_list)
    out_dict["time_signals"] = out_signal_list
    return out_dict

def perform_profile_analysis(analysis_dict, **kwargs):
    out_dict = {}
    out_signal_list = []
    #average_absolute_error_flag = kwargs.pop("average_absolute_error", False)
    average_error_flag = kwargs.pop("average_error", False)
    error_type = kwargs.pop("error_type", 'absolute')
    integrate_errors = kwargs.pop("integrate_errors", False)
    if average_error_flag:
        avg_abs_err_dict = compute_average_error_for_all_profiles(analysis_dict, error_type = error_type, integrate_errors = integrate_errors)
        signal_list = avg_abs_err_dict.pop("profile_signals")
        out_dict.update(avg_abs_err_dict)
        for signal in signal_list:
            if signal not in out_signal_list:
                out_signal_list.append(signal)
        #out_signal_list.extend(signal_list)
    out_dict["time_signals"] = out_signal_list
    return out_dict

def perform_sign_correction(raw_dict, ref_tag):
    # Only handles the index levels in order from get_onedict()
    for tag in raw_dict:
        if tag != ref_tag:
            for key in raw_dict[tag]:
                if not key.endswith(".x") and not key.endswith(".t") and key in raw_dict[ref_tag]:
                    if tag != 'time_signals' and tag != 'profile_signals':
                        if isinstance(raw_dict[tag][key][0], dict):
                            if np.mean(raw_dict[ref_tag][key][0]["y"]) * np.mean(raw_dict[tag][key][0]["y"]) < 0.0:
                                for ii in range(len(raw_dict[tag][key])):
                                    raw_dict[tag][key][ii]["y"] = -raw_dict[tag][key][ii]["y"]
                        elif np.mean(raw_dict[ref_tag][key]) * np.mean(raw_dict[tag][key]) < 0.0:
                            raw_dict[tag][key] = -raw_dict[tag][key]
    return raw_dict

def standardize_basis_vectors(raw_dict, ref_tag, time_basis=None):

    # Transform reference data into the required field names and store in reference container
    ref_dict = {}
    for key in raw_dict[ref_tag]:
        if not key.endswith(".x") and not key.endswith(".t"):

            if key not in ref_dict:
                ref_dict[key] = {}

            if time_basis is None:

                ref_dict[key][ref_tag] = copy.deepcopy(raw_dict[ref_tag][key])
                if key+".x" in raw_dict[ref_tag]:
                    ref_dict[key+".x"] = copy.deepcopy(raw_dict[ref_tag][key+".x"])
                ref_dict[key+".t"] = copy.deepcopy(raw_dict[ref_tag][key+".t"])

            else:    # Apply user-defined time vector as interpolation basis

                # User-defined time vector takes priority over the time vector inside the user-defined reference run
                ytable = np.atleast_2d(raw_dict[ref_tag][key])

                # Radial interpolation will take place in the next loop
                if key+".x" in raw_dict[ref_tag]:
                    ref_dict[key+".x"] = copy.deepcopy(raw_dict[ref_tag][key+".x"])
                ref_dict[key+".t"] = copy.deepcopy(time_basis)

                # Perform time vector interpolation, always present
                t_new = ref_dict[key+".t"]
                if len(raw_dict[ref_tag][key+".t"]) > 1:
                    ytable_new = None
                    for ii in range(ytable.shape[1]):
                        y_new = fit_and_substitute(raw_dict[ref_tag][key+".t"], t_new, ytable[:, ii])
                        ytable_new = np.vstack((ytable_new, y_new)) if ytable_new is not None else np.atleast_2d(y_new)
                    ref_dict[key][ref_tag] = ytable_new.T
                else:
                    # Copies existing time slice multiple times if only one time slice is present in the run
                    ytable_new = None
                    for ii in range(len(t_new)):
                        ytable_new = np.vstack((ytable_new, ytable)) if ytable_new is not None else np.atleast_2d(ytable)
                    ref_dict[key][ref_tag] = copy.deepcopy(ytable_new)

    # Loop over all runs in order to maintain run[0] for analysis purposes
    std_dict = {}
    for tag, run_dict in raw_dict.items():
        # tag contains the id of the run, key the variable to be plotted
        if tag == ref_tag:
            for key in ref_dict:
                if not key.endswith(".x") and not key.endswith(".t"):
                    if key not in std_dict:
                        std_dict[key] = {}
                    std_dict[key][tag] = ref_dict[key][tag]
                else:
                    std_dict[key] = ref_dict[key]
        elif tag not in ["time_signals", "profile_signals"]:
            for key in run_dict:
                if not key.endswith(".x") and not key.endswith(".t"):

                    if key not in std_dict:
                        std_dict[key] = {}

                    ytable = np.atleast_2d(run_dict[key])
                    ytable_temp = None
                    # Perform radial vector interpolation, if radial vector is present in signal
                    if key+".x" in ref_dict:
                        x_new = ref_dict[key+".x"]
                        for ii in range(ytable.shape[0]):
                            y_new = fit_and_substitute(run_dict[key+".x"], x_new, ytable[ii, :])
                            ytable_temp = np.vstack((ytable_temp, y_new)) if ytable_temp is not None else np.atleast_2d(y_new)
                    else:
                        ytable_temp = np.atleast_2d(ytable)

                    ytable_new = None
                    # Perform time vector interpolation, always present
                    t_new = ref_dict[key+".t"]
                    if len(run_dict[key+".t"]) > 1:
                        if key+".x" in run_dict:
                            for ii in range(ytable_temp.shape[1]):
                                y_new = fit_and_substitute(run_dict[key+".t"], t_new, ytable_temp[:, ii])
                                ytable_new = np.vstack((ytable_new, y_new)) if ytable_new is not None else np.atleast_2d(y_new)
                        else:
                            ytable_new = fit_and_substitute(run_dict[key+".t"], t_new, ytable_temp[0])
                        if ytable_new is not None:
                            ytable_new = ytable_new.T
                    else:
                        # Copies existing time slice multiple times if only one time slice is present in the run
                        for ii in range(len(t_new)):
                            ytable_new = np.vstack((ytable_new, ytable_temp)) if ytable_new is not None else np.atleast_2d(ytable_temp)

                    std_dict[key][tag] = copy.deepcopy(ytable_new)
        else:
            std_dict[tag] = run_dict

    return std_dict

def compute_user_string_functions(data_dict, signal_operations, standardized=False, keep_op_signals = False):

    taglist = get_taglist(data_dict)

    # Oof, this is not the safest implementation (due to eval) but it takes a lot to make this both general and clean
    if not standardized:          # This is for the raw vector branch
        for tag in taglist:
            for op, sigopvec in signal_operations.items():

                # Standardizing basis vectors across operated signals, which may be different between different IDSs
                operation_dict = {}
                time_ref = None
                radial_ref = None
                for key in sigopvec:
                    if time_ref is None and key+".t" in data_dict[tag]:
                        time_ref = data_dict[tag][key+".t"]
                    if radial_ref is None and key+".x" in data_dict[tag]:
                        radial_ref = data_dict[tag][key+".x"]

                    ytable = np.atleast_2d(data_dict[tag][key])
                    ytable_temp = None
                    if radial_ref is not None and key+".x" in data_dict[tag]:
                        for ii in range(ytable.shape[0]):
                            y_new = fit_and_substitute(data_dict[tag][key+".x"], radial_ref, ytable[ii, :])
                            ytable_temp = np.vstack((ytable_temp, y_new)) if ytable_temp is not None else np.atleast_2d(y_new)
                    else:
                        ytable_temp = np.atleast_2d(ytable)

                    ytable_new = None
                    if time_ref is not None and len(data_dict[key+".t"]) > 1:
                        if key+".x" in data_dict:
                            for ii in range(ytable_temp.shape[1]):
                                y_new = fit_and_substitute(data_dict[tag][key+".t"], time_ref, ytable_temp[:, ii])
                                ytable_new = np.vstack((ytable_new, y_new)) if ytable_new is not None else np.atleast_2d(y_new)
                        else:
                            ytable_new = fit_and_substitute(data_dict[tag][key+".t"], time_ref, ytable_temp[0])
                    else:
                        ytable_new = copy.deepcopy(ytable_temp)
                    if ytable_new is not None:
                        ytable_new = ytable_new.T

                    operation_dict[key] = copy.deepcopy(ytable_new)

                # Substituting the key since otherwise it will think that the dots define attributes
                sfunc = op.replace('.', '_')
                nkey_list = []
                fready = True
                for key in sigopvec:
                    new_key = key.replace('.', '_')
                    nkey_list.append(new_key)
                    if key not in operation_dict:
                        fready = False

                if fready:
                    for new_key in nkey_list:
                        globals()[new_key] = copy.deepcopy(operation_dict[key])
                    op_result = eval(sfunc)
                    data_dict[tag][op] = op_result
                    data_dict[tag][op+'.t'] = time_ref
                    if radial_ref is not None:
                        data_dict[tag][op+'.x'] = radial_ref
                        data_dict["profile_signals"].append(op)
                    else:
                        data_dict["time_signals"].append(op)
                    for new_key in nkey_list:
                        del globals()[new_key]

    else:                            # This is for the standardized vector branch
        for op, sigopvec in signal_operations.items():

            # Substituting the key since otherwise it will think that the dots define attributes
            sfunc = op.replace('.', '_')
            # Removing parenthesis
            sfunc = sfunc.replace('[', '')
            sfunc = sfunc.replace(']', '')
            nkey_list = []
            fready = True
            fradial = False

            for key in sigopvec:
                if key not in data_dict:
                    fready = False
                if key+".x" in data_dict:
                    fradial = True

            for key in sigopvec:
                new_key = key.replace('.', '_')
                new_key = new_key.replace('[', '')
                new_key = new_key.replace(']', '')

                nkey_list.append(new_key)

            if fready:
                operation_dict = {}
                for tag in taglist:
                    for new_key, key in zip(nkey_list, sigopvec):
                        globals()[new_key] = copy.deepcopy(data_dict[key][tag])
                    op_result = eval(sfunc)
                    operation_dict[tag] = op_result
                    for new_key in nkey_list:
                        del globals()[new_key]
                data_dict[op] = copy.deepcopy(operation_dict)
                data_dict[op + '.t'] = copy.deepcopy(data_dict[sigopvec[0] + '.t'])
                if fradial:
                    data_dict["profile_signals"].append(op)
                    data_dict[op + '.x'] = copy.deepcopy(data_dict[sigopvec[0] + '.x'])
                else:
                    data_dict["time_signals"].append(op)

    if not keep_op_signals:
        for signal in sigopvec:
            del data_dict[signal]
            del data_dict[signal + '.t']
            if signal + '.x' in data_dict:
                del data_dict[signal + '.x']


    return data_dict

def generate_data_tables(run_tags, signals, time_begin, time_end, signal_operations=None, correct_sign=False, reference_index=None, standardize=False, time_basis=None, backend = None, keep_op_signals = False):

    jruns_home = os.environ['JRUNS']
    current_user = os.getlogin()

    ref_idx = reference_index if isinstance(reference_index, int) else 0

    # Adding the variables for comparison of functions when they are not available
    sigvec = copy.deepcopy(signals)
    sigopdict = {}
    fops = False
    if isinstance(signal_operations, list):
        for sigop in signal_operations:
            sigop_tmp = copy.deepcopy(sigop)
            for op in operations:
                sigop_tmp = sigop_tmp.replace(op, ';')
            sigop_vars = sigop_tmp.split(';')
            fops = True
            empty_vars = []
            for ii, sigop_var in enumerate(sigop_vars):
                if sigop_var == '':
                    empty_vars.append(ii)
            for ii in empty_vars[::-1]:
                del sigop_vars[ii]
            for var in sigop_vars:
                if not sigvec: sigvec = []
                if var not in sigvec:
                    sigvec.append(var)
            sigopdict[sigop] = sigop_vars

    raw_dict = {}
    ref_tag = None

    taglist = []
    t_fields = []
    xt_fields = []
    for ii, tag in enumerate(run_tags):

        stag = tag.strip().split('/')
        while not stag[-1]:
            stag = stag[:-1]
        db = stag[-3].strip()
        shot = int(stag[-2].strip())
        runid = int(stag[-1].strip())
        user = '/'.join(stag[:-3]) if len(stag) > 4 else stag[0]
        sid = None
        tid = None

        if user.startswith('jruns/'):
            requested_user = stag[1]
            jloc = jruns_home.replace(current_user, requested_user)
            user = jloc + '/' + '/'.join(stag[2:-3])

        backend_run = None
        if not backend: backend_run = get_backend(db, shot, runid, username=user)

        onedict = get_onedict(sigvec, user, db, shot, runid, time_begin, time_end=time_end, sid=sid, tid=tid, interpolate=standardize, backend = backend_run)
        if ii == ref_idx:
            ref_tag = tag
        if "time_signals" in onedict:
            for signame in onedict["time_signals"]:
                if signame not in t_fields:
                    t_fields.append(signame)
            del onedict["time_signals"]
        if "profile_signals" in onedict:
            for signame in onedict["profile_signals"]:
                if signame not in xt_fields:
                    xt_fields.append(signame)
            del onedict["profile_signals"]
        raw_dict[tag] = onedict
        taglist.append(tag)

    raw_dict["time_signals"] = t_fields
    raw_dict["profile_signals"] = xt_fields

    out_dict = {}
    if raw_dict and ref_tag is not None:

        out_dict = copy.deepcopy(raw_dict)

        # Changes the sign of the variable if it is mismatching with reference run. Useful for q profile in some instances.
        # NOTE: this implementation requires the index level order from get_onedict()
        if correct_sign:
            out_dict = perform_sign_correction(out_dict, ref_tag)

        # Standardizes the radial and time vectors to the reference run
        # NOTE: this implementation inverts the run tag and signal index levels within the nested dict!!!
        if standardize:
            out_dict = standardize_basis_vectors(out_dict, ref_tag, time_basis=time_basis)

        # Applies user-defined string operations to single and/or multiple variables
        # NOTE: this implementation uses eval to process string operations!!! BE CAREFUL!!!
        if fops:
            out_dict = compute_user_string_functions(out_dict, sigopdict, standardized=standardize, keep_op_signals = keep_op_signals)

    return out_dict, ref_tag

####### SCRIPT #######

def get_taglist(data_dict):

    taglist = []
    for key in data_dict:
        if not key.endswith('t') and not key.endswith('x') and not key == 'time_signals' and not key == 'profile_signals':
            for tag in data_dict[key]:
                if tag not in taglist:
                    taglist.append(tag)

    return taglist


def get_siglist(data_dict):
    siglist = []
    for key in data_dict:
        if not key.endswith('t') and not key.endswith('x') and not key == 'time_signals' and not key == 'profile_signals':
            siglist.append(key)

    return siglist


def rebase_x_datadict(data_dict):

    #first create a new variable with a regular grid in x

    for signal in data_dict:
        if signal.endswith('.x'):
            variable_name = signal[:-2]
            x_new = np.arange(0,1,1/np.shape(data_dict[signal])[0])
            for run in data_dict[variable_name]:
                ys_new = np.asarray([])
                for time_slice in data_dict[variable_name][run]:
                    ys_new = np.hstack((ys_new, fit_and_substitute(data_dict[signal], x_new, time_slice))) #Recalculate the variable on regular grid
                ys_new = ys_new.reshape(np.shape(data_dict[variable_name][run]))

                data_dict[variable_name][run] = ys_new
            data_dict[signal] = x_new

    return data_dict


def compare_runs(signals, idslist, time_begin, time_end=None, time_basis=None, plot=False, analyze=False, error_type = 'absolute', correct_sign=False, steady_state=False, uniform=False, signal_operations=None, backend = 'hdf5', keep_op_signals = False, integrate = False, integrate_errors = False, use_regular_rho_grid = False, labels = None, y_limits=[None, None], verbose = False):

    ref_idx = 1 if steady_state else 0
    standardize = (uniform or analyze or isinstance(time_basis, (list, tuple, np.ndarray)))

    if 'volume' in error_type:
        signals.append('core_profiles.profiles_1d[].grid.volume')

    data_dict, ref_tag = generate_data_tables(idslist, signals, time_begin, time_end=time_end, signal_operations=signal_operations, correct_sign=correct_sign, reference_index=ref_idx, standardize=standardize, time_basis=time_basis, backend = backend, keep_op_signals = keep_op_signals)

    if use_regular_rho_grid:
        data_dict = rebase_x_datadict(data_dict)

    if integrate and standardize:
        siglist = get_siglist(data_dict)
        taglist = get_taglist(data_dict)
        for sig in siglist:
            for tag in taglist:
                data_dict[sig][tag] = cumtrapz(data_dict[sig][tag], x=data_dict[sig + '.t'])
            data_dict[sig + '.t'] = (data_dict[sig + '.t'][:-1] + data_dict[sig + '.t'][1:]) / 2.0

    if plot:
        if standardize:
            if 'volume' in error_type:
                data_dict_no_vol = copy.deepcopy(data_dict)
                del data_dict_no_vol['core_profiles.profiles_1d[].grid.volume']

                plot_interpolated_traces(data_dict_no_vol, labels = labels, y_limits=y_limits)
                plot_gif_interpolated_profiles(data_dict_no_vol, labels = labels)
                if verbose:
                    plot_interpolated_profiles(data_dict_no_vol, labels = labels)
            else:
                plot_interpolated_traces(data_dict, labels = labels, y_limits=y_limits)
                plot_gif_interpolated_profiles(data_dict, labels = labels)
                if verbose:
                    plot_interpolated_profiles(data_dict, labels = labels)
        else:
            plot_traces(data_dict, single_time_reference=steady_state, labels = labels)
            plot_gif_profiles(data_dict, single_time_reference=steady_state, labels = labels)

    time_averages = print_time_traces(data_dict, inverted_layout=standardize)
    time_error_averages = {}
    profile_error_averages = {}

    if analyze:

        options = {"error": True, "error_type": error_type, "integrate_errors": integrate_errors}
        time_error_dict = perform_time_trace_analysis(data_dict, **options)

        if plot:
            plot_interpolated_traces(time_error_dict, labels=labels, y_limits=y_limits)

        options = {"average_error": True, "error_type": error_type}

        profile_error_dict = perform_profile_analysis(data_dict, **options)

        if 'squared' in error_type:
            for signal in profile_error_dict:
                if signal.endswith('error'):
                    for run in profile_error_dict[signal]:
                        profile_error_dict[signal][run] = profile_error_dict[signal][run]*profile_error_dict[signal][run]

        if 'volume' in error_type:
            error_type_split = error_type.split(' ')
            del profile_error_dict['core_profiles.profiles_1d[].grid.volume.average_' + error_type_split[0] + '_error_volume']
            del profile_error_dict['core_profiles.profiles_1d[].grid.volume.average_' + error_type_split[0] + '_error_volume.t']

        if plot:
            plot_interpolated_traces(profile_error_dict, labels=labels, y_limits=y_limits)

        time_error_averages = print_time_trace_errors(time_error_dict, integrate_errors = integrate_errors)
        if 'squared' in error_type:
            if time_error_averages:
                for signal in time_error_averages:
                    for run in time_error_averages[signal]:
                        time_error_averages[signal][run] = np.sqrt(time_error_averages[signal][run])

        profile_error_averages = print_profile_errors(profile_error_dict, integrate_errors = integrate_errors)
        if 'squared' in error_type:
            if profile_error_averages:
                for signal in profile_error_averages:
                    for run in profile_error_averages[signal]:
                        profile_error_averages[signal][run] = np.sqrt(profile_error_averages[signal][run])

    return time_averages, time_error_averages, profile_error_averages

####### COMMAND LINE INTERFACE #######

def main():

    args = input()

    if args.backend == 'hdf5':
        args.backend = imas.imasdef.HDF5_BACKEND
    elif args.backend == 'mdsplus':
        args.backend = imas.imasdef.MDSPLUS_BACKEND

    do_plot = not args.calc_only
    time_averages, time_error_averages, profiles_error_averages = compare_runs(
        signals=args.signal,
        idslist=args.ids,
        backend=args.backend,
        time_begin=args.time_begin,
        time_end=args.time_end,
        time_basis=args.time_out,
#        sourcelist=args.source,
#        transportlist=args.transport,
        plot=do_plot,
        analyze=args.analyze,
        error_type=args.error_type,
        correct_sign=args.correct_sign,
        steady_state=args.steady_state,
        uniform=args.uniform,
        signal_operations=args.function,
        keep_op_signals=args.keep_op_signals,
        integrate=args.integrate,
        integrate_errors=args.integrate_errors,
        use_regular_rho_grid=args.use_regular_rho_grid,
        labels=args.labels,
        y_limits=args.y_limits,
        verbose=args.verbose
    )
    # Arugments not used: save_plot, version

if __name__ == "__main__":
    main()

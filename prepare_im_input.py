import json
import os,datetime,sys
import shutil
import getpass
import numpy as np
import pickle
import math
import functools
import copy
import json
from scipy import integrate
from scipy.interpolate import interp1d, UnivariateSpline
#import idstools
#from idstools import *
from packaging import version
from os import path
import inspect
import types
import pdb
import xarray as xr

import duqtools
from duqtools.api import ImasHandle
from duqtools.api import Variable
from duqtools.api import rebase_on_time

import matplotlib.pyplot as plt

#from matplotlib.animation import FuncAnimation
from IPython import display

import xml.sax
import xml.sax.handler
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

variables_summary = [
    Variable(name='ip',
             ids = 'summary',
             path = 'global_quantities/ip/value',
             dims = ['time']),
    Variable(name='v_loop',
             ids = 'summary',
             path = 'global_quantities/v_loop/value',
             dims = ['time']),
    Variable(name='li',
             ids = 'summary',
             path = 'global_quantities/li/value',
             dims = ['time']),
    Variable(name='li_mhd',
             ids = 'summary',
             path = 'global_quantities/li_mhd/value',
             dims = ['time']),
    Variable(name='energy_diamagnetic',
             ids = 'summary',
             path = 'global_quantities/energy_diamagnetic/value',
             dims = ['time']),
    Variable(name='energy_mhd',
             ids = 'summary',
             path = 'global_quantities/energy_mhd/value',
             dims = ['time']),
    Variable(name='energy_thermal',
             ids = 'summary',
             path = 'global_quantities/energy_thermal/value',
             dims = ['time']),
    Variable(name='beta_pol',
             ids = 'summary',
             path = 'global_quantities/beta_pol/value',
             dims = ['time']),
    Variable(name='beta_pol_mhd',
             ids = 'summary',
             path = 'global_quantities/beta_pol_mhd/value',
             dims = ['time']),
    Variable(name='beta_tor_norm',
             ids = 'summary',
             path = 'global_quantities/beta_tor_norm/value',
             dims = ['time']),
    Variable(name='power_radiated',
             ids = 'summary',
             path = 'global_quantities/power_radiated/value',
             dims = ['time']),
    Variable(name='volume',
             ids = 'summary',
             path = 'global_quantities/volume/value',
             dims = ['time']),
    Variable(name='neutron flux total',
             ids = 'summary',
             path = 'fusion/neutron_fluxes/total/value',
             dims = ['time']),
    Variable(name='power_nbi',
             ids = 'summary',
             path = 'heating_current_drive/power_nbi/value',
             dims = ['time']),
    Variable(name='power_ic',
             ids = 'summary',
             path = 'heating_current_drive/power_ic/value',
             dims = ['time']),
    Variable(name='power_ec',
             ids = 'summary',
             path = 'heating_current_drive/power_ec/value',
             dims = ['time']),
    Variable(name='power_lh',
             ids = 'summary',
             path = 'heating_current_drive/power_lh/value',
             dims = ['time'])
]

variable_summary_time = [
    Variable(name='time',
             ids = 'summary',
             path = 'time',
             dims = ['time'])
]

variables_core_profiles = [
    Variable(name='r0',
             ids = 'core_profiles',
             path = 'vacuum_toroidal_field/r0',
             dims = []),
    Variable(name='b0',
             ids = 'core_profiles',
             path = 'vacuum_toroidal_field/b0',
             dims = ['time']),
    Variable(name='q',
             ids = 'core_profiles',
             path = 'profiles_1d/*/q',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='ne thermal',
             ids = 'core_profiles',
             path = 'profiles_1d/*/electrons/density_thermal',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='ne',
             ids = 'core_profiles',
             path = 'profiles_1d/*/electrons/density',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='v tor',
             ids = 'core_profiles',
             path = 'profiles_1d/*/electrons/velocity_tor',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='pressure thermal',
             ids = 'core_profiles',
             path = 'profiles_1d/*/electrons/pressure_thermal',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='t_i_ave',
             ids = 'core_profiles',
             path = 'profiles_1d/*/t_i_average',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='zeff',
             ids = 'core_profiles',
             path = 'profiles_1d/*/zeff',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='rho tor',
             ids = 'core_profiles',
             path = 'profiles_1d/*/grid/rho_tor',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='rho pol norm',
             ids = 'core_profiles',
             path = 'profiles_1d/*/grid/rho_tor_norm',
             dims = ['time', 'rho_tor_norm_core_profiles']),
    Variable(name='rho tor norm time',
             ids = 'core_profiles',
             path = 'profiles_1d/*/grid/rho_tor_norm',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='ion temperature',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/temperature',
             dims = ['time', 'ion', 'rho_tor_norm']),
    Variable(name='ni thermal',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/density_thermal',
             dims = ['time', 'ion', 'rho_tor_norm']),
    Variable(name='ni',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/density',
             dims = ['time', 'ion', 'rho_tor_norm']),
    Variable(name='pressure thermal',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/pressure_thermal',
             dims = ['time', 'ion', 'rho_tor_norm']),
    Variable(name='pressure',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/pressure',
             dims = ['time', 'ion', 'rho_tor_norm']),
    Variable(name='ip',
             ids = 'core_profiles',
             path = 'global_quantities/ip',
             dims = ['time']),
    Variable(name='v loop',
             ids = 'core_profiles',
             path = 'global_quantities/v_loop',
             dims = ['time']),
    Variable(name='li 3',
             ids = 'core_profiles',
             path = 'global_quantities/li_3',
             dims = ['time']),
    Variable(name='energy diamagnetic',
             ids = 'core_profiles',
             path = 'global_quantities/energy_diamagnetic',
             dims = ['time']),
    Variable(name='z ion',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/z_ion',
             dims = ['time', 'ion']),
    Variable(name='multiple states flag',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/multiple_states_flag',
             dims = ['time', 'ion']),
    Variable(name='label',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/label',
             dims = ['time', 'ion']),
    Variable(name='ion a',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/element/*/a',
             dims = ['time', 'ion', 'element']),
    Variable(name='ion z n',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/element/*/z_n',
             dims = ['time', 'ion', 'element']),
    Variable(name='ion atoms n',
             ids = 'core_profiles',
             path = 'profiles_1d/*/ion/*/element/*/atoms_n',
             dims = ['time', 'ion', 'element'])
]

variable_core_profiles_time = [
    Variable(name='time',
             ids = 'core_profiles',
             path = 'time',
             dims = ['time'])
]

variable_core_profiles_rho = [
    Variable(name='rho_tor_norm',
             ids = 'core_profiles',
             path = 'profiles_1d/0/grid/rho_tor_norm',
             dims = ['rho_tor_norm'])
]

variables_equilibrium = [
    Variable(name='r0',
             ids = 'equilibrium',
             path = 'vacuum_toroidal_field/r0',
             dims = []),
    Variable(name='b0',
             ids = 'equilibrium',
             path = 'vacuum_toroidal_field/b0',
             dims = ['time']),
    Variable(name='psi',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_1d/psi',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='phi',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_1d/phi',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='f',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_1d/f',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='q',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_1d/q',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='pressure',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_1d/pressure',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='rho tor',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_1d/rho_tor',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='rho tor norm time',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_1d/rho_tor_norm',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='area',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_1d/area',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='volume',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_1d/volume',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='r',
             ids = 'equilibrium',
             path = 'time_slice/*/boundary/outline/r',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='z',
             ids = 'equilibrium',
             path = 'time_slice/*/boundary/outline/z',
             dims = ['time', 'rho_tor_norm']),
    Variable(name='ip',
             ids = 'equilibrium',
             path = 'time_slice/*/global_quantities/ip',
             dims = ['time']),
    Variable(name='li 3',
             ids = 'equilibrium',
             path = 'time_slice/*/global_quantities/li_3',
             dims = ['time']),
    Variable(name='beta pol',
             ids = 'equilibrium',
             path = 'time_slice/*/global_quantities/beta_pol',
             dims = ['time']),
    Variable(name='beta tor',
             ids = 'equilibrium',
             path = 'time_slice/*/global_quantities/beta_tor',
             dims = ['time']),
    Variable(name='magnetic axis r',
             ids = 'equilibrium',
             path = 'time_slice/*/global_quantities/magnetic_axis/r',
             dims = ['time']),
    Variable(name='magnetic axis z',
             ids = 'equilibrium',
             path = 'time_slice/*/global_quantities/magnetic_axis/z',
             dims = ['time']),
    Variable(name='dim1',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_2d/*/grid/dim1',
             dims = ['time','profile_2d','dim1']),
    Variable(name='dim2',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_2d/*/grid/dim2',
             dims = ['time','profile_2d','dim2']),
    Variable(name='psi 2d',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_2d/*/psi',
             dims = ['time','profile_2d','dim1','dim2']),
    Variable(name='grid name',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_2d/*/grid_type/name',
             dims = ['time','profile_2d']),
    Variable(name='grid index',
             ids = 'equilibrium',
             path = 'time_slice/*/profiles_2d/*/grid_type/index',
             dims = ['time','profile_2d'])
]

variable_equilibrium_time = [
    Variable(name='time',
             ids = 'equilibrium',
             path = 'time',
             dims = ['time'])
]

variable_equilibrium_rho = [
    Variable(name='rho_tor_norm',
             ids = 'equilibrium',
             path = 'time_slice/0/profiles_1d/rho_tor_norm',
             dims = ['rho_tor_norm'])
]

def construct_variables_lists(ids_names_to_extract):

    variables, variables_time, variables_rho = {}, {}, {}
    if 'summary' in ids_names_to_extract:
        variables['summary'] = variables_summary
        variables_time['summary'] = variable_summary_time
    if 'core_profiles' in ids_names_to_extract:
        variables['core_profiles'] = variables_core_profiles
        variables_time['core_profiles'] = variable_core_profiles_time
        variables_rho['core_profiles'] = variable_core_profiles_rho
    if 'equilibrium' in ids_names_to_extract:
        variables['equilibrium'] = variables_equilibrium
        variables_time['equilibrium'] = variable_equilibrium_time
        variables_rho['equilibrium'] = variable_equilibrium_rho

    return variables, variables_time, variables_rho

def extract_integrated_modelling(ids_names_to_extract, username, db, shot, run):

    handle = ImasHandle(user = username, db = db, shot = shot, run = run)
    dataset_integrated_modelling = {}
    variables, variables_time, variables_rho = construct_variables_lists(ids_names_to_extract)

    single_dataset = handle.get_variables([variables['core_profiles'][0], variables_time['core_profiles'][0], variables_rho['core_profiles'][0]])

    for key in variables:
        dataset = xr.Dataset()
        time = handle.get_variables(variables_time[key])

        for variable in variables[key]:
            try:
                if key in variables_rho.keys(): 
                    single_dataset = handle.get_variables([variable, variables_time[key][0], variables_rho[key][0]])
                else:
                    single_dataset = handle.get_variables([variable, variables_time[key][0]])
                dataset = xr.merge([dataset, single_dataset])
            except (ValueError, duqtools.ids._mapping.EmptyVarError):
                pass

        dataset_integrated_modelling[key] = dataset

    return dataset_integrated_modelling


def select_interval(ds, ids_list, time_start, time_end):

    # Could be only one for loop
    for ids_key in ids_list:

        ds[ids_key] = ds[ids_key].where(ds[ids_key].time>time_start, drop = True)
        ds[ids_key] = ds[ids_key].where(ds[ids_key].time<time_end, drop = True)

        #ds[ids_key] = ds[ids_key].sel(ds[ids_key]time_summary=0.0300001)

        #dimx = self.ds.dimx.where(self.ds.normni.sel(nions=0) > normni_max, drop = True).values

        #index_time_start = np.abs(time - time_start).argmin(0)
        #index_time_end = np.abs(time - time_end).argmin(0)

        
    return ds

def average_integrated_modelling_duqtools(ds):

    for ids_key in ds:
        time = {}
        time['time'] = np.average(ds[ids_key].time)

        if 'rho_tor_norm' in ds[ids_key].coords:
            ds[ids_key] = fit_x(ds[ids_key])

        ds[ids_key] = ds[ids_key].mean(dim='time')

        ds[ids_key] = ds[ids_key].fillna(0)
        # Still to be tested with infs
        ds[ids_key] = ds[ids_key].where(ds[ids_key] != np.inf, 0, drop = False)
        ds[ids_key] = ds[ids_key].where(ds[ids_key] != -np.inf, 0, drop = False)

        ds[ids_key] = ds[ids_key].assign_coords(coords=time)
        ds[ids_key] = ds[ids_key].expand_dims({'time':1})

    return ds


def remove_point_rho_tor_norm(rho_tor_norm_stand):

    distances = []
    for p, e in zip(rho_tor_norm_stand, rho_tor_norm_stand[1:]):
        distances.append(abs(p-e))

    d_min = min(distances)

    for ip, p, e in zip(np.arange(np.shape(rho_tor_norm_stand)[0]), rho_tor_norm_stand, rho_tor_norm_stand[1:]):
        if abs(p-e) == d_min:
            rho_tor_norm_stand = np.delete(rho_tor_norm_stand, ip)
            break

    return rho_tor_norm_stand, d_min


def create_stand_rho_tor_norm(rho_tor_norm_time, len_rho_tor_norm, grid_option = 'keep length', distance = 0.0001):

    rho_tor_norm_stand = np.sort(rho_tor_norm_time.values, axis = None)

    if grid_option == 'distance':
        d_min = 0
        while(d_min<distance):
            rho_tor_norm_stand, d_min = remove_point_rho_tor_norm(rho_tor_norm_stand)

    elif grid_option == 'keep length':
        while(np.shape(rho_tor_norm_stand)[0] != len_rho_tor_norm):
            rho_tor_norm_stand, d_min = remove_point_rho_tor_norm(rho_tor_norm_stand)

    elif grid_option == 'all':
        d_min = 0
        while(d_min<1.0e-10):
            rho_tor_norm_stand, d_min = remove_point_rho_tor_norm(rho_tor_norm_stand)

    return rho_tor_norm_stand


def fit_x(ds_ids):

    rho_tor_norm_stand = create_stand_rho_tor_norm(ds_ids['rho tor norm time'], ds_ids.dims['rho_tor_norm'], grid_option = 'keep length', distance = 0.0001)

    ds_ids_new = copy.deepcopy(ds_ids)

    # This purges everything that has a rho_tor_norm dimension
    ds_ids_new = ds_ids_new.drop_dims('rho_tor_norm')

    # Create a new standardize rho tor norm to allow interpolation in time
    coord_rho_tor_norm_stand = {}
    coord_rho_tor_norm_stand['rho_tor_norm'] = rho_tor_norm_stand
    ds_ids_new = ds_ids_new.assign_coords(coord_rho_tor_norm_stand)

    for variable in ds_ids:
        if 'rho_tor_norm' in ds_ids[variable].coords:
            variable_new = np.asarray([])
            dims_variable = [ds_ids[variable].sizes[dim_variable] for dim_variable in ds_ids[variable].dims]
            dims_variable_names = [dim_variable for dim_variable in ds_ids[variable].dims]
            coords_variable = [coord_variable for coord_variable in ds_ids[variable].coords]

            dims_variable_stripped = copy.deepcopy(dims_variable_names)
            dims_variable_stripped.remove('rho_tor_norm')

            # Handling the case when there dimensions other than time
            dim_reshape = 1
            for dim_variable_stripped in dims_variable_stripped:
                dim_reshape *= ds_ids[variable].sizes[dim_variable_stripped]

            dim_reshape_rho_tor_norm = dim_reshape/ds_ids[variable].sizes['time']
            flatten_variable = ds_ids[variable].values.reshape(dim_reshape, ds_ids[variable].sizes['rho_tor_norm'])

            # Build rho tor norm time so it fits with the variable when interpolating
            rho_tor_norm_time = np.asarray([])
            for rho_tor_norm_time_slice in ds_ids['rho tor norm time']:
                for i in np.arange(dim_reshape_rho_tor_norm):
                    rho_tor_norm_time = np.hstack((rho_tor_norm_time, rho_tor_norm_time_slice))

            rho_tor_norm_time = rho_tor_norm_time.reshape(dim_reshape, ds_ids[variable].sizes['rho_tor_norm'])

            for slice_rho_tor_norm, slice_variable in zip(rho_tor_norm_time, flatten_variable):
                slice_new = fit_and_substitute(slice_rho_tor_norm, rho_tor_norm_stand, slice_variable)
                variable_new = np.hstack((variable_new, slice_new))

            variable_new = np.reshape(variable_new, dims_variable)
            ds_ids_new[variable] = (dims_variable_names, variable_new)

    return ds_ids_new


# Rebasing the profiles in time
def rebase_integrated_modelling_duqtools(ds, changing_idss, time_new):
    ds_new = {}
    for changing_ids in changing_idss:
        ds[changing_ids] = fit_x(ds[changing_ids])

        # Interpolate over time here
        ds_new[changing_ids] = ds[changing_ids].interp(time=time_new, kwargs={'fill_value':'extrapolate'})

        # test on ti
        #index_time = np.abs(ds[changing_ids].time.values - time_new[1]).argmin(0)
        #ds_new.isel(time=1).plot.scatter(x='rho_tor_norm',y='t_i_ave')
        #ds[changing_ids].isel(time=index_time).plot.scatter(x='rho_tor_norm',y='t_i_ave')        
        #plt.show()

    return ds_new


'''
--------------- AVAILABLE FUNCTIONS: ------------------


1 - setup_input_baserun(db, shot, run_exp, run_input, zeff_option = None, instructions = [], time_start = 0, time_end = 100)
'''

# passing json file name now. Maybe would be better to add the option to pass the dictionary already...

def setup_input(db, shot, run_input, run_start, json_input, time_start = 0, time_end = 100, force_input_overwrite = False, verbose = False, core_profiles = None, equilibrium = None):

    '''

    Instructions is a list of strings with a list of actions that should be performed.
    Possibilities are: average, rebase, nbi heating, flat q profile, correct zeff, correct boundaries, flipping ip

    zeff_option needs a string with a key word about what to do with zeff.
    Possibilities are: None, flat maximum, flat minimum, flat median, impurity from flattop. Correct zeff is an extra action, needs to be in istructions

    '''

    username=getpass.getuser()
    ids_names_to_extract = ['core_profiles', 'equilibrium', 'summary']

    dataset_integrated_modelling = extract_integrated_modelling(ids_names_to_extract, username, db, shot, run_input)

    dataset_integrated_modelling = select_interval(dataset_integrated_modelling, ['equilibrium'], 0.2, 0.3)

    #rebase_integrated_modelling_duqtools(dataset_integrated_modelling, ['equilibrium'], [0.1,0.2,0.3])
    rebase_integrated_modelling_duqtools(dataset_integrated_modelling, ['equilibrium'], [0.1,0.2,0.3])

    dataset_integrated_modelling = average_integrated_modelling_duqtools(dataset_integrated_modelling)


    exit()

    '''
    #handle = ImasHandle(user = username, db = 'tcv', shot = 64862, run = 5)
    #print(handle.path())
    #handle.exists()

    target = ImasHandle(user = username, db = 'tcv', shot = 64862, run = 5000)

    #handle.copy_data_to(target)
    #target.exists()

    core_profiles = target.get('core_profiles')
    t_i_average = core_profiles.get('profiles_1d/1/t_i_average')

    a = core_profiles.get('profiles_1d/*/ions/*/z')

    #average, rebase, nbi_heating, flat_q_profile = False, False, False, False
    #correct_Zeff, set_boundaries, correct_boundaries, adding_early_profiles = False, False, False, False

    variable = [
        Variable(name='ion_temperature',
                 ids = 'core_profiles',
                 path = 'profiles_1d/*/ion/*/z_ion',
                 dims = ['time', 'ion']),
        't_i_ave',
        'rho_tor_norm',
        'time'
    ]

    #merge_data(handle, target, 'zeff'):
    #core_profiles = target.get('core_profiles')
    zeff = target.get_variables(['zeff','rho_tor_norm','time'])

    dataset_all = target.get_all_variables()

    print(dataset_all)

    dataset = target.get_variables(variable)

    # This will rewrite the IDS
    #core_profiles.sync(target)

    times = [0.1,0.2,0.3]

    rebased_dataset = rebase_on_time(dataset, time_dim='time', new_coords = times)

    print(rebased_dataset)
    '''
    # And then add strings and special cases separately

    # Need to import variable_dict
    #variables = variable_dict.values()

    #target = ImasHandle(user = username, db = db, shot = shot, run = run_input)
    #zeff = target.get_variables(['zeff','rho_tor_norm','time'])

    #print(zeff)

    #exit()

    Zeff = ZeffIntegratedModelling(db, shot, run_input, run_start)
    option = 'median'
    #Zeff.set_flat_zeff(option)

    zeff_param = 3
    Zeff.set_parabolic_zeff(zeff_param = zeff_param)

    exit()


    # Checking that everything is fine with the input options
    if json_input['zeff profile'] not in json_input['zeff profile options']:
        print('Unrecognized zeff profile options. Do not know what to do. Aborting')
        exit()

    if json_input['zeff evolution'] not in json_input['zeff evolution options']:
        print('Unrecognized zeff evolution options. Do not know what to do. Aborting')
        exit()

    if json_input['instructions']['set boundaries']:
        if json_input['boundary instructions']['method te'] not in json_input['boundary instructions']['method options']:
            print('Unrecognized boundary options. Do not know what to do. Aborting')
            exit()

    # To save computational time, equilibrium and core profiles might be passed if they have already been extracted
    if not core_profiles:
        core_profiles = open_and_get_core_profiles(db, shot, run_input)
    if not equilibrium:
        equilibrium = open_and_get_equilibrium(db, shot, run_input)

    # Boundaries instructions are read only if 'set boundaries' is true (Moved in the function itself)
    #if json_input['instructions']['set boundaries']:
    #    boundary_method_te = json_input['boundary instructions']['method te']
    #    boundary_method_ti = json_input['boundary instructions']['method ti']
    #    boundary_sep_te = json_input['boundary instructions']['te sep']
    #    if json_input['boundary instructions']['ti sep']:
    #        boundary_sep_ti = json_input['boundary instructions']['ti sep']
    #    else:
    #        boundary_sep_ti = False

    ip = equilibrium.time_slice[0].global_quantities.ip
    b0 = equilibrium.vacuum_toroidal_field.b0

    # Trying to deactivate the ip flipping, seems to work without...
    if json_input['instructions']['flipping ip'] == 'auto':
        flipping_ip = False
        json_input['instructions']['flipping ip'] = False
        #if ip > 0:
        #    flipping_ip = True
        #    json_input['instructions']['flipping ip'] = True
        #    print('ip will be flipped. Still necessary because of bugs but should not happen')
    else:
        flipping_ip = json_input['instructions']['flipping ip']

    if json_input['instructions']['average'] and json_input['instructions']['rebase']:
        print('rebase and average cannot be done at the same time. Aborting')
        exit()

    # Counting how many operations will be performed
    generated_idss_length = 0
    for key in json_input['instructions']:
        if json_input['instructions'][key] != False:
            generated_idss_length +=1

    if json_input['zeff evolution'] != 'original':
        generated_idss_length += 1

    if json_input['zeff profile'] != 'flat':
        generated_idss_length += 1

    # Checking that all the idss that will be used are free
    if not force_input_overwrite:
        for index in range(run_start - generated_idss_length, run_start, 1):
            data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, index, user_name=username)
            op = data_entry.open()

            if op[0]==0:
                print('One of the data entries needed to manipulate the input already exists, aborting. Try increasing run start.')
                exit()

            data_entry.close()

    run_start = run_start - generated_idss_length + 1

    time_eq = equilibrium.time
    time_cp = core_profiles.time

    if time_start == None:
        time_start = max(min(time_eq), min(time_cp))
    elif time_start == 'core_profile':
        time_start = min(time_cp)
    elif time_start == 'equilibrium':
        time_start = min(time_eq)
    else:
        time_start = time_start

    if json_input['instructions']['average']:
        average_integrated_modelling(db, shot, run_input, run_start, time_start, time_end)
        print('Averaging on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

    # Rebase sets the times of the equilibrium equal to the times in core profiles. Usually the grid in core profiles is more coarse.
    # This should help with the speed and with the noisiness in vloop

    elif json_input['instructions']['rebase']:
        rebase_option, rebase_num_times = json_input['rebase']['option'], json_input['rebase']['num times']
        rebase_integrated_modelling(db, shot, run_input, run_start, ['equilibrium'], option = rebase_option, num_times = rebase_num_times)
        print('Rebasing on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

    # If necessary, flipping ip here. Runs for sensitivities should be possible from this index (unless Zeff is weird)

    if json_input['instructions']['flipping ip']:
        # Currently flips both Ip and b0
        flip_ip(db, shot, run_input, shot, run_start)
        print('flipping ip on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

        # option of flipping the q profile might be added here
        #flip_q_profile(db, shot, run_input, run_start)
        #print('flipping q profile on index ' + str(run_start))
        #run_input, run_start = run_start, run_start+1

    # Temporarily like this. The function itself will need to be updated
    if json_input['instructions']['nbi heating']:
        prepare_nbi(db, shot, run_input, shot, run_start)
        print('Preparing nbi on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

    ion_number = check_ion_number(db, shot, run_input)

    if json_input['instructions']['add early profiles']:
        add_early_profiles(db, shot, run_input, run_start, extra_early_options = json_input['extra early options'])
        print('Adding early profiles on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

    if json_input['instructions']['set boundaries']:
        #set_boundaries(db, shot, run_input, run_start, te_sep = boundary_sep_te, ti_sep = boundary_sep_ti, method_te = boundary_method_te, method_ti = boundary_method_ti, bound_te_down = json_input['boundary instructions']['te bound down'], bound_te_up = json_input['boundary instructions']['te bound up'])
        set_boundaries(db, shot, run_input, run_start, extra_boundary_instructions = json_input['boundary instructions'])
        print('Setting boundaries te and ti on index ' + str(run_start))
        run_input, run_start = run_start, run_start + 1

    if json_input['instructions']['peak temperature']:
        peak_temperature(db, shot, run_input, run_start, mult = json_input['instructions']['peak temperature'])
        print('Peaking temperature on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

    if json_input['instructions']['multiply electron temperature']:
        shift_profiles('te', db, shot, run_input, run_start, mult = json_input['instructions']['multiply electron temperature'])
        print('Multipling electron temperature on index ' + str(run_start))
        run_input, run_start = run_start, run_start + 1

    if json_input['instructions']['multiply ion temperature']:
        shift_profiles('ti', db, shot, run_input, run_start, mult = json_input['instructions']['multiply ion temperature'])
        print('Multipling ion temperature on index ' + str(run_start))
        run_input, run_start = run_start, run_start + 1

    if json_input['instructions']['correct ion temperature']:
        correct_ion_temperature(db, shot, run_input, run_start, ratio_limit = json_input['instructions']['correct ion temperature'])
        print('Correcting the ion temperature ratio on index ' + str(run_start))
        run_input, run_start = run_start, run_start + 1

    if json_input['instructions']['multiply electron density']:
        shift_profiles('ne', db, shot, run_input, run_start, mult = json_input['instructions']['multiply electron density'])
        print('Multipling electron density on index ' + str(run_start))
        run_input, run_start = run_start, run_start + 1

    if json_input['instructions']['multiply q profile']:
        alter_q_profile_same_q95(db, shot, run_input, run_start, mult = json_input['instructions']['multiply q profile'])
        print('Multipling q profile, maintaining q95, on index ' + str(run_start))
        run_input, run_start = run_start, run_start + 1

    if json_input['instructions']['correct boundaries']:
        correct_boundaries_te(db, shot, run_input, db, shot, run_start)
        print('Correcting te at the boundaries on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

    #if 'add early profiles' in instructions:
    #    add_early_profiles(db, shot, run_input, run_start)
    #    print('Adding early profiles on index ' + str(run_start))
    #    run_input, run_start = run_start, run_start+1

    if json_input['zeff evolution'] != 'original':
        zeff_option = json_input['zeff evolution']
        if zeff_option == 'flat maximum':
            set_flat_Zeff(db, shot, run_input, run_start, 'maximum')
            print('Setting flat Zeff with maximum value on index ' + str(run_start))
            run_input, run_start = run_start, run_start+1
        elif zeff_option == 'flat minimum':
            set_flat_Zeff(db, shot, run_input, run_start, 'minimum')
            print('Setting flat Zeff with minimum value on index ' + str(run_start))
            run_input, run_start = run_start, run_start+1
        elif zeff_option == 'flat median':
            set_flat_Zeff(db, shot, run_input, run_start, 'median')
            print('Setting flat Zeff with median value on index ' + str(run_start))
            run_input, run_start = run_start, run_start+1
        elif ion_number > 1 and not json_input['instructions']['average'] and zeff_option == 'impurity from flattop':
            set_impurity_composition_from_flattop(db, shot, run_input, run_start, verbose = verbose)
            print('Setting impurity composition from flattop on index ' + str(run_start))
            run_input, run_start = run_start, run_start+1
        elif ion_number > 1 and json_input['instructions']['average'] and zeff_option == 'impurity from flattop':
            print('Cannot extract the impurities from flattop when averaging')
            exit()
        elif ion_number > 1 and not json_input['instructions']['average'] and zeff_option == 'linear descending zeff':
            set_linear_descending_zeff(db, shot, run_input, run_start)
            print('Setting descending initial impurity composition on index ' + str(run_start))
            run_input, run_start = run_start, run_start+1
        elif ion_number > 1 and not json_input['instructions']['average'] and zeff_option == 'ip ne scaled':
            set_ip_ne_scaled_zeff(db, shot, run_input, run_start)
            print('Setting impurity composition using ne and ip scaling on index ' + str(run_start))
            run_input, run_start = run_start, run_start+1
        elif ion_number > 1 and not json_input['instructions']['average'] and zeff_option == 'hyperbole':
            set_hyperbole_zeff(db, shot, run_input, run_start, zeff_param = json_input['zeff evolution parameter'], zeff_max = json_input['zeff max evolution'])
            print('Setting descending hiperbolic initial impurity composition on index ' + str(run_start))
            run_input, run_start = run_start, run_start+1
        else:
            print('Option for Zeff initialization not recognized. Aborting generation')
            exit()

    if json_input['zeff profile'] == 'parabolic zeff':
        set_parabolic_zeff(db, shot, run_input, run_start, zeff_param = json_input['zeff profile parameter'])
        print('Setting parabolic zeff profile on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1
    elif json_input['zeff profile'] == 'peaked zeff':
        set_peaked_zeff_profile(db, shot, run_input, run_start, zeff_param = json_input['zeff profile parameter'])
        print('Setting peaked zeff profile on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1
    elif json_input['zeff profile'] == 'peaked zeff evolved':
        set_peaked_ev_zeff_profile(db, shot, run_input, run_start, zeff_param = json_input['zeff profile parameter'])
        print('Setting peaked zeff profile on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1
    elif json_input['zeff profile'] == 'low edge zeff':
        set_low_edge_zeff(db, shot, run_input, run_start, zeff_param = json_input['zeff profile parameter'])
        print('Setting zeff profile low at the edge on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

    if json_input['instructions']['correct zeff']:
        correct_zeff(db, shot, run_input, db, shot, run_start)
        print('correcting zeff on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

    if json_input['instructions']['flat q profile']:
        use_flat_q_profile(db, shot, run_input, run_start)
        print('setting a flat q profile on index ' + str(run_start))
        run_input, run_start = run_start, run_start+1

    # To save time, core_profiles and equilibrium are saved and passed

    return core_profiles, equilibrium


class IntegratedModellingDict:

    def __init__(self, db, shot, run, username='', backend='mdsplus'):

        # It might be possible to generalize part of the following functions. Left for future work
        # The extraction might also be simplified using partial_get. I am not sure if the filling part can also be simplified...

        # ------------------------------ LITS WITH ALL THE KEYS -----------------------------------
        self.db = db
        self.shot = shot
        self.run = run
        self.all_keys = copy.deepcopy(keys_list)
        self.ids_struct = open_integrated_modelling(db, shot, run, username=username, backend=backend)
        self.extract_ids_dict()

    def extract_ids_dict(self):
    
        self.ids_dict = {'time' : {}, 'traces' : {}, 'profiles_1d' : {}, 'profiles_2d' : {}, 'extras' : {}}
    
        # -------------------- Extract summary -----------------------------------
    
        traces, time = self.extract_summary()
    
        self.ids_dict['traces'] = {**self.ids_dict['traces'], **traces}
        self.ids_dict['time']['summary'] = time
    
        # -------------------- Extract core profiles -----------------------------
    
        #Maybe also everything in grid?
    
        profiles_1d, traces, time = self.extract_core_profiles()
    
        self.ids_dict['profiles_1d'] = {**self.ids_dict['profiles_1d'], **profiles_1d}
        self.ids_dict['traces'] = {**self.ids_dict['traces'], **traces}
        self.ids_dict['time']['core_profiles'] = time
    
        # --------------------- Extract equilibrium ------------------------------
    
        profiles_2d, profiles_1d, traces, extras, time = self.extract_equilibrium()

        self.ids_dict['profiles_2d'] = {**self.ids_dict['profiles_2d'], **profiles_2d}
        self.ids_dict['profiles_1d'] = {**self.ids_dict['profiles_1d'], **profiles_1d}
        self.ids_dict['traces'] = {**self.ids_dict['traces'], **traces}
        self.ids_dict['extras'] = {**self.ids_dict['extras'], **extras}         # Might not need this one when new_classes actually becomes a class
        self.ids_dict['time']['equilibrium'] = time
    
        # --------------------- Extract core_sources ----------------------------

        if 'core_sources' in self.ids_struct:   
            profiles_1d, traces, time = self.extract_core_sources()
    
            self.ids_dict['profiles_1d'] = {**self.ids_dict['profiles_1d'], **profiles_1d}
            self.ids_dict['traces'] = {**self.ids_dict['traces'], **traces}
            self.ids_dict['time']['core_sources'] = time
    
        # --------------------- Extract nbi data --------------------------------
        
        if 'nbi' in self.ids_struct:

            traces, time = self.extract_nbi()
    
            self.ids_dict['traces'] = {**self.ids_dict['traces'], **traces}
            self.ids_dict['time']['nbi'] = time
        
        # Pellets maybe in the future
    
    def extract_core_profiles(self):
    
        profiles_1d, traces = {}, {}
        ids_iden = 'core_profiles'
    
        for tag in keys_list['profiles_1d']['core_profiles']:
            parts = tag.split('[')
            # Initializing profiles
            if len(parts) == 1:
                profiles_1d[tag] = []
            elif len(parts) == 2:
    
                # A new list needs to be created, appending the keys including the values of the indexes of the nested elements
                self.all_keys['profiles_1d']['core_profiles'].append(parts[0] + '[' + str(0) + parts[1])
                profiles_1d[parts[0] + '[' + str(0) + parts[1]] = []
    
            # Filling profiles
            for profile in self.ids_struct[ids_iden].profiles_1d:
                parts = tag.split('[')
                if len(parts) == 1:
                    point = eval('profile.' + tag)
                    profiles_1d[tag].append(point)
    
            for profile in self.ids_struct[ids_iden].profiles_1d:
                if len(parts) == 2:
    
                    array = eval('profile.' + parts[0])
                    for index, element in enumerate(array):
                        point = eval('element.' + parts[1][2:])
                        if parts[0] + '[' + str(index) + parts[1] not in profiles_1d:
                            self.all_keys['profiles_1d']['core_profiles'].append(parts[0] + '[' + str(index) + parts[1])  # adding the keys
                            profiles_1d[parts[0] + '[' + str(index) + parts[1]] = [point]
                        else:
                            profiles_1d[parts[0] + '[' + str(index) + parts[1]].append(point)
    
        # The values need to be arrays, not lists
        for key in profiles_1d:
            profiles_1d[key] = np.asarray(profiles_1d[key])
    
        for tag in keys_list['traces']['core_profiles']:
            parts = tag.split('[')
            if len(parts) == 1:
                traces[tag] = eval('self.ids_struct[ids_iden].global_quantities.'+ tag)
                traces[tag] = np.asarray(traces[tag])
    
            if len(parts) == 2:
                for profile in self.ids_struct[ids_iden].profiles_1d:
                    array = eval('profile.' + parts[0])
                    for index, element in enumerate(array):
                        point = eval('element.' + parts[1][2:])
                        if parts[0] + '[' + str(index) + parts[1] not in traces:
                            self.all_keys['traces']['core_profiles'].append(parts[0] + '[' + str(index) + parts[1])  # adding the keys
                            traces[parts[0] + '[' + str(index) + parts[1]] = [point]
                        else:
                            traces[parts[0] + '[' + str(index) + parts[1]].append(point)

                # atoms_n and multiple_states_flag need to be int type

                index = 0
                while parts[0] + '[' + str(index) + parts[1] in self.all_keys['traces']['core_profiles']:
                    if tag.endswith('multiple_states_flag'):
                        traces[parts[0] + '[' + str(index) + parts[1]] = np.asarray([int(point) for point in traces[parts[0] + '[' + str(index) + parts[1]]])
                    else:
                        # Avoid transformation to np.str_
                        if type(traces[parts[0] + '[' + str(index) + parts[1]][0]) != str:
                            traces[parts[0] + '[' + str(index) + parts[1]] = np.asarray(traces[parts[0] + '[' + str(index) + parts[1]])
                    index += 1

            elif len(parts) == 3:
                for profile in self.ids_struct[ids_iden].profiles_1d:
                    array1 = eval('profile.' + parts[0])
                    for index1, element1 in enumerate(array1):
                        array2 = eval('element1.' + parts[1][2:])
                        for index2, element2 in enumerate(array2):
                            point = eval('element2.' + parts[2][2:])
                            if parts[0] + '[' + str(index1) + parts[1]+ '[' + str(index2) + parts[2] not in self.all_keys['traces']['core_profiles']:
                                # Adding the keys
                                self.all_keys['traces']['core_profiles'].append(parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2])
                                traces[parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2]] = [point]
                            else:
                                traces[parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2]].append(point)

                index1 = 0
                while parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2] in self.all_keys['traces']['core_profiles']:
                    while parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2] in self.all_keys['traces']['core_profiles']:
                        if tag.endswith('atoms_n'):
                            traces[parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2]] = np.asarray([int(point) for point in traces[parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2]]])
                        else:
                            traces[parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2]] = np.asarray(traces[parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2]])
                        index2 += 1
                    index2 = 0
                    index1 += 1

        time = self.ids_struct[ids_iden].time

        return profiles_1d, traces, time
    
    
    def extract_summary(self):
    
        traces = {}
        ids_iden = 'summary'

        for tag in keys_list['traces']['summary']:
            traces[tag] = eval('self.ids_struct[ids_iden].'+ tag)
    
        time = self.ids_struct[ids_iden].time

        # The n_e line averaged, at least for TCV, does not have data very early or very late. Will interpolate missing data.
        # Will assume that the density is not zero at the very beginning and very end. 0.5e19 is arbitrary...
        # This could be a single function
        # It is not necessary if the first value is already 0 (It would actually cause problems with the interpolation)

        if 'line_average.n_e.value' in traces:
            time_ne_average = time[np.where(np.isnan(traces['line_average.n_e.value']), False, True)]
            ne_average = traces['line_average.n_e.value'][np.where(np.isnan(traces['line_average.n_e.value']), False, True)]

            if np.size(traces['line_average.n_e.value']) != 0 and time_ne_average[0] != 0:
                #traces['line_average.n_e.value'] = fit_and_substitute(time_ne_average, time, ne_average)

                #if traces['line_average.n_e.value'][0] < 0:
                time_ne_average = np.insert(time_ne_average, 0, 0)
                ne_average = np.insert(ne_average, 0, 0.5e19)

                #if traces['line_average.n_e.value'][-1] < 0:
                time_ne_average = np.insert(time_ne_average, -1, time[-1]+0.1)
                ne_average = np.insert(ne_average, -1, 0.5e19)

                traces['line_average.n_e.value'] = fit_and_substitute(time_ne_average, time, ne_average)

        return traces, time
    
    
    def extract_equilibrium(self):
    
        profiles_2d, profiles_1d, traces, extras = {}, {}, {}, {}
        ids_iden = 'equilibrium'    

        for tag in keys_list['profiles_1d'][ids_iden]:
            parts = tag.split('[')
            # Initializing profiles
            if len(parts) == 1:
                profiles_1d[tag] = []
            elif len(parts) == 2:
    
                # A new list needs to be created, appending the keys including the values of the indexes of the nested elements
                self.all_keys['profiles_1d'][ids_iden].append(parts[0] + '[' + str(0) + parts[1])
                profiles_1d[parts[0] + '[' + str(0) + parts[1]] = []
    
            # Filling profiles
            for time_slice in self.ids_struct[ids_iden].time_slice:
                parts = tag.split('[')
                if len(parts) == 1:
                    point = eval('time_slice.' + tag)
                    profiles_1d[tag].append(point)
                elif len(parts) == 2:
                    point = eval('time_slice.' + parts[0] + '[' + str(0) + parts[1])
                    profiles_1d[parts[0] + '[' + str(0) + parts[1]].append(point)
    
        # Transforming to numpy arrays
        for key in profiles_1d:
            profiles_1d[key] = np.asarray(profiles_1d[key])
    
        # Filling traces
        for tag in keys_list['traces'][ids_iden]:
            parts = tag.split('[')
            if len(parts) == 1:
                traces[tag] = []
                for time_slice in self.ids_struct[ids_iden].time_slice:
                    point = eval('time_slice.' + tag)
                    traces[tag].append(point)

            if len(parts) == 2:
                for time_slice in self.ids_struct[ids_iden].time_slice:
                    array = eval('time_slice.' + parts[0])
                    for index, element in enumerate(array):
                        point = eval('element.' + parts[1][2:])
                        if parts[0] + '[' + str(index) + parts[1] not in traces:
                            self.all_keys['traces'][ids_iden].append(parts[0] + '[' + str(index) + parts[1])  # adding the key

                            traces[parts[0] + '[' + str(index) + parts[1]] = [point]
                        else:
                            traces[parts[0] + '[' + str(index) + parts[1]].append(point)

                # index needs to be an int
                index = 0
                while parts[0] + '[' + str(index) + parts[1] in self.all_keys['traces']['equilibrium']:
                    if tag.endswith('index'):
                        traces[parts[0] + '[' + str(index) + parts[1]] = np.asarray([int(point) for point in traces[parts[0] + '[' + str(index) + parts[1]]])
                    else:
                        # Avoid np.str_ type
                        if type(traces[parts[0] + '[' + str(index) + parts[1]][0]) != str:
                            traces[parts[0] + '[' + str(index) + parts[1]] = np.asarray(traces[parts[0] + '[' + str(index) + parts[1]])
                    index += 1

        # Transforming to numpy arrays. Not to be done when dealing with strings, imas does not recognize np.str_
        for key in traces:
            if type(traces[key][0]) != str:
                traces[key] = np.asarray(traces[key])
 
        for tag in keys_list['profiles_2d']['equilibrium']:
            parts = tag.split('[')
            # Initializing profiles
            if len(parts) == 1:
                profiles_2d[tag] = []
            elif len(parts) == 2:
                # A new list needs to be created, appending the keys including the values of the indexes of the nested elements
                self.all_keys['profiles_2d']['equilibrium'].append(parts[0] + '[' + str(0) + parts[1])
                profiles_2d[parts[0] + '[' + str(0) + parts[1]] = []

            for time_slice in self.ids_struct[ids_iden].time_slice:
                parts = tag.split('[')
                if len(parts) == 1:
                    point = eval('time_slice.' + tag)
                    profiles_2d[tag].append(point)
                elif len(parts) == 2:
                    point = eval('time_slice.' + parts[0] + '[' + str(0) + parts[1])
                    profiles_2d[parts[0] + '[' + str(0) + parts[1]].append(point)

        for key in profiles_2d:
            profiles_2d[key] = np.asarray(profiles_2d[key])

        time = self.ids_struct[ids_iden].time
    
        extras['b0'] = self.ids_struct[ids_iden].vacuum_toroidal_field.b0
        extras['r0'] = self.ids_struct[ids_iden].vacuum_toroidal_field.r0
    
        return profiles_2d, profiles_1d, traces, extras, time
    
    
    def extract_core_sources(self):
    
        profiles_1d, traces = {}, {}
        ids_iden = 'core_sources'
    
        for tag in keys_list['profiles_1d']['core_sources']:
            split = tag.split('#')
    
            if len(self.ids_struct[ids_iden].source) == 0:
                break
    
            i_source = self.get_index_source(split[0])
            parts = split[1].split('[')
    
            if len(parts) == 1:
                profiles_1d[tag] = []
            elif len(parts) == 2:
            # The new list is updated, appending the keys including the values of the indexes of the nested elements
                self.all_keys['profiles_1d']['core_sources'].append(parts[0] + '[' + str(0) + parts[1])
                profiles_1d[split[0] + '#' + parts[0] + '[' + str(0) + parts[1]] = []
    
            for profile in self.ids_struct[ids_iden].source[i_source].profiles_1d:
                if len(parts) == 1:
                    point = eval('profile.'+ split[1])
                    profiles_1d[split[0] + '#' + split[1]].append(point)
                elif len(parts) == 2:
                    point = eval('profile.' + parts[0] + '[' + str(0) + parts[1])
                    profiles_1d[split[0] + '#' + parts[0] + '[' + str(0) + parts[1]].append(point)
    
        for tag in keys_list['traces']['core_sources']:
            split = tag.split('#')
    
            if len(self.ids_struct[ids_iden].source) == 0:
                break
    
            i_source = self.get_index_source(split[0])
    
            parts = split[1].split('[')
    
             # How I would do it counting the number of tags and building the number of for loops accordingly
    
            if len(parts) == 1:
                traces[tag] = []
                for profile in self.ids_struct[ids_iden].source[i_source].profiles_1d:
                    point = eval('profile.' + tag)
                    self.ids_dicttraces[tag].append(point)
    
            if len(parts) == 2:
                for profile in self.ids_struct[ids_iden].source[i_source].profiles_1d:
                    array = eval('profile.' + parts[0])
                    for index, element in enumerate(array):
                        point = eval('element.' + parts[1][2:])
                        if not traces[split[0] + parts[0] + str(index) + parts[1]]:
                            self.all_keys['traces']['core_sources'].append(split[0] + '#' + parts[0] + '[' + str(index) + parts[1])  # adding the keys
                            traces[split[0] + '#' + parts[0] + '[' + str(index) + parts[1]] = [point]
                        else:
                            traces[split[0] + '#' + parts[0] + '[' + str(index) + parts[1]].append(point)
    
            if len(parts) == 3:
                for profile in self.ids_struct[ids_iden].source[2].profiles_1d:
                    array1 = eval('profile.' + parts[0])
                    for index1, element1 in enumerate(array1):
                        array2 = eval('element1.' + parts[1][2:])
                        for index2, element2 in enumerate(array2):
                            point = eval('element2.' + parts[2][2:])
                            if split[0] + '#' + parts[0] + '[' + str(index1) + parts[1] + str(index2) + '[' + parts[2] not in traces:
                                # Adding the keys
                                self.all_keys['traces']['core_sources'].append(split[0] + '#' + parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2])
                                traces[split[0] + '#' + parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2]] = [point]
                            else:
                                traces[split[0] + '#' + parts[0] + '[' + str(index1) + parts[1] + '[' + str(index2) + parts[2]].append(point)
    
        time = self.ids_struct[ids_iden].time
    
        return profiles_1d, traces, time
    
    
    #For now nbi works weird. Will fix later when I know how the nbi enters pencil
    def extract_nbi(self):
    
        profiles_1d, traces = {}, {}
        ids_iden = 'nbi' 

        times = {}

        for tag in keys_list['traces'][ids_iden]:

            parts = tag.split('[')
            for index, unit in enumerate(self.ids_struct[ids_iden].unit):
                traces[parts[0] + '[' + str(index) + parts[1]] = eval('unit' + parts[1][1:])
                self.all_keys['traces'][ids_iden].append(parts[0] + '[' + str(index) + parts[1])
                # Extract the time using the most coarse grid
                if tag.endswith('time'):
                    times[parts[0] + '[' + str(index) + parts[1]] = traces[parts[0] + '[' + str(index) + parts[1]]

        time = [0]*5 # Not very elegant
        # Extract the time using the most coarse grid
        # TODO: might want to add conditions on start and end times when choosing which time array to use for all the traces.
        for tag in times:
            if len(times[tag]) >= len(time) and len(times[tag]) != 0:
                time = times[tag]

        # Extract the time using the most fine grid. It is not that much data anyway
        for tag in self.all_keys['traces'][ids_iden]:
            if tag in traces.keys():
                if tag.endswith('time'):
                    traces[tag] = time
                    # Search for corresponding data to fit
                    tag_quantity = tag.replace('time', 'data')
                    # This excludes the case where the length of times is 0, so when an NBI is not used.
                    if len(traces[tag_quantity]) == 0:
                        traces[tag_quantity] = np.asarray([])
                    elif len(np.shape(traces[tag_quantity])) > 1:
                        new_quantity = np.asarray([])
                        for single_trace in traces[tag_quantity]:
                            new_quantity = np.hstack((new_quantity, fit_and_substitute(times[tag], time, single_trace)))
                        traces[tag_quantity] = new_quantity.reshape(np.shape(traces[tag_quantity])[0], np.shape(time)[0])
                    else:
                        traces[tag_quantity] = fit_and_substitute(times[tag], time, traces[tag_quantity])

        traces = self.set_nbi_consistency(traces, time)

        return traces, time

    # Correcting nbi traces for consistency here. Should be reusable if needed
    def set_nbi_consistency(self, traces, time):

        for tag in ['unit[].beam_current_fraction.data', 'unit[].beam_power_fraction.data']:

            parts = tag.split('[')
            index = 0
            while parts[0] + '[' + str(index) + parts[1] in traces.keys():
                key = parts[0] + '[' + str(index) + parts[1]
                # Fill with two dimensional data with right shape and zeros if a unit is not used for the shot.
                # Need to search a unit that is not empty
                if np.size(traces[key]) == 0:
                    index_search, size_unit = 0, 0
                    while (parts[0] + '[' + str(index_search) + parts[1] in traces.keys()) and (size_unit == 0):
                        size_unit = np.size(traces[parts[0] + '[' + str(index_search) + parts[1]])
                        shape_unit = np.shape(traces[parts[0] + '[' + str(index_search) + parts[1]])
                        index += 1
                    # If the size is still 0 and you are trying to set up the nbi you have a problem, because there are no fractions available
                    if size_unit == 0:
                        # Should be a raise error
                        print('Warning, your NBI IDS does not have beam_current_fraction or beam_power_fraction. The generation will stop')
                        exit()
                    else:
                        traces[key] = np.zeros(shape_unit)

                # Values should be bounded between 0 and 1.
                traces[key][traces[key] < 0] = 0
                traces[key][traces[key] > 1] = 1
                # Values should add up to 1.
                traces[key][2,:] = 1 - traces[key][0,:] - traces[key][1,:]
                
                index += 1

        # Energy and power should not be negative
        for tag in ['unit[].energy.data', 'unit[].power_launched.data']:
            parts = tag.split('[')
            index = 0
            while parts[0] + '[' + str(index) + parts[1] in traces.keys():
                key = parts[0] + '[' + str(index) + parts[1]
                if np.size(traces[key]) == 0:
                    traces[key] = np.zeros(np.shape(time))
                traces[key][traces[key] < 0] = 0
                index += 1

        # Try to kill spurious extrapolated data by setting energy and power to 0 when the power is really small
        parts_power = 'unit[].power_launched.data'.split('[')
        parts_energy = 'unit[].energy.data'.split('[')
        parts_beam_fraction = 'unit[].beam_current_fraction.data'.split('[')
        parts_power_fraction = 'unit[].beam_power_fraction.data'.split('[')

        index = 0
        dummy_fractions = [1.0, 0.0, 0.0]
        while parts_power[0] + '[' + str(index) + parts_power[1] in traces.keys():
            key_power = parts_power[0] + '[' + str(index) + parts_power[1]
            key_energy = parts_energy[0] + '[' + str(index) + parts_energy[1]
            key_beam_fraction = parts_beam_fraction[0] + '[' + str(index) + parts_beam_fraction[1]
            key_power_fraction = parts_power_fraction[0] + '[' + str(index) + parts_power_fraction[1]

            traces[key_power][traces[key_power] < 10000] = 0
            traces[key_energy][traces[key_power] < 10000] = 0
            # Should substitute the fractions in the same way here, with the fractions at the first timestep

            for i, dummy_fraction in enumerate(dummy_fractions):
                traces[key_beam_fraction][i,:][traces[key_power] < 10000] = dummy_fraction
                traces[key_power_fraction][i,:][traces[key_power] < 10000] = dummy_fraction

            index +=1

        return traces
    
    # ------------------------------------ MANIPULATION -------------------------------
    
    def select_interval(self, time_start, time_end):
        
        # Could be only one for loop
        for element_key in self.ids_dict['traces'].keys():
            for ids_key in ids_list:
                if element_key in self.all_keys['traces'][ids_key]:
                
                    time = self.ids_dict['time'][ids_key]
    
                    index_time_start = np.abs(time - time_start).argmin(0)
                    index_time_end = np.abs(time - time_end).argmin(0)
    
                    self.ids_dict['traces'][element_key] = self.ids_dict['traces'][element_key][index_time_start:index_time_end]
    
        for element_key in self.ids_dict['profiles_1d'].keys():
            for ids_key in ids_list:
                if element_key in self.all_keys['profiles_1d'][ids_key]:
    
                    time = self.ids_dict['time'][ids_key]
    
                    index_time_start = np.abs(time - time_start).argmin(0)
                    index_time_end = np.abs(time - time_end).argmin(0)
    
                    self.ids_dict['profiles_1d'][element_key] = self.ids_dict['profiles_1d'][element_key][index_time_start:index_time_end]

        for element_key in self.ids_dict['profiles_2d'].keys():
            # profiles 2d only availabe in equilibrium
            if element_key in self.all_keys['profiles_2d']['equilibrium']:

                time = self.ids_dict['time']['equilibrium']

                index_time_start = np.abs(time - time_start).argmin(0)
                index_time_end = np.abs(time - time_end).argmin(0)

                self.ids_dict['profiles_2d'][element_key] = self.ids_dict['profiles_2d'][element_key][index_time_start:index_time_end]

        # Also fill b0. Might want to change this in the future (but b0 in traces)
        time = self.ids_dict['time']['equilibrium']
        index_time_start = np.abs(time - time_start).argmin(0)
        index_time_end = np.abs(time - time_end).argmin(0)
        self.ids_dict['extras']['b0'] = self.ids_dict['extras']['b0'][index_time_start:index_time_end]

        for ids_key in ids_list:
            if ids_key in self.ids_dict['time'].keys():
                if len(self.ids_dict['time'][ids_key]) != 0:
                    time = self.ids_dict['time'][ids_key]
    
                    index_time_start = np.abs(time - time_start).argmin(0)
                    index_time_end = np.abs(time - time_end).argmin(0)
    
                    self.ids_dict['time'][ids_key] = self.ids_dict['time'][ids_key][index_time_start:index_time_end]
    
        self.fill_ids_struct()

    def average_traces_profile(self):
    
        for key in self.ids_dict['traces']:
            if len(self.ids_dict['traces'][key]) != 0:
                if type(self.ids_dict['traces'][key][0]) == str or type(self.ids_dict['traces'][key][0]) == np.str_:
                    # np.str_ is not recognized by imas. Signals are converted to str.
                    self.ids_dict['traces'][key] = [str(self.ids_dict['traces'][key][0])]
                elif type(self.ids_dict['traces'][key][0]) == int or type(self.ids_dict['traces'][key][0]) == np.int_:
                    self.ids_dict['traces'][key] = np.asarray([int(np.average(self.ids_dict['traces'][key]))])
                else:
                # The value is given in an array, helps having the same structure later when I will have to deal with more values
                    self.ids_dict['traces'][key] = np.asarray([np.average(self.ids_dict['traces'][key])])
            else:
                self.ids_dict['traces'][key] = np.asarray([])
    
        # averaging profiles
        for key in self.ids_dict['profiles_1d']:
            if len(self.ids_dict['profiles_1d'][key][0]) != 0:
                # Modified to eliminate infs. Might find a more elegant way to do it though
                self.ids_dict['profiles_1d'][key][self.ids_dict['profiles_1d'][key] == np.inf] = 0
                self.ids_dict['profiles_1d'][key][self.ids_dict['profiles_1d'][key] == -np.inf] = 0
    
                self.ids_dict['profiles_1d'][key] = np.average(np.transpose(np.asarray(self.ids_dict['profiles_1d'][key])), axis = 1)
                self.ids_dict['profiles_1d'][key] = np.reshape(self.ids_dict['profiles_1d'][key], (1,len(self.ids_dict['profiles_1d'][key])))
            else:
                self.ids_dict['profiles_1d'][key] = np.asarray([])

        # averaging 2d profiles
        for key in self.ids_dict['profiles_2d']:
            if len(self.ids_dict['profiles_2d'][key][0]) != 0:
                # Modified to eliminate infs. Might find a more elegant way to do it though
                self.ids_dict['profiles_2d'][key][self.ids_dict['profiles_2d'][key] == np.inf] = 0
                self.ids_dict['profiles_2d'][key][self.ids_dict['profiles_2d'][key] == -np.inf] = 0

                self.ids_dict['profiles_2d'][key] = np.average(self.ids_dict['profiles_2d'][key], axis = 0)
                self.ids_dict['profiles_2d'][key] = np.reshape(self.ids_dict['profiles_2d'][key], (1,np.shape(self.ids_dict['profiles_2d'][key])[0], np.shape(self.ids_dict['profiles_2d'][key])[1]))

            else:
                self.ids_dict['profiles_2d'][key] = np.asarray([])
    
        #Selecting the first time to be the place holder. Could also select the middle time
        for ids_key in ids_list:
            if ids_key in self.ids_dict['time'].keys():
                if len(self.ids_dict['time'][ids_key]) != 0:
                    self.ids_dict['time'][ids_key] = np.asarray([self.ids_dict['time'][ids_key][0]])
    
        self.ids_dict['extras']['b0'] = np.asarray([np.average(self.ids_dict['extras']['b0'])])
        self.fill_ids_struct()


    def update_times_traces(self, new_times, changing_idss):
    
        '''
    
        Changes the traces in a list of IDSs to a new time base
    
        '''

        for changing_ids in changing_idss:

            for key in self.ids_dict['traces'].keys():
                # It handles the interpolations for when strings are involved (the usecase I found is with labels)
                #if key in self.all_keys['traces'][changing_ids] and 'label' in key:
                if key in self.all_keys['traces'][changing_ids] and type(self.ids_dict['traces'][key][0]) == str:
                    self.ids_dict['traces'][key] = [str(a) for a in np.full(len(new_times), self.ids_dict['traces'][key][0])]

                # WORK IN PROGRESS

                elif key in self.all_keys['traces'][changing_ids] and type(self.ids_dict['traces'][key][0]) == np.int_:
                    self.ids_dict['traces'][key] = [int(a) for a in np.full(len(new_times), self.ids_dict['traces'][key][0])]


                elif key in self.all_keys['traces'][changing_ids] and len(self.ids_dict['traces'][key]) != 0:
                    self.ids_dict['traces'][key] = fit_and_substitute(self.ids_dict['time'][changing_ids], new_times, self.ids_dict['traces'][key])

        if 'equilibrium' in changing_idss:
            self.ids_dict['extras']['b0'] = fit_and_substitute(self.ids_dict['time']['equilibrium'], new_times, self.ids_dict['extras']['b0'])


    def update_times_profiles(self, new_times, changing_idss):
    
        # Rebasing the profiles in time
        for changing_ids in changing_idss:
            for key in self.ids_dict['profiles_1d'].keys():
                if key in self.all_keys['profiles_1d'][changing_ids] and len(self.ids_dict['profiles_1d'][key]) != 0:

                    old_times = self.ids_dict['time'][changing_ids]
                    # Getting the dimensions of the radial grid and time
                    x_dim = np.shape(self.ids_dict['profiles_1d'][key])[1]
                    time_dim = np.shape(self.ids_dict['profiles_1d'][key])[0]
                    profiles_new = {}
                    '''
                    for i in np.arange(x_dim):
                        if key in profiles_new:
                            profiles_new[key] = np.hstack((profiles_new[key], fit_and_substitute(old_times, new_times, self.ids_dict['profiles_1d'][key][:,i])))
                        else:
                            profiles_new[key] = fit_and_substitute(old_times, new_times, self.ids_dict['profiles_1d'][key][:,i])
                    # New and untested
                    #if x_dim != 0:
                    profiles_new[key] = profiles_new[key].reshape(x_dim, len(new_times))
                    self.ids_dict['profiles_1d'][key] = np.transpose(np.asarray(profiles_new[key]))
                    '''
                    if x_dim != 0:
                        for i in np.arange(x_dim):
                            if key in profiles_new:
                                profiles_new[key] = np.hstack((profiles_new[key], fit_and_substitute(old_times, new_times, self.ids_dict['profiles_1d'][key][:,i])))
                            else:
                                profiles_new[key] = fit_and_substitute(old_times, new_times, self.ids_dict['profiles_1d'][key][:,i])
                    # New and untested
                        profiles_new[key] = profiles_new[key].reshape(x_dim, len(new_times))
                        self.ids_dict['profiles_1d'][key] = np.transpose(np.asarray(profiles_new[key]))

                    else:
                        for i in np.arange(time_dim):
                            if key in profiles_new:
                                profiles_new[key] = np.hstack((profiles_new[key], np.asarray([])))
                            else:
                                profiles_new[key] = np.asarray([])

                        self.ids_dict['profiles_1d'][key] = np.asarray(profiles_new[key])


    def update_times_2d_profiles(self, new_times, changing_idss):

        # Rebasing the profiles in time
        for changing_ids in changing_idss:
            for key in self.ids_dict['profiles_2d'].keys():
                if key in self.all_keys['profiles_2d'][changing_ids] and len(self.ids_dict['profiles_2d'][key]) != 0:

                    old_times = self.ids_dict['time'][changing_ids]
                    # Getting the dimensions of the radial grid and time
                    time_dim = np.shape(self.ids_dict['profiles_2d'][key])[0]
                    x_dim = np.shape(self.ids_dict['profiles_2d'][key])[1]
                    y_dim = np.shape(self.ids_dict['profiles_2d'][key])[2]
                    profiles_new = {}

                    for i in np.arange(x_dim):
                        for j in np.arange(y_dim):
                            if key in profiles_new:
                                profiles_new[key] = np.hstack((profiles_new[key], fit_and_substitute(old_times, new_times, self.ids_dict['profiles_2d'][key][:,i,j])))
                            else:
                                profiles_new[key] = fit_and_substitute(old_times, new_times, self.ids_dict['profiles_2d'][key][:,i,j])

                    profiles_new[key] = profiles_new[key].reshape(x_dim, y_dim, len(new_times))
                    self.ids_dict['profiles_2d'][key] = np.transpose(np.asarray(profiles_new[key]),[2,0,1])
# ---------------------------------------------------------------
   
    def update_times_times(self, new_times, changing_idss):
    
        for changing_ids in changing_idss:
            self.ids_dict['time'][changing_ids] = new_times
    
    def update_times(self, new_times, changing_idss):

        self.update_times_2d_profiles(new_times, changing_idss)
        self.update_times_profiles(new_times, changing_idss)
        self.update_times_traces(new_times, changing_idss)
        self.update_times_times(new_times, changing_idss)

        self.fill_ids_struct()

    def fill_ids_struct(self):

        # For summary r0 and b0 are to be filled with the old values.
        self.fill_summary()

        # --------------------- Fill core profiles -------------------------------
        self.fill_core_profiles()

        # --------------------- Fill equilibrium ---------------------------------
        self.fill_equilibrium()

        # --------------------- Fill core_sources --------------------------------
        if 'core_sources' in self.ids_dict['time'].keys():
            self.fill_core_sources()

        # ------------------------- Filling nbi data -----------------------------
        if 'nbi' in self.ids_dict['time'].keys():
            self.fill_nbi()

        # --------------------- Fill pulse scheduler --------------------------------
        self.fill_pulse_scheduler()



    # Might be cleaner to create a new self.ids_struct, but for how it is coded now it should not matter
    #return self.ids_struct

    def fill_basic_quantities(self, ids_iden):
    
        self.ids_struct[ids_iden] = eval('imas.' + ids_iden + '()')
    
        # Might want to specify this externally
        username=getpass.getuser()
    
        self.ids_struct[ids_iden].code.commit = 'unknown'
        self.ids_struct[ids_iden].code.name = 'Core_profiles_tools'
        self.ids_struct[ids_iden].code.output_flag = np.array([])
        self.ids_struct[ids_iden].code.repository = 'gateway'
        self.ids_struct[ids_iden].code.version = 'unknown'
    
        self.ids_struct[ids_iden].ids_properties.homogeneous_time = imasdef.IDS_TIME_MODE_HOMOGENEOUS
        self.ids_struct[ids_iden].ids_properties.provider = username
        self.ids_struct[ids_iden].ids_properties.creation_date = str(datetime.date)
        self.ids_struct[ids_iden].time = np.asarray([0.1])
    
    # Will need to think what to do here....  # THIS WILL BREAK THE CODE
    def fill_summary(self):
    
        ids_iden = 'summary'
        self.fill_basic_quantities(ids_iden)
   
        #self.ids_struct[ids_iden].global_quantities.r0.value = self.ids_struct_old[ids_iden].global_quantities.r0.value
        #self.ids_struct[ids_iden].global_quantities.b0.value = self.ids_struct_old[ids_iden].global_quantities.b0.value
    
        self.ids_struct[ids_iden].global_quantities.r0.value = self.ids_dict['extras']['r0']
        self.ids_struct[ids_iden].global_quantities.b0.value = self.ids_dict['extras']['b0']

        # Put the traces
        for tag in keys_list['traces']['summary']:
            rsetattr(self.ids_struct[ids_iden], tag, self.ids_dict['traces'][tag])
    
        self.ids_struct[ids_iden].time = self.ids_dict['time']['summary']

    #Need to adapt here to allow for multiple time slices
    
    def fill_core_profiles(self):
    
        ids_iden = 'core_profiles'
        self.fill_basic_quantities(ids_iden)
    
        profile_1d = imas.core_profiles().profiles_1d.getAoSElement()
        self.ids_struct[ids_iden].profiles_1d.append(profile_1d)
    
        # Put the profiles in the structure. A new element is created if it's not available
    
        for tag in keys_list['profiles_1d']['core_profiles']:
            if '[' not in tag:
                for index, profile_1d in enumerate(self.ids_dict['profiles_1d'][tag]):
                    if index >= np.shape(self.ids_struct[ids_iden].profiles_1d)[0]:
                        element = imas.core_profiles().profiles_1d.getAoSElement()
                        self.ids_struct[ids_iden].profiles_1d.append(element)
                    rsetattr(self.ids_struct[ids_iden].profiles_1d[index], tag, profile_1d)
            else:
                parts = tag.split('[')
    
                # put the profiles when there is one nested structure
                index2 = 0
                while parts[0] + '[' + str(index2) + parts[1] in self.ids_dict['profiles_1d'].keys():
                    for index1, profile_1d in enumerate(self.ids_dict['profiles_1d'][parts[0] + '[' + str(index2) + parts[1]]):
                        element = eval('self.ids_struct[ids_iden].profiles_1d[' + str(index1) + '].' + parts[0])
                        if index2 >= len(element):
                            new_profile = eval('self.ids_struct[ids_iden].profiles_1d[' + str(index1) + '].' + parts[0] + '.getAoSElement()')
                            eval('self.ids_struct[ids_iden].profiles_1d[' + str(index1) + '].' + parts[0] + '.append(new_profile)')
    
                        eval('rsetattr(self.ids_struct[ids_iden].profiles_1d[' + str(index1) + '].' + parts[0] + '[' + str(index2) + '], \'' + parts[1][2:] + '\', profile_1d)')
                    index2 += 1
    
        # Put the traces
        tag_type = 'traces'
    
        for tag in keys_list['traces']['core_profiles']:
            parts = tag.split('[')
            if(len(parts)) == 1:
                setattr(self.ids_struct[ids_iden].global_quantities, tag, self.ids_dict['traces'][tag])
    
            elif(len(parts)) == 2:
                index2 = 0
                while parts[0] + '[' + str(index2) + parts[1] in self.ids_dict['traces'].keys():
                    for index1, trace in enumerate(self.ids_dict[tag_type][parts[0] + '[' + str(index2) + parts[1]]):
                        if index1 >= len(self.ids_struct[ids_iden].profiles_1d):
                            new_profile = self.ids_struct[ids_iden].profiles_1d.getAoSElement()
                            self.ids_struct[ids_iden].profiles_1d.append(new_profile)
                        #The addition of new elements might be generalized using eval
                        if index2 >= len(self.ids_struct[ids_iden].profiles_1d[index1].ion):
                            new_item = eval('self.ids_struct[ids_iden].profiles_1d[0].' + parts[0] + '.getAoSElement()')
                            self.ids_struct[ids_iden].profiles_1d[index1].ion.append(new_item)
                            #eval('self.ids_struct[ids_iden].profiles_1d[' + str(index1) + '].' + parts[0] + '.append(new_profile)')

                        eval('rsetattr(self.ids_struct[ids_iden].profiles_1d[' + str(index1) + '].' + parts[0] + '[' + str(index2) + '], \''
 + parts[1][2:] + '\', trace)')

                    index2 += 1

            elif(len(parts)) == 3:
                index2 = 0
                while parts[0] + '[' + str(index2) + parts[1] + '[0' + parts[2] in self.ids_dict[tag_type].keys():
                    index3 = 0
                    while parts[0] + '[' + str(index2) + parts[1] + '[' + str(index3) + parts[2] in self.ids_dict[tag_type].keys():
                        for index1, profile_1d in enumerate(self.ids_dict[tag_type][parts[0] + '[' + str(index2) + parts[1] + '[' + str(index3) + parts[2]]):
                            if index1 >= len(self.ids_struct[ids_iden].profiles_1d):
                                new_profile = self.ids_struct[ids_iden].profiles_1d.getAoSElement()
                                self.ids_struct[ids_iden].profiles_1d.append(new_profile)
    
                            #The addition of new elements might be generalized using eval
                            if index2 >= len(self.ids_struct[ids_iden].profiles_1d[index1].ion):
                                new_item = eval('self.ids_struct[ids_iden].profiles_1d[0].' + parts[0] + '.getAoSElement()')
                                self.ids_struct[ids_iden].profiles_1d[index1].ion.append(new_item)
    
                            if index3 >= len(self.ids_struct[ids_iden].profiles_1d[index1].ion[index2].element):
                                new_item = eval('self.ids_struct[ids_iden].profiles_1d[0].' + parts[0] + '[0' + parts[1] + '.getAoSElement()')
                                self.ids_struct[ids_iden].profiles_1d[index1].ion[index2].element.append(new_item)
    
                            #for index1, profile_1d in enumerate(self.ids_dict[tag_type][split[0] + '#' + parts[0] + '[' + str(index2) + parts[1] + '[' + str(index3) + parts[2]]):
                            eval('rsetattr(self.ids_struct[ids_iden].profiles_1d[' + str(index1) + '].' + parts[0] + '[' + str(index2) + parts[1] + '[' + str(index3) + '] ,\'' + parts[2][2:] +'\', profile_1d)')
                        index3 += 1
                    index2 += 1

        self.ids_struct[ids_iden].time = self.ids_dict['time']['core_profiles']
    
    def fill_equilibrium(self):
    
        ids_iden = 'equilibrium'
        self.fill_basic_quantities(ids_iden)

        for tag in keys_list['profiles_2d']['equilibrium']:
            if '[' not in tag:
                for index, profile_2d in enumerate(self.ids_dict['profiles_2d'][tag]):
                    if index >= np.shape(self.ids_struct['equilibrium'].time_slice)[0]:
                        time_slice = imas.equilibrium().time_slice.getAoSElement()
                        self.ids_struct['equilibrium'].time_slice.append(time_slice)
                        profiles_2d = self.ids_struct['equilibrium'].time_slice[0].profiles_2d.getAoSElement()
                        self.ids_struct['equilibrium'].time_slice[index].profiles_2d.append(profiles_2d)
                    rsetattr(self.ids_struct['equilibrium'].time_slice[index], tag, profile_1d)
            else:
                parts = tag.split('[')

                # Put the profiles in the case of one nested structure
                index2 = 0
                while parts[0] + '[' + str(index2) + parts[1] in self.ids_dict['profiles_2d'].keys():
                    for index1, profile_2d in enumerate(self.ids_dict['profiles_2d'][parts[0] + '[' + str(index2) + parts[1]]):
                        if index1 >= np.shape(self.ids_struct['equilibrium'].time_slice)[0]:
                            time_slice = imas.equilibrium().time_slice.getAoSElement()
                            self.ids_struct['equilibrium'].time_slice.append(time_slice)
                            profiles_2d = self.ids_struct['equilibrium'].time_slice[0].profiles_2d.getAoSElement()
                            self.ids_struct['equilibrium'].time_slice[index1].profiles_2d.append(profiles_2d)

                        eval('rsetattr(self.ids_struct[\'equilibrium\'].time_slice[' + str(index1) + '].' + parts[0] + '[' + str(index2) + '], \'' + parts[1][2:] + '\', profile_2d)')
                    index2 += 1

        # Should append the time slice also for the splitted tags, to avoid ordering problems
        for tag in keys_list['profiles_1d']['equilibrium']:
            if '[' not in tag:
                for index, profile_1d in enumerate(self.ids_dict['profiles_1d'][tag]):
                    if index >= np.shape(self.ids_struct[ids_iden].time_slice)[0]:
                        time_slice = imas.equilibrium().time_slice.getAoSElement()
                        self.ids_struct[ids_iden].time_slice.append(time_slice)
                        profiles_2d = self.ids_struct[ids_iden].time_slice[0].profiles_2d.getAoSElement()
                        self.ids_struct[ids_iden].time_slice[index].profiles_2d.append(profiles_2d)
                    rsetattr(self.ids_struct[ids_iden].time_slice[index], tag, profile_1d)
            else:
                parts = tag.split('[')
    
                # Put the profiles in the case of one nested structure
                index2 = 0
                while parts[0] + '[' + str(index2) + parts[1] in self.ids_dict['profiles_1d'].keys():
                    for index1, profile_1d in enumerate(self.ids_dict['profiles_1d'][parts[0] + '[' + str(index2) + parts[1]]):
                        eval('rsetattr(self.ids_struct[ids_iden].time_slice[' + str(index1) + '].' + parts[0] + '[' + str(index2) + '], \'' + parts[1][2:] + '\', profile_1d)')
                    index2 += 1
    
        tag_type = 'traces'

        for tag in keys_list['traces'][ids_iden]:
            if '[' not in tag:
                for index, trace in enumerate(self.ids_dict['traces'][tag]):
                    if index >= np.shape(self.ids_struct[ids_iden].time_slice)[0]:
                        time_slice = imas.equilibrium().time_slice.getAoSElement()
                        self.ids_struct[ids_iden].time_slice.append(time_slice)
                        profiles_2d = self.ids_struct[ids_iden].time_slice[0].profiles_2d.getAoSElement()
                        self.ids_struct[ids_iden].time_slice[0].profiles_2d.append(profiles_2d)
    
                    rsetattr(self.ids_struct[ids_iden].time_slice[index], tag, trace)

            else:
                parts = tag.split('[')
                index2 = 0
                while parts[0] + '[' + str(index2) + parts[1] in self.ids_dict['traces'].keys():
                    for index1, trace in enumerate(self.ids_dict[tag_type][parts[0] + '[' + str(index2) + parts[1]]):
                        if index1 >= len(self.ids_struct[ids_iden].time_slice):
                            new_slice = self.ids_struct[ids_iden].time_slice.getAoSElement()
                            self.ids_struct[ids_iden].time_slice.append(new_slice)
                        #The addition of new elements might be generalized using eval
                        if index2 >= len(self.ids_struct[ids_iden].time_slice[index1].profiles_2d):
                            new_item = eval('self.ids_struct[ids_iden].time_slice[0].' + parts[0] + '.getAoSElement()')
                            self.ids_struct[ids_iden].time_slice[index1].profiles_2d.append(new_item)
                            #eval('self.ids_struct[ids_iden].profiles_1d[' + str(index1) + '].' + parts[0] + '.append(new_profile)')

                        eval('rsetattr(self.ids_struct[ids_iden].time_slice[' + str(index1) + '].' + parts[0] + '[' + str(index2) + '], \''
 + parts[1][2:] + '\', trace)')

                    index2 += 1


        # These are not in time_slice, so need to be treated separately
    
        self.ids_struct[ids_iden].vacuum_toroidal_field.b0 = self.ids_dict['extras']['b0']
        self.ids_struct[ids_iden].vacuum_toroidal_field.r0 = self.ids_dict['extras']['r0']
    
        self.ids_struct[ids_iden].time = self.ids_dict['time']['equilibrium']
    
    def get_index_source(self, split):
    
        if split == 'total':
            i_source = 0
    
        if split == 'nbi':
            i_source = 1
    
        if split == 'ec':
            i_source = 2
    
        if split == 'lh':
            i_source = 3
    
        if split == 'ic':
            i_source = 4
    
        return(i_source)
    
    def fill_core_sources(self):
    
        ids_iden = 'core_sources'
        self.fill_basic_quantities(ids_iden)

        # Should be possible to generalize here so this piece of code works for all the cases
    
        for tag_type in ['profiles_1d', 'traces']:
            for tag in keys_list[tag_type]['core_sources']:
                # Maybe not the ideal way to do it? It will probably break if there are only one or a few sources.
    
                # Only fills the tags actually in the dictionary
                if tag not in self.ids_dict[tag_type]:
                    break
    
                split = tag.split('#')
                i_source = self.get_index_source(split[0])
    
                parts = split[1].split('[')
                if len(parts) == 1:
    
                    while len(self.ids_struct[ids_iden].source) <= i_source:
                        new_source = self.ids_struct[ids_iden].source.getAoSElement()
                        self.ids_struct[ids_iden].source.append(new_source)
    
                    for index, profile_1d in enumerate(self.ids_dict[tag_type][tag]):
                        if index >= len(self.ids_struct[ids_iden].source[i_source].profiles_1d):
                            new_profile = self.ids_struct[ids_iden].source[i_source].profiles_1d.getAoSElement()
                            self.ids_struct[ids_iden].source[i_source].profiles_1d.append(new_profile)
                        rsetattr(self.ids_struct[ids_iden].source[i_source].profiles_1d[index], split[1], profile_1d)
    
                elif len(parts) == 2:
                    index2 = 0
                    while split[0] + '#' + parts[0] + '[' + str(index2) + parts[1] in self.ids_dict[tag_type].keys():
                        for index1, profile_1d in enumerate(self.ids_dict[tag_type][split[0] + '#' + parts[0] + '[' + str(index2) + parts[1]]):
                            # Inserts the correct structures in the new ids if the don't exist for a given index
                            if index1 >= len(self.ids_struct[ids_iden].source[i_source].profiles_1d):
                                new_profile = self.ids_struct[ids_iden].source[i_source].profiles_1d.getAoSElement()
                                self.ids_struct[ids_iden].source[i_source].profiles_1d.append(new_profile)
    
                            if index2 >= len(self.ids_struct[ids_iden].source[i_source].profiles_1d[index1].ion):
                                new_item = eval('self.ids_struct[ids_iden].source[i_source].profiles_1d[0].' + parts[0] + '.getAoSElement()')
                                self.ids_struct[ids_iden].source[i_source].profiles_1d[index1].ion.append(new_item)
    
                            eval('rsetattr(self.ids_struct[ids_iden].source[' + str(i_source) + '].profiles_1d[' + str(index1) + '].' + parts[0] + '[' + str(index2) + '] ,\'' + parts[1][3:] +'\', profile_1d)')
                        index2 += 1
    
            
                elif(len(parts)) == 3:
                    index2 = 0
                    while split[0] + '#' + parts[0] + '[' + str(index2) + parts[1] + '[0' + parts[2] in self.ids_dict[tag_type].keys():
                        index3 = 0
                        while split[0] + '#' + parts[0] + '[' + str(index2) + parts[1] + '[' + str(index3) + parts[2] in self.ids_dict[tag_type].keys():
                            for index1, profile_1d in enumerate(self.ids_dict[tag_type][split[0] + '#' + parts[0] + '[' + str(index2) + parts[1] + '[' + str(index3) + parts[2]]):
                                if index1 >= len(self.ids_struct[ids_iden].source[i_source].profiles_1d):
                                    new_profile = self.ids_struct[ids_iden].source[i_source].profiles_1d.getAoSElement()
                                    self.ids_struct[ids_iden].source[i_source].profiles_1d.append(new_profile)
    
                                #The addition of new elements might be generalized using eval
                                if index2 >= len(self.ids_struct[ids_iden].source[i_source].profiles_1d[index1].ion):
                                    new_item = eval('self.ids_struct[ids_iden].source[i_source].profiles_1d[0].' + parts[0] + '.getAoSElement()')
                                    self.ids_struct[ids_iden].source[i_source].profiles_1d[index1].ion.append(new_item)
    
                                if index3 >= len(self.ids_struct[ids_iden].source[i_source].profiles_1d[index1].ion[index2].element):
                                    new_item = eval('self.ids_struct[ids_iden].source[i_source].profiles_1d[0].' + parts[0] + '[0' + parts[1] + '.getAoSElement()')
                                    self.ids_struct[ids_iden].source[i_source].profiles_1d[index1].ion[index2].element.append(new_item)
    
                                    #for index1, profile_1d in enumerate(self.ids_dict[tag_type][split[0] + '#' + parts[0] + '[' + str(index2) + parts[1] + '[' + str(index3) + parts[2]]):
                                eval('rsetattr(self.ids_struct[ids_iden].source[' + str(i_source) + '].profiles_1d[' + str(index1) + '].' + parts[0] + '[' + str(index2) + parts[1] + '[' + str(index3) + '] ,\'' + parts[2][2:] +'\', profile_1d)')
    
                            index3 += 1
                        index2 += 1
    
        self.ids_struct[ids_iden].time = self.ids_dict['time']['core_sources']
    
    # nbi does not have a lot of time dependent fields so the structure is a little different. The old IDS is copied and not rebuilt from scratch
    
    def fill_nbi(self):

        # Need to add this part    
        ids_iden = 'nbi'

        self.fill_basic_quantities(ids_iden)

        for tag in keys_list['traces'][ids_iden]:
            # Need to understand why the dictionary is copied when this is not working.

            parts = tag.split('[')
            index = 0
            while parts[0] + '[' + str(index) + parts[1] in self.ids_dict['traces'].keys():
                if index >= len(self.ids_struct[ids_iden].unit):
                    new_item = eval('self.ids_struct[ids_iden].' + parts[0] + '.getAoSElement()')
                    self.ids_struct[ids_iden].unit.append(new_item)

                eval('rsetattr(self.ids_struct[ids_iden].unit[' + str(index) + '] ,\'' + parts[1][2:] + '\', self.ids_dict[\'traces\'][\'' + parts[0] + '[' + str(index) + parts[1] + '\'])')
                index += 1

        self.ids_struct[ids_iden].time = self.ids_dict['time'][ids_iden]

        if self.db == 'tcv':
            self.fill_nbi_iden_tcv()


    def fill_nbi_iden_tcv(self):
        self.ids_struct['nbi'].unit[0].name = '25KeV 1st NBH source'
        self.ids_struct['nbi'].unit[1].name = '50KeV 2nd NBH source'
        self.ids_struct['nbi'].unit[2].name = 'diagnostic NBI'
        self.ids_struct['nbi'].unit[0].identifier = 'NB1'
        self.ids_struct['nbi'].unit[1].identifier = 'NB2'
        self.ids_struct['nbi'].unit[2].identifier = 'DNBI'


    def fill_pulse_scheduler(self):

        ids_iden = 'pulse_schedule'
        self.fill_basic_quantities(ids_iden)

        # And then interpolate all traces to the pulse_schedule.time time schedule

        self.ids_struct[ids_iden].time = self.ids_dict['time']['core_profiles']

        if 'b0' in self.ids_dict['extras']:
            if len(self.ids_dict['extras']['b0']) != 1:
            # Might still be needed becouse of bugs and the flipping ip
                self.ids_struct[ids_iden].tf.b_field_tor_vacuum_r.reference.data = fit_and_substitute(self.ids_dict['time']['equilibrium'], self.ids_struct[ids_iden].time, self.ids_dict['extras']['r0']*self.ids_dict['extras']['b0'])
                #self.ids_struct[ids_iden].tf.b_field_tor_vacuum_r.reference.data = np.abs(fit_and_substitute(self.ids_dict['time']['equilibrium'], self.ids_struct[ids_iden].time, self.ids_dict['extras']['r0']*self.ids_dict['extras']['b0']))
            else:
                #self.ids_struct[ids_iden].tf.b_field_tor_vacuum_r.reference.data = np.abs(self.ids_dict['extras']['r0']*self.ids_dict['extras']['b0'])
                self.ids_struct[ids_iden].tf.b_field_tor_vacuum_r.reference.data = self.ids_dict['extras']['r0']*self.ids_dict['extras']['b0']
            self.ids_struct[ids_iden].tf.b_field_tor_vacuum_r.reference.time = self.ids_dict['time']['core_profiles']

        if len(self.ids_dict['traces']['global_quantities.ip']) !=0:
            if len(self.ids_dict['traces']['global_quantities.ip']) != 1:
                self.ids_struct[ids_iden].flux_control.i_plasma.reference.data = fit_and_substitute(self.ids_dict['time']['equilibrium'], self.ids_struct[ids_iden].time, self.ids_dict['traces']['global_quantities.ip'])
            else:
                self.ids_struct[ids_iden].flux_control.i_plasma.reference.data = self.ids_dict['traces']['global_quantities.ip']
            self.ids_struct[ids_iden].flux_control.i_plasma.reference.time = self.ids_dict['time']['core_profiles']


        if len(self.ids_dict['traces']['line_average.n_e.value']) !=0:
            if len(self.ids_dict['traces']['line_average.n_e.value']) != 1:
                self.ids_struct[ids_iden].density_control.n_e_line.reference.data = fit_and_substitute(self.ids_dict['time']['summary'], self.ids_struct[ids_iden].time, self.ids_dict['traces']['line_average.n_e.value'])
            else:
                self.ids_struct[ids_iden].density_control.n_e_line.reference.data = self.ids_dict['traces']['line_average.n_e.value']
            self.ids_struct[ids_iden].density_control.n_e_line.reference.time = self.ids_dict['time']['core_profiles']


        #fit_and_substitute(old_times, new_times, self.ids_dict['profiles_1d'][key][:,i])

        # Heating, not available yet

        #pulse_schedule.nbi.unit[iunit].power.reference.time
        #pulse_schedule.nbi.unit[iunit].power.reference.data

        #pulse_schedule.ec.launcher[iloop].power.reference.time
        #pulse_schedule.ec.launcher[iloop].power.reference.data

        #pulse_schedule.ec.launcher[iloop].steering_angle_pol.reference.time
        #pulse_schedule.ec.launcher[iloop].steering_angle_pol.reference.data

        #pulse_schedule.ec.launcher[iloop].steering_angle_tor.reference.time
        #pulse_schedule.ec.launcher[iloop].steering_angle_tor.reference.data

        #pulse_schedule.lh.antenna[iloop].power.reference.time
        #pulse_schedule.lh.antenna[iloop].power.reference.data

        #pulse_schedule.ic.antenna[iloop].power.reference.time
        #pulse_schedule.ic.antenna[iloop].power.reference.data


# -------------------------------------------- MANIPULATE IDSS -----------------------------------------------



def select_interval_ids(db, shot, run, run_target, time_start, time_end, username = None):

    if not username:
        username = getpass.getuser()

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_data.select_interval(time_start, time_end)

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)

def average_integrated_modelling(db, shot, run, run_target, time_start, time_end, username = None):

    '''

    Average all the fields in integrated modelling that are useful for the integrated modelling and saves a new ids with just the averages

    '''
    if not username:
        username = getpass.getuser()

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_data.select_interval(time_start, time_end)
    ids_data.average_traces_profile()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)

def rebase_integrated_modelling(db, shot, run, run_target, changing_idss, option = 'core profiles', num_times = 100, username = None):

    '''

    Fits the ids on a new time base and creates a new ids. 'changing_idss' contains the names of the idss that will be rebased in time

    '''

    if not username:
        username = getpass.getuser()

    if option == 'core profiles':
        core_profiles = open_and_get_core_profiles(db, shot, run)
        new_times = core_profiles.times

        ids_data = IntegratedModellingDict(db, shot, run, username = username)
        ids_data.update_times(new_times, changing_idss)

    if option == 'linear':
        for changing_ids in changing_idss:
            #Maintain list structure when passing to general update times
            ids_opened = open_and_get_ids(db, shot, run, changing_ids)
            times = ids_opened.time
            time_start, time_end = min(times), max(times)
            new_times = np.arange(time_start, time_end, (time_end - time_start)/num_times)

            ids_data = IntegratedModellingDict(db, shot, run, username = username)
            #Maintain list structure when passing to general update times
            ids_data.update_times(new_times, [changing_ids])

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)

def smooth_t_and_d_ids_new(db, shot, run, db_target, shot_target, run_target, username = '', username_target = ''):

    '''

    Inserts one more slice for every slice. All the values in new slices are interpolated. A new IDS is created

    '''

    if username == '': username = getpass.getuser()

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    new_times = double_time(core_profiles.time)
    ids_data.update_times(new_times, changing_idss)

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


# Might want to move this guys in another file with other small functions that I use here and there
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# ----------------------------- FILL PULSE SCHEDULER IDS  ---------------------------------

# ------------------------- EXTRA TOOLS TO OPEN AND PUT IDSS ------------------------------


def open_integrated_modelling(db, shot, run, username='', backend='mdsplus'):

    '''

    Opens the idss useful for integrated modelling and saves the fields in a conveniently to use dictionary. This should be done with IMASpy when I learn how to do it.
    Might also just save everything and then check and sort the dimensions later

    '''

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if username == '':
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()

    ids_struct = {}
    #ids_list = ['core_profiles', 'core_sources', 'ec_launchers', 'equilibrium', 'nbi', 'summary', 'thomson_scattering']
    ids_list = ['core_profiles', 'core_sources', 'equilibrium', 'nbi', 'summary', 'thomson_scattering']

    for ids in ids_list:
        ids_struct_single = data_entry.get(ids)
        #if ids_struct_single.time != np.asarray([]):
        if ids_struct_single.time.size != 0:
            ids_struct[ids] = ids_struct_single
        else:
            print('no ' + ids + ' ids')

    data_entry.close()

    return(ids_struct)

def open_and_get_core_profiles(db, shot, run, username='', backend='mdsplus'):

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if username == '':
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

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


def open_and_get_equilibrium(db, shot, run, username='', backend='mdsplus'):

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if username == '':
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()

    if op[0]<0:
        cp=data_entry.create()
        print(cp[0])
        if cp[0]==0:
            print("data entry created")
    elif op[0]==0:
        print("data entry opened")

    equilibrium = data_entry.get('equilibrium')
    data_entry.close()

    return(equilibrium)

#Redundant since I can just use partial get

def open_and_get_equilibrium_tag(db, shot, run, tag, username='', backend='mdsplus', prefix='global_quantities'):

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if username == '':
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()

    if op[0]<0:
        cp=data_entry.create()
        print(cp[0])
        if cp[0]==0:
            print("data entry created")
    elif op[0]==0:
        print("data entry opened")

    time = data_entry.partial_get('equilibrium', 'time(:)')
    variable = data_entry.partial_get('equilibrium', 'time_slice(:)/' + prefix + '/' + tag)
    data_entry.close()

    return(time, variable)

def open_and_get_summary(db, shot, run, username=None, backend='mdsplus'):

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if not isinstance(username, str):
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()

    if op[0]<0:
        cp=data_entry.create()
        print(cp[0])
        if cp[0]==0:
            print("data entry created")
    elif op[0]==0:
        print("data entry opened")

    summary = data_entry.get('summary')
    data_entry.close()

    return(summary)

def open_and_get_core_sources(db, shot, run, username='', backend='mdsplus'):

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if username == '':
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()

    if op[0]<0:
        cp=data_entry.create()
        print(cp[0])
        if cp[0]==0:
            print("data entry created")
    elif op[0]==0:
        print("data entry opened")

    core_sources = data_entry.get('core_sources')
    data_entry.close()

    return(core_sources)


def open_and_get_ids(db, shot, run, ids_name, username='', backend='mdsplus'):

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if username == '':
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()

    if op[0]<0:
        cp=data_entry.create()
        print(cp[0])
        if cp[0]==0:
            print("data entry created")
    elif op[0]==0:
        print("data entry opened")

    ids_opened = data_entry.get(ids_name)
    data_entry.close()

    return(ids_opened)


def open_and_get_nbi(db, shot, run, username='', backend='mdsplus'):

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if username == '':
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()

    if op[0]<0:
        cp=data_entry.create()
        print(cp[0])
        if cp[0]==0:
            print("data entry created")
    elif op[0]==0:
        print("data entry opened")

    nbi = data_entry.get('nbi')
    data_entry.close()

    return(nbi)

def open_and_get_all(db, shot, run, username='', backend='mdsplus'):

    '''

    Opens all the idss create for TCV. This should be done with IMASpy when I learn how to do it.

    '''

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if username == '':
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()

    ids_dict = {}
    ids_list = ['core_profiles', 'core_sources', 'core_transport', 'ec_launchers', 'equilibrium', 'nbi', 'pf_active', 'summary', 'thomson_scattering', 'tf', 'wall']

    for ids in ids_list:
        ids_dict['ids'] = data_entry.get(ids)

    data_entry.close()

    return(ids_dict)

def put_integrated_modelling(db, shot, run, run_target, ids_struct, backend='mdsplus'):

    '''

    Puts the IDSs useful for integrated modelling. This should be done with IMASpy when I learn how to do it.

    '''

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    username = getpass.getuser()
    copy_ids_entry(username, db, shot, run, shot, run_target)

    data_entry = imas.DBEntry(imas_backend, db, shot, run_target, user_name=getpass.getuser())
    ids_list = ['core_profiles', 'core_sources', 'ec_launchers', 'equilibrium', 'nbi', 'summary', 'thomson_scattering', 'pulse_schedule']

    op = data_entry.open()

    for ids in ids_list:
    # If the time vector is empty the IDS is empty or broken, do not put
        if ids in ids_struct:
            if len(ids_struct[ids].time) !=0:
                data_entry.put(ids_struct[ids])

    data_entry.close()


# ------------------------- EXTRA TOOLS TO MODIFY IDSS ------------------------------

# ------------------------------- ZEFF MANIPULATION ---------------------------------

def open_and_copy_handle_target(username, db, shot, run, run_target):

    handle = ImasHandle(user = username, db = db, shot = shot, run = run)
    # Test what this does
    #handle.exists()
    target = ImasHandle(user = username, db = db, shot = shot, run = run_target)
    handle.copy_data_to(target)
    #target.exists()

    return handle, target

def set_default_entries(username, db, shot, username_target, db_target, shot_target):

    if not username:
        username=getpass.getuser()
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot
    if not username_target:
        username_target = username

    return username, username_target, db_target, shot_target


class ZeffIntegratedModelling:

    def __init__(self, db, shot, run, run_target, username=None, username_target=None, db_target=None, shot_target=None, backend='mdsplus'):

        self.db = db
        self.shot = shot
        self.run = run
        self.run_target = run_target
        self.username = username
        self.username_target = username_target
        self.db_target = db_target
        self.shot_target = shot_target
        self.set_default_entries()

        ids_names_to_extract = ['core_profiles','summary']
        dataset_integrated_modelling = extract_integrated_modelling(ids_names_to_extract, self.username, self.db, self.shot, self.run)
        print(dataset_integrated_modelling['core_profiles'])
        self.zeff = dataset_integrated_modelling['core_profiles']['zeff']
        self.Zeff = dataset_integrated_modelling['core_profiles']['zeff'].to_numpy()
        self.rho_tor_norm = dataset_integrated_modelling['core_profiles']['rho_tor_norm'].to_numpy()

        print(dataset_integrated_modelling['core_profiles']['zeff'])
        print(dataset_integrated_modelling['core_profiles']['rho_tor_norm'])
        exit()


        self.time_cp = dataset_integrated_modelling['core_profiles']['time core profiles'].to_numpy()
        #self.time_eq = dataset_integrated_modelling['equilibrium']['time equilibrium'].to_numpy()
        self.ne = dataset_integrated_modelling['core_profiles']['ne'].to_numpy()
        self.ne_ave = np.average(self.ne, axis = 1)
        carbon_density = dataset_integrated_modelling['core_profiles']['ni'].to_numpy()
        self.carbon_density = carbon_density[:,1,:]

        # Equilibrium variables are separated for performance reason
        handle = ImasHandle(user = self.username, db = self.db, shot = self.shot, run = self.run)
        equilibrium = handle.get('equilibrium')
        ip = self.extract_equilibrium_quantity_tag('ip')
        self.Ip = ip['ip'].to_numpy()
        self.time_eq = ip['time equilibrium'].to_numpy()

    def find_index_equilibrium_tag(self, tag):
        index = 0
        for variable in variables_equilibrium:
            if variable.name == tag:
                break
            index += 1
        return index

    def extract_equilibrium_quantity_tag(self, tag):
        index = self.find_index_equilibrium_tag(tag)
        handle = ImasHandle(user = self.username, db = self.db, shot = self.shot, run = self.run)
        equilibrium = handle.get('equilibrium')
        quantity = handle.get_variables([variables_equilibrium[index], variable_equilibrium_time[0]])
        return quantity

    def set_default_entries(self):

        if not self.username:
            username=getpass.getuser()
        if not self.db_target:
            self.db_target = self.db
        if not self.shot_target:
            self.shot_target = self.shot
        if not self.username_target:
            self.username_target = self.username


    def set_flat_zeff(self, option):

        handle, target = open_and_copy_handle_target(self.username, self.db, self.shot, self.run, self.run_target)

        if option == 'maximum':
        # The maximum Zeff might be too large, the not completely ionized carbon close to the LCFS might lead to a negative main ion density
        # Reducing the value in this case
            self.zeff_set_flat_maximum()
        elif option == 'minimum':
            self.zeff_set_flat_minimum()
        elif option == 'median':
            self.zeff_set_flat_median()
        else:
            print('Option not recognized, aborting. This should not happen')

        self.put_zeff()

    def correct_zeff(self):
        self.correct_numpy_Zeff()
        self.put_zeff()


    def set_parabolic_zeff(self, zeff_param = 1):
        self.correct_numpy_Zeff()

        average_zeff = []
        for profile in self.Zeff:
            average_zeff.append(np.average(profile))

        for index in range(np.shape(self.Zeff)[0]):
            norm = zeff_param * (average_zeff[index]-1)/2
            print(1-self.rho_tor_norm[index])
            print(self.rho_tor_norm)
            print(self.rho_tor_norm[index])
            self.Zeff[index] = average_zeff[index] + norm/2 - norm * np.sqrt(1-self.rho_tor_norm[index])
            print(self.Zeff[index])
            exit()

        self.correct_numpy_Zeff()
        self.put_zeff()


    def set_hyperbolic_zeff(self, zeff_param = 0, zeff_max = 3, verbose = False):

        index_start_ft, index_end_ft = find_indexes_start_end(self.Ip, self.time_eq, zeff_param, self.time_cp)

        # The function needs to be continuous. If the parameter is not a list zeff is continuous when the ramp ends. Otherwise a value can be specified.
        zeff_target, time_target = self.Zeff[index_start_ft][0], self.time_cp[index_start_ft]

        # c is the parameter controlling how fast zeff descents after the beginning. Z0 is zeff at t=0
        z0 = zeff_max
        c = 10
        b = (zeff_target-z0)/(-1+1/((c*time_target)**4+1))
        a = -b + z0

        # Only changed at the beginning. Not adequate for ramp down. For that I would need to identify the last flattop, currently not done.
        Zeff_new, index = [], 0
        for z in self.Zeff[:index_start_ft]:
            Zeff_new.append(np.full((np.size(z)), a + b/((c*time_cp[index])**4 +1)))
            index += 1

        for z in self.Zeff[index_start_ft:]:
            Zeff_new.append(self.Zeff[index])
            index += 1

        self.Zeff = np.asarray(Zeff_new)

        #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
        # Especially when combined with add profiles early
        self.correct_numpy_Zeff()

        self.put_zeff()

        print('zeff set hyperbolic')


    def set_peaked_zeff(self, zeff_param = 1):
        self.correct_numpy_Zeff()

        average_zeff = []
        for profile in self.Zeff:
            average_zeff.append(np.average(profile))

        for index in range(np.shape(self.Zeff)[0]):
            norm = zeff_param * (average_zeff[index]-1)/2
            self.Zeff[index] = average_zeff[index] - norm/2 + norm * np.sqrt(1-rho_tor_norm[index])

        self.correct_numpy_Zeff()
        self.put_zeff()

        print('zeff set peaked')

    def set_peaked_ev_zeff_profile(self, verbose = False, zeff_param = 1):

        '''
        Sets a peaked Zeff profiles, trying to (roughly!) keep Zeff the same. The profile is evolved in time to be flat at the beginning and gets peaked later

        '''

        self.correct_numpy_Zeff()

        average_zeff = []
        for profile in self.Zeff:
            average_zeff.append(np.average(profile))

        # Very early want a flat zeff profile, they start to get out slightly later
        for index in range(np.shape(self.Zeff)[0]):
            zeff_param = np.where(time_cp < 0.05, zeff_param*20*time_cp, zeff_param)
            norm = zeff_param * (average_zeff[index]-1)/2
            self.Zeff[index] = average_zeff[index] - norm[index]/2 + norm[index] * np.sqrt(1-rho_tor_norm[index])

        self.correct_numpy_Zeff()
        self.put_zeff()

        print('zeff turned parabolic')


    def set_low_edge_zeff(self, zeff_param = 0):

        '''
        Sets a flat Zeff profile but with the separatrix lower with a gaussian function. Might be better numerically for initialization.

        '''

        self.correct_numpy_Zeff()
        for i in np.arange(len(self.time_cp)):
            rho_profile = self.rho_tor_norm[i]
            self.Zeff[i] = zeff_param-(zeff_param-1)/2*np.exp(-(5*rho_profile-5)**2)

        self.correct_numpy_Zeff()
        self.put_zeff()

        print('zeff set low at the edge')

    def set_ip_ne_scaled_zeff(self, verbose = False):

        ne = core_profiles.to_xarray(['ne','rho_tor_norm','time'])
        ne_ave = np.average(ne['ne'], axis = 1)
        # The steady state is identified
        index_start_ft, index_end_ft = identify_flattop_ip(self.Ip, self.time_eq)

        # Coef1 is calculated for Zeff to be continuous when the ramp up ends
        zeff_target = self.Zeff[index_start_ft][0]
        ip_target, ne_target = self.Ip[index_start_ft], self.ne_ave[index_start_ft]

        coef2 = 3
        coef1 = (zeff_target - 1)/(ip_target/ne_target)**coef2

        # Only changed at the beginning. Not adequate for ramp down. For that I would need to identify the last flattop, currently not done.
        Zeff_new, index = [], 0
        for z in self.Zeff[:index_start_ft]:
            Zeff_new.append(np.full((np.size(z)), 1 + coef1*(Ip[index]/self.ne_ave[index])**coef2))
            index += 1

        for z in self.Zeff[index_start_ft:]:
            Zeff_new.append(self.Zeff[index])
            index += 1

        self.Zeff = np.asarray(Zeff_new)

        #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
        # Especially when combined with add profiles early
        self.correct_numpy_Zeff()
        self.put_zeff()

        print('zeff scaled to ip')


    def set_linear_descending_zeff(self, verbose = False):

        # The steady state is identified
        index_start_ft, index_end_ft = identify_flattop_ip(self.Ip, self.time_eq)
        zeff_target = self.Zeff[index_start_ft][0]

        # Only changed at the beginning. Not adequate for ramp down. For that I would need to identify the last flattop, currently not done.
        Zeff_new = []
        for index, z in enumerate(self.Zeff[:index_start_ft]):
            Zeff_new.append(np.full((np.size(z)), -4/time_cp[index_start_ft]*time_cp[index] + 4))

        for index, z in enumerate(self.Zeff[index_start_ft:]):
            Zeff_new.append(self.Zeff[index])

        self.Zeff = np.asarray(Zeff_new)

        #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
        # Especially when combined with add profiles early
        self.correct_numpy_Zeff()
        self.put_zeff()

        print('zeff linear descent set')

    def set_impurity_composition_from_flattop(self, verbose = False):

        # The steady state is identified
        index_start_ft, index_end_ft = identify_flattop_ip(self.Ip, self.time_eq)
        zeff_target = self.Zeff[index_start_ft][0]

        carbon_density_ave = np.average(self.carbon_density[index_start_ft:index_end_ft], axis = 0)

        # The new carbon density is calculated on the average of the carbon density before the steady state
        carbon_density_new = []

        for carbon_dens in self.carbon_density[:index_start_ft]:
            carbon_density_new.append(carbon_density_ave)

        for carbon_dens in self.carbon_density[index_start_ft:index_end_ft]:
            carbon_density_new.append(carbon_dens)

        for carbon_dens in self.carbon_density[index_end_ft:]:
            carbon_density_new.append(carbon_density_ave)

        # Zeff is calculated normally, the charge is calculated over a profile that considers the lower charge towards the edge. This is TCV specific.
        charge_carbon = []
        for rho_profile in rho_tor_norm:
            charge_carbon.append(6-2*np.exp(-(5*rho_profile-5)**2))
        charge_carbon = np.asarray(charge_carbon).reshape(len(rho_tor_norm), len(rho_tor_norm[0]))

        n_C = carbon_density_new/self.ne
        self.Zeff = 1 + n_C*charge_carbon**2 - n_C*charge_carbon

        # Still want to use a flat z_eff profile. This is imposed here

        Zeff_new = []
        for z in zeff:
            Zeff_new.append(np.full((np.size(z)), np.average(z)))

        self.Zeff = np.asarray(Zeff_new)

        #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
        # Especially when combined with add profiles early
        self.correct_numpy_Zeff()
        self.put_zeff()

        print('zeff set from flattop')


    def zeff_set_flat_maximum(self):
        self.Zeff = np.where(self.Zeff < 4.5, np.max(self.Zeff), 4.5)


    def zeff_set_flat_minimum(self):
        self.Zeff = np.where(self.Zeff > 1.02, np.min(self.Zeff), 1.02)


    #To be tested, might not be the correct syntax
    def zeff_set_flat_median(self):
        self.Zeff[:] = (np.min(self.Zeff) + np.max(self.Zeff))/2


    def put_zeff(self):
        # Put manipulated variable back in the IDS
        self.zeff['zeff'] = xr.DataArray(self.Zeff, coords=self.zeff.coords, dims=self.zeff.dims, attrs=self.zeff.attrs)
        target = ImasHandle(user = self.username_target, db = self.db_target, shot = self.shot_target, run = self.run_target)
        core_profiles = target.get('core_profiles')
        core_profiles.write_array_in_parts('profiles_1d/*/zeff', self.zeff['zeff'])
        core_profiles.sync(target)


    def correct_numpy_Zeff(self):
        self.Zeff = np.where(self.Zeff > 1, self.Zeff, 1.02)
        self.Zeff = np.where(self.Zeff < 4.5, self.Zeff, 4.5)





def set_flat_zeff(db, shot, run, run_target, option, username = None, username_target = None, db_target = None, shot_target = None):

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    core_profiles = handle.get('core_profiles')
    zeff = core_profiles.to_xarray(['zeff','rho_tor_norm','time'])
    Zeff = zeff['zeff'].to_numpy()

    # Body
    if option == 'maximum':
        # The maximum Zeff might be too large, the not completely ionized carbon close to the LCFS might lead to a negative main ion density
        # Reducing the value in this case
        Zeff = zeff_set_maximum(Zeff)
    elif option == 'minimum':
        Zeff = zeff_set_minimum(Zeff)
    elif option == 'median':
        Zeff = zeff_set_median(Zeff)
    else:
        print('Option not recognized, aborting. This should not happen')

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    def zeff_set_maximum(Zeff):
        Zeff = np.where(Zeff < 4.5, np.max(Zeff), 4.5)
        return Zeff

    def zeff_set_minimum(Zeff):
        Zeff = np.where(Zeff > 1.02, np.min(Zeff), 1.02)
        return Zeff

    #To be tested, might not be the correct syntax
    def zeff_set_median(Zeff):
        Zeff = (np.min(Zeff) + np.max(Zeff))/2
        return Zeff


# Name changed. Need to change it in prepare input
def set_flat_Zeff_old(db, shot, run, run_target, option, username = ''):

    '''

    Writes a new ids with a flat Zeff.

    '''
    if username == '':
        username = getpass.getuser()

    core_profiles = open_and_get_core_profiles(db, shot, run, username)
    Zeff = np.array([])

    for profile_1d in core_profiles.profiles_1d:
        Zeff = np.hstack((Zeff, profile_1d.zeff))

    len_time = len(core_profiles.time)
    len_x = len(core_profiles.profiles_1d[0].zeff)

    Zeff.reshape(len_time, len_x)

#    len_time, len_x = np.shape(Zeff)[0], np.shape(Zeff)[1]
    max_Zeff, min_zeff = np.max(Zeff), np.min(Zeff)

    if option == 'maximum':
        # The maximum Zeff might be too large, the not completely ionized carbon close to the LCFS might lead to a negative main ion density.
        # Reducing the value in this case.
        if max_Zeff < 4.5:
            Zeff_value = max_Zeff
        else:
            Zeff_value = 4.5

    elif option == 'minimum':
        if min_zeff > 1.02:
            Zeff_value = min_zeff
        else:
            Zeff_value = 1.02
    elif option == 'median':
        if max_Zeff < 4.5:
            Zeff_value = (max_Zeff + min_zeff)/2
        else:
            Zeff_value = (4.5 + min_zeff)/2

    else:
        print('Option not recognized, aborting. This should not happen')
        exit()

    Zeff = np.full((len_time, len_x), Zeff_value)

    for index, zeff_slice in enumerate(Zeff):
        core_profiles.profiles_1d[index].zeff = zeff_slice

    copy_ids_entry(username, db, shot, run, shot, run_target)

    data_entry_target = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run_target, user_name=username)

    op = data_entry_target.open()
    core_profiles.put(db_entry = data_entry_target)
    data_entry_target.close()

def correct_numpy_Zeff(Zeff):
    Zeff = np.where(Zeff > 1, Zeff, 1.02)
    Zeff = np.where(Zeff < 4.5, Zeff, 4.5)
    return Zeff


def correct_zeff(db, shot, run, run_target, username = None, username_target = None, db_target = None, shot_target = None):

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    '''

    Sets Zeff = 1.02 where Zeff falls below 1 for whatever reason (can happen if experimental data is taken automatically and not properly c
hecked). Also, values greater than
4.5 are clumped (to avoid negative main ion density). The choice of 4.5 can be improved, itś made under the hypotesis that the impurity is c
arbon. Higher than this and it migh
t die close to the boundaries.
    Uses duqtools to do it

    '''

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    core_profiles = handle.get('core_profiles')
    zeff = core_profiles.to_xarray(['zeff','rho_tor_norm','time'])
    Zeff = zeff['zeff'].to_numpy()

    # Body
    Zeff = correct_numpy_Zeff(Zeff)

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    print('zeff corrected')

def correct_zeff_old(db, shot, run, db_target, shot_target, run_target, username = '', username_target = ''):

    '''

    Sets Zeff = 1.02 where Zeff falls below 1 for whatever reason (can happen if experimental data is taken automatically and not properly checked). Also, values greater than
4.5 are clumped (to avoid negative main ion density). The choice of 4.5 can be improved, itś made under the hypotesis that the impurity is carbon. Higher than this and it migh
t die close to the boundaries.
    Uses new_classes to do it

    '''

    if username == '':
        username=getpass.getuser()

    ids_data = IntegratedModellingDict(db, shot, run, username = username)

    ids_dict = ids_data.ids_dict

    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 4.5, ids_dict['profiles_1d']['zeff'], 4.5)

    # Put the data back in the ids structure

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)

    print('zeff corrected')

# WORK IN PROGRESS


def set_parabolic_zeff(db, shot, run, run_target, zeff_param = 1, db_target = None, shot_target = None, username = None, username_target = None):

    '''

    Sets a parabolic Zeff profile

    '''

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    core_profiles = handle.get('core_profiles')
    zeff = core_profiles.to_xarray(['zeff','rho_tor_norm','time'])
    Zeff = zeff['zeff'].to_numpy()
    rho_tor_norm = zeff['rho_tor_norm'].to_numpy()

    # Body
    Zeff = correct_numpy_Zeff(Zeff)

    average_zeff = []
    for profile in Zeff:
        average_zeff.append(np.average(profile))

    for index in range(np.shape(Zeff)[0]):
        norm = zeff_param * (average_zeff[index]-1)/2
        Zeff[index] = average_zeff[index] + norm/2 - norm * np.sqrt(1-rho_tor_norm[index])

    Zeff = correct_numpy_Zeff(Zeff)

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    print('zeff turned parabolic')


def set_parabolic_zeff_old(db, shot, run, run_target, zeff_param = 1, db_target = None, shot_target = None, username = None, username_target = None):

    '''

    Sets a parabolic Zeff profile

    '''

    if not username:
        username=getpass.getuser()
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot
    if not username_target:
        username_target = username

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 4.5, ids_dict['profiles_1d']['zeff'], 4.5)

    average_zeff = []

    for profile in ids_dict['profiles_1d']['zeff']:
        average_zeff.append(np.average(profile))

    for index in range(np.shape(ids_dict['profiles_1d']['zeff'])[0]):
        #if type(zeff_param) != list:
        norm = zeff_param * (average_zeff[index]-1)/2
        #else:
        #    norm = zeff_param[0] * (average_zeff[index]-1)/2
        ids_dict['profiles_1d']['zeff'][index] = average_zeff[index] + norm/2 - norm * np.sqrt(1-ids_dict['profiles_1d']['grid.rho_tor_norm'][index])

    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 4.5, ids_dict['profiles_1d']['zeff'], 4.5)

    # Put the data back in the ids structure

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)

    print('zeff turned parabolic')

def set_peaked_zeff_profile(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose = False, zeff_param = 1):

    '''
    Sets a peaked Zeff profiles, trying to (roughly!) keep Zeff the same

    '''

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    core_profiles = handle.get('core_profiles')
    zeff = core_profiles.to_xarray(['zeff','rho_tor_norm','time'])
    Zeff = zeff['zeff'].to_numpy()
    rho_tor_norm = zeff['rho_tor_norm'].to_numpy()

    # Body
    Zeff = correct_numpy_Zeff(Zeff)

    average_zeff = []
    for profile in Zeff:
        average_zeff.append(np.average(profile))

    for index in range(np.shape(Zeff)[0]):
        norm = zeff_param * (average_zeff[index]-1)/2
        Zeff[index] = average_zeff[index] - norm/2 + norm * np.sqrt(1-rho_tor_norm[index])

    Zeff = correct_numpy_Zeff(Zeff)

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    print('zeff turned parabolic')


def set_peaked_zeff_profile_old(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose = False, zeff_param = 1):

    if not username:
        username=getpass.getuser()
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot
    if not username_target:
        username_target = username

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 5, ids_dict['profiles_1d']['zeff'], 5)

    average_zeff = []

    for profile in ids_dict['profiles_1d']['zeff']:
        average_zeff.append(np.average(profile))

    for index in range(np.shape(ids_dict['profiles_1d']['zeff'])[0]):
        #if type(zeff_param) != list:
        norm = zeff_param * (average_zeff[index]-1)/2
        #else:
        #    norm = zeff_param[0] * (average_zeff[index]-1)/2
        ids_dict['profiles_1d']['zeff'][index] = average_zeff[index] - norm/2 + norm * np.sqrt(1-ids_dict['profiles_1d']['grid.rho_tor_norm'][index])

    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 5, ids_dict['profiles_1d']['zeff'], 5)

    # Put the data back in the ids structure

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


# ----------------------------- WORK IN PROGRESS ----------------------------------

def set_peaked_ev_zeff_profile(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose =
 False, zeff_param = 1):

    '''
    Sets a peaked Zeff profiles, trying to (roughly!) keep Zeff the same. The profile is evolved in time to be flat at the beginning and gets peaked later

    '''

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    core_profiles = handle.get('core_profiles')
    zeff = core_profiles.to_xarray(['zeff','rho_tor_norm','time'])
    Zeff = zeff['zeff'].to_numpy()
    rho_tor_norm = zeff['rho_tor_norm'].to_numpy()
    time_cp = zeff['time'].to_numpy()

    # Body
    Zeff = correct_numpy_Zeff(Zeff)

    average_zeff = []
    for profile in Zeff:
        average_zeff.append(np.average(profile))

    # Very early want a flat zeff profile, they start to get out slightly later
    for index in range(np.shape(Zeff)[0]):
        zeff_param = np.where(time_cp < 0.05, zeff_param*20*time_cp, zeff_param)
        norm = zeff_param * (average_zeff[index]-1)/2
        Zeff[index] = average_zeff[index] - norm[index]/2 + norm[index] * np.sqrt(1-rho_tor_norm[index])

    Zeff = correct_numpy_Zeff(Zeff)

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    print('zeff turned parabolic')


def set_peaked_ev_zeff_profile_old(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose = False, zeff_param = 1):

    if not username:
        username=getpass.getuser()
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot
    if not username_target:
        username_target = username

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 5, ids_dict['profiles_1d']['zeff'], 5)

    #time_eq = ids_dict['time']['equilibrium']
    #ip = ids_dict['traces']['global_quantities.ip']
    #index_start_ft_eq, index_end_ft_eq = identify_flattop_ip(ip, time_eq)
    time_cp = ids_dict['time']['core_profiles']

    average_zeff = []

    for profile in ids_dict['profiles_1d']['zeff']:
        average_zeff.append(np.average(profile))

    # Very early want a flat zeff profile, they start to get out slightly later
    for index in range(np.shape(ids_dict['profiles_1d']['zeff'])[0]):
        #if type(zeff_param) != list:
        zeff_param = np.where(time_cp < 0.05, zeff_param*20*time_cp, zeff_param)
        norm = zeff_param * (average_zeff[index]-1)/2
        #else:
        #    zeff_param = np.where(time_cp < 0.05, zeff_param[0]*20*time_cp, zeff_param[0])
        #    norm = zeff_param * (average_zeff[index]-1)/2
        ids_dict['profiles_1d']['zeff'][index] = average_zeff[index] - norm[index]/2 + norm[index] * np.sqrt(1-ids_dict['profiles_1d']['grid.rho_tor_norm'][index])

    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 5, ids_dict['profiles_1d']['zeff'], 5)

    # Put the data back in the ids structure

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)

def set_low_edge_zeff(db, shot, run, run_target, zeff_param = 0, db_target = None, shot_target = None, username = None, username_target = None):

    '''
    Sets a flat Zeff profile but with the separatrix lower with a gaussian function. Might be better numerically for initialization.

    '''

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    core_profiles = handle.get('core_profiles')
    zeff = core_profiles.to_xarray(['zeff','rho_tor_norm','time'])
    Zeff = zeff['zeff'].to_numpy()
    rho_tor_norm = zeff['rho_tor_norm'].to_numpy()
    time_cp = zeff['time'].to_numpy()

    # Body
    Zeff = correct_numpy_Zeff(Zeff)
    for i in np.arange(len(time_cp)):
        rho_profile = rho_tor_norm[i]
        Zeff[i] = zeff_param-(zeff_param-1)/2*np.exp(-(5*rho_profile-5)**2)

    Zeff = correct_numpy_Zeff(Zeff)

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    print('zeff turned parabolic')


def set_low_edge_zeff_old(db, shot, run, run_target, zeff_param = 0, db_target = None, shot_target = None, username = None, username_target = None):

    if not username:
        username=getpass.getuser()
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot
    if not username_target:
        username_target = username

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    time = ids_dict['time']['core_profiles']
    for i in np.arange(len(time)):
        rho_profile = ids_dict['profiles_1d']['grid.rho_tor_norm'][i]
        #zeff = np.average(ids_dict['profiles_1d']['zeff'][i])
        ids_dict['profiles_1d']['zeff'][i] = zeff_param-(zeff_param-1)/2*np.exp(-(5*rho_profile-5)**2)

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


# WORK IN PROGRESS LATEST #3.31.2023

def find_indexes_start_end(ip, time_eq, zeff_param, time_cp):

    if not zeff_param:
        index_start_ft_eq, index_end_ft_eq = identify_flattop_ip(ip, time_eq)
    else:
        #The rationale behind specifying a value is that zeff might decrease more or less independently from the current (if it depends on other variables)
        index_start_ft_eq = np.abs(time_eq - zeff_param).argmin(0)
        index_end_ft_eq = len(time_eq) - 1

    #Need the indexes for core_profile
    index_start_ft = np.abs(time_cp - time_eq[index_start_ft_eq]).argmin(0)
    index_end_ft = np.abs(time_cp - time_eq[index_end_ft_eq]).argmin(0)


def set_hyperbole_zeff(db, shot, run, run_target, zeff_param = 0, zeff_max = 3, db_target = None, shot_target = None, username = None, username_target = None, verbose = False):

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    core_profiles = handle.get('core_profiles')
    zeff = core_profiles.to_xarray(['zeff','rho_tor_norm','time'])
    Zeff = zeff['zeff'].to_numpy()
    rho_tor_norm = zeff['rho_tor_norm'].to_numpy()
    time_cp = zeff['time'].to_numpy()

    equilibrium = handle.get('equilibrium')
    ip = equilibrium.to_xarray(['ip','time'])
    Ip = ip['ip']
    time_eq = ip['time']

    # Body
    index_start_ft, index_end_ft = find_indexes_start_end(Ip, time_eq, zeff_param, time_cp)

    # The function needs to be continuous. If the parameter is not a list zeff is continuous when the ramp ends. Otherwise a value can be specified.
    zeff_target, time_target = Zeff[index_start_ft][0], time_cp[index_start_ft]

    # c is the parameter controlling how fast zeff descents after the beginning. Z0 is zeff at t=0
    z0 = zeff_max
    c = 10
    b = (zeff_target-z0)/(-1+1/((c*time_target)**4+1))
    a = -b + z0

    # Only changed at the beginning. Not adequate for ramp down. For that I would need to identify the last flattop, currently not done.
    Zeff_new, index = [], 0
    for z in Zeff[:index_start_ft]:
        Zeff_new.append(np.full((np.size(z)), a + b/((c*time_cp[index])**4 +1)))
        index += 1

    for z in Zeff[index_start_ft:]:
        Zeff_new.append(Zeff[index])
        index += 1

    Zeff = np.asarray(Zeff_new)

    #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
    # Especially when combined with add profiles early
    Zeff = correct_numpy_Zeff(Zeff)

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    print('zeff set hyperbolic')


def set_hyperbole_zeff_old(db, shot, run, run_target, zeff_param = 0, zeff_max = 3, db_target = None, shot_target = None, username = None, username_target = None, verbose = False):

    if not username:
        username=getpass.getuser()
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot
    if not username_target:
        username_target = username

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    time_eq = ids_dict['time']['equilibrium']
    time_cp = ids_dict['time']['core_profiles']
    ip = ids_dict['traces']['global_quantities.ip']

    ''' 
    # The steady state is identified
    if type(zeff_param) != list:
        index_start_ft_eq, index_end_ft_eq = identify_flattop_ip(ip, time_eq)
    else:
        # The rationale behind specifying a value is that zeff might decrease more or less independently from the current (if it depends on other variables)
        index_start_ft_eq = np.abs(time_eq - zeff_param[1]).argmin(0)
        #index_end_ft = np.abs(time_eq - (time_eq[-1] - zeff_param[1])).argmin(0)
        index_end_ft_eq = len(time_eq) - 1
    '''

    if not zeff_param:
        index_start_ft_eq, index_end_ft_eq = identify_flattop_ip(ip, time_eq)
    else:
        #The rationale behind specifying a value is that zeff might decrease more or less independently from the current (if it depends on other variables)
        index_start_ft_eq = np.abs(time_eq - zeff_param).argmin(0)
        index_end_ft_eq = len(time_eq) - 1

    #Need the indexes for core_profile
    index_start_ft = np.abs(time_cp - time_eq[index_start_ft_eq]).argmin(0)
    index_end_ft = np.abs(time_cp - time_eq[index_end_ft_eq]).argmin(0)

    # The function needs to be continuous. If the parameter is not a list zeff is continuous when the ramp ends. Otherwise a value can be specified.
    zeff_target, time_target = ids_dict['profiles_1d']['zeff'][index_start_ft][0], time_cp[index_start_ft]

    # c is the parameter controlling how fast zeff descents after the beginning. Z0 is zeff at t=0
    #z0 = 4
    z0 = zeff_max
    #z0 = ids_dict['profiles_1d']['zeff'][0][0]
    c = 10
    #b = (zeff_target - z0)/(-1 + 1/((c*time_target)**2 +1))
    b = (zeff_target-z0)/(-1+1/((c*time_target)**4+1))
    a = -b + z0

    time = ids_dict['time']['core_profiles']

    # Only changed at the beginning. Not adequate for ramp down. For that I would need to identify the last flattop, currently not done.
    zeff_new, index = [], 0
    for z in ids_dict['profiles_1d']['zeff'][:index_start_ft]:
        zeff_new.append(np.full((np.size(z)), a + b/((c*time[index])**4 +1)))
        index += 1

    for z in ids_dict['profiles_1d']['zeff'][index_start_ft:]:
        zeff_new.append(ids_dict['profiles_1d']['zeff'][index])
        index += 1

    ids_dict['profiles_1d']['zeff'] = np.asarray(zeff_new)

    #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
    # Especially when combined with add profiles early
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 4.0, ids_dict['profiles_1d']['zeff'], 4.0)

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


def set_ip_ne_scaled_zeff(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose = False):

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    core_profiles = handle.get('core_profiles')
    zeff = core_profiles.to_xarray(['zeff','rho_tor_norm','time'])
    Zeff = zeff['zeff'].to_numpy()
    rho_tor_norm = zeff['rho_tor_norm'].to_numpy()
    time_cp = zeff['time'].to_numpy()
    ne = core_profiles.to_xarray(['ne','rho_tor_norm','time'])
    ne_ave = np.average(ne['ne'], axis = 1)

    equilibrium = handle.get('equilibrium')
    ip = equilibrium.to_xarray(['ip','time'])
    Ip = ip['ip']
    time_eq = ip['time']

    # Body
    # The steady state is identified
    index_start_ft, index_end_ft = identify_flattop_ip(Ip, time_eq)

    # Coef1 is calculated for Zeff to be continuous when the ramp up ends
    zeff_target = Zeff[index_start_ft][0]
    ip_target, ne_target = Ip[index_start_ft], ne_ave[index_start_ft]

    coef2 = 3
    coef1 = (zeff_target - 1)/(ip_target/ne_target)**coef2

    # Only changed at the beginning. Not adequate for ramp down. For that I would need to identify the last flattop, currently not done.
    Zeff_new, index = [], 0
    for z in Zeff[:index_start_ft]:
        Zeff_new.append(np.full((np.size(z)), 1 + coef1*(Ip[index]/ne_ave[index])**coef2))
        index += 1

    for z in Zeff[index_start_ft:]:
        Zeff_new.append(Zeff[index])
        index += 1

    Zeff = np.asarray(Zeff_new)

    #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
    # Especially when combined with add profiles early
    Zeff = correct_numpy_Zeff(Zeff)

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    print('zeff scaled to ip')


def set_ip_ne_scaled_zeff_old(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose = False):

    if not username:
        username=getpass.getuser()
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot
    if not username_target:
        username_target = username

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    ip = ids_dict['traces']['global_quantities.ip']
    ne = np.average(ids_dict['profiles_1d']['electrons.density'], axis = 1)
    time = ids_dict['time']['equilibrium']


    # The steady state is identified
    index_start_ft, index_end_ft = identify_flattop_ip(ip, time)

    # Coef1 is calculated for Zeff to be continuous when the ramp up ends
    zeff_target = ids_dict['profiles_1d']['zeff'][index_start_ft][0]
    ip_target, ne_target = ip[index_start_ft], ne[index_start_ft]

    coef2 = 3
    coef1 = (zeff_target - 1)/(ip_target/ne_target)**coef2

    # Only changed at the beginning. Not adequate for ramp down. For that I would need to identify the last flattop, currently not done.
    zeff_new, index = [], 0
    for z in ids_dict['profiles_1d']['zeff'][:index_start_ft]:
        zeff_new.append(np.full((np.size(z)), 1 + coef1*(ip[index]/ne[index])**coef2))
        index += 1

    for z in ids_dict['profiles_1d']['zeff'][index_start_ft:]:
        zeff_new.append(ids_dict['profiles_1d']['zeff'][index])
        index += 1

    ids_dict['profiles_1d']['zeff'] = np.asarray(zeff_new)

    #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
    # Especially when combined with add profiles early
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 4.0, ids_dict['profiles_1d']['zeff'], 4.0)

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


def extract_variables_zeff_manipulation(username, db, shot, run, run_target):

    handle = ImasHandle(user = username, db = db, shot = shot, run = run)
    core_profiles = handle.get('core_profiles')
    zeff = core_profiles.to_xarray(['zeff','rho_tor_norm','time'])
    Zeff = zeff['zeff'].to_numpy()
    rho_tor_norm = zeff['rho_tor_norm'].to_numpy()
    time_cp = zeff['time'].to_numpy()
    ne = core_profiles.to_xarray(['ne','rho_tor_norm','time'])
    ne_ave = np.average(ne['ne'], axis = 1)

    equilibrium = handle.get('equilibrium')
    ip = equilibrium.to_xarray(['ip','time'])
    Ip = ip['ip']
    time_eq = ip['time']

    return rho_tor_norm, time_cp, time_eq, Zeff, Ip, ne, ne_ave

def set_linear_descending_zeff(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose = False):

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    rho_tor_norm, time_cp, time_eq, Zeff, Ip, ne, ne_ave = extract_variables_zeff_manipulation(username, db, shot, run, run_target)

    # Body
    # The steady state is identified
    index_start_ft, index_end_ft = identify_flattop_ip(Ip, time_eq)
    zeff_target = Zeff[index_start_ft][0] 

    # Only changed at the beginning. Not adequate for ramp down. For that I would need to identify the last flattop, currently not done.
    Zeff_new = []
    for index, z in enumerate(Zeff[:index_start_ft]):
        Zeff_new.append(np.full((np.size(z)), -4/time_cp[index_start_ft]*time_cp[index] + 4))

    for index, z in enumerate(Zeff[index_start_ft:]):
        Zeff_new.append(Zeff[index])

    Zeff = np.asarray(Zeff_new)

    #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
    # Especially when combined with add profiles early
    Zeff = correct_numpy_Zeff(Zeff)

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    print('zeff linear descent set')


def set_linear_descending_zeff_old(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose = False):

    if not username:
        username=getpass.getuser()
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot
    if not username_target:
        username_target = username

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    ip = ids_dict['traces']['global_quantities.ip']
    time = ids_dict['time']['equilibrium']

    # The steady state is identified
    index_start_ft, index_end_ft = identify_flattop_ip(ip, time)

    zeff_target = ids_dict['profiles_1d']['zeff'][index_start_ft][0]

    # Only changed at the beginning. Not adequate for ramp down. For that I would need to identify the last flattop, currently not done.
    zeff_new = []
    for index, z in enumerate(ids_dict['profiles_1d']['zeff'][:index_start_ft]):
        zeff_new.append(np.full((np.size(z)), -4/time[index_start_ft]*time[index] + 4))

    for index, z in enumerate(ids_dict['profiles_1d']['zeff'][index_start_ft:]):
        zeff_new.append(ids_dict['profiles_1d']['zeff'][index])


    ids_dict['profiles_1d']['zeff'] = np.asarray(zeff_new)

    #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
    # Especially when combined with add profiles early
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 4.0, ids_dict['profiles_1d']['zeff'], 4.0)

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


def set_impurity_composition_from_flattop(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose = False):

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    rho_tor_norm, time_cp, time_eq, Zeff, Ip, ne, ne_ave = extract_variables_zeff_manipulation(username, db, shot, run, run_target)

    # Body
    # The steady state is identified
    index_start_ft, index_end_ft = identify_flattop_ip(Ip, time_eq)
    zeff_target = Zeff[index_start_ft][0]

    variable = [
        Variable(name='ion_0_density',
                 ids = 'core_profiles',
                 path = 'profiles_1d/1/ion/*/density',
                 dims = ['time', 'rho_tor_norm']),
        'rho_tor_norm',
        'time'
    ]

    carbon_density = handle.get_variables(variable)
    carbon_density = carbon_density['ion_0_density']
    carbon_density_ave = np.average(carbon_density[index_start_ft:index_end_ft], axis = 0)

    # The new carbon density is calculated on the average of the carbon density before the steady state
    carbon_density_new = []

    for carbon_dens in carbon_density[:index_start_ft]:
        carbon_density_new.append(carbon_density_ave)

    for carbon_dens in carbon_density[index_start_ft:index_end_ft]:
        carbon_density_new.append(carbon_dens)

    for carbon_dens in carbon_density[index_end_ft:]:
        carbon_density_new.append(carbon_density_ave)

    # Zeff is calculated normally, the charge is calculated over a profile that considers the lower charge towards the edge. This is TCV specific.
    charge_carbon = []
    for rho_profile in rho_tor_norm:
        charge_carbon.append(6-2*np.exp(-(5*rho_profile-5)**2))
    charge_carbon = np.asarray(charge_carbon).reshape(len(rho_tor_norm), len(rho_tor_norm[0]))

    n_C = carbon_density_new/ne['ne']
    Zeff = 1 + n_C*charge_carbon**2 - n_C*charge_carbon

    # Still want to use a flat z_eff profile. This is imposed here

    Zeff_new = []
    for z in zeff:
        Zeff_new.append(np.full((np.size(z)), np.average(z)))

    Zeff = np.asarray(Zeff_new)

    #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
    # Especially when combined with add profiles early
    Zeff = correct_numpy_Zeff(Zeff)

    # Put manipulated variable back in the IDS
    zeff['zeff'] = xr.DataArray(Zeff, coords=zeff.coords, dims=zeff.dims, attrs=zeff.attrs)
    core_profiles.write_array_in_parts('profiles_1d/*/zeff', zeff['zeff'])
    core_profiles.sync(target)

    print('zeff linear descent set')


def set_impurity_composition_from_flattop_old(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, verbose = False):

    if not username:
        username=getpass.getuser()
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot
    if not username_target:
        username_target = username

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    ip = ids_dict['traces']['global_quantities.ip']
    time = ids_dict['time']['equilibrium']

    # The steady state is identified
    index_start_ft, index_end_ft = identify_flattop_ip(ip, time)

    carbon_density = ids_dict['profiles_1d']['ion[1].density']
    carbon_density_ave = np.average(carbon_density[index_start_ft:index_end_ft], axis = 0)

    # The new carbon density is calculated on the average of the carbon density before the steady state
    carbon_density_new = []

    for carbon_dens in carbon_density[:index_start_ft]:
        carbon_density_new.append(carbon_density_ave)

    for carbon_dens in carbon_density[index_start_ft:index_end_ft]:
        carbon_density_new.append(carbon_dens)

    for carbon_dens in carbon_density[index_end_ft:]:
        carbon_density_new.append(carbon_density_ave)

    # Zeff is calculated normally, the charge is calculated over a profile that considers the lower charge towards the edge. This is TCV specific.
    charge_carbon = []
    rho_tor_norm = ids_dict['profiles_1d']['grid.rho_tor_norm']
    for rho_profile in rho_tor_norm:
        charge_carbon.append(6-2*np.exp(-(5*rho_profile-5)**2))
    charge_carbon = np.asarray(charge_carbon).reshape(len(rho_tor_norm), len(rho_tor_norm[0]))

    n_C = carbon_density_new/ids_dict['profiles_1d']['electrons.density']
    zeff = 1 + n_C*charge_carbon**2 - n_C*charge_carbon

    # Still want to use a flat z_eff profile. This is imposed here

    zeff_new = []
    for z in zeff:
        zeff_new.append(np.full((np.size(z)), np.average(z)))

    ids_dict['profiles_1d']['zeff'] = np.asarray(zeff_new)

    #4 should be fine as a limit for Zeff. Lower than usual due to possible lower temperature at the beginning
    # Especially when combined with add profiles early
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] > 1, ids_dict['profiles_1d']['zeff'], 1.02)
    ids_dict['profiles_1d']['zeff'] = np.where(ids_dict['profiles_1d']['zeff'] < 4.0, ids_dict['profiles_1d']['zeff'], 4.0)

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


def identify_flattop(db, shot, run, verbose = False):

    '''

    Automatically identifies the flattop. Not very robust but should work for setting Zeff correctly

    '''

    username, username_target, db_target, shot_target = set_default_entries(username, db, shot, username_target, db_target, shot_target)

    # Opens IDSs
    handle, target = open_and_copy_handle_target(db, shot, run, run_target)

    # Open Variable to be manipulated
    rho_tor_norm, time_cp, time_eq, Zeff, Ip, ne, ne_ave = extract_variables_zeff_manipulation(username, db, shot, run, run_target)
    core_profiles = handle.get('core_profiles')
    summary = handle.get('summary')
    t_i_ave = core_profiles.to_xarray(['t_i_ave','rho_tor_norm','time'])
    te = core_profiles.to_xarray(['te','rho_tor_norm','time'])

    core_profiles = open_and_get_core_profiles(db, shot, run, username)
    summary = open_and_get_summary(db, shot, run, username)

    variable = [
        Variable(name='ip',
                 ids = 'summary',
                 path = 'global_quantities.ip.value',
                 dims = ['time']),
        'rho_tor_norm',
        'time'
    ]

    ip = handle.get_variables(variable)
    Ip = ip['ip']

    if verbose:
        plt.subplot(1,1,1)
        plt.plot(time_cp, ne['ne'][:,0], 'c-', label = 'Te')
        plt.plot(time_cp, t_i_ave['t_i_ave'][:,0], 'r-', label = 'Ti')
        plt.legend()
        plt.show()

#    Searching for start and stop of the flattop for various variables. The the start of the flattop is set not to be on the first index

    time_start_ft_te, time_end_ft_te = identify_flattop_variable(te, time_cp)
    time_start_ft_ti, time_end_ft_ti = identify_flattop_variable(t_i_ave, time_cp)

# Removing nan from the ip array. Need to also remove the corresponding times. ip and zeff might be useful for the future or to find mistakes

    ip_map = np.where(np.isnan(Ip), False, True)

    Ip = Ip[ip_map]
    ip_time = ip[ip_map]

    time_start_ft_zeff, time_end_ft_zeff = identify_flattop_variable(Zeff, time_cp)
    time_start_ft_ip, time_end_ft_ip = identify_flattop_variable(ip, ip_time)
    time_start_ft_ne, time_end_ft_ne = identify_flattop_variable(ne['ne'], time_cp)

    if verbose:
        print('flattop start and end for Te are ')
        print(time_start_ft_te, time_end_ft_te)
        print('flattop start and end for Ti are ')
        print(time_start_ft_ti, time_end_ft_ti)
        print('flattop start and end for zeff are ')
        print(time_start_ft_zeff, time_end_ft_zeff)
        print('flattop start and end for ne are ')
        print(time_start_ft_ne, time_end_ft_ne)

# The intervals should cross somewhere. Then minimum and maximum are taken. If they do not, an average of start and finish is taken.
# Still kinda ugly but should work. Might want to identify better methods later

    if time_end_ft_te <= time_start_ft_ti or time_end_ft_ti <= time_start_ft_te:
        time_start_ft = (time_start_ft_te + time_start_ft_ti)/2
        time_end_ft = (time_end_ft_te + time_end_ft_ti)/2
    else:
        time_start_ft = max(time_start_ft_te, time_start_ft_ti)
        time_end_ft = min(time_end_ft_te, time_end_ft_ti)

    print('flattop starts and ends at:')
    print(time_start_ft, time_end_ft)

# Might find a better way to include the current start and end of the flattop

    print('current flattop is')
    print(time_start_ft_ip, time_end_ft_ip)
    print('it should be similar. If not, there might be a problem')

    return(time_start_ft, time_end_ft)


def identify_flattop_old(db, shot, run, verbose = False):

    '''

    Automatically identifies the flattop. Not very robust but should work for setting Zeff correctly

    '''

    username = getpass.getuser()

    core_profiles = open_and_get_core_profiles(db, shot, run, username)
    summary = open_and_get_summary(db, shot, run, username)

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    if verbose:
        plt.subplot(1,1,1)
        plt.plot(ids_dict['time']['core_profiles'], ids_dict['profiles_1d']['electrons.temperature'][:,0], 'c-', label = 'Te')
        plt.plot(ids_dict['time']['core_profiles'], ids_dict['profiles_1d']['ion[0].temperature'][:,0], 'r-', label = 'Ti')
        plt.legend()
        plt.show()

#    Searching for start and stop of the flattop for various variables. The the start of the flattop is set not to be on the first index

    time_start_ft_te, time_end_ft_te = identify_flattop_variable(ids_dict['profiles_1d']['electrons.temperature'], ids_dict['time']['core_profiles'])
    time_start_ft_ti, time_end_ft_ti = identify_flattop_variable(ids_dict['profiles_1d']['ion[0].temperature'], ids_dict['time']['core_profiles'])

# Removing nan from the ip array. Need to also remove the corresponding times. ip and zeff might be useful for the future or to find mistakes

    ip_map = np.where(np.isnan(summary.global_quantities.ip.value), False, True)

    ip = summary.global_quantities.ip.value[ip_map]
    ip_time = summary.time[ip_map]

    time_start_ft_zeff, time_end_ft_zeff = identify_flattop_variable(ids_dict['profiles_1d']['zeff'], ids_dict['time']['core_profiles'])
    time_start_ft_ip, time_end_ft_ip = identify_flattop_variable(ip, ip_time)
    time_start_ft_ne, time_end_ft_ne = identify_flattop_variable(ids_dict['profiles_1d']['electrons.density'], ids_dict['time']['core_profiles'])

    if verbose:
        print('flattop start and end for Te are ')
        print(time_start_ft_te, time_end_ft_te)
        print('flattop start and end for Ti are ')
        print(time_start_ft_ti, time_end_ft_ti)
        print('flattop start and end for zeff are ')
        print(time_start_ft_zeff, time_end_ft_zeff)
        print('flattop start and end for ne are ')
        print(time_start_ft_ne, time_end_ft_ne)

# The intervals should cross somewhere. Then minimum and maximum are taken. If they do not, an average of start and finish is taken.
# Still kinda ugly but should work. Might want to identify better methods later

    if time_end_ft_te <= time_start_ft_ti or time_end_ft_ti <= time_start_ft_te:
        time_start_ft = (time_start_ft_te + time_start_ft_ti)/2
        time_end_ft = (time_end_ft_te + time_end_ft_ti)/2
    else:
        time_start_ft = max(time_start_ft_te, time_start_ft_ti)
        time_end_ft = min(time_end_ft_te, time_end_ft_ti)

    print('flattop starts and ends at:')
    print(time_start_ft, time_end_ft)

# Might find a better way to include the current start and end of the flattop

    print('current flattop is')
    print(time_start_ft_ip, time_end_ft_ip)
    print('it should be similar. If not, there might be a problem')

    return(time_start_ft, time_end_ft)

#  ------ Should implement, and it might be a good time to try to program the equilibrium interface using attributes ------

#    equilibrium = open_and_get_equilibrium(db, shot, run, username)
#    time_start_ft_ip, time_end_ft_ip = identify_flattop_variable(equilibrium.global_quantities.ip, equilibrium.time)

#    print(time_start_ft_ip, time_end_ft_ip)

def identify_flattop_variable(variables, time):

    '''

    Identifies the flattop for single variables

    '''

    if len(variables.shape) == 2:
        variables = np.average(variables, axis = 1)

    flattop_begin, index_flattop_begin, index_flattop_end = False, 0, len(time)-1
    average, spread = np.average(variables), np.std(variables)/2

#    print(average, spread)

    for index, variable in enumerate(variables):
        if variable > (average - spread) and not flattop_begin and not index == 0:
             index_flattop_begin = index
             flattop_begin = True
        if flattop_begin and variable < (average - spread):
             index_flattop_end = index
             break

    return(time[index_flattop_begin], time[index_flattop_end])

def identify_flattop_ip(ip, time):

    '''

    A more rubust way to identify the steady state for ip
    min_derivative are arbitrary. starting limit_derivative is arbitrary. When derivative is close to 0 I define it as a steady state
    limit derivative is increased is the steady state is too short. Too short is arbitrary, 200ms in the case

    '''

    time_interval, limit_derivative = 0.1, 20000
    if ip[0] > 0:
        ip = -ip

    while time_interval < 0.2:

        smooth_ip = smooth(ip, window_len=17)
        dip_dt = np.gradient(smooth_ip, time)  

        index, index_flattop_begin, index_flattop_end = 0, 0, 0
        #while dip_dt[index] < -limit_derivative and index < (len(dip_dt)-1):
        # Asimmetric limits try to delay the beginning to when the current is more stable
        while dip_dt[index] < -(limit_derivative-4000) and index < (len(dip_dt)-1):
            index += 1
        index_flattop_begin = index

        if index == len(dip_dt-1):
            limit_derivative -= 4000
            continue

        #while dip_dt[index] > -limit_derivative and dip_dt[index] < limit_derivative and index < (len(dip_dt)-1):
        while dip_dt[index] > -(limit_derivative-4000) and dip_dt[index] < limit_derivative and index < (len(dip_dt)-1):
            index += 1
        index_flattop_end = index

        time_interval = time[index_flattop_end] - time[index_flattop_begin]
        limit_derivative += 4000

    return(index_flattop_begin, index_flattop_end)


def check_ion_number(db, shot, run):

    handle = ImasHandle(user = username, db = db, shot = shot, run = run)
    variable = [
        Variable(name='ions',
                 ids = 'core_profiles',
                 path = 'profiles_1d/ion/*',
                 dims = ['time', 'rho_tor_norm']),
        'rho_tor_norm',
        'time'
    ]

    ions = handle.get_variables(variable)
    ions = ions['ions']

    ion_number = len(ions)

def check_ion_number(db, shot, run):

    core_profiles = open_and_get_core_profiles(db, shot, run)
    ion_number = len(core_profiles.profiles_1d[0].ion)

    return(ion_number)

# ------------------------------- Q PROFILE MANIPULATION ---------------------------------

def flip_q_profile(db, shot, run, run_target, username = ''):

    '''

    Writes a new ids with the opposite sign of the q profile

    '''
    if username == '':
        username = getpass.getuser()

    core_profiles = open_and_get_core_profiles(db, shot, run, username)

    for index, profile_1d in enumerate(core_profiles.profiles_1d):
        core_profiles.profiles_1d[index].q = -core_profiles.profiles_1d[index].q

    copy_ids_entry(username, db, shot, run, shot, run_target)

    data_entry_target = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run_target, user_name=username)

    op = data_entry_target.open()
    core_profiles.put(db_entry = data_entry_target)
    data_entry_target.close()

def use_flat_q_profile(db, shot, run, run_target, username = ''):

    '''

    Writes a new ids with a flat q profile

    '''
    if username == '':
        username = getpass.getuser()

    core_profiles = open_and_get_core_profiles(db, shot, run, username)
    q = []

    len_time = len(core_profiles.time)
    len_x = len(core_profiles.profiles_1d[0].q)
    q_edge_1 = core_profiles.profiles_1d[0].q[-1]

    q = np.full((len_time, len_x), q_edge_1)

    for index, q_slice in enumerate(q):
        core_profiles.profiles_1d[index].q = q_slice

    copy_ids_entry(username, db, shot, run, shot, run_target)

    data_entry_target = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run_target, user_name=username)

    op = data_entry_target.open()
    core_profiles.put(db_entry = data_entry_target)
    data_entry_target.close()

def use_flat_vloop(db, shot, run_relaxed, run_target, username = ''):

    '''

    Substitute the q profile from a run where the q profile was relaxed.
    Such run should be specified

    '''

    if username == '':
        username = getpass.getuser()

    core_profiles = open_and_get_core_profiles(db, shot, run_relaxed, username)
    q_slice = core_profiles.profiles_1d[-1].q

    for index, q_slice in enumerate(q):
        core_profiles.profiles_1d[index] = q_slice

    copy_ids_entry(username, db, shot, run_relaxed, shot_target, run_target)

    data_entry_target = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run_target, user_name=username)

    op = data_entry_target.open()
    core_profiles.put(db_entry = data_entry_target)
    data_entry_target.close()


def check_and_flip_ip(db, shot, run, shot_target, run_target):

    equilibrium = open_and_get_equilibrium(db, shot, run)
    if equilibrium.time_slice[0].global_quantities.ip > 0:
        flip_ip(db, shot, run, shot_target, run_target)

        print('ip was positive for shot ' + str(shot) + ' and was flipped to negative')

def flip_ip(db, shot, run, shot_target, run_target):

    username = getpass.getuser()

    equilibrium = open_and_get_equilibrium(db, shot, run)
    copy_ids_entry(username, db, shot, run, shot_target, run_target)

    equilibrium_new = copy.deepcopy(equilibrium)

    for itime, time_slice in enumerate(equilibrium.time_slice):
        equilibrium_new.time_slice[itime].global_quantities.ip = -equilibrium.time_slice[itime].global_quantities.ip

    equilibrium_new.vacuum_toroidal_field.b0 = -equilibrium.vacuum_toroidal_field.b0

    data_entry_target = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot_target, run_target, user_name=getpass.getuser())

    op = data_entry_target.open()
    equilibrium_new.put(db_entry = data_entry_target)
    data_entry_target.close()

# ------------------------------- KINETIC PROFILES MANIPULATION ---------------------------------

def peak_temperature(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, mult = 1):

    '''

    Writes a new IDS with a more (or less) peaked electron temperature profile. The value at the boundary is kept constant. The new version is still untested

    '''

    if not username: username = getpass.getuser()
    if not username_target: username_target = username
    if not db_target: db_target = db
    if not shot_target: shot_target = shot

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    e_temperatures = ids_dict['profiles_1d']['electrons.temperature']
    new_e_temperature = []

    for e_temperature in e_temperatures:
        new_e_temperature.append(mult*(e_temperature - e_temperature[-1]) + e_temperature[-1])

    ids_dict['profiles_1d']['electrons.temperature'] = np.asarray(new_e_temperature)

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)

    print('temperature_peaked')

def correct_boundaries_te(db, shot, run, db_target, shot_target, run_target, username = None, username_target = None, verbose = False):

    '''

    Writes a new IDS with a corrected value at the boundaries. With 'corrected' it is meant a value larger than 20 eV, since I
    do not think that a lower value at the separatrix would be physical. The boundary is raised and then everything is shifted linearly.
    The same value is kept for the axis. The same is done for the ion temperature

    '''

    if not username:
        username=getpass.getuser()

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    e_temperatures = ids_dict['profiles_1d']['electrons.temperature']
    i_temperatures = ids_dict['profiles_1d']['t_i_average']
    rhos = ids_dict['profiles_1d']['grid.rho_tor_norm']

    new_e_temperatures, new_i_temperatures = [], []

    for rho, e_temperature, index in zip(rhos, e_temperatures, range(len(rhos))):
        if e_temperature[-1] < 20:
            new_e_temperatures.append(e_temperature+(20-e_temperature[-1])*rho)
            index_modified = index
        else:
            new_e_temperatures.append(e_temperature)

    for rho, i_temperature, index in zip(rhos, i_temperatures, range(len(rhos))):
        if i_temperature[-1] < 20:
            new_i_temperatures.append(i_temperature+(20-i_temperature[-1])*rho)
            index_modified = index
        else:
            new_i_temperatures.append(i_temperature)

    new_e_temperatures = np.asarray(new_e_temperatures).reshape(len(e_temperatures),len(e_temperatures[0]))
    ids_dict['profiles_1d']['electrons.temperature'] = new_e_temperatures

    new_i_temperatures = np.asarray(new_i_temperatures).reshape(len(i_temperatures),len(i_temperatures[0]))
    ids_dict['profiles_1d']['t_i_average'] = new_i_temperatures
    # Not really needed but trying to maintain consistency in case is needed later. Might put a loop later, only 2 imp supported now
    ids_dict['profiles_1d']['ion[0].temperature'] = new_i_temperatures
    ids_dict['profiles_1d']['ion[1].temperature'] = new_i_temperatures

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)

    if verbose and index_modified:
        fig, axs = plt.subplots(1,1)
        axs.plot(rhos[index_modified], e_temperatures[index_modified], 'r-', label = 'Te old')
        axs.plot(rhos[index_modified], new_e_temperatures[index_modified], 'b-', label = 'Te new')
        fig.legend()
        plt.show()

    print('Boundaries corrected')


#def set_boundaries(db, shot, run, run_target, te_sep, ti_sep = None, method_te = 'constant', method_ti = None, bound_te_up = False, bound_te_down = False, db_target = None, shot_target = None, username = None):

def set_boundaries(db, shot, run, run_target, extra_boundary_instructions = {}, db_target = None, shot_target = None, username = None):

    '''

    Writes a new IDS with a corrected value at the boundaries. With 'corrected' it is meant a value larger than 20 eV, since I
    do not think that a lower value at the separatrix would be physical. The boundary is raised and then everything is shifted linearly.
    The same value is kept for the axis. The same is done for the ion temperature

    '''

    if not username: username = getpass.getuser()
    if not db_target: db_target = db
    if not shot_target: shot_target = shot
    #if not ti_sep: ti_sep = te_sep

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    e_temperatures = ids_dict['profiles_1d']['electrons.temperature']
    i_temperatures = ids_dict['profiles_1d']['t_i_average']
    e_densities = ids_dict['profiles_1d']['electrons.density']
    rhos = ids_dict['profiles_1d']['grid.rho_tor_norm']
    times = ids_dict['time']['core_profiles']

    method_te = extra_boundary_instructions['method te']
    method_ti = extra_boundary_instructions['method ti']
    te_sep = extra_boundary_instructions['te sep']
    if extra_boundary_instructions['ti sep']:
        ti_sep = extra_boundary_instructions['ti sep']
    else:
        ti_sep = False

    if extra_boundary_instructions['bound te up']:
        bound_te_up = extra_boundary_instructions['bound te up']
    else:
        bound_te_up = False

    if extra_boundary_instructions['bound te down']:
        bound_te_down = extra_boundary_instructions['bound te down']
    else:
        bound_te_down = False

    time_continuity = extra_boundary_instructions['time continuity']
    temp_start = extra_boundary_instructions['temp start']

    te_sep_time, ti_sep_time = [], []
    for itime, time in enumerate(times):

        # Setting te
        if method_te == 'constant':
            te_sep_time.append(te_sep)

        elif method_te == 'linear':
            if te_sep is list:
                te_sep_time.append((te_sep[1]-te_sep[0])*(time-time[0])/time[-1])
            else:
                print('te sep needs to be a list with the first and the last value when method is linear. Aborting')
                exit()

        elif method_te == 'add':
            te_sep_time.append(e_temperatures[itime][-1] + te_sep)

        elif method_te == 'add no start':
            if itime > 1 and time < time_continuity:
                te_sep_time.append(e_temperatures[itime][-1] + te_sep*(time_continuity-time)/time_continuity)
            elif itime > 1 and time > time_continuity:
                te_sep_time.append(e_temperatures[itime][-1] + te_sep)
            else:
                te_sep_time.append(e_temperatures[itime][-1])

        elif method_te == 'add early':
            # Sets initial temperature at temp_start eV and goes linearly to whatever value there is at temp_continuity. Still adds te_sep
            f_space = interp1d(times, e_temperatures[:,-1])
            temp_continuity = f_space(time_continuity)

            if time < time_continuity:
                te_sep_time.append((temp_continuity - temp_start + te_sep)/time_continuity*time + temp_start)
            else:
                te_sep_time.append(e_temperatures[itime][-1] + te_sep)

        elif method_te == 'add early high':
            # Sets initial temperature at 100 eV and goes linearly to whatever value there is at 0.05. Still adds. Now that I added flexibility should be the same as add early
            f_space = interp1d(times, e_temperatures[:,-1])
            temp_continuity = f_space(time_continuity)

            if time < time_continuity:
                te_sep_time.append((temp_continuity - temp_start + te_sep)/time_continuity*time + temp_start)
            else:
                te_sep_time.append(e_temperatures[itime][-1] + te_sep)

        else:
            print('method for boundary settings not recognized. Aborting')
            exit()

        # Setting ti

        if not method_ti:
            # Still want something here. I will secretly correct Ti when is clearly too low or too high.
            ti_sep_time.append(i_temperatures[itime][-1])

        elif method_ti == 'constant':
            if ti_sep == 'te':
                ti_sep_time.append(te_sep)
            else:
                ti_sep_time.append(ti_sep)

        elif method_ti == 'linear':
            if ti_sep is list:
                ti_sep_time.append((ti_sep[1]-ti_sep[0])*(time-time[0])/time[-1])
            elif ti_sep == 'te':
                ti_sep_time.append((te_sep[1]-te_sep[0])*(time-time[0])/time[-1])
            else:
                print('ti sep needs to be a list with the first and the last value when method is linear. Aborting')
                exit()

        elif method_ti == 'add':
            if ti_sep == 'te':
                ti_sep_time.append(e_temperatures[itime][-1] + te_sep)
            else:
                ti_sep_time.append(i_temperatures[itime][-1] + ti_sep)

        elif method_ti == 'add on te':
            ti_sep_time.append(e_temperatures[itime][-1] + ti_sep)

        elif method_ti == 'add on te profile':
            ti_sep_time.append(e_temperatures[itime][-1] + ti_sep)

        elif method_ti == 'add no start':
            if itime > 1 and time < time_continuity:
                if ti_sep == 'te':
                    ti_sep_time.append(e_temperatures[itime][-1] + te_sep*(time_continuity-time)/time_continuity)
                else:
                    ti_sep_time.append(i_temperatures[itime][-1] + ti_sep*(time_continuity-time)/time_continuity)

            elif itime > 1 and time > time_continuity:
                if ti_sep == 'te':
                    ti_sep_time.append(e_temperatures[itime][-1] + te_sep)
                else:
                    ti_sep_time.append(i_temperatures[itime][-1] + ti_sep)
            else:
                if ti_sep == 'te':
                    ti_sep_time.append(e_temperatures[itime][-1])
                else:
                    ti_sep_time.append(i_temperatures[itime][-1])

        elif method_ti == 'add early':
            # Sets initial temperature at 20 eV and goes linearly to whatever value there is at 0.05. Still adds
            f_space = interp1d(times, i_temperatures[:,-1])
            t_continuity = f_space(time_continuity)

            if time < time_continuity:
                if ti_sep == 'te':
                    ti_sep_time.append((t_continuity - temp_start + te_sep)/time_continuity*time + temp_start)
                else:
                    ti_sep_time.append((t_continuity - temp_start + ti_sep)/time_continuity*time + temp_start)
            else:
                if ti_sep == 'te':
                    ti_sep_time.append(e_temperatures[itime][-1] + te_sep)
                else:
                    ti_sep_time.append(i_temperatures[itime][-1] + ti_sep)

        elif method_ti == 'add early high':
            # Sets initial temperature at 20 eV and goes linearly to whatever value there is at 0.05. Still adds
            f_space = interp1d(times, i_temperatures[:,-1])
            t_continuity = f_space(time_continuity)

            if time < time_continuity:
                ti_sep_time.append((t_continuity - temp_start + ti_sep)/time_continuity*time + temp_start)
            else:
                ti_sep_time.append(i_temperatures[itime][-1] + ti_sep)

        else:
            print('method for boundary settings not recognized. Aborting')
            exit()


    ne_sep_time = []
    for itime, time in enumerate(times):
        # Setting ne
        if extra_boundary_instructions['method ne'] == 'constant':
            ne_sep_time.append(extra_boundary_instructions['ne sep'])
        elif extra_boundary_instructions['method ne'] == 'limit':
            ne_sep_time.append(ids_dict['profiles_1d']['electrons.density'][itime][-1])

    ne_sep_time = np.asarray(ne_sep_time)

    if extra_boundary_instructions['method ne'] == 'limit':
        ne_sep_time = np.where(ne_sep_time < extra_boundary_instructions['ne sep'], ne_sep_time, extra_boundary_instructions['ne sep'])

    te_sep_time, ti_sep_time = np.asarray(te_sep_time), np.asarray(ti_sep_time)
    if bound_te_down:
        te_sep_time = np.where(te_sep_time > bound_te_down, te_sep_time, bound_te_down)
    if bound_te_up:
        te_sep_time = np.where(te_sep_time < bound_te_up, te_sep_time, bound_te_up)


    # Ti is corrected when clearly too low or high. It would unecessarily slow down the simulation
    ti_sep_time = np.where(ti_sep_time > 15, ti_sep_time, 15)
    ti_sep_time = np.where(ti_sep_time < 500, ti_sep_time, 500)

    '''
    # limit is stricter at the very beginning, where measurements are difficult
    te_sep_time_tmp = []
    for te_sep_t, time in zip(te_sep_time, times):
        if te_sep_t > 80 and time < 0.1:
            te_sep_time_tmp.append(80)
        else:
            te_sep_time_tmp.append(te_sep_t)
    te_sep_time = np.asarray(te_sep_time_tmp)

    ti_sep_time = np.where(ti_sep_time > 30, ti_sep_time, 30)
    ti_sep_time = np.where(ti_sep_time < 100, ti_sep_time, 100)

    ti_sep_time_tmp = []
    for ti_sep_t, time in zip(ti_sep_time, times):
        if ti_sep_t > 80 and time < 0.1:
            ti_sep_time_tmp.append(80)
        else:
            ti_sep_time_tmp.append(ti_sep_t)
    ti_sep_time = np.asarray(ti_sep_time_tmp)
    '''

    new_e_temperatures, new_i_temperatures = [], []

    for rho, e_temperature, index in zip(rhos, e_temperatures, range(len(rhos))):
        new_e_temperatures.append(e_temperature+(te_sep_time[index]-e_temperature[-1])*rho)

    new_e_temperatures = np.asarray(new_e_temperatures).reshape(len(e_temperatures),len(e_temperatures[0]))
    ids_dict['profiles_1d']['electrons.temperature'] = new_e_temperatures

    '''
    if ti_sep:
        for rho, e_temperature, i_temperature, index in zip(rhos, e_temperatures, i_temperatures, range(len(rhos))):
            if method_ti == 'add on te profile':
                new_i_temperatures.append(e_temperature+ti_sep)
            else:
                new_i_temperatures.append(i_temperature+(ti_sep_time[index]-i_temperature[-1])*rho)

    if ti_sep:
        new_i_temperatures = np.asarray(new_i_temperatures).reshape(len(i_temperatures),len(i_temperatures[0]))
        ids_dict['profiles_1d']['t_i_average'] = new_i_temperatures
        # Not really needed but trying to maintain consistency in case is needed later. Might put a loop later, only 2 imp supported now
        ids_dict['profiles_1d']['ion[0].temperature'] = new_i_temperatures
        ids_dict['profiles_1d']['ion[1].temperature'] = new_i_temperatures
    '''

    for rho, e_temperature, i_temperature, index in zip(rhos, e_temperatures, i_temperatures, range(len(rhos))):
        if method_ti == 'add on te profile':
            new_i_temperatures.append(e_temperature+ti_sep)
        else:
            new_i_temperatures.append(i_temperature+(ti_sep_time[index]-i_temperature[-1])*rho)

    new_i_temperatures = np.asarray(new_i_temperatures).reshape(len(i_temperatures),len(i_temperatures[0]))
    ids_dict['profiles_1d']['t_i_average'] = new_i_temperatures
    # Not really needed but trying to maintain consistency in case is needed later. Might put a loop later, only 2 imp supported now
    ids_dict['profiles_1d']['ion[0].temperature'] = new_i_temperatures
    ids_dict['profiles_1d']['ion[1].temperature'] = new_i_temperatures


    if extra_boundary_instructions['method ne']:
        new_e_densities = []
        for rho, e_density, index in zip(rhos, e_densities, range(len(rhos))):
            new_e_densities.append(e_density+(ne_sep_time[index]-e_density[-1])*rho)

        new_e_densities = np.asarray(new_e_densities).reshape(len(e_densities),len(e_densities[0]))
        ids_dict['profiles_1d']['electrons.density'] = new_e_densities
        # Not really needed but trying to maintain consistency in case is needed later. Might put a loop later, only 2 imp supported now
        ids_dict['profiles_1d']['electrons.density_thermal'] = new_e_densities

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)

    print('Set boundaries completed')


def alter_q_profile_same_q95(db, shot, run, run_target, db_target = None, shot_target = None,  username = None, username_target = None, mult = 1):

    '''

    Writes a new IDS with the same q95, but changing the value of q0. This version is untested

    '''

    if not username: username = getpass.getuser()
    if not username_target: username_target = username
    if not db_target: db_target = db
    if not shot_target: shot_target = shot

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    equilibrium = open_and_get_ids(db, shot, run, 'equilibrium')
    r0, b0s = equilibrium.vacuum_toroidal_field.r0, equilibrium.vacuum_toroidal_field.b0
    mu0 = 4*np.pi*1.0e-7
    volumes, ips = [], []
    for slice_eq in equilibrium.time_slice:
        volumes.append(slice_eq.profiles_1d.volume[-1])
        ips.append(slice_eq.global_quantities.ip)

    volumes, ips = np.asarray(volumes), np.asarray(ips)
    q_old, rho = ids_dict['profiles_1d']['profiles_1d.q'], ids_dict['profiles_1d']['profiles_1d.rho_tor_norm']

    # Changing the q profile both in equilibrium and core profiles
    q_new = []
    for q_slice, rho_slice, volume, ip, b0 in zip(q_old, rho, volumes, ips, b0s):
        # This normalizes the q95 from the extrapolation to the value expected from the other parameters
        index_rho_95 = np.abs(rho_slice - 0.95).argmin(0)
        q95 = abs(q_slice[index_rho_95])

        q95_norm = abs(2*volume*b0/(np.pi*mu0*r0*r0*ip))
        #q_slice = q_slice/q95*q95_norm

        #print(q95_norm/q95)
        #print(q95)
        #print(q95_norm)

        # This makes it easier to decide a value for q[0]. Could live it as an option.
        mult_slice = abs(mult/q_slice[0])
        q_slice = q_slice*((1-mult_slice)/0.95*rho_slice + mult_slice)
        q_new.append(q_slice)

    ids_dict['profiles_1d']['profiles_1d.q'] = np.asarray(q_new)

    q_old, rho = ids_dict['profiles_1d']['q'], ids_dict['profiles_1d']['grid.rho_tor_norm']

    # Changing the q profile both in equilibrium and core profiles
    q_new = []
    for q_slice, rho_slice, volume, ip, b0 in zip(q_old, rho, volumes, ips, b0s):
        # This normalizes the q95 from the extrapolation to the value expected from the other parameters
        index_rho_95 = np.abs(rho_slice - 0.95).argmin(0)
        q95 = abs(q_slice[index_rho_95])

        q95_norm = abs(2*volume*b0/(np.pi*mu0*r0*r0*ip))
        #q_slice = q_slice/q95*q95_norm

        # This makes it easier to decide a value for q[0]. Could live it as an option.
        mult_slice = abs(mult/q_slice[0])
        q_slice = q_slice*((1-mult_slice)/0.95*rho_slice + mult_slice)
        q_new.append(q_slice)

    ids_dict['profiles_1d']['q'] = np.asarray(q_new)

    '''
    q_new = []
    for q_slice, rho_slice in zip(ids_dict['profiles_1d']['q'], ids_dict['profiles_1d']['grid.rho_tor_norm']):
        q_slice = q_slice*((1-mult)/0.95*rho_slice + mult)
        q_new.append(q_slice)

    ids_dict['profiles_1d']['q'] = np.asarray(q_new)
    '''

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


def correct_ion_temperature(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, ratio_limit = 2):

    '''

    Puts a limit to ti in case the ti/te ratio is too large.

    '''
    if not username: username = getpass.getuser()
    if not username_target: username_target = username
    if not db_target: db_target = db
    if not shot_target: shot_target = shot

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    #ion_temperature_keys = ['ion[0].temperature', 'ion[0].pressure_thermal', 'ion[1].temperature', 'ion[1].pressure_thermal', 't_i_average']
    # Maybe in the future allow for consistency with the pressure
    ion_temperature_keys = ['ion[0].temperature', 'ion[1].temperature', 't_i_average']
    for key in ion_temperature_keys:

        ion_temperatures = ids_dict['profiles_1d'][key]
        electron_temperatures = ids_dict['profiles_1d']['electrons.temperature']
        new_profiles = np.where(ion_temperatures/electron_temperatures < ratio_limit, ion_temperatures, ratio_limit*electron_temperatures)
        ids_dict['profiles_1d'][key] = new_profiles

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


def shift_profiles(profile_tag, db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, mult = 1):

    '''

    Multiplies the profiles for all timeslices for a fixed value. Needs a tag to work (te, ne, ti, zeff)

    '''

    if not username: username = getpass.getuser()
    if not username_target: username_target = username
    if not db_target: db_target = db
    if not shot_target: shot_target = shot

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    if profile_tag == 'te':
        dict_key = 'electrons.temperature'
    elif profile_tag == 'ne':
        dict_key = 'electrons.density'
    elif profile_tag == 'ti':
        dict_key = 'ion[0].temperature'
    elif profile_tag == 'zeff':
# Modified and untested. Not sure it maintains ambipolarity by default
        dict_key = 'ion[1].density'
#        dict_key = 'zeff'

# Could also use lists for everything and use only one of these. For example, electron pressure should also be changed.

    if dict_key == 'ion[0].temperature':
        ion_temperature_keys = ['ion[0].temperature', 'ion[0].pressure_thermal', 'ion[1].temperature', 'ion[1].pressure_thermal', 't_i_average']
        for key in ion_temperature_keys:

            new_profiles = copy.deepcopy(ids_dict['profiles_1d'][key])
            for index, profile in enumerate(ids_dict['profiles_1d'][key]):
                new_profiles[index] = mult*profile

            ids_dict['profiles_1d'][key] = new_profiles

    elif dict_key == 'electrons_density':
        density_keys = ['electrons.density', 'electrons.density_thermal']
        for key in density_keys:

            new_profiles = copy.deepcopy(ids_dict['profiles_1d'][key])
            for index, profile in enumerate(ids_dict['profiles_1d'][key]):
                new_profiles[index] = mult*profile

            ids_dict['profiles_1d'][key] = new_profiles

    else:
        new_profiles = copy.deepcopy(ids_dict['profiles_1d'][dict_key])

        for index, profile in enumerate(ids_dict['profiles_1d'][dict_key]):
            new_profiles[index] = mult*profile

        ids_dict['profiles_1d'][dict_key] = new_profiles

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


def add_early_profiles_duqtools(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, extra_early_options = []):

    set_default_entries(username, db, shot, username_target, db_target, shot_target)
    ids_names_to_extract = ['core_profiles','summary']
    dataset_im = extract_integrated_modelling(ids_names_to_extract, self.username, self.db, self.shot, self.run)

    

    ne_peaking_0 = extra_early_options['ne peaking 0']
    te_peaking_0 = extra_early_options['te peaking 0']
    ti_peaking_0 = extra_early_options['ti peaking 0']
    electron_density_option = extra_early_options['electron density option']
    ion_density_option = extra_early_options['ion density option']
    electron_temperature_option = extra_early_options['electron temperature option']
    ion_temperature_option = extra_early_options['ion temperature option']

    old_times_core_profiles = dataset_im['core_profiles'].time
    old_times_summary = dataset_im['summary'].time

    new_times_core_profiles = old_times_core_profiles[:]

    # Go backwards with the same timestep
    while new_times_core_profiles[0] > 0.01:
        new_times_core_profiles = np.insert(new_times_core_profiles, 0, new_times_core_profiles[0] - (new_times_core_profiles[1] - new_times_core_profiles[0]))

    len_added_times = len(new_times_core_profiles) - len(old_times_core_profiles)

    # The grid needs to be calculated before the first profile since the first grid might change and this can create problems

    x_dim = np.shape(dataset_im['core_profiles']['rho tor norm time'])[1]
    time_dim = np.shape(dataset_im['core_profiles']['rho tor norm time'])[0]
    first_rho_norm_old = dataset_im['core_profiles']['rho tor norm time'][0]

    # Calculate the last rho_tor (non normalized) to extrapolate the last of the rho_tor for the first time step. The rest will be equidistant
    last_radial_rho = fit_and_substitute(old_times_core_profiles, new_times_core_profiles, dataset_im['core_profiles']['rho tor'][:,-1])[0]
    first_rho_tor_norm = np.arange(x_dim)/x_dim
    first_rho_tor = first_rho_tor_norm*last_radial_rho

    # WORKING ON NOW

    first_rho_tor_norm = np.arange(x_dim)/x_dim
    first_rho_tor = first_rho_tor_norm*last_radial_rho
    # This is just to make the formulas later readable
    x = first_rho_tor_norm

    # The grid should not be back extrapolated freely (causes problems) but should be back extrapolated starting from a regularly spaced grid
    # Can probably be incorporated in the later code
    old_rhos_norm = ids_dict['profiles_1d']['grid.rho_tor_norm'][:]
    old_rhos_norm = np.insert(old_rhos_norm, 0, first_rho_tor_norm, axis = 0)
    old_times_core_profiles = np.insert(old_times_core_profiles, 0, 0.0)

    new_rhos_norm = np.asarray([])
    for i in np.arange(x_dim):
        if np.size(new_rhos_norm) == 0:
            new_rhos_norm = fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_rhos_norm[:,i])
        else:
            new_rhos_norm = np.hstack((new_rhos_norm, fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_rhos_norm[:,i])))

    new_rhos_norm = new_rhos_norm.reshape(x_dim, len(new_times_core_profiles))
    new_rhos_norm = np.transpose(new_rhos_norm)

    old_rhos = ids_dict['profiles_1d']['grid.rho_tor'][:]
    old_rhos = np.insert(old_rhos, 0, first_rho_tor, axis = 0)

    new_rhos = np.asarray([])
    for i in np.arange(x_dim):
        if np.size(new_rhos) != 0:
            new_rhos = np.hstack((new_rhos, fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_rhos[:,i])))
        else:
            new_rhos = fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_rhos[:,i])

    new_rhos = new_rhos.reshape(x_dim, len(new_times_core_profiles))
    new_rhos = np.transpose(new_rhos)

    # WORKING ON NOW



def add_early_profiles(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, extra_early_options = []):

    '''

    Inserts time_slices at the beginning of core_profiles, to try to model the early stages.
    No normalization to energy diamagnetic is done right now. Agreement with the profiles is not always great, so need to think what to do there
    For now a flat te profile is imposed at 0.01, using the last boundary value.

    '''

    if not username:
        username = getpass.getuser()
    if not username_target:
        username_target = username
    if not db_target:
        db_target = db
    if not shot_target:
        shot_target = shot

    ids_data = IntegratedModellingDict(db, shot, run, username = username)
    ids_dict = ids_data.ids_dict

    ne_peaking_0 = extra_early_options['ne peaking 0']
    te_peaking_0 = extra_early_options['te peaking 0']
    ti_peaking_0 = extra_early_options['ti peaking 0']
    electron_density_option = extra_early_options['electron density option']
    ion_density_option = extra_early_options['ion density option']
    electron_temperature_option = extra_early_options['electron temperature option']
    ion_temperature_option = extra_early_options['ion temperature option']

    old_times_core_profiles = ids_dict['time']['core_profiles']
    old_times_summary = ids_dict['time']['summary']
    new_times_core_profiles = old_times_core_profiles[:]

    # Go backwards with the same timestep
    while new_times_core_profiles[0] > 0.01:
        new_times_core_profiles = np.insert(new_times_core_profiles, 0, new_times_core_profiles[0] - (new_times_core_profiles[1] - new_times_core_profiles[0]))

    len_added_times = len(new_times_core_profiles) - len(old_times_core_profiles)

    # The grid needs to be calculated before the first profile since the first grid might change and this can create problems
    x_dim = np.shape(ids_dict['profiles_1d']['grid.rho_tor_norm'])[1]
    time_dim = np.shape(ids_dict['profiles_1d']['grid.rho_tor_norm'])[0]
    first_rho_norm_old = ids_dict['profiles_1d']['grid.rho_tor_norm'][0]

    # Calculate the last rho_tor (non normalized) to extrapolate the last of the rho_tor for the first time step. The rest will be equidistant
    last_radial_rho = fit_and_substitute(old_times_core_profiles, new_times_core_profiles, ids_dict['profiles_1d']['grid.rho_tor'][:,-1])[0]
    first_rho_tor_norm = np.arange(x_dim)/x_dim
    first_rho_tor = first_rho_tor_norm*last_radial_rho

    first_rho_tor_norm = np.arange(x_dim)/x_dim
    first_rho_tor = first_rho_tor_norm*last_radial_rho
    # This is just to make the formulas later readable
    x = first_rho_tor_norm

    # The grid should not be back extrapolated freely (causes problems) but should be back extrapolated starting from a regularly spaced grid
    # Can probably be incorporated in the later code
    old_rhos_norm = ids_dict['profiles_1d']['grid.rho_tor_norm'][:]
    old_rhos_norm = np.insert(old_rhos_norm, 0, first_rho_tor_norm, axis = 0)
    old_times_core_profiles = np.insert(old_times_core_profiles, 0, 0.0)

    new_rhos_norm = np.asarray([])
    for i in np.arange(x_dim):
        if np.size(new_rhos_norm) == 0:
            new_rhos_norm = fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_rhos_norm[:,i])
        else:
            new_rhos_norm = np.hstack((new_rhos_norm, fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_rhos_norm[:,i])))

    new_rhos_norm = new_rhos_norm.reshape(x_dim, len(new_times_core_profiles))
    new_rhos_norm = np.transpose(new_rhos_norm)

    old_rhos = ids_dict['profiles_1d']['grid.rho_tor'][:]
    old_rhos = np.insert(old_rhos, 0, first_rho_tor, axis = 0)

    new_rhos = np.asarray([])
    for i in np.arange(x_dim):
        if np.size(new_rhos) != 0:
            new_rhos = np.hstack((new_rhos, fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_rhos[:,i])))
        else:
            new_rhos = fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_rhos[:,i])

    new_rhos = new_rhos.reshape(x_dim, len(new_times_core_profiles))
    new_rhos = np.transpose(new_rhos)

    '''
    for i in np.arange(x_dim):
        if np.size(new_rho) == 0:
            new_rho = fit_and_substitute(old_times, new_times, ids_dict['profiles_1d']['grid.rho_tor_norm'][:,i])
        else:
            new_rho = np.hstack((new_rho, fit_and_substitute(old_times, new_times, ids_dict['profiles_1d']['grid.rho_tor_norm'][:,i])))

    new_rho = new_rho.reshape(x_dim, len(new_times))
    #x = np.transpose(new_rho)[0]
    new_rho = np.transpose(new_rho)
    '''

    new_profiles = {}

    '''
    # The fit in time of rho tor norm might mix the grid, which needs to be sorted.
    new_rho_tor_norm = np.asarray([])

    #The grid needs to be sorted in crescent order and normalized to the maximum or the interpolation might bring problems
    for rho_tor_norm_profile in new_rho:
        if np.size(new_rho_tor_norm) == 0:
            new_rho_tor_norm = np.sort(rho_tor_norm_profile)/max(rho_tor_norm_profile)
        else:
            new_rho_tor_norm = np.hstack((new_rho_tor_norm, np.sort(rho_tor_norm_profile)/max(rho_tor_norm_profile)))

    new_rho_tor_norm = new_rho_tor_norm.reshape(len(new_times), x_dim)
    x = new_rho_tor_norm[0]
    '''

    for variable in ['electrons.density_thermal', 'electrons.density', 'electrons.temperature', 'q', 't_i_average', 'ion[0].density']:

        old_profiles = ids_dict['profiles_1d'][variable]
        if variable == 'electrons.density_thermal' or variable == 'electrons.density':
            if electron_density_option == 'flat':
                first_profile = np.full(np.size(ids_dict['profiles_1d'][variable][0]), ids_dict['profiles_1d'][variable][0][-1])
            elif electron_density_option == 'first profile':
                # Need to remap the profile to ensure first shape for the first profile
                first_profile = fit_and_substitute(first_rho_norm_old, x, ids_dict['profiles_1d'][variable][0])
                #first_profile = ids_dict['profiles_1d'][variable][0]
            elif electron_density_option == 'linear':
                first_profile = ids_dict['profiles_1d'][variable][0][-1] + ne_peaking_0*ids_dict['profiles_1d'][variable][0][-1]*(1-x)
            elif electron_density_option == 'parabolic':
                first_profile = ids_dict['profiles_1d'][variable][0][-1] + ne_peaking_0*ids_dict['profiles_1d'][variable][0][-1]*(1-x)*(1-x)
            else:
                print('option for the first density profile not recognized. Aborting')
        # Assuming (empirically, and it makes sense) that the initial temperature at the boundaries is lower at the very beginning, while the plasma warms up
        # That did not work. Assuming boundaries are fixed, but temperature is slightly peaked
        # Not activate now. Comments left to remember the history of changes
        elif variable == 'electrons.temperature':
            if electron_temperature_option == 'linear':
                first_profile = ids_dict['profiles_1d'][variable][0][-1] + te_peaking_0*ids_dict['profiles_1d'][variable][0][-1]*(1-x)
            elif electron_temperature_option == 'parabolic':
                first_profile = ids_dict['profiles_1d'][variable][0][-1] + te_peaking_0*ids_dict['profiles_1d'][variable][0][-1]*(1-x)*(1-x)
            elif electron_temperature_option == 'first profile':
                first_profile = fit_and_substitute(first_rho_norm_old, x, ids_dict['profiles_1d'][variable][0])
            elif electron_temperature_option == 'flat':
                first_profile = np.full(np.size(ids_dict['profiles_1d'][variable][0]), ids_dict['profiles_1d'][variable][0][-1])
        # Assumes that ions and electrons are thermalized at the beginning. Should be a fair assumption
        elif variable == 't_i_average':
            # Need to polish Ti at the boundary when too large already here
            t_i_bound = ids_dict['profiles_1d'][variable][0][-1]
            if t_i_bound > 4*ids_dict['profiles_1d']['electrons.temperature'][0][-1]:
                t_i_bound = 4*ids_dict['profiles_1d']['electrons.temperature'][0][-1]
            if ion_temperature_option == 'linear':
                first_profile = t_i_bound + ti_peaking_0*t_i_bound*(1-x)
            elif ion_temperature_option == 'parabolic':
                first_profile = t_i_bound + ti_peaking_0*t_i_bound*(1-x)*(1-x)
            elif ion_temperature_option == 'first profile':
                first_profile = fit_and_substitute(first_rho_norm_old, x, ids_dict['profiles_1d'][variable][0])
            elif ion_temperature_option == 'flat':
                first_profile = np.full(np.size(ids_dict['profiles_1d'][variable][0]), t_i_bound)
            elif ion_temperature_option == 'electron first profile':
                first_profile = fit_and_substitute(first_rho_norm_old, x, ids_dict['profiles_1d']['electrons.temperature'][0])
        # Probably not necessary but to avoid crashes. Could also use the zeff routines to imposed consistency with zeff
        elif variable == 'ion[0].density':
            if ion_density_option == 'linear':
                first_profile = ids_dict['profiles_1d'][variable][0][-1] + ti_peaking_0*ids_dict['profiles_1d'][variable][0][-1]*(1-x)
            elif ion_density_option == 'parabolic':
                first_profile = ids_dict['profiles_1d'][variable][0][-1] + ti_peaking_0*ids_dict['profiles_1d'][variable][0][-1]*(1-x)*(1-x)
            elif ion_density_option == 'first profile':
                first_profile = fit_and_substitute(first_rho_norm_old, x, ids_dict['profiles_1d'][variable][0])
            elif ion_density_option == 'flat':
                first_profile = np.full(np.size(ids_dict['profiles_1d'][variable][0]), ids_dict['profiles_1d'][variable][0][-1])

        # Setting a parabolic and not flat initial q profile
        elif variable == 'q':
            if extra_early_options['flat q profile']:
                first_profile = np.full(np.size(ids_dict['profiles_1d'][variable][0]), ids_dict['profiles_1d'][variable][0][-1])
            else:
                norm = ids_dict['profiles_1d']['q'][0][-1]/2
                ave_q_profile = np.average(ids_dict['profiles_1d']['q'][0])
                first_profile = ids_dict['profiles_1d']['q'][0][-1] - norm * np.sqrt(1-x)
        else:
            first_profile = np.full(np.size(ids_dict['profiles_1d'][variable][0]), ids_dict['profiles_1d'][variable][0][-1])

        old_profiles = np.insert(old_profiles, 0, first_profile, axis = 0)

        #x_dim = np.shape(old_profiles)[1]
        #time_dim = np.shape(old_profiles)[0]
        new_profiles[variable] = np.asarray([])

        #if variable == 'electrons.density_thermal':
        #    print('first density profile is')
        #    print(first_profile)

        for i in np.arange(x_dim):

            if np.size(new_profiles[variable]) != 0:
                new_profiles[variable] = np.hstack((new_profiles[variable], fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_profiles[:,i])))
            else:
                new_profiles[variable] = fit_and_substitute(old_times_core_profiles, new_times_core_profiles, old_profiles[:,i])

        new_profiles[variable] = np.transpose(new_profiles[variable].reshape(x_dim, len(new_times_core_profiles)))

    #for variable in ['electrons.density_thermal', 'electrons.density', 'electrons.temperature', 'q', 't_i_average']:
    #    ids_dict['profiles_1d'][variable] = np.transpose(np.asarray(new_profiles[variable]))

    #When the current is negative, the fit extrapolation might flip it back to positive. Enforcing 0 current a t=0
    old_current = np.insert(ids_dict['traces']['global_quantities.ip'], 0, 0)
    old_times_equilibrium = np.insert(ids_dict['time']['equilibrium'], 0, 0)

    # This might not belong in add_early_profiles. Ideally want to do it even when not adding the early profiles. Maybe in rebase?
    # When measurement for the density are available and line averaged density is not, the line averaged density is corrected instead of just extrapolated.
    # The formula is still raw but this is necessary for predictive runs to have the feedback puff activated correctly
    first_density_boundary_measured = ids_dict['profiles_1d']['electrons.density'][0][-1]
    i_time_over_0_1 = np.argmax(old_times_summary>0.1) + 1
    #line_average_measured_point = np.sum(new_profiles[variable][i_time_over_0_1])/x_dim

    # Summary and core profile do not have the same time trace.
    # Need to build the line averaged density in the core profile time trace and the proxy in the summary time trace.
    # Actually, I will build 3 proxys for electrons.density_thermal, electrons.density, ion[0].density
    line_averaged_proxy, line_average_measured_point = {}, {}
    for variable in ['electrons.density_thermal', 'electrons.density', 'ion[0].density']:
        line_averaged_proxy[variable] = []
        for i in range(len(ids_dict['profiles_1d'][variable])):
            line_averaged_proxy[variable].append(np.sum(ids_dict['profiles_1d'][variable][i])/x_dim)

        # Modifying here the line averaged density at the beginning to correlate it to the first measured boundary value
        # Instead of having a fixed 0.5e19 value. The assumption is basically a mainly flat density profile
        # 1.2 Is arbitrary, represents a sligthly peaked profile
        line_averaged_proxy[variable] = np.insert(line_averaged_proxy[variable], 0, ids_dict['profiles_1d']['electrons.density'][0][-1]*1.2)
        f_space = interp1d(old_times_core_profiles, line_averaged_proxy[variable])
        line_average_measured_point[variable] = f_space(0.1)

    line_averaged_proxy_summary_time = {}
    for variable in ['electrons.density_thermal', 'electrons.density', 'ion[0].density']:
        line_averaged_proxy_summary_time[variable] = fit_and_substitute(old_times_core_profiles, old_times_summary, line_averaged_proxy[variable])

    # Will still try to generate reasonable data if the data is not available in the original IDS
    if len(ids_dict['traces']['line_average.n_e.value']) == 0:
        ids_dict['traces']['line_average.n_e.value'] = []
        for i in range(len(old_times_summary)):
            ids_dict['traces']['line_average.n_e.value'].append(line_averaged_proxy_summary_time['electrons.density'][i])
        print('Careful! Generating the data for the line averaged density')
        ids_dict['traces']['line_average.n_e.value'] = np.asarray(ids_dict['traces']['line_average.n_e.value'])
    else:
        for i in range(i_time_over_0_1):
            ids_dict['traces']['line_average.n_e.value'][i] = ids_dict['traces']['line_average.n_e.value'][i_time_over_0_1]*line_averaged_proxy_summary_time['electrons.density'][i]/line_average_measured_point['electrons.density']

    line_ave_density_core_profiles_time = fit_and_substitute(old_times_summary, old_times_core_profiles, ids_dict['traces']['line_average.n_e.value'])
    # In the pulse scheduler, where this is in the end taken from, the time trace is the one in core profiles.
    # This might break things when I am NOT using add_early_profiles
    #ids_dict['traces']['line_average.n_e.value'] = fit_and_substitute(old_times_summary, new_times_core_profiles, ids_dict['traces']['line_average.n_e.value'])

    # This also secretly rebase. Could turn off this option...
    ids_data.update_times(new_times_core_profiles, ['core_profiles'])
    # Maybe it does not work, especially if the coordinates start to merge. Testing
    ids_data.update_times(new_times_core_profiles, ['equilibrium'])

    # Adding the option of normalizing the initial density to the initial line average density (which might or might not be extrapolated)
    if extra_early_options['normalize density to line ave']:
        if len(ids_dict['traces']['line_average.n_e.value']) == 0:
            print('No line average density data available. Cannot normalize density. Option should be changed to false')
            exit()

        for variable in ['electrons.density_thermal', 'electrons.density', 'ion[0].density']:
            for i in range(len_added_times):
                # The line average here is a raw approximation, should be done correctly taking the radial coordinate into account
                # Also, the right profile to do this should be the first one actually measured. Here it might be extrapolated.
                # Could also implement something that does not change the boundaries but it might be a little complicated
                line_average = np.sum(new_profiles[variable][i])
                line_average_measured_point = np.sum(new_profiles[variable][len_added_times])
                new_profiles[variable][i] = new_profiles[variable][i]*line_average_measured_point/line_average*line_ave_density_core_profiles_time[i]/line_ave_density_core_profiles_time[len_added_times]

    for variable in ['electrons.density_thermal', 'electrons.density', 'electrons.temperature', 'q', 't_i_average', 'ion[0].density']:
        ids_dict['profiles_1d'][variable] = new_profiles[variable]

    ids_dict['profiles_1d']['grid.rho_tor_norm'] = new_rhos_norm
    ids_dict['profiles_1d']['grid.rho_tor'] = new_rhos
    ids_dict['traces']['global_quantities.ip'] = fit_and_substitute(old_times_equilibrium, new_times_core_profiles, old_current)

    # Should check that the correct coordinate for the HFPS is also updated. Or is it necessary?

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run, run_target, ids_data.ids_struct)


# -------------------------------- EXTRA TOOLS TO MAKE THE REST WORK -------------------------------------

class Parser(xml.sax.handler.ContentHandler):
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self.idss = []

    def startElement(self, name, attrs):
        if name == 'IDS':
            ids = dict()
            for i in attrs.getNames():
                ids[i] = attrs.getValue(i)
            self.idss.append(ids)


def fit_and_substitute(x_old, x_new, data_old):

    f_space = interp1d(x_old, data_old, fill_value = 'extrapolate')

    variable = np.array(f_space(x_new))
    variable[variable > 1.0e25] = 0

    return variable

def fit_and_substitute_nbi(x_old, x_new, data_old):

    f_space = interp1d(x_old, data_old, bounds_error = False, fill_value = 0)

    variable = np.array(f_space(x_new))
    variable[variable > 1.0e25] = 0

    return variable

def double_time(times):

    '''

    Insert middle times in a time array

    '''

    time_doubled = []

    for time_pre, time in zip(times, times[1:]):
        time_doubled.append(time_pre)
        time_doubled.append(time_pre+(time - time_pre)/2)

    time_doubled.append(times[-1])
    time_doubled.append(times[-1]+(times[-1] - times[-2])/2)

    return(time_doubled)


def create_line_list():

    color_list = 'b', 'g', 'r', 'c', 'm', 'y','k'
    line_list = '-', '--', '-.', ':', '.'

    style_list = []
    for line in line_list:
        for color in color_list:
            style_list.append(color+line)

    return style_list

def get_label(profile_tag):

    if profile_tag in profile_tag_list:
        y_label, units = profile_tag_list[profile_tag][0], profile_tag_list[profile_tag][1]
    else:
        y_label, units = profile_tag, '[-]'

    return y_label, units

def smooth(x,window_len=7,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        print('smooth only accepts 1 dimension arrays.')
        raise ValueError

    if x.size < window_len:
        print('Input vector needs to be bigger than window size.')
        raise ValueError

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print('Window is on of \'flat\', \'hanning\', \'hamming\', \'bartlett\', \'blackman\'')
        raise ValueError


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[int(window_len/2+1):-int(window_len/2-1)]

def copy_ids_entry(username, db, shot, run, shot_target, run_target, ids_list = [], backend = 'mdsplus'):

    '''

    Copies an entire IDS entry

    '''

    if username == '':
        username = getpass.getuser()

    # open input pulsefile and create output one

# path hardcoded for now, not ideal but avoids me to insert the version everytime. Might improve later
    path = '/gw/swimas/core/installer/src/3.34.0/ual/4.9.3/xml/IDSDef.xml'
    parser = Parser()
    xml.sax.parse(path, parser)

    vsplit = imas.names[0].split("_")
    imas_version = version.parse(".".join(vsplit[1:4]))
    imas_major_version = str(imas_version)[0]
    ual_version = version.parse(".".join(vsplit[5:]))

    print('Opening', username, db, imas_version, shot, run)
    idss_in = imas.ids(shot, run)
    op = idss_in.open_env(username, db, imas_major_version)

    if op[0]<0:
        print('The entry you are trying to copy does not exist')
        exit()

    print('Creating', username, db, imas_version, shot_target, run_target)
    idss_out = imas.ids(shot_target, run_target)
    idss_out.create_env(username, db, imas_major_version)
    if backend == 'mdsplus':
        idss_out = imas.DBEntry(imasdef.MDSPLUS_BACKEND, 'tcv', shot_target, run_target)
    if backend == 'hdf5':
        idss_out = imas.DBEntry(imasdef.HDF5_BACKEND, 'tcv', shot_target, run_target)
    idx = idss_out.create()[1]
    #idx = idss_out.expIdx
    #ids_list = None
    # read/write every IDS

    for ids_info in parser.idss:
        name = ids_info['name']
        maxoccur = int(ids_info['maxoccur'])
        if ids_list and name not in ids_list:
            continue
        if name == 'ec_launchers' or name == 'numerics' or name == 'sdn':
#        if name == 'ec_launchers' or name == 'numerics' or name == 'sdn' or name == 'nbi':    # test for nbi ids, temporary
            continue
            print('continue on ec launchers')  # Temporarily down due to a malfunctioning of ec_launchers ids
            print('skipping numerics')  # Not in the newest version of IMAS
        for i in range(maxoccur + 1):
            if not i:
                print('Processing', ids_info['name'])
#            if i:
#                print('Processing', ids_info['name'], i)
#            else:
#                print('Processing', ids_info['name'])

            ids = idss_in.__dict__[name]
            stdout = sys.stdout
            sys.stdout = open('/afs/eufus.eu/user/g/g2mmarin/warnings_imas.txt', 'w') # suppress warnings
            ids.get(i)
#            sys.stdout = stdout
            ids.setExpIdx(idx)
            ids.put(i)
            sys.stdout.close()
            sys.stdout = stdout

    idss_in.close()
    idss_out.close()

# -------------------------------- LISTS OF KEYS -------------------------------------

keys_list = {
    'traces': {},
    'profiles_1d': {},
    'profiles_2d': {}
}

ids_list = [
    'core_profiles',
    'core_sources',
    'ec_launchers',
    'equilibrium',
    'nbi',
    'summary',
    'thomson_scattering'
]

keys_list['profiles_2d']['core_profiles'] = [
]

#electrons.velocity_tor and grid.rho_pol_norm are untested
keys_list['profiles_1d']['core_profiles'] = [
    'q',
    'electrons.density_thermal',
    'electrons.density',
    'electrons.temperature',
    'electrons.velocity_tor',
    'electrons.pressure_thermal',
    'electrons.pressure',
    'ion[].temperature',
    'ion[].density_thermal',
    'ion[].density',
    'ion[].pressure_thermal',
    'ion[].pressure',
    't_i_average',
    'zeff',
    'grid.rho_tor',
    'grid.rho_tor_norm',
    'grid.rho_pol_norm'
]

keys_list['traces']['core_profiles'] = [
    'ip',
    'v_loop',
    'li_3',
    'energy_diamagnetic',
    'ion[].z_ion',
    'ion[].multiple_states_flag',
# This might break when interpolating (It should not, but it is still untested)
    'ion[].label',
# These two are new and might break things
    'ion[].element[].a',
    'ion[].element[].z_n',
    'ion[].element[].atoms_n'
]
keys_list['traces']['nbi'] = [
    'unit[].species.a',
    'unit[].species.z_n',
    'unit[].species.label',
    'unit[].energy.data',
    'unit[].energy.time',
    'unit[].power_launched.data',
    'unit[].power_launched.time',
    'unit[].beam_current_fraction.data',
    'unit[].beam_current_fraction.time',
    'unit[].beam_power_fraction.data',
    'unit[].beam_power_fraction.time'
]
keys_list['profiles_1d']['nbi'] = []

# This list should be passed to the pulse_schedule IDS
keys_list['profiles_1d']['summary'] = []
keys_list['traces']['summary'] = [
    'global_quantities.ip.value',
    'heating_current_drive.power_nbi.value',
    'heating_current_drive.power_ic.value',
    'heating_current_drive.power_ec.value',
    'heating_current_drive.power_lh.value',
    'stationary_phase_flag.value',
    'line_average.n_e.value',
    'global_quantities.v_loop.value',
    'global_quantities.li.value',
    'global_quantities.li_mhd.value',
    'global_quantities.energy_diamagnetic.value',
    'global_quantities.energy_mhd.value',
    'global_quantities.energy_thermal.value',
    'global_quantities.beta_pol.value',
    'global_quantities.beta_pol_mhd.value',
    'global_quantities.beta_tor_norm.value',
    'global_quantities.power_radiated.value',
    'global_quantities.volume.value',
    'fusion.neutron_fluxes.total.value'
]

keys_list['profiles_1d']['equilibrium'] = [
    'profiles_1d.psi',
    'profiles_1d.phi',
    'profiles_1d.f',
    'profiles_1d.q',
    'profiles_1d.pressure',
    'profiles_1d.rho_tor',
    'profiles_1d.rho_tor_norm',
    'profiles_1d.area',
    'profiles_1d.volume',
    'boundary.outline.r',
    'boundary.outline.z',
    'profiles_2d[].grid.dim1',
    'profiles_2d[].grid.dim2'
]

keys_list['traces']['equilibrium'] = [
    'global_quantities.ip',
    'global_quantities.li_3',
    'global_quantities.beta_pol',
    'global_quantities.beta_tor',
    'global_quantities.magnetic_axis.r',
    'global_quantities.magnetic_axis.z',
    'profiles_2d[].grid_type.name',
    'profiles_2d[].grid_type.index'
]

keys_list['profiles_2d']['equilibrium'] = ['profiles_2d[].psi']

keys_list['profiles_1d']['core_sources'] = [
    'total#electrons.energy',
    'total#total_ion_energy',
    'total#j_parallel',
    'total#momentum_tor',
    'total#ion[].particles',
    'total#grid.rho_tor_norm',
    'nbi#electrons.energy',
    'nbi#total_ion_energy',
    'nbi#j_parallel',
    'nbi#momentum_tor',
    'nbi#ion[].particles',
    'nbi#grid.rho_tor_norm',
    'ec#electrons.energy',
    'ec#total_ion_energy',
    'ec#j_parallel',
    'ec#momentum_tor',
    'ec#ion[].particles',
    'ec#grid.rho_tor_norm',
    'lh#electrons.energy',
    'lh#total_ion_energy',
    'lh#j_parallel',
    'lh#momentum_tor',
    'lh#ion[].particles',
    'lh#grid.rho_tor_norm',
    'ic#electrons.energy',
    'ic#total_ion_energy',
    'ic#j_parallel',
    'ic#momentum_tor',
    'ic#ion[].particles',
    'ic#grid.rho_tor_norm'
]

keys_list['traces']['core_sources'] = [
    'total#ion[].element[].a',
    'total#ion[].element[].z_n',
    'total#ion[].element[].atoms_n',
    'nbi#ion[].element[].a',
    'nbi#ion[].element[].z_n',
    'nbi#ion[].element[].atoms_n',
    'ec#ion[].element[].a',
    'ec#ion[].element[].z_n',
    'ec#ion[].element[].atoms_n',
    'lh#ion[].element[].a',
    'lh#ion[].element[].z_n',
    'lh#ion[].element[].atoms_n',
    'ic#ion[].element[].a',
    'ic#ion[].element[].z_n',
    'ic#ion[].element[].atoms_n'
]

#keys_list['traces']['nbi'] = ['unit[]energy.data']
#keys_list['profiles_1d']['nbi'] = ['unit[]beam_power_fraction.data']

keys_list['traces']['ec_launchers'] = []
keys_list['profiles_1d']['ec_launchers'] = []

keys_list['traces']['thomson_scattering'] = []
keys_list['profiles_1d']['thomson_scattering'] = []

profile_tag_list = {
    'q_' : ['q profile', '[-]'],
    'electrons_temperature' : ['electron temperature', '[eV]'],
    't_i_average' : ['ion temperature', '[eV]'],
    'electrons_density' : ['electron density', r'[$m^{-3}$]']
}

if __name__ == "__main__":

    #setup_input('tcv', 64965, 5, 1500, json_dict, time_start = 0, time_end = 100, verbose = False, core_profiles = None, equilibrium = None)
    copy_ids_entry('g2mmarin', 'tcv', 64965, 1010, 64965, 1, backend = 'hdf5')







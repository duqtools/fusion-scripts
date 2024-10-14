import os,datetime,sys
from scipy.interpolate import interp1d, UnivariateSpline
from packaging import version

import duqtools
from duqtools.api import ImasHandle
from imas2xarray import Variable
from imas2xarray import rebase_on_time

import xarray as xr

import copy
import numpy as np
import jetto_tools
import functools
import logging
import warnings
import getpass
import xml.sax
import xml.sax.handler
import argparse
import shutil
import math

import imas
if imas is not None:
    from imas import imasdef
    vsplit = imas.names[0].split("_")
    imas_version = version.parse(".".join(vsplit[1:4]))
    ual_version = version.parse(".".join(vsplit[5:]))

variables_nbi = [
    Variable(name='a',
        ids = 'nbi',
        path = 'unit/*/species/a',
        dims = ['unit']),
    Variable(name='z_n',
        ids = 'nbi',
        path = 'unit/*/species/z_n',
        dims = ['unit']),
    Variable(name='energy_data',
        ids = 'nbi',
        path = 'unit/*/energy/data',
        dims = ['unit', 'energy_time']),
    Variable(name='energy_time',
        ids = 'nbi',
        path = 'unit/*/energy/time',
        dims = ['unit', 'energy_time']),
    Variable(name='power_launched_data',
        ids = 'nbi',
        path = 'unit/*/power_launched/data',
        dims = ['unit', 'power_launched_time']),
    Variable(name='power_launched_time',
        ids = 'nbi',
        path = 'unit/*/power_launched/time',
        dims = ['unit', 'power_launched_time']),
    Variable(name='beam_current_fraction_data',
        ids = 'nbi',
        path = 'unit/*/beam_current_fraction/data',
        dims = ['unit', 'energies', 'beam_current_fraction_time']),
    Variable(name='beam_current_fraction_time',
        ids = 'nbi',
        path = 'unit/*/beam_current_fraction/time',
        dims = ['unit', 'beam_current_fraction_time']),
    Variable(name='beam_power_fraction_data',
        ids = 'nbi',
        path = 'unit/*/beam_power_fraction/data',
        dims = ['unit', 'energies', 'beam_power_fraction_time']),
    Variable(name='beam_power_fraction_time',
        ids = 'nbi',
        path = 'unit/*/beam_power_fraction/time',
        dims = ['unit', 'beam_power_fraction_time'])
]

variable_time_nbi = [
    Variable(name='time',
        ids = 'nbi',
        path = 'time',
        dims = ['time']),
]

variables_ec_launchers = [
   Variable(name='beam_name',
        ids = 'ec_launchers',
        path = 'beam/*/name',
        dims = ['beam']),

   Variable(name='power_launched_data',
        ids = 'ec_launchers',
        path = 'beam/*/power_launched/data',
        dims = ['beam', 'power_launched_time']),
   Variable(name='power_launched_time',
        ids = 'ec_launchers',
        path = 'beam/*/power_launched/time',
        dims = ['beam', 'power_launched_time']),

    Variable(name='frequency_data',
        ids = 'ec_launchers',
        path = 'beam/*/frequency/data',
        dims = ['beam', 'frequency_time']),
    Variable(name='frequency_time',
        ids = 'ec_launchers',
        path = 'beam/*/frequency/time',
        dims = ['beam', 'frequency_time']),

    Variable(name='beam_launching_position_r_data',
        ids = 'ec_launchers',
        path = 'beam/*/launching_position/r',
        dims = ['beam', 'beam_launching_position_r_time']),
    Variable(name='beam_launching_position_r_time',
        ids = 'ec_launchers',
        path = 'beam/*/time',
        dims = ['beam', 'beam_launching_position_r_time']),

    Variable(name='beam_launching_position_z_data',
        ids = 'ec_launchers',
        path = 'beam/*/launching_position/z',
        dims = ['beam', 'beam_launching_position_z_time']),
    Variable(name='beam_launching_position_z_time',
        ids = 'ec_launchers',
        path = 'beam/*/time',
        dims = ['beam', 'beam_launching_position_z_time']),

    Variable(name='beam_launching_position_phi_data',
        ids = 'ec_launchers',
        path = 'beam/*/launching_position/phi',
        dims = ['beam', 'beam_launching_position_phi_time']),
    Variable(name='beam_launching_position_phi_time',
        ids = 'ec_launchers',
        path = 'beam/*/time',
        dims = ['beam', 'beam_launching_position_phi_time']),

    Variable(name='steering_angle_tor_data',
        ids = 'ec_launchers',
        path = 'beam/*/steering_angle_tor',
        dims = ['beam', 'steering_angle_tor_time']),
    Variable(name='steering_angle_tor_time',
        ids = 'ec_launchers',
        path = 'beam/*/time',
        dims = ['beam', 'steering_angle_tor_time']),

    Variable(name='steering_angle_pol_data',
        ids = 'ec_launchers',
        path = 'beam/*/steering_angle_pol',
        dims = ['beam', 'steering_angle_pol_time']),
    Variable(name='steering_angle_pol_time',
        ids = 'ec_launchers',
        path = 'beam/*/time',
        dims = ['beam', 'steering_angle_pol_time']),

    Variable(name='spot_size_data',
        ids = 'ec_launchers',
        path = 'beam/*/spot/size',
        dims = ['beam', 'sizes', 'spot_size_time']),
    Variable(name='spot_size_time',
        ids = 'ec_launchers',
        path = 'beam/*/time',
        dims = ['beam', 'spot_size_time']),

    Variable(name='spot_angle_data',
        ids = 'ec_launchers',
        path = 'beam/*/spot/angle',
        dims = ['beam', 'spot_angle_time']),
    Variable(name='spot_angle_time',
        ids = 'ec_launchers',
        path = 'beam/*/time',
        dims = ['beam', 'spot_angle_time']),

    Variable(name='phase_curvature_data',
        ids = 'ec_launchers',
        path = 'beam/*/phase/curvature',
        dims = ['beam', 'sizes', 'phase_curvature_time']),
    Variable(name='phase_curvature_time',
        ids = 'ec_launchers',
        path = 'beam/*/time',
        dims = ['beam', 'phase_curvature_time']),

    Variable(name='phase_angle_data',
        ids = 'ec_launchers',
        path = 'beam/*/phase/angle',
        dims = ['beam', 'phase_angle_time']),
    Variable(name='phase_angle_time',
        ids = 'ec_launchers',
        path = 'beam/*/time',
        dims = ['beam', 'phase_angle_time']),
]

variable_time_ec_launchers = [
    Variable(name='time',
        ids = 'ec_launchers',
        path = 'time',
        dims = ['time']),
]

def get_variable_system(name, system):

    variable_found = None
    if system == 'nbi':
        variables = variables_nbi
    elif system == 'ec_launchers':
        variables = variables_ec_launchers

    for variable in variables:
        if variable.name == name:
            variable_found = variable

    return variable_found


def create_dummy_dataset(name_variable, name_variable_time):

    data_template_not_empty = xr.Dataset(
        {
            name_variable: ([name_variable_time], [0.1,0.1])
        },
        coords={
            name_variable_time: [0,1000]
        }
    )
    return data_template_not_empty



def extract_system_data(user, db, shot, run, system, backend = None):

    if not backend: backend = get_backend(db, shot, run_input)

    handle = ImasHandle(user = user, db = db, shot = shot, run = run)
    dataset = xr.Dataset()

    # There can be different PINIs, beam, units or launchers depending on if it is a ECRH or a NBI. Here we are trying to be flexible with this
    subsystem_name = 'unit'

    if system == 'nbi':
        variables_system = variables_nbi
        subsystem_name = 'unit'
    elif system == 'ec_launchers':
        variables_system = variables_ec_launchers
        subsystem_name = 'beam'

    num_subsystems = 0

    for variable in variables_system:
        if variable.name.endswith('data'):
            variable_time_name = variable.name.replace('data', 'time')
            variable_time = get_variable_system(variable_time_name, system)
            try:
                single_dataset = handle.get_variables([variable, variable_time])
                dataset = xr.merge([dataset, single_dataset])
            except (ValueError, duqtools.ids._mapping.EmptyVarError):
                single_array = []
                # Check if at least one of the units has data
                if subsystem_name not in dataset.dims:
                    # I think that if the first one is empty this crashes. Trying to prevent that
                    # Should put the check for empty IDS later I guess
                    if dataset.variables or dataset.dims:
                        print('there is no ' + system + ' IDS to work with, aborting generation')
                        exit()

                for index in range(dataset.dims[subsystem_name]):
                    try:
                        variable_single, variable_time_single = copy.deepcopy(variable), copy.deepcopy(variable_time)
                        variable_single.path = variable.path.replace('*',str(index))
                        variable_time_single.path = variable_time.path.replace('*',str(index))
                        variable_single.dims.remove(subsystem_name)
                        variable_time_single.dims.remove(subsystem_name)
                        dataset_slice = handle.get_variables([variable_single, variable_time_single])
                        single_array.append(dataset_slice[variable_single.name].data)
                    except (ValueError, duqtools.ids._mapping.EmptyVarError):
                        pass
                # fill the dataset using zeros where there is no data
                if len(single_array) != 0:
                    dataset_slice_template = dataset_slice
                    single_dataset = xr.Dataset()
                    single_dataset = single_dataset.expand_dims(subsystem_name)
                    for index in range(dataset.dims[subsystem_name]):
                        variable_single, variable_time_single = copy.deepcopy(variable), copy.deepcopy(variable_time)
                        variable_single.path = variable.path.replace('*',str(index))
                        variable_time_single.path = variable_time.path.replace('*',str(index))
                        variable_single.dims.remove(subsystem_name)
                        variable_time_single.dims.remove(subsystem_name)
                       # This will work when there is only one value in the ids but it is supposed to be an array
                        try:
                            dataset_slice = handle.get_variables([variable_single, variable_time_single])
                        except ValueError:
                            dataset_slice_variable = handle.get_variables([variable_single])
                            dataset_slice_time = handle.get_variables([variable_time_single])

                            if dataset_slice_variable[variable_single.name].values.size == 1:
                                constant_value = dataset_slice_variable[variable_single.name].values[0]
                                # Not sure this is the most elegant way to do this
                                dataset_slice = copy.deepcopy(dataset_slice_time)
                                dataset_slice[variable_single.name] = xr.full_like(dataset_slice_time[variable_time_single.name], constant_value)

                        if dataset_slice[variable_time_single.name].values.size == 0:

                            if dataset_slice_template[variable_time_single.name].size == 0:
                                dataset_slice_template = create_dummy_dataset(variable_single.name, variable_time_single.name)

                            dataset_slice = xr.zeros_like(dataset_slice_template)

                        if not np.array_equal(dataset_slice[variable_time_single.name].values, dataset_slice_template[variable_time_single.name].values):
                            time_instance = dataset_slice[variable_time_single.name].values
                            time_template = dataset_slice_template[variable_time_single.name].values
                            # Creates a new time array, which needs to include all the times and be in crescent order
                            time = np.sort(np.asarray(list(set(np.hstack((time_instance, time_template))))))
                            dataset_slice = eval("dataset_slice.interp(" + variable_time_single.name + "=time,kwargs={'fill_value':'extrapolate'})")
                            if index != 0:
                                single_dataset = eval("single_dataset.interp(" + variable_time_single.name + "=time,kwargs={'fill_value':'extrapolate'})")
                            dataset_slice_template = dataset_slice

                        dataset_slice = dataset_slice.assign_coords({subsystem_name:index})
                        dataset_slice[subsystem_name] = index
                        if index == 0:
                            single_dataset = xr.merge([single_dataset, dataset_slice])
                        else:
                            single_dataset = xr.concat([single_dataset, dataset_slice], dim = subsystem_name)

            dataset = xr.merge([dataset, single_dataset])

        else:
            try:
                single_dataset = handle.get_variables([variable])
                dataset = xr.merge([dataset, single_dataset])
            except (ValueError, duqtools.ids._mapping.EmptyVarError):
                len_subsystem = count_subsystem(db, shot, run, system)
                if len(dataset.data_vars) == 0 and len(dataset.dims) == 0:
                    dataset = xr.Dataset({'dummy': (['beam'], np.zeros(len_subsystem))}, coords={'beam': np.arange(len_subsystem)})

    if system == 'nbi':
        extra_info = get_nbi_extra(db, shot, run, user = user, backend = backend)
    elif system == 'ec_launchers':
        extra_info = get_ec_launchers_extra(db, shot, run, user = user, backend = backend)

    # Dummy is not always created, delete it if present
    if 'dummy' in dataset.data_vars:
        dataset = dataset.drop_vars('dummy')

    return dataset, extra_info


def count_subsystem(db, shot, run, system, user = None, backend = None):

    if not backend: backend = get_backend(db, shot, run)
    if not user: username=getpass.getuser()

    system_ids = open_and_get_ids(db, shot, run, system, username=user, backend=backend)

    if system == 'nbi':
        len_subsystem = len(system_ids.unit)
    elif system == 'ec_launchers':
        len_subsystem = len(system_ids.beam)

    return len_subsystem


def get_nbi_extra(db, shot, run, user = None, backend = None):

    if not backend: backend = get_backend(db, shot, run)
    if not user: user=getpass.getuser()

    nbi = open_and_get_ids(db, shot, run, 'nbi', username=user, backend=backend)

    extra_info = {}
    extra_info['labels'] = []
    for unit in nbi.unit:
        extra_info['labels'].append(unit.species.label)

    return extra_info


def get_ec_launchers_extra(db, shot, run, user = None, backend = None):

    if not backend: backend = get_backend(db, shot, run)
    if not user: username=getpass.getuser()

    ec_launchers = open_and_get_ids(db, shot, run, 'ec_launchers', username=user, backend=backend)

    extra_info = {}
    extra_info['name'], extra_info['identifier'], extra_info['mode']  = [], [], []
    for beam in ec_launchers.beam:
        extra_info['name'].append(beam.name)
        extra_info['identifier'].append(beam.identifier)
        extra_info['mode'].append(beam.mode)

    return extra_info


def fill_nbi_iden_ids(nbi_ids, db):
    if db == 'tcv':
        nbi_ids.unit[0].name = '25KeV 1st NBH source'
        nbi_ids.unit[1].name = '50KeV 2nd NBH source'
        nbi_ids.unit[2].name = 'diagnostic NBI'
        nbi_ids.unit[0].identifier = 'NB1'
        nbi_ids.unit[1].identifier = 'NB2'
        nbi_ids.unit[2].identifier = 'DNBI'


def force_column_sum_to_one(array):
    array_new = []
    for array_unit in array:
        array_unit_new = force_column_sum_to_one_unit(array_unit)
        array_new.append(array_unit_new)

    array = np.asarray(array_new)

    return array

def force_column_sum_to_one_unit(array):
    col_sums = np.sum(array, axis=0)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    normalized_array = array / col_sums

    return normalized_array


def substitute_dummy_when_zero(array):
    array_new = []
    for array_unit in array:
        array_unit_new = substitute_dummy_when_zero_unit(array_unit)
        array_new.append(array_unit_new)

    array = np.asarray(array_new)

    return array


def substitute_dummy_when_zero_unit(array):

    dummy_fractions = [1.0, 0.0, 0.0]
    col_sums = np.sum(array, axis=0)

    zero_sum_cols = np.where(col_sums == 0)[0]

    normalized_array = np.copy(array)
    normalized_array[:, zero_sum_cols] = np.transpose(np.array(dummy_fractions*np.size(zero_sum_cols)).reshape(np.size(zero_sum_cols),3))

    return normalized_array


def substitute_dummy_when_low_power(array, array_power):
    array_new = []
    for array_unit in array:
        array_unit_new = substitute_dummy_when_low_power_unit(array_unit, array_power)
        array_new.append(array_unit_new)

    array = np.asarray(array_new)

    return array


def create_dummy_core_sources(db, shot, run_input, run_start, username = None, db_target = None, shot_target = None, username_target = None, backend = None):

    if not username: username=getpass.getuser()
    if not db_target: db_target = db
    if not shot_target: shot_target = shot
    if not username_target: username_target = username
    if not backend: backend = get_backend(db, shot, run_input)

    # Could rewrite this not to need fusion script, but this is just a workaround so probably not useful...
    try:
        from prepare_im_input import IntegratedModellingDict
    except ImportError:
        warnings.warn("dummy core sources requires fusion scripts. Please download from git", UserWarning)
        exit()

    ids_data = IntegratedModellingDict(db, shot, run_input, username = username, backend = backend)

    # Here is where the course sources is created
    ids_data.fill_basic_quantities('core_sources')
    ids_data.ids_struct['core_sources'].time = np.asarray([0.1])

    ids_data.ids_struct['core_sources'].vacuum_toroidal_field.r0 = 0.88
    ids_data.ids_struct['core_sources'].vacuum_toroidal_field.b0 = np.asarray([1.0])

    source_dummy = imas.core_sources().source.getAoSElement()
    profiles_1d_dummy = source_dummy.profiles_1d.getAoSElement()
    global_quantities_dummy = source_dummy.global_quantities.getAoSElement()

    #profiles_1d_dummy.total_ion_energy = np.asarray([0])
    #global_quantities_dummy.power = 0.0

    dummy_array = np.linspace(0, 1, 50)

    ids_data.ids_struct['core_sources'].source.append(source_dummy)
    #ids_data.ids_struct['core_sources'].source.append(source_dummy)

    ids_data.ids_struct['core_sources'].source[0].species.type.name = 'electron'
    ids_data.ids_struct['core_sources'].source[0].species.type.index = 1


    ids_data.ids_struct['core_sources'].source[0].profiles_1d.append(profiles_1d_dummy)
    ids_data.ids_struct['core_sources'].source[0].global_quantities.append(global_quantities_dummy)

    for i in range(1):
        #ids_data.ids_struct['core_sources'].source.append(source_dummy)
        #ids_data.ids_struct['core_sources'].source[i].profiles_1d.append(profiles_1d_dummy)
        #ids_data.ids_struct['core_sources'].source[i].global_quantities.append(global_quantities_dummy)

        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].grid.rho_tor_norm = dummy_array
        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].grid.rho_tor = dummy_array
        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].grid.rho_pol_norm = dummy_array
        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].grid.psi = dummy_array
        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].grid.volume = dummy_array

        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].electrons.energy = dummy_array
        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].electrons.particles = dummy_array

        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].j_parallel = dummy_array
        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].total_ion_energy = dummy_array
        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].current_parallel_inside = dummy_array
        ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].conductivity_parallel = dummy_array

        ids_data.ids_struct['core_sources'].source[i].global_quantities[0].power = 1.0
        ids_data.ids_struct['core_sources'].source[i].global_quantities[0].total_ion_particles = 1.0
        ids_data.ids_struct['core_sources'].source[i].global_quantities[0].total_ion_power = 1.0
        ids_data.ids_struct['core_sources'].source[i].global_quantities[0].electrons.power = 1.0
        ids_data.ids_struct['core_sources'].source[i].global_quantities[0].electrons.particles = 1.0

        ids_data.ids_struct['core_sources'].source[i].identifier.index = 3
        ids_data.ids_struct['core_sources'].source[i].identifier.name = 'ec'

    '''
    for i in range(6):
        if i < len(ids_data.ids_struct['core_sources'].source):
            ids_data.ids_struct['core_sources'].source[i].profile_1d[0].grid.rho_tor_norm = dummy_array
            ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].electrons.energy = dummy_array
            ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].j_parallel = dummy_array
            ids_data.ids_struct['core_sources'].source[i].profiles_1d[0].total_ion_energy = dummy_array
            #ids_data.ids_struct['core_sources'].source[i].global_quantities.power = np.asarray([0])
            ids_data.ids_struct['core_sources'].source[i].global_quantities.power = 0

            ids_data.ids_struct['core_sources'].source[i].identfier.index = 3
            ids_data.ids_struct['core_sources'].source[i].identfier.name = 'ec'
        else:
            ids_data.ids_struct['core_sources'].source.append(source_dummy)
    '''


    ids_dict = ids_data.ids_dict

    # Put the data back in the ids structure

    ids_data.ids_dict = ids_dict
    ids_data.fill_ids_struct()

    put_integrated_modelling(db, shot, run_input, run_start, ids_data.ids_struct, backend = backend)


def substitute_dummy_when_low_power_unit(array, array_power):

    dummy_fractions = [1.0, 0.0, 0.0]
    low_power_cols = np.where(array_power < 10000)[0]
    normalized_array = np.copy(array)
    normalized_array[:, low_power_cols] = np.transpose(np.array(dummy_fractions*np.size(low_power_cols)).reshape(np.size(low_power_cols),3))

    return normalized_array


def adapt_fractions(array, array_power):

    # Make sure that fractions are not above 1 or below 0
    array = np.where(array > 0, array, 0)
    array = np.where(array < 1, array, 1)

    array = substitute_dummy_when_zero(array)
    array = force_column_sum_to_one(array)
    array = substitute_dummy_when_low_power(array, array_power)

    return(array)


def get_time_system(dataset, user, db, shot, run, system, time = []):

    handle = ImasHandle(user = user, db = db, shot = shot, run = run)

    if system == 'nbi':
        variable_time_system = variable_time_nbi
    if system == 'ec_launchers':
        variable_time_system = variable_time_ec_launchers

    if not time:
        try:
            time = handle.get_variables([variable_time_system])
        except (ValueError, duqtools.ids._mapping.EmptyVarError):
            time = None

    if not time:
        time = np.asarray([0]*5) # Not very elegant
        # Should create an epmty dataset. If it fails here now it means that there is no nbi data
        for time_variable in dataset.coords:
            if np.size(dataset[time_variable]) >= np.size(time) and np.size(dataset[time_variable]) != 0:
                time = dataset[time_variable]

    return time


def set_nbi_consistency(dataset, user, db, shot, run, time = None):

    # Energy, power and fractions should not be negative
    dataset['energy_data'].data = np.where(dataset['energy_data'].data > 0, dataset['energy_data'].data, 0)
    dataset['power_launched_data'].data = np.where(dataset['power_launched_data'].data > 0, dataset['power_launched_data'].data, 0)
    dataset['beam_current_fraction_data'].data = np.where(dataset['beam_current_fraction_data'].data > 0, dataset['beam_current_fraction_data'].data, 0)
    dataset['beam_power_fraction_data'].data = np.where(dataset['beam_power_fraction_data'].data > 0, dataset['beam_power_fraction_data'].data, 0)

    time = get_time_system(dataset, user, db, shot, run, 'nbi', time = time)

    for coord in time.coords:
        time = time.rename({coord: 'time'})

    time = time[time>0]

    # Setting all times to be the same
    for time_variable in dataset.coords:
        if 'time' in time_variable:
            dataset = eval("dataset.interp(" + time_variable + "=time,kwargs={'fill_value':0.0})")
            dataset = dataset.reset_coords(time_variable, drop = True)

    # Fractions should sum to 1
    dataset['beam_current_fraction_data'].data = adapt_fractions(dataset['beam_current_fraction_data'].data, dataset['power_launched_data'].data)
    dataset['beam_power_fraction_data'].data = adapt_fractions(dataset['beam_power_fraction_data'].data, dataset['power_launched_data'].data)

    # Setting energy and power to 0 when is too low (probably just noise)
    dataset['energy_data'].data = np.where(dataset['power_launched_data'].data > 10000, dataset['energy_data'].data, 0)
    dataset['power_launched_data'].data = np.where(dataset['power_launched_data'].data > 10000, dataset['power_launched_data'].data, 0)

    return dataset


def set_ec_launchers_consistency(dataset, user, db, shot, run, time = None):

    # Energy, power and fractions should not be negative
    # It seems that for now a non zero value is required
    dataset['frequency_data'].data = np.where(dataset['frequency_data'].data > 0, dataset['frequency_data'].data, 1.0)
    dataset['power_launched_data'].data = np.where(dataset['power_launched_data'].data > 0, dataset['power_launched_data'].data, 1.0)
    # Might need to have this non zero or it is not recognized?
    # dataset['phase_angle_data'].data = np.where(dataset['phase_angle_data'].data > 0, dataset['phase_angle_data'].data, 0.001)

    time = get_time_system(dataset, user, db, shot, run, 'ec_launchers', time = time)

    for coord in time.coords:
        time = time.rename({coord: 'time'})

    time = time[time>0]

    # Setting all times to be the same
    for time_variable in dataset.coords:
        if 'time' in time_variable:
            dataset = eval("dataset.interp(" + time_variable + "=time,kwargs={'fill_value':0.0})")
            dataset = dataset.reset_coords(time_variable, drop = True)

    # Setting energy and power to 0 when is too low (probably just noise)
    dataset['power_launched_data'].data = np.where(dataset['power_launched_data'].data > 10000, dataset['power_launched_data'].data, 0)

    return dataset


def multiply_power(dataset, power_multiplier):

    dataset['power_launched_data'].data = dataset['power_launched_data'].data*power_multiplier

    return dataset

def substitute_strings_with_time(string_list):

    substituted_list = [s if 'time' not in s else 'time' for s in string_list]
    return substituted_list


def remove_duplicates(list_input):
    unique_list = []
    seen_entries = set()
    for item in list_input:
        if item not in seen_entries:
            unique_list.append(item)
            seen_entries.add(item)

    return unique_list


def update_time_dimension(string_list):

    substituted_list = substitute_strings_with_time(string_list)
    substituted_list = remove_duplicates(substituted_list)

    return substituted_list


def get_backend(db, shot, run, username=None):

    if not username: username = getpass.getuser()

    imas_backend = imasdef.HDF5_BACKEND
    data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)

    op = data_entry.open()
    if op[0]<0:
        imas_backend = imasdef.MDSPLUS_BACKEND

    data_entry.close()

    data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)
    op = data_entry.open()
    if op[0]<0:
        print('Input does not exist. Aborting generation')

    data_entry.close()

    return imas_backend


def fill_basic_quantities(ids_iden):

    ids_struct = eval('imas.' + ids_iden + '()')

    # Might want to specify this externally
    username=getpass.getuser()

    ids_struct.code.commit = 'unknown'
    ids_struct.code.name = 'Nbi/ec_heating consistency_tools'
    ids_struct.code.output_flag = np.array([])
    ids_struct.code.repository = 'gateway'
    ids_struct.code.version = 'unknown'

    ids_struct.ids_properties.homogeneous_time = imasdef.IDS_TIME_MODE_HOMOGENEOUS
    ids_struct.ids_properties.provider = username
    ids_struct.ids_properties.creation_date = str(datetime.date)
    ids_struct.time = np.asarray([0.1])

    return ids_struct

def put_system_ids(dataset, user, db, shot, run, system, extra_info = {}, backend = None):

    if not backend: backend = get_backend(db, shot, run)
    if not user: user = getpass.getuser()

    #data_entry = imas.DBEntry(backend, db, shot, run, user_name=user)
    #op = data_entry.open()

    ids_struct = fill_basic_quantities(system)
    #ids_struct = data_entry.get(system)

    for variable in dataset:

        ids_struct = put_single_variable_ids(dataset, variable, ids_struct, system)

        if 'data' in variable:
            variable_time = variable.replace('data', 'time')
            ids_struct = put_single_variable_ids(dataset, variable_time, ids_struct, system)
            #variable_time = variable.replace('time', 'data')

    ids_struct.time = dataset['time'].values

    if system == 'nbi':
        if 'label' in extra_info:
            for iunit in range(len(ids_struct.unit)):
                ids_struct.unit[iunit].species.label = extra_info['label']

    if system == 'ec_launchers':
        if 'mode' in extra_info:
            for ibeam in range(len(ids_struct.beam)):
                if extra_info['mode'][ibeam] != 1 and extra_info['mode'][ibeam] != -1:
                    ids_struct.beam[ibeam].mode = -1
                else:
                    ids_struct.beam[ibeam].mode = extra_info['mode'][ibeam]

    return ids_struct


def put_single_variable_ids(dataset, variable, ids_struct, system):

    # Handling the fact that I want only one time to be in the final IDS
    variable_dataset = copy.deepcopy(variable)

    if 'time' in variable_dataset: variable_dataset = 'time'

    # Need not to modify the nbi variables
    tag_bound = get_variable_system(variable, system)
    tag = copy.deepcopy(tag_bound)

    tag.dims = update_time_dimension(tag.dims)

    tag.path = tag.path.replace('/','.')
    parts = tag.path.split('*')

    size_dim1 = dataset.dims[tag.dims[0]]
    size_values = np.size(dataset[variable_dataset].values)
    size_time = np.size(dataset['time'].values)

    if 'time' not in variable_dataset:
        values = dataset[variable_dataset].values.reshape(size_dim1,size_values//size_dim1)
    else:
        values = np.concatenate([dataset[variable_dataset].values]*size_dim1).reshape(size_dim1,size_values)

    if len(dataset[variable_dataset].dims) > len(parts):
        if size_values//size_dim1 != size_time:
            values = values.reshape(size_dim1,size_values//(size_dim1*size_time),size_time)

    for index, value in enumerate(values):
        ids_subsystem = eval('ids_struct.' + parts[0][:-1])
        if index >= len(ids_subsystem):
            new_item = eval('ids_struct.' + parts[0][:-1] + '.getAoSElement()')
            eval('ids_struct.' + parts[0][:-1] + '.append(new_item)')

        if variable == 'a' or variable == 'z_n':
            value = float(value[0])
        eval('rsetattr(ids_struct.' + parts[0][:-1] + '[' + str(index) + '], \'' + parts[1][1:] + '\', value)')
        index += 1

    return ids_struct


def put_integrated_modelling(db, shot, run, run_target, ids_struct, backend = None):

    '''

    Puts the IDSs useful for integrated modelling. This should be done with IMASpy when I learn how to do it.

    '''

    if not backend: backend = get_backend(db, shot, run)

    username = getpass.getuser()

    print(db, shot, run, run_target, backend)

    copy_ids_entry(db, shot, run, run_target, backend = backend)

    data_entry = imas.DBEntry(backend, db, shot, run_target, user_name=getpass.getuser())
    ids_list = ['core_profiles', 'core_sources', 'ec_launchers', 'equilibrium', 'nbi', 'summary', 'thomson_scattering', 'pulse_schedule']

    op = data_entry.open()

    for ids in ids_list:
    # If the time vector is empty the IDS is empty or broken, do not put
        if ids in ids_struct:
            if len(ids_struct[ids].time) !=0:
                data_entry.put(ids_struct[ids])

    data_entry.close()


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


class LoggingContext:
    """Context manager to Temporarily change logging configuration.

    From https://docs.python.org/3/howto/logging-cookbook.html

    Parameters
    ----------
    logger : None, optional
        Logging instance to change, defaults to root logger.
    level : None, optional
        New log level, i.e. `logging.CRITICAL`.
    handler : None, optional
        Log handler to use.
    close : bool, optional
        Whether to close the handler after use.
    """

    def __init__(self, logger=None, level=None, handler=None, close=True):
        if not logger:
            logger = logging.getLogger()
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()

def copy_ids_entry(db, shot, run, run_target, db_target = None, shot_target = None, username = None, username_target = None, ids_list = [], backend = None, verbose = False):

    '''

    Copies an entire IDS entry

    '''

    if not username: username = getpass.getuser()
    if not username_target: username_target = username
    if not db_target: db_target = db
    if not shot_target: shot_target = shot
    if not backend: backend = get_backend(db, shot, run, username = username)

    username_personal = getpass.getuser()
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

    idss_in = imas.DBEntry(backend, db, shot, run, user_name=username)
    idss_in = imas.ids(shot, run)

    op = idss_in.open_env_backend(username, db, imas_major_version, backend)
    if op[0]<0:
        print('The entry you are trying to copy does not exist')
        exit()

    print('Creating', username_target, db, imas_version, shot_target, run_target)

    #idss_out = imas.ids(shot_target, run_target)
    #idss_out.create_env_backend(username_target, db_target, imas_major_version, backend)

    idss_out = imas.DBEntry(backend, db_target, shot_target, run_target)
    idx = idss_out.create()[1]

    with LoggingContext(level=logging.CRITICAL):

        for ids_info in parser.idss:
            name = ids_info['name']
            maxoccur = int(ids_info['maxoccur'])
            if ids_list and name not in ids_list:
                continue
            #if name == 'ec_launchers':
            #    print('continue on ec launchers')  # Temporarily down due to a malfunctioning of ec_launchers ids
            #    continue
            #if name in idss_in.__dict__:
            for i in range(maxoccur + 1):
                if not i and verbose:
                    print('Processing', ids_info['name'])

                ids = idss_in.__dict__[name]
                ids.get(i)
                ids.setExpIdx(idx)
                ids.put(i)

    idss_in.close()


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
 

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def fit_and_substitute_nbi(x_old, x_new, data_old):

    f_space = interp1d(x_old, data_old, bounds_error = False, fill_value = 0)

    variable = np.array(f_space(x_new))
    variable[variable > 1.0e25] = 0

    return variable


def open_and_get_ids(db, shot, run, ids_name, username=None, backend = None):

    if not backend: backend = get_backend(db, shot, run)

    if not username:
        data_entry = imas.DBEntry(backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(backend, db, shot, run, user_name=username)

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


def change_line_nbi(line, start, values):

    values_str = []
    len_values_str = []
    new_line = line

    for value in values:
        if start == 'E1 [keV]':
            values_str.append('{0:.2f}'.format(value*1e-3))
        elif start == 'f1' or start == 'f2' or start == 'f3':
            values_str.append('{0:.2f}'.format(value))
        elif start == 'ANum' or start == 'ZNum':
            values_str.append('{0:.0f}'.format(value))
        len_values_str.append(len(values_str[-1]))

    if line.startswith(start):
        num_spaces_start = 22 - len(start) - len_values_str[0]
        num_spaces1 = 8 - len_values_str[1]
        num_spaces2 = 8 - len_values_str[2]
        new_line = start + ' '*num_spaces_start + values_str[0]
        new_line += ' '*num_spaces1 + values_str[1] + ' '*num_spaces2 + values_str[2] + '\n'

    return new_line

def add_line_power(lines, time, powers):

    power_str = []
    len_power_str = []

    for power in powers:
        power_str.append('{0:.3f}'.format(power*1e-6))
        len_power_str.append(len(power_str[-1]))

    if time > 1:
        num_digits = math.floor(math.log10(abs(time)))
    else:
        num_digits = 0

    time_str = eval('\'{0:.' + str(4-num_digits) + 'f}\'.format(time)')
    time_str = ' ' + time_str

    num_spaces_start = 15 - len_power_str[0]
    num_spaces1 = 8 - len_power_str[1]
    num_spaces2 = 8 - len_power_str[2]
    new_line = time_str + ' '*num_spaces_start + power_str[0]
    new_line += ' '*num_spaces1 + power_str[1] + ' '*num_spaces2 + power_str[2]  + '\n'

    lines.append(new_line)

    return lines


def modify_jset_nbi(run_path, nbi_config_name):

    '''

    Modifies the jset file to accomodate a new run name, username, shot and run. Database not really implemented yet

    '''

    line_start_list = [
        'NBIAscotRef.configFileName',
        'NBIAscotRef.configPrvDir'
    ]

    new_content_list = [
        nbi_config_name,
        '/afs/eufus.eu/user/g/g2ethole/public/tcv_inputs'
    ]

    for line_start, new_content in zip(line_start_list, new_content_list):
        modify_jset_line(run_path, line_start, new_content)


def modify_jset_line(run_path, line_start, new_content):

    '''

    Modifies a line of the jset file. Maybe it would be better to change all the lines at once but future work, not really speed limited now

    '''
    read_data = []

    len_line_start = len(line_start)
    with open(run_path + '/' + 'jetto.jset') as f:
        lines = f.readlines()
        for line in lines:
            read_data.append(line)

        for index, line in enumerate(read_data):
            if line[:len_line_start] == line_start:
                read_data[index] = read_data[index][:62] + new_content + '\n'

    with open(run_path + '/' + 'jetto.jset', 'w') as f:
        for line in read_data:
            f.writelines(line)


def modify_jetto_nbi_config(db, shot, run, run_name, path_nbi_config, time = []):

    nbi_ids = open_and_get_ids(db, shot, run, 'nbi')

    # Set incipit
    energy, A, Z = [], [], []
    for unit in nbi_ids.unit:
        energy.append(np.average(unit.energy.data[unit.energy.data != 0]))
        A.append(unit.species.a)
        Z.append(unit.species.z_n)

    fractions = []
    for unit in nbi_ids.unit:
        # Assume that there are only 3 energy fractions. This should be general (1, 1/2, 1/3 energy)
        for i in range(3):
            fractions_array = unit.beam_power_fraction.data[i][unit.beam_power_fraction.data[i] != 0]
            fractions_array = fractions_array[fractions_array != 1]
            fractions.append(np.average(fractions_array))

    energy, fractions = np.asarray(energy), np.asarray(fractions)
    energy[np.isnan(energy)] = 1000
    energy[energy == 0] = 1000
    fractions[np.isnan(fractions)] = 0.01
    fractions = np.around(fractions.reshape(len(nbi_ids.unit), 3), 2)

    for i, fraction in enumerate(fractions):
        fractions[i,0] = 1 - fractions[i,1] - fractions[i,2]

    fractions = fractions.T

    ntimes = np.size(nbi_ids.time)

    lines = []
    with open(path_nbi_config) as f:
        lines_file = f.readlines()
        for line in lines_file:
            lines.append(line)

        starts = ['E1 [keV]', 'f1', 'f2', 'f3', 'ANum', 'ZNum']
        all_values = [energy, fractions[0], fractions[1], fractions[2], A, Z]

        for start, values in zip(starts, all_values):
            new_lines = []
            for line in lines:
                new_lines.append(change_line_nbi(line, start, values))
            lines = new_lines

    # Set Ntimes
    lines = lines[:-3]
    if not time:
        num_times = 2500
    else:
        num_times = len(time)

    line_ntimes = 'Ntimes' + ' '*12 + str(num_times) + '\n'
    lines.append(line_ntimes)

    # Set new time
    time_min = max(0, min(nbi_ids.time))
    time_max = max(nbi_ids.time)
    new_times = np.linspace(time_min, time_max, num=num_times)

    # Set powers
    powers = np.asarray([])
    for unit in nbi_ids.unit:
        power = fit_and_substitute_nbi(nbi_ids.time, new_times, unit.power_launched.data)
        powers = np.hstack((powers, power))

    powers = powers.reshape(3, num_times)
    powers = powers.T

    lines_power = []
    for time, power in zip(new_times, powers):
        add_line_power(lines_power, time, power)

    lines.append(lines_power)

    path_nbi_target = path_nbi_config.replace('.nbicfg', '') + run_name + '.nbicfg'

    with open(path_nbi_target, 'w') as f:
        for line in lines:
            f.writelines(line)

def modify_ascot_cntl_line(run_name, line_start, new_content):

    '''

    Modifies a line of the ascot_cntl file. Maybe it would be better to change all the lines at once but future work, not really speed limited now

    '''
    read_data = []

    len_line_start = len(line_start)
    with open(run_name + '/' + 'ascot.cntl') as f:
        lines = f.readlines()
        for line in lines:
            read_data.append(line)

        for index, line in enumerate(read_data):
            if line[:len_line_start] == line_start:
                read_data[index] = read_data[index][:42] + new_content + '\n'

    with open(run_name + '/' + 'ascot.cntl', 'w') as f:
        for line in read_data:
            f.writelines(line)

def modify_ascot_cntl(run_path):

    line_start = 'Creation Name'
    modify_ascot_cntl_line(run_path, line_start, run_path + '/ascot.cntl')


def check_nbi_data(db, shot, run):

    nbi = open_and_get_ids(db, shot, run, 'nbi')
    if nbi.time.size == 0:
        print('You are trying to setup the nbi but the nbi ids is empty. Aborting')
        exit()


def get_ids_params(ids):

    stag = ids.strip().split('/')
    while not stag[-1]:
        stag = stag[:-1]
    db = stag[-3].strip()
    shot = int(stag[-2].strip())
    runid = int(stag[-1].strip())
    user = '/'.join(stag[:-3]) if len(stag) > 4 else stag[0]

    if not user: user = getpass.getuser()

    return user, db, shot, runid


def setup_ec_launchers(db, shot, run, path_run, user = None, run_target = None, backend = None):

    if not run_target: run_target = run + 1
    if not user: user = getpass.getuser()
    if not backend: backend = get_backend(db, shot, run_input)

    if run != run_target:
        setup_system_ids(db, shot, run, run_target, 'ec_launchers', user = user, backend = backend)


def setup_nbi(db, shot, run, path_run, user = None, run_target = None, backend = None, path_nbi_config = '/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg'):

    if not run_target: run_target = run + 1
    if not user: user = getpass.getuser()
    if not backend: backend = get_backend(db, shot, run)
    if not path_nbi_config: path_nbi_config = '/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg'

    if run != run_target:
        check_nbi_data(db, shot, run)
        setup_system_ids(db, shot, run, run_target, 'nbi', user = user, backend = backend)

    if path_run:
        if '/jetto/runs/' in path_run:
            run_name = path_run.split('/')[-1]
            path_nbi_target = path_nbi_config.replace('.nbicfg', '') + run_name + '.nbicfg'
            #modify_jset_nbi(path_run, nbi_config_name = path_nbi_config)
            modify_jset_nbi(path_run, nbi_config_name = path_nbi_target)
            modify_jetto_nbi_config(db, shot, run_target, run_name = run_name, path_nbi_config = path_nbi_config)
            modify_ascot_cntl(path_run)
            shutil.copyfile(path_nbi_target, path_run + '/jetto.nbicfg')
        else:
            run_name = 'new'
            modify_jetto_nbi_config(db, shot, run_target, run_name = run_name, path_nbi_config = path_nbi_config)
            if os.path.exists(path_run + '/ascot.cntl'):
                modify_ascot_cntl(path_run)
            shutil.copyfile(path_nbi_config + run_name, path_run + '/jetto.nbicfg')


def setup_system_ids(db, shot, run, run_target, system, user = None, time = [], backend = None, power_multiplier = 1):

    if not user: user = getpass.getuser()
    if not backend: backend = get_backend(db, shot, run_input)

    dataset, extra_info = extract_system_data(user, db, shot, run, system, backend=backend)
    if system == 'nbi':
        dataset = set_nbi_consistency(dataset, user, db, shot, run, time = time)
    if system == 'ec_launchers':
        dataset = set_ec_launchers_consistency(dataset, user, db, shot, run, time = time)

    if power_multiplier != 1:
        dataset = multiply_power(dataset,  power_multiplier)

    ids_stuct_system = put_system_ids(dataset, user, db, shot, run, system, extra_info = extra_info)

    ids_struct = {}
    ids_struct[system] = ids_stuct_system
    put_integrated_modelling(db, shot, run, run_target, ids_struct, backend=backend)


def setup_ec_launchers_ids(db, shot, run, run_target, system, user = None, time = [], backend = None, power_multiplier = 1):

    if not user: user = getpass.getuser()
    if not backend: backend = get_backend(db, shot, run_input)

    dataset, extra_info = extract_system_data(user, db, shot, run, system, backend=backend)
    dataset = set_ec_launchers_consistency(dataset, user, db, shot, run, time = time)

    if power_multiplier != 1:
        dataset = multiply_power(dataset,  power_multiplier)

    ids_stuct_ec_launchers = put_system_ids(dataset, user, db, shot, run, system, extra_info = extra_info)

    ids_struct = {}
    ids_struct[system] = ids_stuct_ec_launchers
    put_integrated_modelling(db, shot, run, run_target, ids_struct, backend=backend)


def input():

    parser = argparse.ArgumentParser(
        description=
    """Modifies an ids and a jetto run to setup the nbi ids and the input files for ASCOT, created by M. Marin.\n
    If the nbi ids is already correct, specify the same run for the ids_input and run_target
    ---
    Examples:\n
            python setup_nbi_input.py --ids 'g2mmarin/tcv/73388/2' --run_target 3 --path_run runtest_ascot_setup/ \n
    ---
    """,
    epilog="",
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--backend",    "-b",            type=str,   default=None, choices=[None, "mdsplus", "hdf5"],  help="Backend with which to access data")
    parser.add_argument("--ids_input",  "-i",            type=str,   default=None,                                     help="IDS identifiers in which data is stored")
    parser.add_argument("--run_target", "-r",            type=int,   default=None,                                     help="Run number for the output ids")
    parser.add_argument("--version",    "-v",            type=str,   default="3",                                      help="UAL version")
    parser.add_argument("--path_run",                    type=str,   default=None,                                     help="Path of the run (or just of the config file)")
    parser.add_argument("--path_nbi_config",             type=str,   default='/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg',  help="List with the densities")
    parser.add_argument("--time_trace",      nargs='+',  type=float, default=None,                                     help="List with the corresponding times")
    parser.add_argument("--create_dummy_source",                     default=False, action='store_true',               help="Creates a dummy core_sources")

    args=parser.parse_args()

    return args


def main():

    args = input()
    ids = args.ids_input
    run_target = args.run_target
    backend = args.backend
    version = args.version
    path_run = args.path_run
    path_nbi_config = args.path_nbi_config
    time_trace = args.time_trace
    create_dummy_source = args.create_dummy_source

    if backend == 'hdf5':
        backend = imasdef.HDF5_BACKEND
    else:
        backend = imasdef.MDSPLUS_BACKEND

    user, db, shot, run = get_ids_params(ids)
    if path_run:
        if not path_run.startswith('/pfs/'):
            path_run = '/pfs/work/' + user + '/jetto/runs/' + path_run

    setup_nbi(db, shot, run, path_run, run_target = run_target, user = user, path_nbi_config = '/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg')
    setup_ec_launchers(db, shot, run, path_run, run_target = run_target, user = user)

    if create_dummy_source:
        create_dummy_core_sources(db, shot, run_target, run_target, backend = backend)


if __name__ == "__main__":
    #user, db, shot, run = 'g2mmarin', 'tcv', 73388, 2
    #set_nbi_consistency(user, db, shot, run, time = [])
    #modify_nbi_ids(user, db, shot, run)
    #extract_system_data(user, db, shot, run, 'nbi')
    #setup_system_ids(db, shot, run, run + 1, 'nbi')
    #setup_nbi(db, shot, run, path_run = '/pfs/work/g2mmarin/jetto/runs/test_ascot_setup', path_nbi_config = '/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg')
    #create_dummy_core_sources('tcv', 80599, 1, 11)
    main()


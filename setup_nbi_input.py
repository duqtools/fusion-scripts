import os,datetime,sys
from scipy.interpolate import interp1d, UnivariateSpline
from packaging import version
import xarray as xr

import copy
import numpy as np
import imas
import jetto_tools
import functools
import getpass
import xml.sax
import xml.sax.handler
import argparse
import shutil

import imas
if imas is not None:
    from imas import imasdef
    vsplit = imas.names[0].split("_")
    imas_version = version.parse(".".join(vsplit[1:4]))
    ual_version = version.parse(".".join(vsplit[5:]))

import math
import duqtools
from duqtools.api import ImasHandle
from duqtools.api import Variable
from duqtools.api import rebase_on_time

variables_nbi = [
    Variable(name='a',
        ids = 'nbi',
        path = 'unit/*/species/a',
        dims = ['unit']),
    Variable(name='z_n',
        ids = 'nbi',
        path = 'unit/*/species/z_n',
        dims = ['unit']),
    #Variable(name='label',
    #    ids = 'nbi',
    #    path = 'unit/*/species/label',
    #    dims = ['unit']),
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

def get_variable_nbi(name):

    variable_found = None
    for variable in variables_nbi:
        if variable.name == name:
            variable_found = variable

    return variable_found


def extract_nbi_data(user, db, shot, run, backend = 'mdsplus'):

    handle = ImasHandle(user = user, db = db, shot = shot, run = run)
    dataset = xr.Dataset()

    for variable in variables_nbi:
        if variable.name.endswith('data'):
            variable_time_name = variable.name.replace('data', 'time')
            variable_time = get_variable_nbi(variable_time_name)
            try:
                single_dataset = handle.get_variables([variable, variable_time])
                dataset = xr.merge([dataset, single_dataset])
            except (ValueError, duqtools.ids._mapping.EmptyVarError):
                single_array = []
                # Check if at least one of the units has data
                for index in range(dataset.dims['unit']):
                    try:
                        variable_single, variable_time_single = copy.deepcopy(variable), copy.deepcopy(variable_time)
                        variable_single.path = variable.path.replace('*',str(index))
                        variable_time_single.path = variable_time.path.replace('*',str(index))
                        variable_single.dims.remove('unit')
                        variable_time_single.dims.remove('unit')
                        dataset_slice = handle.get_variables([variable_single, variable_time_single])

                        single_array.append(dataset_slice[variable_single.name].data)
                    except (ValueError, duqtools.ids._mapping.EmptyVarError):
                        pass
                # fill the dataset using zeros where there is no data
                if len(single_array) != 0:
                    dataset_slice_template = dataset_slice
                    single_dataset = xr.Dataset()
                    single_dataset = single_dataset.expand_dims('unit')
                    for index in range(dataset.dims['unit']):
                        variable_single, variable_time_single = copy.deepcopy(variable), copy.deepcopy(variable_time)
                        variable_single.path = variable.path.replace('*',str(index))
                        variable_time_single.path = variable_time.path.replace('*',str(index))
                        variable_single.dims.remove('unit')
                        variable_time_single.dims.remove('unit')
                        dataset_slice = handle.get_variables([variable_single, variable_time_single])
                        if dataset_slice[variable_time_single.name].values.size == 0:
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

                        dataset_slice = dataset_slice.assign_coords({'unit':index})
                        dataset_slice['unit'] = index
                        if index == 0:
                            single_dataset = xr.merge([single_dataset, dataset_slice])
                        else:
                            single_dataset = xr.concat([single_dataset, dataset_slice], dim = 'unit')

            dataset = xr.merge([dataset, single_dataset])

        else:
            try:
                single_dataset = handle.get_variables([variable])
                dataset = xr.merge([dataset, single_dataset])
            except (ValueError, duqtools.ids._mapping.EmptyVarError):
                pass

    extra_info = get_nbi_extra(db, shot, run, user = user, imas_backend = 'mdsplus')

    return dataset, extra_info


def get_nbi_extra(db, shot, run, user = None, imas_backend = 'mdsplus'):

    nbi = open_and_get_ids(db, shot, run, 'nbi', username=user, backend=imas_backend)

    extra_info = {}
    extra_info['labels'] = []
    for unit in nbi.unit:
        extra_info['labels'].append(unit.species.label)

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


def get_time_nbi(dataset, user, db, shot, run, time = []):

    handle = ImasHandle(user = user, db = db, shot = shot, run = run)

    if not time:
        try:
            time = handle.get_variables([variable_time_nbi])
        except (ValueError, duqtools.ids._mapping.EmptyVarError):
            time = None

    if not time:
        time = np.asarray([0]*5) # Not very elegant
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

    time = get_time_nbi(dataset, user, db, shot, run, time = time)

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


def fill_basic_quantities(ids_iden):

    ids_struct = eval('imas.' + ids_iden + '()')

    # Might want to specify this externally
    username=getpass.getuser()

    ids_struct.code.commit = 'unknown'
    ids_struct.code.name = 'Nbi consistency_tools'
    ids_struct.code.output_flag = np.array([])
    ids_struct.code.repository = 'gateway'
    ids_struct.code.version = 'unknown'

    ids_struct.ids_properties.homogeneous_time = imasdef.IDS_TIME_MODE_HOMOGENEOUS
    ids_struct.ids_properties.provider = username
    ids_struct.ids_properties.creation_date = str(datetime.date)
    ids_struct.time = np.asarray([0.1])

    return ids_struct

def put_nbi_ids(dataset, user, db, shot, run, extra_info = {}):

    # Need to add this part
    ids_iden = 'nbi'

    imas_backend = imasdef.MDSPLUS_BACKEND

    if not user:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=user)

    op = data_entry.open()

    ids_struct = fill_basic_quantities(ids_iden)
    #ids_struct = data_entry.get(ids_iden)

    for variable in dataset:

        ids_struct = put_single_variable_ids(dataset, variable, ids_struct)

        if 'data' in variable:
            variable_time = variable.replace('data', 'time')
            ids_struct = put_single_variable_ids(dataset, variable_time, ids_struct)

    ids_struct.time = dataset['time'].values

    if 'label' in extra_info:
        for iunit in range(len(ids_struct.unit)):
            ids_struct.unit[iunit].species.label = extra_info['label']

    return ids_struct


def put_single_variable_ids(dataset, variable, ids_struct):

    # Handling the fact that I want only one time to be in the final IDS
    variable_dataset = copy.copy(variable)
    if 'time' in variable_dataset: variable_dataset = 'time'

    tag = get_variable_nbi(variable)
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

    for ids_info in parser.idss:
        name = ids_info['name']
        maxoccur = int(ids_info['maxoccur'])
        if ids_list and name not in ids_list:
            continue
        if name == 'ec_launchers' or name == 'numerics' or name == 'sdn':
            continue
            print('continue on ec launchers')  # Temporarily down due to a malfunctioning of ec_launchers ids
            print('skipping numerics')  # Not in the newest version of IMAS
        for i in range(maxoccur + 1):
            if not i:
                print('Processing', ids_info['name'])

            ids = idss_in.__dict__[name]
            stdout = sys.stdout
            sys.stdout = open('/afs/eufus.eu/user/g/g2mmarin/warnings_imas.txt', 'w') # suppress warnings
            ids.get(i)
            ids.setExpIdx(idx)
            ids.put(i)
            sys.stdout.close()
            sys.stdout = stdout

    idss_in.close()
    idss_out.close()


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


def open_and_get_ids(db, shot, run, ids_name, username=None, backend='mdsplus'):

    imas_backend = imasdef.MDSPLUS_BACKEND
    if backend == 'hdf5':
        imas_backend = imasdef.HDF5_BACKEND

    if not username:
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
    energy[np.isnan(energy)] = 0
    fractions[np.isnan(fractions)] = 0
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

    with open(path_nbi_config + run_name, 'w') as f:
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
    if nbi.time.size != 0:
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


def setup_nbi(db, shot, run, path_run, user = None, run_target = None, path_nbi_config = '/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg'):

    if not run_target: run_target = run + 1

    if not user: user = getpass.getuser()

    if run != run_target:
        check_nbi_data(db, shot, run)
        setup_nbi_ids(db, shot, run, run_target, user = user, backend='mdsplus')

    if path_run:
        if '/jetto/runs/' in path_run:
            modify_jset_nbi(path_run, nbi_config_name = path_nbi_config)
            run_name = path_run.split('/')[-1]
            modify_jetto_nbi_config(db, shot, run_target, run_name = run_name, path_nbi_config = path_nbi_config)
            modify_ascot_cntl(path_run)
            shutil.copyfile(path_nbi_config + run_name, path_run + '/jetto.nbicfg')
        else:
            run_name = 'new'
            modify_jetto_nbi_config(db, shot, run_target, run_name = run_name, path_nbi_config = path_nbi_config)
            if os.path.exists(path_run + '/ascot.cntl'):
                modify_ascot_cntl(path_run)
            shutil.copyfile(path_nbi_config + run_name, path_run + '/jetto.nbicfg')


def setup_nbi_ids(db, shot, run, run_target, user = None, time = [], backend='mdsplus', power_multiplier = 1):

    if not user: user = getpass.getuser()

    dataset, extra_info = extract_nbi_data(user, db, shot, run, backend=backend)
    dataset = set_nbi_consistency(dataset, user, db, shot, run, time = time)

    if power_multiplier != 1:
        dataset = multiply_power(dataset,  power_multiplier)

    ids_stuct_nbi = put_nbi_ids(dataset, user, db, shot, run, extra_info = extra_info)
    ids_struct = {}
    ids_struct['nbi'] = ids_stuct_nbi
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

    parser.add_argument("--backend",    "-b",            type=str,   default="mdsplus", choices=["mdsplus", "hdf5"], help="Backend with which to access data")
    parser.add_argument("--ids_input",  "-i",            type=str,   default=None,                                   help="IDS identifiers in which data is stored")
    parser.add_argument("--run_target", "-r",            type=int,   default=None,                                   help="Run number for the output ids")
    parser.add_argument("--version",    "-v",            type=str,   default="3",                                    help="UAL version")
    parser.add_argument("--path_run",                    type=str,   default=None,                                   help="Path of the run (or just of the config file)")
    parser.add_argument("--path_nbi_config",             type=str,   default='/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg',  help="List with the densities")
    parser.add_argument("--time_trace",      nargs='+',  type=float, default=None,                                   help="List with the corresponding times")

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

    user, db, shot, run = get_ids_params(ids)
    if path_run:
        if not path_run.startswith('/pfs/'):
            path_run = '/pfs/work/' + user + '/jetto/runs/' + path_run

    setup_nbi(db, shot, run, path_run, run_target = run_target, user = user, path_nbi_config = '/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg')


if __name__ == "__main__":
    #user, db, shot, run = 'g2mmarin', 'tcv', 73388, 2
    #set_nbi_consistency(user, db, shot, run, time = [])
    #modify_nbi_ids(user, db, shot, run)
    #extract_nbi_data(user, db, shot, run)
    #setup_nbi_ids(db, shot, run, run + 1)

    #setup_nbi(db, shot, run, path_run = '/pfs/work/g2mmarin/jetto/runs/test_ascot_setup', path_nbi_config = '/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg')
    main()


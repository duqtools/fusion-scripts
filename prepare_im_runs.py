import json
import os,datetime,sys
import shutil
import getpass
import numpy as np
import pickle
import math
import functools
import re
#import scipy
from scipy import integrate
from scipy.interpolate import interp1d, UnivariateSpline
#import idstools

from packaging import version
from os import path
from pathlib import Path

import inspect
import types

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display

import xml.sax
import xml.sax.handler
import pdb
import xarray as xr
import copy

'''
The tools in this script are useful to:

Setup integrated modelling simulations
Setup sensitivities
Run sensitivities
Compare integrated modelling with experimental data
'''

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

import sys
sys.path.insert(0, '/afs/eufus.eu/user/g/g2mmarin/python_tools/jetto-pythontools')
import jetto_tools

import duqtools
from duqtools.jetto import BaseJettoSystem

path_in = '/pfs/work/g2mmarin/jetto/runs/run465_ohmic_predictive_new7'
path_out = '/pfs/work/g2mmarin/jetto/runs/runtest_duqtools'


import logging
import warnings
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from duqtools.apply_model import apply_model
from duqtools.cleanup import remove_run
from duqtools.config import Config, cfg
from duqtools.ids import ImasHandle
from duqtools.matrix_samplers import get_matrix_sampler
from duqtools.models import Locations, Run, Runs
#from duqtools.models import Locations
from duqtools.operations import add_to_op_queue, op_queue
from duqtools.system import get_system
from duqtools.create import CreateManager, create, create_entry

from duqtools.jetto._jettovar_to_json import jettovar_to_json
from duqtools.system import get_system


#global cfg global config to unhack?


path_in = Path('/pfs/work/g2mmarin/jetto/runs/rungenerator_ohmic_duq')
path_out = Path('/pfs/work/g2mmarin/jetto/runs/run_duq_test')

test = BaseJettoSystem()
test.copy_from_template(path_in, path_out)


'''
cfg = cfg.parse_file('duqtools.yaml')

path = Locations().jruns_path
print(path)

cfg.create.jruns = '/pfs/work/g2mmarin/jetto/runs/'
cfg.create.runs_dir = 'test_duqtools'

# global config to unhack?

#print(cfg.create)
#print(type(cfg.create))

created = CreateManager(cfg)
#created = CreateManager()

ops_dict = created.generate_ops_dict(base_only=True)

runs = created.make_run_models(ops_dict=ops_dict, absolute_dirpath=False)

#for operation in runs[0].operations:
#    print(operation)

#for operation in runs[0].operations[0].__dir__():
#    if not operation.startswith('_'):
#        print(operation)

#jetto_lookup = lookup.from_file(lookup_file)

new_jetto_operation = runs[0].operations[2]

print(new_jetto_operation)
new_jetto_operation.variable.name = 'ibtsign'
new_jetto_operation.value = 1
#print(new_jetto_operation.variable.lookup)
new_jetto_operation.variable.lookup.doc = 'Orientation of magnetic field'
new_jetto_operation.variable.lookup.name = 'ibtsign'
new_jetto_operation.variable.lookup.type = 'int'
new_jetto_operation.variable.lookup.keys[1].file = 'jetto.in'
new_jetto_operation.variable.lookup.keys[1].field = 'IBTSIGN'
new_jetto_operation.variable.lookup.keys[1].section= 'nlist1'


new_jetto_variable = new_jetto_operation.variable.lookup


#new_jetto_variable.name = 'ibtsign'
#new_jetto_variable.lookup = None

path =  '/pfs/work/g2mmarin/jetto/runs/rungenerator_ohmic_duq/'

#jetto_template = jetto_tools.template.from_directory(path)

#if new_jetto_variable:
#    extra_lookup = jetto_tools.lookup.from_json(jettovar_to_json(new_jetto_variable))
#    jetto_template._lookup.update(extra_lookup)




#jetto_config = jetto_tools.config.RunConfig(jetto_template)

#jetto_config.export(path)

#new_jetto_operation.operator = 'copyto'
#new_jetto_operation.scale_to_error = False
new_jetto_operation.variable.name = 'atmi'
new_jetto_operation.value = 1
new_jetto_operation.variable.lookup.doc = 'mass of first impurity'
new_jetto_operation.variable.lookup.name = 'atmi'
new_jetto_operation.variable.lookup.type = 'real'
#JsetField
new_jetto_operation.variable.lookup.keys[0].file = 'jetto.jset'
new_jetto_operation.variable.lookup.keys[0].field = 'ImpOptionPanel.impurityMass[0]'
#NamelistField
new_jetto_operation.variable.lookup.keys[1].file = 'jetto.in'
new_jetto_operation.variable.lookup.keys[1].field = 'ATMI'
new_jetto_operation.variable.lookup.keys[1].section= 'nlist4'

new_jetto_variable = new_jetto_operation.variable.lookup

system = get_system()


#system.set_jetto_variable(run_dir, model.variable.name, model.value, model.variable.lookup)
#system.set_jetto_variable(path, 'ibtsign', -1, new_jetto_variable)
system.set_jetto_variable(path, 'atmi', -1, new_jetto_variable)


runs[0].operations.append(new_jetto_operation)

print(runs[0].operations[-1])

#'ibtsign', 'null', 'NLIST1', 'int', 'scalar'

#'OutputExtraNamelist.selItems'

#print(runs[0].operations)
#print(runs[0].dict)
#print(runs[0].json)
#runs[0].dirname = '/pfs/work/g2mmarin/jetto/runs/runtest_duqtools' would need to construct type
#runs[0].shortname = 'runtest_duqtools'


created.create_run(runs[0], force=False)

#print(runs)
'''

exit()


'''

--------------- AVAILABLE FUNCTIONS: ------------------

Class IntegratedModellingRuns: sets up everything needed for an integrated modelling simulation.

instructions_list: possibilities are 'setup base', 'setup sens', 'create base', 'create sens', 'run base', 'run sens'

'setup base'  - Setup the input for a baserun
'setup sens'  - Setup the input for a sensitivity
'create base' - Create the baserun folder
'create sens' - Create the sensitivities folders
'run base'    - Run the baserun
'run sens'    - Run the sensitivities

Action: setup_create_compare()

db:                    Name of the ids database
run input:             The run where the experimental data are
run start:             Where the actual input for the run is taken after manipulation
generator_name:        Name of the generator from which all the settings that will not be changed will be taken
time_start:            Starting time for the simulation. Options are: time_start, 'core_profiles', 'equilibrium'. For the latter 2 the first time in the respective ids will be used
time_end:              Time_end, auto (untested)
esco_timesteps:        Number of times esco will be called (homogeneous)
output_timesteps:      Number of times the output will be printed (homogeneous)
force_run:             If true, will not stop if the output ids aready exists
density_feedback:      If True, will setup the density feedback, with the density in summary.line_averaged.density. Does not use the pulse scheduler (It does not work yet)
zeff_options:          Describe how to set the time trace for zeff
-- 'flat maximum'           Sets the maximum zeff everywhere
-- 'flat minimum'           Sets the minimum zeff everywhere
-- 'flat median'            Sets the median zeff everywhere
-- 'impurity from flattop'  Auto detects the flattop, averages the impurity composition there and imposes during the ramp-up
-- 'linear descending zeff' Zeff descends linearly
-- 'ip ne scaled'           Uses scaling from ASDEX
-- 'hyperbole'              Decreases zeff rapidly, starting from 4 and merging to the zeff value at the end of the ramp-up

sensitivity_list:      Can contain some basic sensitivities. Best to use the duqtools unless interested in peaking of te or setting initial q profile keeping q95
input_instructions:    Options for the input. They will be applied one by one generating new idss, starting from time_start-len(input_instructions)
-- average - rebase    Averages relevant IDSs or rebase the equilibrium IDS with the core profiles time base
-- flipping ip         Can be added, but will be added authomatically when the current is positive because I still cannot make positive current work
-- nbi heating         Sets up NBI. Not ready yet
-- set boundaries      Will setup the boundaries for te and ti. 
   -- 'constant'       Can set them constant
   -- 'add'            To add a constant value and keep the time evolution
   -- 'linear'         To increase linearly between two extremes
-- correct boundaries  Will increase the boundaries when below 20 eV
-- add early profiles  Will extrapolate the profiles to 0.01. Not implemented for the 2d equilibrium yet
-- parabolic zeff, peaked zeff  Sets a hollow or a peaked profile for Zeff
-- correct zeff        Corrects Zeff where it is below 1.02 or above 4
-- flat q profile      Sets up a flat q profile with the boundary values everywhere


boundary_instructions: Options to modify the boundaries for te and ti. It modifies the edge and keeps the axis the same, linearly


1 - setup_input_baserun(verbose = False):
2 - setup_input_sensitivities()
3 - create_baserun()
4 - create_sensitivities(force_run = False)
5 - run_baserun()
6 - run_sensitivities(force_run = False)


# Setting up a single folder ready for integrated modelling
setup_jetto_simulation()
setup_feedback_on_density()


# Tools: modify the jset and llcmd files
modify_jset(path, sensitivity_name, ids_number, ids_output_number, db, username, shot)
modify_jset_line(sensitivity_name, line_start, new_content)
modify_llcmd(sensitivity_name, baserun_name)
add_item_lookup


# Used to modify jetto extranamelist
get_extraname_fields()
add_extraname_fields()
put_extraname_fields()


# Small utilities, hopefully temporary
check_and_flip_ip(db, shot, run, shot_target, run_target)
flip_ip(db, shot, run, shot_target, run_target)


'''

'''
class BaseJettoSystem(AbstractSystem):
    def submit_slurm(job: Job):
        if not job.has_submit_script:
            raise FileNotFoundError(job.submit_script)

        submit_cmd = job.cfg.submit.submit_command.split()
        cmd: list[Any] = [*submit_cmd, str(job.submit_script)]

        logger.info(f'submitting script via slurm {cmd}')

        ret = sp.run(cmd, check=True, capture_output=True)
        logger.info('submission returned: ' + str(ret.stdout))
        with open(job.lockfile, 'wb') as f:
            f.write(ret.stdout)

    def copy_from_template(source_drc: Path, target_drc: Path):
        jetto_jset = jset.read(source_drc / 'jetto.jset')
        jetto_namelist = namelist.read(source_drc / 'jetto.in')

        jetto_sanco = None
        if (source_drc / 'jetto.sin').exists():
            jetto_sanco = namelist.read(source_drc / 'jetto.sin')

        all_files = os.listdir(source_drc)

        jetto_extra = _get_jetto_extra(all_files, jset=jetto_jset)

        jetto_extra = [str(source_drc / file) for file in jetto_extra]

        jetto_template = template.Template(jset=jetto_jset,
                                           namelist=jetto_namelist,
                                           lookup=jetto_lookup,
                                           sanco_namelist=jetto_sanco,
                                           extra_files=jetto_extra)

        JettoSystem._apply_patches_to_template(jetto_template)

        jetto_config = config.RunConfig(jetto_template)

        jetto_config.export(target_drc)
        lookup.to_file(jetto_lookup, target_drc /
                       'lookup.json')  # TODO, this should be copied as well

        for filename in (
                'rjettov',
                'utils_jetto',
        ):
            src = source_drc / filename
            dst = target_drc / filename
            shutil.copyfile(src, dst)
            dst.chmod(dst.stat().st_mode | stat.S_IXUSR)


class JettoSystemV220922(BaseJettoSystem):
    """System that can be used to create runs for jetto.
    The backend that is  assumed is jetto-v220922. The most important
    difference with v210921 is that the IMAS data is handled locally
    instead of via a public `imasdb`
    This system can submit to various backends like docker, prominence
    and the gateway.
    ```yaml title="duqtools.yaml"
    system: jetto-v220922
    submit:
      submit_system: docker
    ```
    """
    def get_data_in_handle(*, dirname: Path, source: ImasHandle, seq_number: int, options):
        return ImasHandle

    def get_data_out_handle(*, dirname: Path, source: ImasHandle, seq_number: int, options):
        return ImasHandle


def _get_jetto_extra(filenames: List[str], *, jset):
    jetto_extra: List[str] = []
    for regex in _EXTRA_FILE_REGEXES:
        jetto_extra.extend(filter(regex.match, filenames))

  def write_batchfile(run_dir: Path, cfg: Config):
        jetto_jset = jset.read(run_dir / 'jetto.jset')
        _write_batchfile(run_dir, jetto_jset, tag=cfg.tag)

    @staticmethod
    def submit_job(job: Job):
        # Make sure we get a new correct status
        if (job.dir / 'jetto.status').exists():
            os.remove(job.dir / 'jetto.status')

        submit_system = job.cfg.submit.submit_system


class Job:

    def __init__(self, dir: Path):

   def start(self):
        click.echo(f'Submitting {self}\033[K')
        self.submit()

class Operation(OperatorMixin, BaseModel):
    variable: str
    value: float

    def convert(self):
        """Expand variable and convert to correct type."""
        from duqtools.config import var_lookup
        variable = var_lookup[self.variable]

        if isinstance(variable, JettoVariableModel):
            cls = JettoOperation
        elif isinstance(variable, IDSVariableModel):
            cls = IDSOperation
        else:
            raise NotImplementedError(
                f'{self.variable} convert not implemented')

        mapping = self.dict()
        mapping['variable'] = variable

        return cls(**mapping)

class VariableConfigModel(BaseModel):
    __root__: list[Union[JettoVariableModel, IDSVariableModel,
                         IDS2JettoVariableModel]]

    def to_variable_dict(self) -> dict:
        """Return dict of variables."""
        return {variable.name: variable for variable in self}


class JettoVariableModel(BaseModel):
    """Variable for describing variables specific to Jetto, The lookup table
    can be defined as a JettoVar under the lookup key."""

    type: str = Field('jetto-variable',
                      description='Discriminator for the variable type.')

    name: str = Field(description=f("""
        Name of the variable.
        Used for the lookup table to find actual fields.
        """))
    lookup: Optional[JettoVar] = Field(description=f("""
    Description of the fields that have to be updated for a Jetto Variable
    """))


class JettoVar(BaseModel):
    """These describe the jetto variables."""
    doc: str = Field(description='Docstring for the variable.')
    name: str = Field(description='Name of the variable.')
    type: Literal['str', 'int', 'float'] = Field(
        description=f('Type of the variable (str, int, float)'))
    keys: list[JettoField] = Field(description=f(
        'Jetto keys to update when this jetto variable is requested'))

    available: get_type

#Basemodel is separately described in pydantic. Can use dict(), json(), parse_obj(), schema(), schema_json()
'''

class IntegratedModellingRuns:
    def __init__(
	self, 
	shot, 
	instructions_list, 
	generator_name, 
	baserun_name, 
	db = 'tcv', 
	run_input = 1,
        run_start = None,
	run_output = 100,
	time_start = None,
	time_end = 100,
        esco_timesteps = None,
        output_timesteps = None,
	force_run = False,
        force_input_overwrite = False,
	density_feedback = False,
        setup_time_polygon_flag = False,
        setup_nbi_flag = False,
        path_nbi_config = None,
        json_input = None,
        sensitivity_list = [],
    ):

        # db is the name of the machine. Needs to be the name of the imas database.
        # shot is the shot number. It is an int.
        # run input is where the input is. It will not be the input for the simulations though, since it needs to be massaged
        # run output is the output number for the baserun. The sensitivities will start here and increase by 1 as in the list
        # generator name is the name of the generator as it appears in the run folder

        self.username = getpass.getuser()
        self.db = db
        self.shot = shot
        self.run_input = run_input
        self.run_start = run_start
        self.run_output = run_output
        self.time_start = time_start
        self.time_end = time_end
        self.esco_timesteps = esco_timesteps
        self.output_timesteps = output_timesteps
        self.force_run = force_run
        self.force_input_overwrite = force_input_overwrite
        self.density_feedback = density_feedback
        self.setup_time_polygon_flag = setup_time_polygon_flag
        self.setup_nbi_flag = setup_nbi_flag
        self.path_nbi_config = path_nbi_config
        self.core_profiles = None
        self.equilibrium = None
        self.line_ave_density = None
        self.json_input = json_input
        self.sensitivity_list = sensitivity_list

        # Trying to be a little flexible with the generator name. It is not used if I am only setting the input.
        # Still mandatory argument, should not be forgotten

        self.path = '/pfs/work/' + self.username + '/jetto/runs/'
        self.generator_username = ''

        if generator_name.startswith('/pfs/work'):
            self.path_generator = generator_name
            self.generator_name = generator_name.split('/')[-2]
            self.generator_username = generator_name.split('/')[3]
        elif generator_name.startswith('rungenerator_'):
            self.generator_name = generator_name
            self.path_generator = self.path + self.generator_name
            self.generator_username = self.username  # new
        else:
            self.generator_name = 'rungenerator_' + generator_name
            self.path_generator = self.path + self.generator_name
            self.generator_username = self.username # new

        self.baserun_name = baserun_name

        # Default instructions: do nothing
        self.instructions = {
            'setup base' : False,
            'setup sens' : False,
            'create base' : False,
            'create sens' : False,
            'run base' : False,
            'run sens' : False
        }

        for key in instructions_list:
            if key in self.instructions:
                self.instructions[key] = True

        # Default sensitivity list. The sensitivity list can be omitted and will not be used when only dealing with the baserun

        # Example of sensitivity list. Not default.
        #if not sensitivity_list:
        #    self.sensitivity_list = ['te 0.8', 'te 1.2', 'ne 0.8', 'ne 1.2', 'zeff 0.8', 'zeff 1.2', 'q95 0.8', 'q95 1.2']

        # Default baserun name is 'run000'. It is not used if I am only setting the input. Baserun name should always start with 'run###'

        if self.baserun_name == '':
            self.baserun_name = 'run000'  + str(self.shot) + 'base'

        self.path_baserun = self.path + self.baserun_name


        self.tag_list = []
        for sensitivity in self.sensitivity_list:
            tag = sensitivity.replace(' ', '_')
            tag = tag.replace('.', '_')
            tag = '_' + tag
            self.tag_list.append(tag)

        # Default nbi path is in public.
        if not self.path_nbi_config:
            self.path_nbi_config = '/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg'

        # New instructions are just an array with six True/False (or 0/1). They correspond orderly to what to do in the instruction list

    def update_instructions(self, new_instructions):

        for i, key in enumerate(self.instructions):
            self.instructions[key] = new_instructions[i]

    def update_sensitivities(self, new_sensitivities_list):

        self.sensitivity_list = new_sensitivities_list

    def setup_create_compare(self, verbose = False):


        # Could use the list directly but this should be more readable. instructions_list needs to be a list of six values, connected with the instructions
        # Put checks on the list to make sure that is fool proof

        if self.instructions['setup base']:
            self.setup_input_baserun(verbose = False)
        if self.instructions['setup sens']:
            self.setup_input_sensitivities()
        if self.instructions['create base']:
            self.create_baserun()
        if self.instructions['create sens']:
            self.create_sensitivities()
        if self.instructions['run base']:
            self.run_baserun()
        if self.instructions['run sens']:
            self.run_sensitivities()

    def setup_input_baserun(self, verbose = False):
    
        '''
    
        Modified the setup function to have it in a separate file as an extra option. The new script can be used standalone.
        If the setup is not used importing the script will not be necessary    

        '''

        try:
            import prepare_im_input
        except ImportError:
            print('prepare_input.py not found and needed for this option. Aborting')
            exit()

        if not self.json_input:
            json_file_name = '/afs/eufus.eu/user/g/g2mmarin/public/scripts/template_prepare_input.json'
            print('json input to prepare the runs not specified, using dummy file')
            json_input_raw = open(json_file_name)
            self.json_input = json.load(json_input_raw)

        self.core_profiles, self.equilibrium = prepare_im_input.setup_input(self.db, self.shot, self.run_input, self.run_start, json_input = self.json_input, time_start = self.time_start, time_end = self.time_end, force_input_overwrite = self.force_input_overwrite, core_profiles = self.core_profiles, equilibrium = self.equilibrium)


    def setup_input_sensitivities(self):
    
        '''
    
        Automatically sets up the IDSs to be used as an input for a sensitivity study. More sensitivities can be added
    
        '''
    
        try:
            import prepare_im_input
        except ImportError:
            print('prepare_input.py not found and needed for this option. Aborting')
            exit()

    # Maybe here I am already creating all the entries, which might be a problem if I change 'shift profiles', but should be allright for now

        for index in range(1,len(self.tag_list),1):
            data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, self.db, self.shot, self.run_start+index, user_name=self.username)
            op = data_entry.open()
    
            if op[0]==0:
                print('one of the data entries already exists, aborting')
                exit()
    
            data_entry.close()
    
        # Could check that there are no idss here before I overwrite evverything
    
        name, mult = [], []
    
    # Give the option to the user to decide which sensitivities should be done
    
        index = 1

        for run in self.sensitivity_list:
            name, mult = run.split(' ')
            name = name
            mult = float(mult)
    
            if name == 'tepeak':
                prepare_im_input.peak_temperature(self.db, self.shot, self.run_start, self.db, self.shot, self.run_start+index, mult = mult)
            if name == 'q95':
                prepare_im_input.alter_q_profile_same_q95(self.db, self.shot, self.run_start, self.db, self.shot, self.run_start+index, mult = mult)
            else:
                prepare_im_input.shift_profiles(name, self.db, self.shot, self.run_start, self.run_start+index, mult = mult)
    
            index += 1
    
    def create_baserun(self):
    
        '''
    
        Automatically sets up the folder for the baserun of a specific scan.
        The type of the baserun will determine which kind of runs the sensitivity should be carried out from. Default options are given
    
        '''
  
        os.chdir(self.path)

        if os.path.exists(self.path_generator):
            shutil.copytree(self.path_generator, self.path_baserun)
        else:
            print('generator not recognized. Aborting')
            exit()

        # To save time, equilibrium and core profiles are not extracted if they already exist
        if not self.core_profiles:
            self.core_profiles = open_and_get_core_profiles(self.db, self.shot, self.run_input)

        if not self.equilibrium:
            self.equilibrium = open_and_get_equilibrium(self.db, self.shot, self.run_input)

        time_eq = self.equilibrium.time
        time_cp = self.core_profiles.time

        if self.time_start == None:
            self.time_start = max(min(time_eq), min(time_cp))
        elif self.time_start == 'core_profiles':
            self.time_start = min(time_cp)
        elif self.time_start == 'equilibrium':
            self.time_start = min(time_eq)

        if self.time_end == 100:
            self.time_end = min(max(time_eq), max(time_cp))
        if self.time_end == 'auto':
            summary = open_and_get_summary(self.db, self.shot, self.run_input)
            kfactor = 0.05
            mu0 = 4 * np.pi * 1.0e-7
            time_sim = kfactor * mu0 * np.abs(summary.global_quantities.ip.value[0] * summary.global_quantities.r0.value)
            time_end = self.time_start + time_sim

        b0, r0 = self.get_r0_b0()

        if self.density_feedback == True:
            self.get_feedback_on_density_quantities()        

        # This should not be needed and should be handled by the jetto_tools. It's not though...
        # This is untested and might break
        imp_data = []
        first_imp_density = None
        for ion in self.core_profiles.profiles_1d[0].ion:
            imp_density = np.average(ion.density)
            z_ion = ion.element[0].z_n
            a_ion = ion.element[0].a
            z_bundle = round(z_ion)
            if z_ion > 1:
                if not first_imp_density:
                    imp_relative_density = 1.0
                    first_imp_density = imp_density
                else:
                    imp_relative_density = first_imp_density/imp_density

                imp_data.append([imp_relative_density, a_ion, z_bundle, z_ion])

#        imp_data = [[1.0, 12.0, 6, 6.0]]

        if (b0 > 0 and self.equilibrium.time_slice[0].global_quantities.ip < 0) or (b0 < 0 and self.equilibrium.time_slice[0].global_quantities.ip > 0):
            ibtsign = 1
        else:
            ibtsign = -1

        if 'interpretive' not in self.path_generator:
            interpretive_flag = False
        else:
            interpretive_flag = True

        # Still cannot run with positive current...
        self.modify_jetto_in(self.baserun_name, r0, abs(b0), self.time_start, self.time_end, imp_datas_ids = imp_data, num_times_print = self.output_timesteps, num_times_eq = self.esco_timesteps, ibtsign = ibtsign, interpretive_flag = interpretive_flag)
        #self.modify_jetto_in(self.baserun_name, r0, b0, self.time_start, end_time, imp_datas_ids = imp_data, num_times_print = self.output_timesteps, ibtsign = ibtsign)

        self.setup_jetto_simulation()
    
        if self.density_feedback == True:
            self.setup_feedback_on_density()

        self.modify_jset(self.path, self.baserun_name, self.run_start, self.run_output, abs(b0), r0)
        #self.modify_jset(self.path, self.baserun_name, self.run_start, self.run_output, b0, r0)

        if self.setup_nbi_flag:
            nbi = open_and_get_nbi(self.db, self.shot, self.run_start)
            if nbi.time != np.asarray([]):
                self.setup_nbi(path_nbi_config = self.path_nbi_config)
            else:
                print('You are trying to setup the nbi but the nbi ids is empty. Aborting')
                exit()

        if self.setup_time_polygon_flag:
            self.setup_time_polygon()

        # Selecting the impurity correctly in the jset

        impurity_jset_linestarts = ['ImpOptionPanel.impuritySelect[]',
                                    'ImpOptionPanel.impurityMass[]',
                                    'ImpOptionPanel.impurityCharge[]',
                                    'ImpOptionPanel.impuritySuperStates[]'
                                   ]
        for index in range(6):
            if index < len(imp_data):
                for jset_linestart in impurity_jset_linestarts:
                    line_start = jset_linestart[:-2] + str(index) + jset_linestart[-1]
                    if jset_linestart == 'ImpOptionPanel.impuritySelect[]':
                        new_content = '1'
                    elif jset_linestart == 'ImpOptionPanel.impurityMass[]':
                        new_content = str(imp_data[index][1])
                    elif jset_linestart == 'ImpOptionPanel.impurityCharge[]':
                        new_content = str(imp_data[index][2])
                    elif jset_linestart == 'ImpOptionPanel.impuritySuperStates[]':
                        new_content = str(imp_data[index][3])

                    modify_jset_line(self.baserun_name, line_start, new_content)

            else:
                line_start = 'ImpOptionPanel.impuritySelect[' + str(index) + ']'
                new_content = 'false'
                modify_jset_line(self.baserun_name, line_start, new_content)

        modify_llcmd(self.baserun_name, self.generator_name, self.generator_username)

        self.copy_ids_input()


    # This will work with MDSPLUS. Should code something else for HDF5
    def copy_ids_input(self):

        if self.run_start < 10:
            run_str = '000' + str(self.run_start)
        elif self.run_start < 100:
            run_str = '00' + str(self.run_start)
        elif self.run_start < 1000:
            run_str = '0' + str(self.run_start)
        else:
            run_str = str(self.run_start)

        path_ids_input = '/afs/eufus.eu/user/g/' + self.username + '/public/imasdb/' + self.db + '/3/0/ids_' + str(self.shot) + run_str
        path_characteristics = path_ids_input + '.characteristics'
        path_datafile = path_ids_input + '.datafile'
        path_tree = path_ids_input + '.tree'

        path_output = self.path_baserun+ '/imasdb/' + self.db + '/3/0/ids_' + str(self.shot) + '0001'

        # This creates the folder when the machine is not the same as in the generator case
        if not os.path.exists(self.path_baserun+ '/imasdb/' + self.db):
            db_generator = os.listdir(self.path_baserun+ '/imasdb/')[0]
            shutil.copytree(self.path_baserun+ '/imasdb/' + db_generator, self.path_baserun+ '/imasdb/' + self.db)
            shutil.rmtree(self.path_baserun+ '/imasdb/' + db_generator)

        # This deletes the IDS of the generator
        for filename in os.listdir(self.path_baserun+ '/imasdb/' + self.db + '/3/0/'):
            file_path = os.path.join(self.path_baserun+ '/imasdb/' + self.db + '/3/0/', filename)
            os.remove(file_path)

        shutil.copyfile(path_ids_input + '.characteristics', path_output + '.characteristics')
        shutil.copyfile(path_ids_input + '.datafile', path_output + '.datafile')
        shutil.copyfile(path_ids_input + '.tree', path_output + '.tree')


    def create_sensitivities(self):
    
        '''
    
        Sets up and runs the simulations created by setup_input_sensitivities(). Runs are expected to be numbered as run###something
    
        '''

        # To save time, equilibrium and core profiles are not extracted if they already exist
        if not self.core_profiles:
            self.core_profiles = open_and_get_core_profiles(self.db, self.shot, self.run_input)

        if not self.equilibrium:
            self.equilibrium = open_and_get_equilibrium(self.db, self.shot, self.run_input)

        time_eq = self.equilibrium.time
        time_cp = self.core_profiles.time

        if self.time_start == None:
            self.time_start = max(min(time_eq), min(time_cp))
        elif self.time_start == 'core_profile':
            self.time_start = min(time_cp)
        elif self.time_start == 'equilibrium':
            self.time_start = min(time_eq)

        if self.time_end == 100:
            self.time_end = min(max(time_eq), max(time_cp))

        baserun_number = int(self.baserun_name[3:6])
    
        if not self.force_run:
            for index in range(1,len(self.tag_list),1):
                data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, self.db, self.shot, self.run_output+index, user_name=self.username)
                op = data_entry.open()
    
                if op[0]==0:
                    print('one of the data entries already exists, aborting')
                    exit()
    
                data_entry.close()
    
        number_list = range(baserun_number+1,baserun_number+len(self.tag_list)+1,1)
        number_list = [str(i) for i in number_list]
        ids_list = range(self.run_start+1,self.run_start+len(self.tag_list)+1,1)
        ids_list = [str(i) for i in ids_list]
        ids_output_list = range(self.run_output+1,self.run_output+len(self.tag_list)+1,1)
        ids_output_list = [str(i) for i in ids_output_list]
    
        os.chdir(self.path)
    
        sensitivity_names_list = []

        for number, tag in zip(number_list, self.tag_list):
            sensitivity_names_list.append(self.baserun_name[:3] + number + self.baserun_name[6:] + tag)
    
        for sensitivity_name in sensitivity_names_list:
            shutil.copytree(self.baserun_name, sensitivity_name)
    
        for sensitivity_name, ids_number, ids_output_number in zip(sensitivity_names_list, ids_list, ids_output_list):
            os.chdir(self.path)
    
            b0, r0 = self.get_r0_b0()

            self.modify_jset(self.path, sensitivity_name, ids_number, ids_output_number, b0, r0)
            modify_llcmd(sensitivity_name, self.baserun_name, self.generator_username)
    
    def run_baserun(self):
    
        # ------- Not working yet, jetto_tool automatically setup a slurm environment -------

        #    manager = jetto_tools.job.JobManager()
        #    manager.submit_job_to_batch(config, baserun_name + 'tmp', run=False)

        # ------- Substitute this to the custom automatic run when available --------

        os.chdir(self.path + self.baserun_name)
        print('running ' + self.baserun_name)
        os.system('sbatch ./.llcmd')
    
    
    def run_sensitivities(self):
    
        '''
    
        It assumes that the inputs for the sensitivities and the run folders already exists, and runs them.
    
        '''
    
        baserun_number = int(self.baserun_name[3:6])
    
        # If force run is true it will overwrite whaterver is in the target runs. You might lose the output from previous simulations.
    
        if not self.force_run:
            for index in range(1,len(self.tag_list),1):
                data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, self.db, self.shot, self.run_output+index, user_name=self.username)
                op = data_entry.open()
    
                if op[0]==0:
                    print('one of the data entries already exists, aborting')
                    exit()
    
                data_entry.close()
    
        number_list = range(baserun_number+1,baserun_number+len(self.tag_list)+1,1)
        number_list = [str(i) for i in number_list]

        sensitivity_names_list = []

        for number, tag in zip(number_list, self.tag_list):
            sensitivity_names_list.append(self.baserun_name[:3] + number + self.baserun_name[6:] + tag)
    
        os.chdir(self.path)
    
        for sensitivity_name in sensitivity_names_list:
            os.chdir(self.path + '/' + sensitivity_name)
            print('running ' + sensitivity_name)
            os.system('sbatch ./.llcmd')


    def get_r0_b0(self):

        # -------------------- GET b0 and r0 ---------------------
        # Only extract once. This takes time so it's only for speed purposes

        if not self.core_profiles:
            self.core_profiles = open_and_get_core_profiles(self.db, self.shot, self.run_input)
        if not self.equilibrium:
            self.equilibrium = open_and_get_equilibrium(self.db, self.shot, self.run_input)

        # Here I can set the initial time as the time where I can find the first measurement in core profiles or equilibrium

        time_eq = self.equilibrium.time
        time_cp = self.core_profiles.time

        index_start = np.abs(time_eq - self.time_start).argmin(0)
        index_end = np.abs(time_eq - self.time_end).argmin(0)

        if index_start != index_end:
            b0 = np.average(self.equilibrium.vacuum_toroidal_field.b0[index_start:index_end])
        else:
            b0 = self.equilibrium.vacuum_toroidal_field.b0[0]

        r0 = self.equilibrium.vacuum_toroidal_field.r0*100

# -----------------------------------------------------

        return b0, r0

    
    def setup_jetto_simulation(self):
    
        '''
    
        Uses the jetto_tools to setup various parameters for the jetto simulation.
        Updates the magnetic field and the radius. Can be used to update the output and the and the impurity composition.
        Updates the jetto starting time as the first time for which data are available both for the
        equilibrium and the core profiles. Strongly advised!
    
        '''
    
        # In the future an option to operate with the correct ip sign could be added here. Not currently working
        # Also, IBTSIGN seems not to be in the list as it should. not sure what is happening...
    
        #lookup = jetto_tools.lookup.from_file(self.path + '/lookup_json/lookup.json')
        #jset = jetto_tools.jset.read(self.path_generator + '/jetto.jset')
        #namelist = jetto_tools.namelist.read(self.path_generator + '/jetto.in')

        b0, r0 = self.get_r0_b0()

        if not self.core_profiles:
            self.core_profiles = open_and_get_core_profiles(self.db, self.shot, self.run_input)
        if not self.equilibrium:
            self.equilibrium = open_and_get_equilibrium(self.db, self.shot, self.run_input)

        if not os.path.exists(self.path_baserun):
            shutil.copytree(self.path_generator, self.path_baserun)
        else:
            shutil.copyfile(self.path + '/lookup_json/lookup.json', self.path_baserun + '/lookup.json')  # Just this line should be fine
#            shutil.copyfile(self.path_generator + '/jetto.in', self.path_baserun + '/jetto.in')

        # Changing the orientation when necessary
        # Add IBTSING if ip sign and b0 sign are opposite. There is still a bug.

        extranamelist = get_extraname_fields(self.path_baserun)
        if 'interpretive' not in self.path_generator:
            add_item_lookup('btin', 'EquilEscoRefPanel.BField.ConstValue', 'NLIST1', 'real', 'scalar', self.path_baserun)
            add_item_lookup('rmj', 'EquilEscoRefPanel.refMajorRadius', 'NLIST1', 'real', 'scalar', self.path_baserun)
            add_item_lookup('ibtsign', 'null', 'NLIST1', 'int', 'scalar', self.path_baserun)

            if (b0 > 0 and self.equilibrium.time_slice[0].global_quantities.ip < 0) or (b0 < 0 and self.equilibrium.time_slice[0].global_quantities.ip > 0):
                extranamelist = add_extraname_fields(extranamelist, 'IBTSIGN', ['1'])
            else:
                extranamelist = add_extraname_fields(extranamelist, 'IBTSIGN', ['-1'])

        put_extraname_fields(self.path_baserun, extranamelist)

        # A temporary function to handle arrays since the pythontools do not do it yet. When the option comes online again use that.
        self.tmp_handle_arrays_open()

        template = jetto_tools.template.from_directory(self.path_baserun)
        config = jetto_tools.config.RunConfig(template)

        if 'interpretive' not in self.path_generator:
            if (b0 > 0 and self.equilibrium.time_slice[0].global_quantities.ip < 0) or (b0 < 0 and self.equilibrium.time_slice[0].global_quantities.ip > 0):
                config['ibtsign'] = 1
            else:
                config['ibtsign'] = -1

        if 'interpretive' not in self.path_generator:
            config['btin'] = abs(b0)
            # Absolute value should not be needed anymore since the fix on the ip sign
            #config['btin'] = b0
            config['rmj'] = r0

# Try to build a template as is done in system.py

        if self.esco_timesteps:
            config.esco_timesteps = self.esco_timesteps
        if self.output_timesteps:
            config.profile_timesteps = self.output_timesteps
            config['ntint'] = self.output_timesteps

        config.start_time = self.time_start
        config.end_time = self.time_end

        # I could introduce a way not to do this if there are no impurities. Need to add impurities if not there, modifying the various files. Maybe in the future. For now I modify the jset anyway so this part does nothing...
   
        config['atmi'] = 6.0
        config['nzeq'] = 12.0
        config['zipi'] = 6

        config.export(self.path_baserun + 'tmp')
        shutil.copyfile(self.path_baserun + 'tmp' + '/jetto.jset', self.path_baserun + '/jetto.jset')
        #shutil.copyfile(self.path_baserun + 'tmp' + '/jetto.in', self.path_baserun + '/jetto.in')
        shutil.rmtree(self.path_baserun + 'tmp')

        # A temporary function to handle arrays since the pythontools do not do it yet. When the option comes online again use that.
        self.tmp_handle_arrays_close()


    def tmp_handle_arrays_open(self):
        extranamelist = get_extraname_fields(self.path_baserun)
        for key in extranamelist:
            if '(' in extranamelist[key][0] and ')' in extranamelist[key][0]:
                extranamelist[key][0] = '\'' + extranamelist[key][0] + '\''
        put_extraname_fields(self.path_baserun, extranamelist)


    def tmp_handle_arrays_close(self):
        extranamelist = get_extraname_fields(self.path_baserun)
        for key in extranamelist:
            if '(' in extranamelist[key][0] and ')' in extranamelist[key][0]:
                extranamelist[key][0] = extranamelist[key][0].strip('\'')
        put_extraname_fields(self.path_baserun, extranamelist)


    def increase_processors(self, processors = 8, walltime = 24):

        binary, userid = 'v210921_gateway_imas', 'g2fkoech'

        template = jetto_tools.template.from_directory(self.path_baserun)
        config = jetto_tools.config.RunConfig(template)

        config.binary = binary
        config.userid = userid
        config.processors = processors
        config.walltime = walltime

    def get_feedback_on_density_quantities(self):

        # Could add a check if run_exp exists. Should become the runinput though...

        #summary = open_and_get_summary(self.db, self.shot, self.run_exp)
        #self.summary_time = summary.time
        #self.line_ave_density = summary.line_average.n_e.value


        pulse_schedule = open_and_get_pulse_schedule(self.db, self.shot, self.run_start)
        self.dens_feedback_time = pulse_schedule.time
        self.line_ave_density = pulse_schedule.density_control.n_e_line.reference.data


    def setup_feedback_on_density(self):
        '''
    
        Still deciding what exactly this will be. Some step to setup a run with automatic density feedback control
        Strategy should be: add this when setting up the correponding baserun.
    
        '''

        dneflfb_strs = []
        for density in self.line_ave_density*1e-6:
            dneflfb_strs.append(str(density))

        dtneflfb_strs = []
        for time in self.dens_feedback_time:
            dtneflfb_strs.append(str(time))

        extranamelist = get_extraname_fields(self.path_baserun)

        if not dneflfb_strs:
            print('No quantity to set the density feedback. Aborting')
            exit()

        extranamelist = add_extraname_fields(extranamelist, 'DNEFLFB', dneflfb_strs)
        extranamelist = add_extraname_fields(extranamelist, 'DTNEFLFB', dtneflfb_strs)
        put_extraname_fields(self.path_baserun, extranamelist)

        # Ideally the following should be enough. Currently it is not working.
        '''
        add_item_lookup('dneflfb', 'null', 'NLIST4', 'real', 'vector', self.path_baserun)
        add_item_lookup('dtneflfb', 'null', 'NLIST4', 'real', 'vector', self.path_baserun)

        template = jetto_tools.template.from_directory(self.path_baserun)
        config = jetto_tools.config.RunConfig(template)

        config['dneflfb'] = self.line_ave_density*1e-6
        config['dtneflfb'] = self.dens_feedback_time
    
        # ------- Can use to create the baseruns when I understand how to create a template from a run without the lookup file (probably just creating the lookup file there)
    
        config.export(self.path_baserun + 'tmp')
        shutil.copyfile(self.path_baserun + 'tmp' + '/jetto.jset', self.path_baserun + '/jetto.jset')
        shutil.rmtree(self.path_baserun + 'tmp')
        '''

    def modify_jset(self, path, run_name, ids_number, ids_output_number, b0, r0):
    
        '''
    
        Modifies the jset file to accomodate a new run name, username, shot and run. Database not really implemented yet
    
        '''
    
        # Might want more flexibility with the run list here. Maybe set more options in the future
        # The last values with the final times should be handled within the config, but are not. They should be temporary

        line_start_list = [
            'Creation Name', 
            'JobProcessingPanel.runDirNumber', 
            'SetUpPanel.idsIMASDBRunid', 
            'JobProcessingPanel.idsRunid', 
            'AdvancedPanel.catMachID', 
            'AdvancedPanel.catMachID_R', 
            'SetUpPanel.idsIMASDBMachine',
            'SetUpPanel.machine',
            'SetUpPanel.idsIMASDBUser',
            'AdvancedPanel.catOwner', 
            'AdvancedPanel.catOwner_R', 
            'AdvancedPanel.catShotID', 
            'AdvancedPanel.catShotID_R',
            'SetUpPanel.idsIMASDBShot', 
            'SetUpPanel.shotNum',
            'SetUpPanel.endTime',
            'EquilEscoRefPanel.tvalue.tinterval.endRange',
            'EquilIdsRefPanel.rangeEnd',
            'OutputStdPanel.profileRangeEnd',
            'SetUpPanel.startTime',
            'EquilEscoRefPanel.tvalue.tinterval.startRange',
            'EquilIdsRefPanel.rangeStart',
            'OutputStdPanel.profileRangeStart',
            'EquilEscoRefPanel.BField.ConstValue',
            'EquilEscoRefPanel.BField ',
            'EquilEscoRefPanel.refMajorRadius'
        ]
    
        new_content_list = [
            path + run_name + '/jetto.jset', 
            run_name[3:], 
            str(ids_number), 
            str(ids_output_number), 
            self.db, 
            self.db,
            self.db,
            self.db, 
            self.username,
            self.username, 
            self.username, 
            str(self.shot), 
            str(self.shot), 
            str(self.shot), 
            str(self.shot),
            str(self.time_end),
            str(self.time_end),
            str(self.time_end),
            str(self.time_end),
            str(self.time_start),
            str(self.time_start),
            str(self.time_start),
            str(self.time_start),
            str(b0),
            str(b0),
            str(r0)
        ]
    
        # ImpOptionPanel.impuritySelect[1]                            : false to deselect the impurity
    
        for line_start, new_content in zip(line_start_list, new_content_list):
            modify_jset_line(run_name, line_start, new_content)


    def modify_ascot_cntl(self, run_name):

        line_start = 'Creation Name'
        modify_ascot_cntl_line(run_name, line_start, run_name + '/ascot.cntl')


    def modify_jset_nbi(self, run_name, nbi_config_name):

        '''

        Modifies the jset file to accomodate a new run name, username, shot and run. Database not really implement
ed yet

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
            modify_jset_line(run_name, line_start, new_content)


    def modify_jetto_in(self, sensitivity_name, r0, b0, time_start, time_end, num_times_print = None, num_times_eq = None, imp_datas_ids = [[1.0, 12.0, 6, 6.0]], ibtsign = 1, interpretive_flag = False):

        '''
    
        modifies the jset file to accomodate a new run name. Default impurity is carbon
    
        '''

        imp_datas = []
        for index in range(7):
            imp_datas.append([0.0, 0.0, 0, 0.0])

        for index, imp_data in enumerate(imp_datas_ids):
            imp_datas[index] = imp_data

        imp_density, imp_mass, imp_super, imp_charge = '', '', '', ''

        for index in range(7):
            imp_density += str(imp_datas[index][0]) + '      ,  '
            imp_mass += str(imp_datas[index][1]) + '      ,  '
            imp_super += str(imp_datas[index][2]) + '      ,  '
            imp_charge += str(imp_datas[index][3]) + '      ,  '

        read_data = []
    
        with open(sensitivity_name + '/' + 'jetto.in') as f:
            lines = f.readlines()
            for line in lines:
                read_data.append(line)
    
        # Could also use a list here as well, but just trying now
        index_btin, index_nlist1, index_nlist4 = 0, 0, 0

        original_num_tprint = 1

        for index, line in enumerate(read_data):
            if line.startswith('  NTINT'):
                original_num_tprint = int(re.search(r'\d+', line).group())

        jetto_in_nameslist = {
            '  RMJ': str(r0),
            '  BTIN': str(b0),
            '  TBEG': str(time_start),
            '  TMAX': str(time_end),
            '  MACHID': '\'' + self.db + '\'',
            '  NPULSE': str(self.shot),
            '  NIMP': str(len(imp_datas_ids))
            #'  NIMP': '1'  # Needs to be changed
        }

        if num_times_print != None:
           jetto_in_nameslist['  NTINT'] = str(num_times_print)
           jetto_in_nameslist['  NTPR'] = str(num_times_print - 2)

        if interpretive_flag:
           del jetto_in_nameslist['  RMJ']
           del jetto_in_nameslist['  BTIN']

        for index, line in enumerate(read_data):
            if line[:6] == '  BTIN':
                index_btin = index
            elif line[:18] == ' Namelist : NLIST4':
                index_nlist4 = index
            elif line[:8] == ' &NLIST1':
                index_nlist1 = index

        for index, line in enumerate(read_data):
            for jetto_name in jetto_in_nameslist:
                if line.startswith(jetto_name):
                    read_data[index] = read_data[index][:14] + jetto_in_nameslist[jetto_name] + '    ,'  + '\n'

            # needs to be modified
            if line[:8] == '  TIMEQU':
                if num_times_eq:
                    read_data[index] = read_data[index][:14] + str(time_start) + ' , ' + str(time_end) + ' , '
                    # Testing, do not leave like this
                    #read_data[index] += str(num_times_eq) + ' , ' + '\n'
                    read_data[index] += str((time_end - time_start)/num_times_eq) + ' , ' + '\n'
                else:
                    original_time_eq = re.findall("\d+\.\d+", line)
                    if len(original_time_eq) == 1:
                        num_times_eq = 1
                    else:
                        num_times_eq = int(round((float(original_time_eq[1]) - float(original_time_eq[0]))/float(original_time_eq[-1])))
                        # The meaning of the numbers is different when interpretive equilibrium. Trying to handle that here.
                        if num_times_eq == 0:
                            num_times_eq = int(round(float(original_time_eq[-1])))
                    read_data[index] = read_data[index][:14] + str(time_start) + ' , ' + str(time_end) + ' , '
                    # Testing, do not leave like this
                    read_data[index] += str((time_end - time_start)/num_times_eq) + ' , ' + '\n'
                    #read_data[index] += str(num_times_eq) + ' , ' + '\n'

        # Nexessary for interpretive runs when there is no btin
        if index_btin == 0:
            index_btin = index_nlist1 + 2
            read_data.insert(index_btin, '  RMJ   =  ' + str(r0) + '     ,'  + '\n')
            read_data.insert(index_btin, '  BTIN  =  ' + str(b0) + '     ,'  + '\n')

        #if not interpretive_flag:
        #    if ibtsign == 1:
        #        read_data.insert(index_btin, '  IBTSIGN  =  1        ,'  + '\n')
        #    elif ibtsign == -1 :
        #        read_data.insert(index_btin, '  IBTSIGN  =  -1       ,'  + '\n')
         
        if ibtsign == 1:
            read_data.insert(index_btin, '  IBTSIGN  =  1        ,'  + '\n')
        elif ibtsign == -1 :
            read_data.insert(index_btin, '  IBTSIGN  =  -1       ,'  + '\n')


        if self.line_ave_density is not None:
            # -------------- First delete lines if they are already there -----------------
            lines_to_kill = []
            for line in read_data:
                if line.startswith('  DNEFLFB'):
                    lines_to_kill.append(line)

            for line_to_kill in lines_to_kill:
                read_data.remove(line_to_kill)

            # -------------- Feedback density density -----------------
            dneflfb_lines = []
            for index_dens, dens_value in enumerate(self.line_ave_density):
                dneflfb_line = '  DNEFLFB' + '(' + str(index_dens+1) + ')' + ' =  ' + str(dens_value*1e-6) + '    , \n'
                dneflfb_lines.append(dneflfb_line)

            for index, dneflfb_line in enumerate(dneflfb_lines):
                read_data.insert(index_nlist4+10+index, dneflfb_line)
    
            # -------------- Feedback density time --------------------
            # -------------- First delete lines if they are already there -----------------
            lines_to_kill = []
            for line in read_data:
                if line.startswith('  DTNEFLFB'):
                    lines_to_kill.append(line)

            for line_to_kill in lines_to_kill:
                read_data.remove(line_to_kill)

            dtneflfb_lines = []
            for index_time, time_value in enumerate(self.dens_feedback_time):
                dtneflfb_line = '  DTNEFLFB' + '(' + str(index_time+1) + ')' + ' =  ' + str(time_value) + '    , \n'
                dtneflfb_lines.append(dtneflfb_line)

            for index, dtneflfb_line in enumerate(dtneflfb_lines):
                read_data.insert(index_nlist4+12+len(dneflfb_lines)+index, dtneflfb_line)

        # Need to extract the previous TPRINT and adapt the array to the new start time-end time
        if not num_times_print:
            num_times_print = original_num_tprint

        tprint_start = 0
        for index, line in enumerate(read_data):
            if line[:8] == '  TPRINT':
                tprint_start = index
                tprint = ''
                for index_print in range(num_times_print):
                    new_time = time_start + (time_end - time_start)/num_times_print*index_print
                    tprint = tprint + str(new_time) + ' , '
                read_data[index] = read_data[index][:14] + tprint + '\n'
            if line[:6] == '      ' and tprint_start != 0 and index > tprint_start and index < tprint_start +10:
                read_data[index] = '             ' + '\n'


        # Could simplify   
        # ALFP, ALFINW might need to be modified as well

        jetto_in_multiline_nameslist = {
            '  ALFI': imp_density,
            '  ATMI': imp_mass,
            '  NZEQ': imp_super,
            '  ZIPI': imp_charge,
        }

        def change_jetto_in_multiline(item):
            print_start = 0
            for index, line in enumerate(read_data):
                if line.startswith(item[0]):
                    print_start = index
                    read_data[index] = read_data[index][:14] + item[1]  + '\n'
                if line[:6] == '      ' and print_start != 0 and index == print_start + 1:
                    del read_data[index]
        
        # for item in jetto_in_multiline_nameslist:
        # change_jetto_in_multiline(item)


        print_start = 0
        for index, line in enumerate(read_data):
            if line[:6] == '  ALFI':
                print_start = index
                read_data[index] = read_data[index][:14] + imp_density  + '\n'
            if line[:6] == '      ' and print_start != 0 and index == print_start + 1:
                del read_data[index]
    
        print_start = 0
        for index, line in enumerate(read_data):
            if line[:6] == '  ATMI':
                print_start = index
                read_data[index] = read_data[index][:14] + imp_mass  + '\n'
            if line[:6] == '      ' and print_start != 0 and index == print_start + 1:
                del read_data[index]
    
        print_start = 0
        for index, line in enumerate(read_data):
            if line[:6] == '  NZEQ':
                print_start = index
                read_data[index] = read_data[index][:14] + imp_super  + '\n'
            if line[:6] == '      ' and print_start != 0 and index == print_start + 1:
                read_data[index] = '             ' + '\n'
    
        print_start = 0
        for index, line in enumerate(read_data):
            if line[:6] == '  ZIPI':
                print_start = index
                read_data[index] = read_data[index][:14] + imp_charge  + '\n'
            if line[:6] == '      ' and print_start != 0 and index == print_start + 1:
                read_data[index] = '             ' + '\n'
    
        with open(sensitivity_name + '/' + 'jetto.in', 'w') as f:
            for line in read_data:
                f.writelines(line)


    def setup_nbi(self, path_nbi_config = '/afs/eufus.eu/user/g/g2mmarin/public/tcv_inputs/jetto.nbicfg'):

        self.modify_jetto_nbi_config(path_nbi_config = path_nbi_config)
        self.modify_jset_nbi(self.path + self.baserun_name, path_nbi_config + self.baserun_name)
        self.modify_ascot_cntl(self.path + self.baserun_name)
        shutil.copyfile(path_nbi_config + self.baserun_name, self.path + self.baserun_name + '/jetto.nbicfg')


    def setup_time_polygon(self):

        modify_jset_time_polygon(self.path + self.baserun_name, self.time_start, self.time_end)
        modify_jettoin_time_polygon(self.path + self.baserun_name + '/jetto.in', self.time_start, self.time_end)


    def modify_jetto_nbi_config(self, path_nbi_config):

        nbi_ids = open_and_get_nbi(self.db, self.shot, self.run_start)


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
        num_times = 2500
        line_ntimes = 'Ntimes' + ' '*12 + str(num_times) + '\n'
        lines.append(line_ntimes)

        # Set new time
        time_min = max(0, min(nbi_ids.time))
        time_max = max(nbi_ids.time)
        new_times = np.linspace(time_min, time_max, num=num_times)

        # Set powers
        powers = np.asarray([])
        for unit in nbi_ids.unit:
            power = fit_and_substitute(nbi_ids.time, new_times, unit.power_launched.data)
            powers = np.hstack((powers, power))

        powers = powers.reshape(3, num_times)
        powers = powers.T

        lines_power = []
        for time, power in zip(new_times, powers):
            add_line_power(lines_power, time, power)
        
        lines.append(lines_power)

        with open(path_nbi_config + self.baserun_name, 'w') as f:
            for line in lines:
                f.writelines(line)


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


def modify_jset_time_polygon(run_name, time_start, time_end):

    times = [time_start, time_start+0.001, time_start+0.01, time_start+0.02, time_end]
    values = [2.0e-6, 4.0e-6, 2.0e-4, 1.0e-3, 1.0e-3]

    line_start_list = ['SetUpPanel.maxTimeStep.option']
    for itime in range(len(times)):
        line_start_list.append('SetUpPanel.maxTimeStep.tpoly.select[' + str(itime) + ']')
    for itime in range(len(times)):
        line_start_list.append('SetUpPanel.maxTimeStep.tpoly.time[' + str(itime) + ']')
    for itime in range(len(times)):
        line_start_list.append('SetUpPanel.maxTimeStep.tpoly.value[' + str(itime) + ']')

    new_content_list = ['Time Dependent']
    for itime in range(len(times)):
        new_content_list.append('true')
    for time in times:
        new_content_list.append(str(time))
    for value in values:
        new_content_list.append(str(value))

    for line_start, new_content in zip(line_start_list, new_content_list):
        modify_jset_line(run_name, line_start, new_content)


def modify_jettoin_time_polygon(path_jetto_in, time_start, time_end):

    fields_single = ['  DTMAX', '  NDTMAX']
    fields_array = ['  PDTMAX', '  TDTMAX']

    max_step_mults = [1.0, 2.0, 100.0, 500.0,  500.0]
    times = [time_start, time_start+0.001, time_start+0.01, time_start+0.02, time_end]

    numbers_single = [2.0e-6, 5]
    numbers_array = [max_step_mults, times]

    modify_jettoin_time_polygon_single(path_jetto_in, fields_single, numbers_single)
    modify_jettoin_time_polygon_array(path_jetto_in, fields_array, numbers_array)

def modify_jettoin_time_polygon_single(path_jetto_in, fields, numbers):

    jetto_in_nameslist, lines = {}, []

    for field, number in zip(fields, numbers):
        jetto_in_nameslist[field] = str(number)

    with open(path_jetto_in) as f:
        lines_file = f.readlines()
        for line in lines_file:
            lines.append(line)

    for index, line in enumerate(lines):
        for jetto_name in jetto_in_nameslist:
            if line.startswith(jetto_name):
                lines[index] = lines[index][:14] + jetto_in_nameslist[jetto_name] + '    ,'  + '\n'

    with open(path_jetto_in, 'w') as f:
        for line in lines:
            f.writelines(line)

def modify_jettoin_time_polygon_array(path_jetto_in, fields, numbers):

    jetto_in_nameslist, lines = {}, []

    for field, number in zip(fields, numbers):
        jetto_in_nameslist[field] = number

    str_array = {}
    for field in jetto_in_nameslist:
        str_array[field] = ''
        for number in jetto_in_nameslist[field]:
            num_spaces = 9 - len(str(number))
            str_array[field] += '  ' + str(number) + ' '*num_spaces + ','
        str_array[field] += '\n'
        str_array[field] = str_array[field][2:]

    with open(path_jetto_in) as f:
        lines_file = f.readlines()
        for line in lines_file:
            lines.append(line)

    for index, line in enumerate(lines):
        for jetto_name in jetto_in_nameslist:
            if line.startswith(jetto_name):
                lines[index] = lines[index][:14] + str_array[jetto_name]

    with open(path_jetto_in, 'w') as f:
        for line in lines:
            f.writelines(line)


def get_extraname_fields(path):

    '''

    Gets all the extranamelist fields in the jset file. This is necessary to put it there when it is not already there, ready to be modified by the jetto tools.

    '''

    read_lines = []

    with open(path + '/' + 'jetto.jset') as f:
        lines = f.readlines()
        for line in lines:
            read_lines.append(line)

    extranamelist_lines = []
    for line in read_lines:
        if line.startswith('OutputExtraNamelist.selItems.cell'):
            extranamelist_lines.append(line)

    indexs1, indexs2, values = [], [], []
    for line in extranamelist_lines:
        indexs1.append(int(line.split('[')[1].split(']')[0]))
        indexs2.append(int(line.split('[')[2].split(']')[0]))
        values.append(line[62:-1])

    # The following assumes that the elements in the extranamelist are always in order 0-1-2-3

    extranamelist = {}
    array_flag = False
    for index1, index2, value in zip(indexs1, indexs2, values):
        if index2 == 0:
            key = value
            extranamelist[key] = []

        if index2 == 1:
            if value != '':
                array_flag = True
            else:
                array_flag = False

        if index2 == 2:
            if array_flag:
                extranamelist[key].append(value)
            else:
                extranamelist[key] = [value]

        # If the item in the extranamelist is not active just kill it
        if index2 == 3:
            if value == 'false':
                extranamelist.pop(key)


    # Could code something that kills the namelist entry if the third entry does not exists, for legacy purposes

    return extranamelist

def add_extraname_fields(extranamelist, key, values):

    # Values needs to be an array of strings

    extranamelist[key] = values
    extranamelist_ordered = {key: value for key, value in sorted(extranamelist.items())}

    return extranamelist_ordered

def put_extraname_fields(path, extranamelist):

    index_start = 0
    namelist_start = 'OutputExtraNamelist.selItems.cell'

    read_lines = []
    with open(path + '/' + 'jetto.jset') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if line.startswith('NeutralSourcePanel'):
                index_start = index
            if not line.startswith(namelist_start):
                read_lines.append(line)

    index_start += 1

    # Could keep this as a legacy option
    '''
    extranamelist_lines = []
    ilist = 0
    for index_item, item in enumerate(extranamelist.items()):
        for ielement, element in enumerate(item[1]):
            new_line1 = namelist_start + '[' + str(ilist) + ']' + '[' + str(0) + ']'
            new_line2 = namelist_start + '[' + str(ilist) + ']' + '[' + str(1) + ']'
            new_line3 = namelist_start + '[' + str(ilist) + ']' + '[' + str(2) + ']'
            new_line4 = namelist_start + '[' + str(ilist) + ']' + '[' + str(3) + ']'

            spaces = ' '*(60 - len(new_line1))

            new_line1 = new_line1 + spaces + ': ' + item[0] + '\n'
            if len(item[1]) == 1:
                new_line2 = new_line2 + spaces + ': \n'
            else:
                new_line2 = new_line2 + spaces + ': ' + str(ielement+1) + '\n'
            new_line3 = new_line3 + spaces + ': ' + element + '\n'

            new_line4 = new_line4 + spaces + ': true \n'

            extranamelist_lines.append(new_line1)
            extranamelist_lines.append(new_line2)
            extranamelist_lines.append(new_line3)
            extranamelist_lines.append(new_line4)

            ilist += 1

    '''
    extranamelist_lines = []
    ilist = 0
    for index_item, item in enumerate(extranamelist.items()):
        new_line1 = namelist_start + '[' + str(ilist) + ']' + '[' + str(0) + ']'
        new_line2 = namelist_start + '[' + str(ilist) + ']' + '[' + str(1) + ']'
        new_line3 = namelist_start + '[' + str(ilist) + ']' + '[' + str(2) + ']'
        new_line4 = namelist_start + '[' + str(ilist) + ']' + '[' + str(3) + ']'

        spaces = ' '*(60 - len(new_line1))

        new_line1 = new_line1 + spaces + ': ' + item[0] + '\n'
        new_line2 = new_line2 + spaces + ': \n'
        if len (item[1]) == 1:
            new_line3 = new_line3 + spaces + ': ' + item[1][0] + '\n'
        else:
            new_line3 = new_line3 + spaces + ': ('
            for element in item[1]:
                new_line3 = new_line3 + str(element) + ' ,'
            new_line3 = new_line3[:-2] + ') \n'

        new_line4 = new_line4 + spaces + ': true \n'

        extranamelist_lines.append(new_line1)
        extranamelist_lines.append(new_line2)
        extranamelist_lines.append(new_line3)
        extranamelist_lines.append(new_line4)

        ilist += 1

    read_lines.insert(index_start, extranamelist_lines)

    with open(path + '/' + 'jetto.jset', 'w') as f:
        for line in read_lines:
            f.writelines(line)

    modify_jset_line(path, 'OutputExtraNamelist.selItems.rows', str(ilist))
    

#  This function is just for testing, can be deleted later
def get_put_namelist(path):

    # the extranamelist might be updated even without the extra.. = something

    extranamelist = get_extraname_fields(path)
    extranamelist = add_extraname_fields(extranamelist, 'IBTSIGN', ['1'])
    extranamelist = add_extraname_fields(extranamelist, 'DNEFLFB', ['1e13', '2e13'])
    extranamelist = add_extraname_fields(extranamelist, 'DTNEFLFB', ['1', '2'])
    put_extraname_fields(path, extranamelist)


def modify_jset_line(run_name, line_start, new_content):

    '''

    Modifies a line of the jset file. Maybe it would be better to change all the lines at once but future work, not really speed limited now

    '''
    read_data = []

    len_line_start = len(line_start)
    with open(run_name + '/' + 'jetto.jset') as f:
        lines = f.readlines()
        for line in lines:
            read_data.append(line)

        for index, line in enumerate(read_data):
            if line[:len_line_start] == line_start:
                read_data[index] = read_data[index][:62] + new_content + '\n'

    with open(run_name + '/' + 'jetto.jset', 'w') as f:
        for line in read_data:
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


def modify_llcmd(run_name, baserun_name, generator_username):

    '''

    modifies the jset file to accomodate a new run name

    '''

    read_data = []
    username = username = getpass.getuser()

    with open(run_name + '/' + '.llcmd') as f:
        lines = f.readlines()
        for line in lines:
            read_data.append(line)

        for index, line in enumerate(read_data):
            read_data[index] = line.replace(baserun_name, run_name)
            if generator_username:
#                print('changing' + generator_username + 'in' + username)
                read_data[index] = read_data[index].replace(generator_username, username)


    with open(run_name + '/' + '.llcmd', 'w') as f:
        for line in read_data:
            f.writelines(line)

def add_item_lookup(name, name_jset, namelist, name_type, name_dim, path):

    read_data = []

    with open(path + '/' + 'lookup.json') as f:
        lines = f.readlines()
        for line in lines:
            read_data.append(line)

    new_item = []

    new_item.append(' \"' + name + '\": { \n')
    if name_jset == 'null':
        new_item.append('  \"jset_id\": ' + name_jset + ',\n')
    else:
        new_item.append('  \"jset_id\": \"' + name_jset + '\",\n')
    new_item.append('  \"nml_id\": { \n')
    new_item.append('   \"namelist\": \"' + namelist + '\",\n')
    new_item.append('   \"field\":  \"' + name.upper() + '\" \n')
    new_item.append('  }, \n')
    new_item.append('  \"type\": \"' + name_type + '\",\n')
    new_item.append('  \"dimension\": \"' + name_dim + '\" \n')
    new_item.append(' }, \n')

    read_data.insert(1, new_item)

    with open(path + '/' + 'lookup.json', 'w') as f:
        for line in read_data:
            f.writelines(line)

# Redefined for when prepare_im_input is not imported
def open_and_get_core_profiles(db, shot, run, username = ''):

    if username == '':
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=username)

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

def open_and_get_equilibrium(db, shot, run, username = ''):

    if username == '':
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=username)

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

def open_and_get_pulse_schedule(db, shot, run, username = ''):

    if username == '':
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=username)

    op = data_entry.open()

    if op[0]<0:
        cp=data_entry.create()
        print(cp[0])
        if cp[0]==0:
            print("data entry created")
    elif op[0]==0:
        print("data entry opened")

    pulse_schedule = data_entry.get('pulse_schedule')
    data_entry.close()

    return(pulse_schedule)

def open_and_get_core_sources(db, shot, run, username = ''):

    if username == '':
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=username)

    op = data_entry.open()

    if op[0]<0:
        cp=data_entry.create()
        print(cp[0])
        if cp[0]==0:
            print("data entry created")
    elif op[0]==0:
        print("data entry opened")

    core_sources = data_entry.get('core_profiles')
    data_entry.close()

    return(core_sources)

def open_and_get_nbi(db, shot, run, username = ''):

    if username == '':
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=getpass.getuser())
    else:
        data_entry = imas.DBEntry(imasdef.MDSPLUS_BACKEND, db, shot, run, user_name=username)

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

def fit_and_substitute(x_old, x_new, data_old):

    f_space = interp1d(x_old, data_old, fill_value = 'extrapolate')

    variable = np.array(f_space(x_new))
    variable[variable > 1.0e25] = 0

    return variable


if __name__ == "__main__":
    print('main')

    print('main, not supported, use commands')

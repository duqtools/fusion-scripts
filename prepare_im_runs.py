"""
Refactored Jetto Integrated Modelling Run Management

This module provides the IntegratedModellingRuns class for orchestrating
Jetto integrated modelling simulations, including setup, creation, and execution
of runs.

Key improvements over prepare_im_runs_old.py:
- Type hints for better code clarity
- Comprehensive docstrings with parameter and return value documentation
- Better organized __init__ with logical grouping of configuration
- Improved method organization and naming conventions
- Better error handling and validation
"""

import json
import os
import datetime
import sys
import shutil
import getpass
import numpy as np
import traceback
import pickle
import math
import functools
import re
import tempfile
from scipy import integrate
from scipy.interpolate import interp1d, UnivariateSpline
from packaging import version
from os import path
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

# Force use of site-provided imas (with imasdef) instead of user imaspy
# Do this BEFORE importing prepare_im_input which also tries to import imas
import importlib.util
imas_site_path = '/gw/swimas/software/IMAS-AL-Python/5.4.0-intel-2023b-DD-3.42.0/lib/python3.11/site-packages/imas/__init__.py'
try:
    spec = importlib.util.spec_from_file_location("imas", imas_site_path)
    imas = importlib.util.module_from_spec(spec)
    sys.modules['imas'] = imas
    spec.loader.exec_module(imas)
    from imas import imasdef
except Exception as e:
    print(f"IMAS Python module not found or not configured properly: {e}")
    imas = None
    imasdef = None

from prepare_im_input import MissingDataError

import inspect
import types

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display

import xml.sax
import xml.sax.handler

try:
    import setup_nbi_input
except ImportError:
    print('setup_input_nbi not present, might cause problems when trying to setup the nbi')

try:
    import jetto_tools
except ImportError:
    print("Jetto tools not available. Please check installation in /python_tools/jetto-pythontools")

import copy

# Version constants
MIN_IMAS_VERSION_STR = "3.28.0"
MIN_IMASAL_VERSION_STR = "4.7.2"


class IntegratedModellingRuns:
    """
    Orchestrates integrated modelling Jetto simulations.
    
    This class manages the complete workflow for setting up, creating, and running
    Jetto integrated modelling simulations, including baserun and sensitivity configurations.
    
    Parameters
    ----------
    shot : int
        Shot number for the simulation
    instructions_list : list
        List of instructions to execute (options: 'setup case', 
        'create case', 'run case')
    generator_name : str
        Name of the generator/template run to use as basis
    baserun_name : str
        Name for the baserun directory
    db : str, optional
        IMAS database name (default: 'tcv')
    run_input : int, optional
        Run number containing input data (default: 1)
    run_start : int, optional
        Run number to use as starting point for configuration (default: None)
    run_output : int, optional
        Run number for baserun output (default: 100)
    time_start : float or str, optional
        Starting time for simulation or 'core_profiles'/'equilibrium' (default: None)
    time_end : float or str, optional
        Ending time for simulation (default: 100)
    esco_timesteps : int, optional
        Number of equilibrium time steps (default: None)
    output_timesteps : int, optional
        Number of output time steps (default: None)
    force_run : bool, optional
        Force run even if output IDS exists (default: False)
    force_input_overwrite : bool, optional
        Force overwrite of input IDS (default: False)
    density_feedback : bool, optional
        Enable density feedback control (default: False)
    set_sep_boundaries : bool, optional
        Set separatrix boundary conditions (default: False)
    boundary_conditions : dict, optional
        Custom boundary conditions (default: {})
    setup_time_polygon_flag : bool, optional
        Setup time-dependent polygon (default: False)
    change_impurity_puff_flag : bool or float, optional
        Modify impurity puff (default: False)
    setup_time_polygon_impurities_flag : bool, optional
        Setup time polygon for impurities (default: False)
    select_impurities_from_ids_flag : bool, optional
        Auto-select impurities from IDS (default: True)
    add_extra_transport_flag : bool or float, optional
        Add extra transport model (default: False)
    setup_nbi_flag : bool, optional
        Setup NBI (default: False)
    path_nbi_config : str, optional
        Path to NBI configuration file (default: None)
    json_input : dict, optional
        JSON configuration for input setup (default: None)
    sensitivity_list : list, optional
        List of sensitivity parameters (default: [])
    """
    
    def __init__(
        self,
        shot: int,
        instructions_list: List[str],
        generator_name: str,
        baserun_name: str,
        db: str = 'tcv',
        run_input: int = 1,
        run_start: Optional[int] = None,
        run_output: int = 100,
        time_start: Optional[float] = None,
        time_end: float = 100,
        esco_timesteps: Optional[int] = None,
        output_timesteps: Optional[int] = None,
        force_run: bool = False,
        force_input_overwrite: bool = False,
        overwrite_baserun: bool = False,
        density_feedback: bool = False,
        set_sep_boundaries: bool = False,
        boundary_conditions: Optional[Dict] = None,
        setup_time_polygon_flag: bool = False,
        change_impurity_puff_flag: Any = False,
        setup_time_polygon_impurities_flag: bool = False,
        select_impurities_from_ids_flag: bool = True,
        add_extra_transport_flag: Any = False,
        setup_nbi_flag: bool = False,
        path_nbi_config: Optional[str] = None,
        json_input: Optional[Dict] = None,
    ):
        """Initialize IntegratedModellingRuns with all configuration parameters.

        Parameters (additional)
        -----------------------
        overwrite_baserun : bool, optional
            If True, delete an existing baserun directory during create. Default False.
        """
        
        # User and database configuration
        self.username = getpass.getuser()
        self.db = db
        
        # Shot and run configuration
        if shot <= 0:
            raise ValueError("Shot number must be positive")
        if run_input <= 0:
            raise ValueError("run_input must be positive")

        self.shot = shot
        self.run_input = run_input
        self.run_start = run_start
        self.run_output = run_output
        
        # Time configuration
        self.time_start = time_start
        self.time_end = time_end
        
        # Simulation timestep configuration
        self.esco_timesteps = esco_timesteps
        self.output_timesteps = output_timesteps

        for name, value in (
            ("esco_timesteps", self.esco_timesteps),
            ("output_timesteps", self.output_timesteps),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        
        # Execution flags
        self.force_run = force_run
        self.force_input_overwrite = force_input_overwrite
        self.overwrite_baserun = overwrite_baserun
        
        # Physics features configuration
        self.density_feedback = density_feedback
        self.set_sep_boundaries = set_sep_boundaries
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else {}
        
        # Time-dependent features
        self.setup_time_polygon_flag = setup_time_polygon_flag
        self.change_impurity_puff_flag = change_impurity_puff_flag
        self.setup_time_polygon_impurities_flag = setup_time_polygon_impurities_flag
        
        # Impurity and transport configuration
        self.select_impurities_from_ids_flag = select_impurities_from_ids_flag
        self.add_extra_transport_flag = add_extra_transport_flag
        
        # NBI configuration
        self.setup_nbi_flag = setup_nbi_flag
        self.path_nbi_config = path_nbi_config
        
        # Cached IDS data
        self.core_profiles = None
        self.equilibrium = None
        self.line_ave_density = None
        self.dens_feedback_time = None
        
        # Input configuration
        self.json_input = json_input

        # Determine IMAS backend
        self.backend_input = self._get_backend(self.db, self.shot, self.run_input)

        # Update NBI flag from JSON if provided
        if self.json_input:
            instructions_cfg = self.json_input.get('instructions', {}) if isinstance(self.json_input, dict) else {}
            self.setup_nbi_flag = bool(instructions_cfg.get('nbi heating', self.setup_nbi_flag))
        
        # Configure paths
        self._setup_paths(generator_name)
        
        # Configure baserun name
        self._setup_baserun_name(baserun_name)
        
        # Setup instructions dictionary
        self._setup_instructions(instructions_list)

    def _setup_paths(self, generator_name: str) -> None:
        """
        Setup all path-related attributes.
        
        Parameters
        ----------
        generator_name : str
            Name or path to the generator/template run
        """
        self.path = '/pfs/work/' + self.username + '/jetto/runs/'
        self.generator_username = ''
        
        if generator_name.startswith('/pfs/work'):
            # Full path provided
            self.path_generator = generator_name
            self.generator_name = generator_name.split('/')[-2]
            self.generator_username = generator_name.split('/')[3]
        elif generator_name.startswith('rungenerator_'):
            # Already has rungenerator_ prefix
            self.generator_name = generator_name
            self.path_generator = self.path + self.generator_name
            self.generator_username = self.username
        else:
            # Add rungenerator_ prefix
            self.generator_name = 'rungenerator_' + generator_name
            self.path_generator = self.path + self.generator_name
            self.generator_username = self.username

    def _setup_baserun_name(self, baserun_name: str) -> None:
        """
        Setup baserun name and path.
        
        Parameters
        ----------
        baserun_name : str
            Name for the baserun directory
        """
        if baserun_name == '':
            self.baserun_name = 'run000' + str(self.shot) + 'base'
        else:
            self.baserun_name = baserun_name
        
        self.path_baserun = self.path + self.baserun_name

    def _setup_instructions(self, instructions_list: List[str]) -> None:
        """
        Setup the instructions dictionary from the instructions list.
        
        Parameters
        ----------
        instructions_list : list
            List of instruction strings to enable
        """
        self.instructions = {
            'setup case': False,
            'create case': False,
            'run case': False
        }
        
        for key in instructions_list:
            if key in self.instructions:
                self.instructions[key] = True


    def update_instructions(self, new_instructions: List[bool]) -> None:
        """
        Update the instructions dictionary.
        
        Parameters
        ----------
        new_instructions : list
            List of boolean values (in order: setup case, create case, run case)
        """
        for i, key in enumerate(self.instructions):
            self.instructions[key] = new_instructions[i]


    def setup_create_compare(self, verbose: bool = False) -> None:
        """
        Main orchestrator method that executes all configured operations.
        
        Runs setup, creation, and execution steps according to the instructions
        dictionary. Steps are executed in order: setup case, create case, run case.
        
        Parameters
        ----------
        verbose : bool, optional
            Enable verbose output (default: False)
        """
        if self.instructions['setup case']:
            self.setup_input_baserun(verbose=verbose)
        if self.instructions['create case']:
            self.create_baserun()
        if self.instructions['run case']:
            self.run_baserun()

    def setup_input_baserun(self, verbose: bool = False) -> None:
        """
        Setup input equilibrium and core profiles for baserun.
        
        Uses the prepare_im_input module to interpolate and configure the
        equilibrium and core profiles from the input IDS according to the
        provided JSON configuration.
        
        Parameters
        ----------
        verbose : bool, optional
            Enable verbose output (default: False)
            
        Raises
        ------
        ImportError
            If prepare_im_input module is not found
        """
        try:
            import prepare_im_input
        except ImportError:
            print('ERROR: prepare_im_input.py not found and needed for this option.')
            exit()
        
        # Load default JSON configuration if not provided
        if not self.json_input:
            json_file_name = '/afs/eufus.eu/user/g/g2mmarin/public/scripts/template_prepare_input.json'
            print('JSON input not specified, using template file:', json_file_name)
            try:
                with open(json_file_name) as f:
                    self.json_input = json.load(f)
            except FileNotFoundError:
                print(f'ERROR: Template JSON file not found at {json_file_name}')
                exit()
        
        # Run the input setup
        try:
            self.core_profiles, self.equilibrium = prepare_im_input.setup_input(
                self.db,
                self.shot,
                self.run_input,
                self.run_start,
                json_input=self.json_input,
                time_start=self.time_start,
                time_end=self.time_end,
                force_input_overwrite=self.force_input_overwrite,
                core_profiles=self.core_profiles,
                equilibrium=self.equilibrium
            )
            if verbose:
                print('Input generated successfully')
                print(f'  Core profiles: {len(self.core_profiles.profiles_1d)} time slices')
                print(f'  Equilibrium: {len(self.equilibrium.time)} time slices')
        except Exception as e:
            print(f'ERROR: Failed to setup input: {e}')
            raise

    def _get_backend(self, db: str, shot: int, run: int, username: Optional[str] = None) -> int:
        """
        Determine the IMAS backend for the given database entry.
        
        Parameters
        ----------
        db : str
            Database name
        shot : int
            Shot number
        run : int
            Run number
        username : str, optional
            Username (default: current user)
            
        Returns
        -------
        int
            IMAS backend constant (HDF5_BACKEND or MDSPLUS_BACKEND)
        """
        if imas is None:
            raise ImportError("IMAS module is not available")
        
        if not username:
            username = getpass.getuser()
        
        imas_backend = imasdef.HDF5_BACKEND
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)
        
        op = data_entry.open()
        if op is None or (isinstance(op, (tuple, list)) and len(op) > 0 and op[0] < 0):
            imas_backend = imasdef.MDSPLUS_BACKEND
        
        data_entry.close()
        
        data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)
        op = data_entry.open()
        if op is None or (isinstance(op, (tuple, list)) and len(op) > 0 and op[0] < 0):
            print(f'ERROR: Input IDS {db}/{shot}/{run} does not exist. Aborting.')
            exit()
        
        data_entry.close()
        
        return imas_backend

    def create_baserun(self) -> None:
        """
        Create and configure the baserun folder structure.
        
        Sets up the baserun directory by copying from the generator template, determining
        time boundaries from equilibrium and core profiles, loading impurity data,
        and configuring all Jetto input files (jetto.in, jetto.jset).
        
        The method orchestrates many sub-tasks in sequence:
        1. Copy generator template to baserun path
        2. Load equilibrium and core profiles IDS if not cached
        3. Determine time boundaries (start/end times)
        4. Extract magnetic field and major radius
        5. Setup impurities if enabled
        6. Configure jetto input files
        7. Setup Jetto via jetto_tools
        8. Setup optional features (density feedback, boundaries, NBI, etc.)
        9. Copy input IDS files to baserun directory
        """
        os.chdir(self.path)
        
        # Copy generator template
        if not os.path.exists(self.path_generator):
            print(f'ERROR: Generator not found at {self.path_generator}')
            exit()
        # Handle existing baserun directory
        if os.path.exists(self.path_baserun):
            if self.overwrite_baserun:
                shutil.rmtree(self.path_baserun)
            else:
                raise FileExistsError(
                    f'Baserun already exists: {self.path_baserun}. '
                    f'Set overwrite_baserun=True to replace it.'
                )
        shutil.copytree(self.path_generator, self.path_baserun)
        
        # Load IDS data if not cached
        self._ensure_ids_loaded()
        
        # Determine time boundaries
        self._setup_time_boundaries()
        
        # Extract equilibrium parameters
        b0, r0 = self.get_r0_b0()
        
        # Load density feedback if enabled
        if self.density_feedback:
            self.get_feedback_on_density_quantities()

        # Pre-load boundary values if requested so jetto_tools namelist can set them
        if self.set_sep_boundaries:
            self.get_boundary_values_quantities()
        
        # Determine if interpretive run
        interpretive_flag = 'interpretive' in self.path_generator
        
        # Setup impurities
        if self.select_impurities_from_ids_flag:
            imp_data_ids = self.select_impurities_from_ids()
            self.modify_jetto_in_impurities(imp_data_ids)
            self.modify_impurities_jset(imp_data_ids)
        
        # Configure jetto.in file
        # Note: Using jetto_tools approach now - handled in setup_jetto_simulation
        
        # Setup jetto via jetto_tools
        self.setup_jetto_simulation()
        
        # Setup optional features
        if self.density_feedback:
            self.setup_feedback_on_density()
            self.setup_feedback_on_density_jset()
        if self.set_sep_boundaries:
            self.setup_boundary_values()
        
        # Configure jset file
        self.modify_jset(self.path, self.baserun_name, self.run_start,
                        self.run_output, abs(b0), r0)
        
        # Setup NBI if enabled
        if self.setup_nbi_flag:
            self._setup_nbi_if_available()
        
        # Setup additional optional features
        if self.setup_time_polygon_flag:
            self.setup_time_polygon()
        if self.change_impurity_puff_flag:
            self.change_impurity_puff()
        if self.add_extra_transport_flag:
            self.add_extra_transport()
        if self.setup_time_polygon_impurities_flag:
            self.setup_time_polygon_impurity_puff()
        
        # Finalize configuration files
        modify_llcmd(self.baserun_name, self.generator_name, self.generator_username)
        
        # Modify hfps.launch with updated parameters
        modify_hfps_launch(
            self.path_baserun,
            self.baserun_name,
            self.shot,
            self.db,
            self.time_start,
            self.time_end,
            self.generator_name
        )
        
        # Setup jintrac launch if available
        if os.path.exists(self.generator_name + '/jintrac.launch'):
            modify_jintrac_launch(self.baserun_name, self.generator_name,
                                self.generator_username, self.db, self.shot,
                                self.time_start, self.time_end)
        
        # Copy input IDS files
        if self.backend_input == imasdef.MDSPLUS_BACKEND:
            self.copy_ids_input_mdsplus()
        elif self.backend_input == imasdef.HDF5_BACKEND:
            self.copy_ids_input_hdf5()

    def _ensure_ids_loaded(self) -> None:
        """Load equilibrium and core_profiles IDS if not already cached."""
        if not self.core_profiles:
            self.core_profiles = open_and_get_ids(
                self.db, self.shot, self.run_input, 'core_profiles'
            )
        if not self.equilibrium:
            self.equilibrium = open_and_get_ids(
                self.db, self.shot, self.run_input, 'equilibrium'
            )

    def _setup_time_boundaries(self) -> None:
        """
        Setup time start and end boundaries from IDS data.
        
        Handles special cases: None (use common range), 'core_profiles' (use min),
        'equilibrium' (use min), 'equilibrium + 1' (use second time point),
        'auto' (calculate from current and field).
        """
        time_eq = self.equilibrium.time
        time_cp = self.core_profiles.time
        
        # Determine time_start
        if self.time_start is None:
            self.time_start = max(min(time_eq), min(time_cp))
        elif self.time_start == 'core_profiles':
            self.time_start = min(time_cp)
        elif self.time_start == 'equilibrium':
            self.time_start = min(time_eq)
        elif self.time_start == 'equilibrium + 1':
            self.time_start = time_eq[1]
        
        # Determine time_end
        if self.time_end == 100:  # Default value
            self.time_end = min(max(time_eq), max(time_cp))
        elif self.time_end == 'auto':
            summary = open_and_get_ids(self.db, self.shot, self.run_input, 'summary')
            kfactor = 0.05
            mu0 = 4 * np.pi * 1.0e-7
            time_sim = kfactor * mu0 * np.abs(
                summary.global_quantities.ip.value[0] *
                summary.global_quantities.r0.value
            )
            self.time_end = self.time_start + time_sim

    def _setup_nbi_if_available(self) -> None:
        """Setup NBI if available in the input IDS."""
        nbi = open_and_get_ids(self.db, self.shot, self.run_start, 'nbi')
        if nbi.time.size != 0:
            setup_nbi_input.setup_nbi(
                self.db, self.shot, self.run_start,
                self.path + self.baserun_name,
                run_target=self.run_start,
                path_nbi_config=self.path_nbi_config
            )
        else:
            print('WARNING: NBI setup requested but NBI IDS is empty. Skipping.')

    def select_impurities_from_ids(self) -> List[List[Any]]:
        """
        Extract impurity data from core profiles IDS.
        
        Reads ion species from the first core profile time slice and extracts
        relative density, mass number, charge state, and nuclear charge for each.
        Excludes hydrogen and deuterium (z_ion <= 1).
        
        Returns
        -------
        list
            List of impurity data [relative_density, mass_number, charge_state, nuclear_charge]
        """
        imp_data = []
        first_imp_density = None
        
        try:
            for ion in self.core_profiles.profiles_1d[0].ion:
                imp_density = np.average(ion.density)
                z_ion = ion.element[0].z_n
                a_ion = ion.element[0].a
                z_bundle = round(z_ion)
                
                # Only include impurities (z > 1)
                if z_ion > 1:
                    if first_imp_density is None:
                        imp_relative_density = 1.0
                        first_imp_density = imp_density
                    else:
                        imp_relative_density = imp_density / first_imp_density
                    
                    imp_data.append([imp_relative_density, a_ion, z_bundle, z_ion])
        except (AttributeError, IndexError) as e:
            print(f'WARNING: Could not extract impurities from IDS: {e}')
        
        return imp_data

    def modify_impurities_jset(self, imp_data: List[List[Any]]) -> None:
        """
        Write impurity configuration to jset file.
        
        Updates jset file with impurity select flags, mass numbers, charge states,
        and super-states for up to 6 impurities.
        
        Parameters
        ----------
        imp_data : list
            List of impurity data from select_impurities_from_ids()
        """
        impurity_jset_fields = [
            'ImpOptionPanel.impuritySelect[]',
            'ImpOptionPanel.impurityMass[]',
            'ImpOptionPanel.impurityCharge[]',
            'ImpOptionPanel.impuritySuperStates[]'
        ]
        
        for index in range(6):
            if index < len(imp_data):
                # Configure this impurity
                for field in impurity_jset_fields:
                    line_start = field[:-2] + str(index) + field[-1]
                    
                    if field == 'ImpOptionPanel.impuritySelect[]':
                        new_content = '1'
                    elif field == 'ImpOptionPanel.impurityMass[]':
                        new_content = str(imp_data[index][1])
                    elif field == 'ImpOptionPanel.impurityCharge[]':
                        new_content = str(imp_data[index][2])
                    else:  # impuritySuperStates
                        new_content = str(imp_data[index][3])
                    
                    modify_jset_line(self.baserun_name, line_start, new_content)
            else:
                # Disable this impurity
                line_start = f'ImpOptionPanel.impuritySelect[{index}]'
                modify_jset_line(self.baserun_name, line_start, 'false')

    def copy_ids_input_mdsplus(self) -> None:
        """
        Copy input IDS files (MDSPlus backend) to baserun directory.
        
        Copies characteristics, datafile, and tree files from the source location
        to the baserun imasdb directory with appropriate formatting for run 0001.
        """
        run_str = self._format_run_string(self.run_start)
        
        path_ids_input = (f'/afs/eufus.eu/user/g/{self.username}/public/imasdb/'
                         f'{self.db}/3/0/ids_{self.shot}{run_str}')
        path_output = (f'{self.path_baserun}/imasdb/{self.db}/3/0/'
                      f'ids_{self.shot}0001')
        
        # Ensure target database directory exists
        self._ensure_db_directory()
        
        # Delete old generator IDS
        self.delete_generator()
        
        # Copy files
        for ext in ['.characteristics', '.datafile', '.tree']:
            shutil.copyfile(path_ids_input + ext, path_output + ext)

    def copy_ids_input_hdf5(self) -> None:
        """
        Copy input IDS files (HDF5 backend) to baserun directory.
        
        Copies the entire directory structure from source to target location
        with appropriate path structure for HDF5 backend format.
        """
        path_ids_input = (f'/afs/eufus.eu/user/g/{self.username}/public/imasdb/'
                         f'{self.db}/3/{self.shot}/{self.run_start}')
        path_output = (f'{self.path_baserun}/imasdb/{self.db}/3/{self.shot}/1')
        
        # Ensure target database directory exists
        self._ensure_db_directory()
        
        # Delete old generator IDS
        self.delete_generator()
        
        # Create output directory if needed
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        
        # Copy files
        copy_files(path_ids_input, path_output)

    def _ensure_db_directory(self) -> None:
        """Ensure database directory exists, migrating from generator if necessary."""
        imasdb_path = f'{self.path_baserun}/imasdb'
        target_db_path = f'{imasdb_path}/{self.db}'
        
        if not os.path.exists(target_db_path):
            # Migrate from generator database
            db_generator = os.listdir(imasdb_path)[0]
            gen_db_path = f'{imasdb_path}/{db_generator}'
            
            shutil.copytree(gen_db_path, target_db_path)
            shutil.rmtree(gen_db_path)

    def _format_run_string(self, run_number: int) -> str:
        """Format run number as zero-padded string (e.g., 0001, 0042, 1234)."""
        if run_number < 10:
            return f'000{run_number}'
        elif run_number < 100:
            return f'00{run_number}'
        elif run_number < 1000:
            return f'0{run_number}'
        else:
            return str(run_number)

    def delete_generator(self) -> None:
        """
        Delete the old generator IDS files from baserun directory.
        
        Removes the original generator database to make room for the new
        input database files.
        """
        imasdb_path = f'{self.path_baserun}/imasdb'
        try:
            self.db_generator = os.listdir(imasdb_path)[0]
            gen_db_path = f'{imasdb_path}/{self.db_generator}'
            
            if os.path.exists(gen_db_path):
                self.shot_generator = os.listdir(f'{gen_db_path}/3/')[0]
                shutil.rmtree(gen_db_path)
        except (IndexError, OSError) as e:
            print(f'WARNING: Could not delete generator IDS: {e}')

    def run_baserun(self) -> None:
        """
        Submit the baserun to the batch queue.
        
        Changes to the baserun directory and submits the .llcmd script
        to the SLURM batch system for execution.
        
        Note: This is a simple implementation. In the future, jetto_tools
        JobManager could be used for more sophisticated job management.
        """
        os.chdir(self.path + self.baserun_name)
        print(f'Submitting baserun: {self.baserun_name}')
        os.system('sbatch .llcmd')

    def get_r0_b0(self) -> Tuple[float, float]:
        """
        Extract major radius and toroidal magnetic field from equilibrium.
        
        Reads the vacuum toroidal field data from equilibrium IDS and averages
        over the time range from time_start to time_end. If start and end times
        are the same, uses the first value.
        
        Returns
        -------
        tuple
            (b0, r0) - Toroidal magnetic field [T] and major radius [cm]
        """
        self._ensure_ids_loaded()
        
        time_eq = self.equilibrium.time
        
        # Find indices closest to time boundaries
        index_start = np.abs(time_eq - self.time_start).argmin()
        index_end = np.abs(time_eq - self.time_end).argmin()
        
        # Average B0 over the time range
        if index_start != index_end:
            b0 = np.average(
                self.equilibrium.vacuum_toroidal_field.b0[index_start:index_end]
            )
        else:
            b0 = self.equilibrium.vacuum_toroidal_field.b0[0]
        
        # Extract major radius (convert from m to cm)
        r0 = self.equilibrium.vacuum_toroidal_field.r0 * 100
        
        return b0, r0

    def _clean_jetto_in_for_namelist_api(self, jetto_in_path: str) -> None:
        """
        Clean jetto.in file for reading with jetto_tools.namelist API.
        
        Removes decorative section headers (dashed lines and "Namelist :" headers)
        that f90nml cannot parse. This allows jetto_tools to read the file.
        """
        try:
            with open(jetto_in_path, 'r') as f:
                lines = f.readlines()
            
            cleaned_lines = []
            in_namelist = False
            
            for line in lines:
                stripped = line.strip()
                
                # Start of namelist (e.g., &NLIST1)
                if stripped.startswith('&') and not stripped.startswith('&END'):
                    in_namelist = True
                    cleaned_lines.append(line)
                # End of namelist
                elif stripped.startswith('&END'):
                    in_namelist = False
                    cleaned_lines.append(line)
                # Inside a namelist - keep everything
                elif in_namelist:
                    cleaned_lines.append(line)
                # Outside namelists - skip decorative lines
                # (dashed lines, "Namelist :" lines, etc.)
                elif '----' not in line and 'Namelist :' not in line:
                    # Keep lines that don't look like decorative headers
                    if line.strip():  # Non-empty lines
                        cleaned_lines.append(line)
            
            # Write cleaned file back
            with open(jetto_in_path, 'w') as f:
                f.writelines(cleaned_lines)
        except Exception as e:
            print(f"WARNING: Could not clean jetto.in: {e}")

    def setup_jetto_simulation(self) -> None:
        """
        Configure Jetto simulation using jetto_tools.
        
        Uses jetto_tools to setup and export Jetto configuration including:
        - Magnetic field and major radius
        - Time boundaries
        - Output and equilibrium timesteps
        - Impurity composition (if needed)
        
        This method handles temporary array workarounds for jetto_tools
        compatibility and updates the jetto.jset file.
        """
        b0, r0 = self.get_r0_b0()
        self._ensure_ids_loaded()
        
        # Ensure baserun directory exists
        if not os.path.exists(self.path_baserun):
            shutil.copytree(self.path_generator, self.path_baserun)
        else:
            # Copy lookup file
            shutil.copyfile(
                f'{self.path}/lookup_json/lookup.json',
                f'{self.path_baserun}/lookup.json'
            )

        # Ensure lookup.json is present for jetto_tools parameter mapping (e.g. atmi/nzeq/zipi)
        lookup_candidates = [
            f'{self.path}/lookup_json/lookup.json',
            '/pfs/work/g2mmarin/jetto/runs/lookup_json/lookup.json'
        ]
        for lookup_src in lookup_candidates:
            if os.path.exists(lookup_src):
                shutil.copyfile(lookup_src, f'{self.path_baserun}/lookup.json')
                break
        else:
            print('WARNING: lookup.json not found; jetto_tools parameter mapping may fail')

        # IMPORTANT: Extract section headers BEFORE sanitizing the file!
        # This must happen on the generator's original jetto.in / jetto.sin with decorative headers intact
        jetto_in_path = os.path.join(self.path_baserun, 'jetto.in')
        section_headers = self._clean_jetto_namelist_file(jetto_in_path)

        # Capture headers for jetto.sin as well so we can restore them later
        jetto_sin_path = os.path.join(self.path_baserun, "jetto.sin")
        self.section_headers_sin = {}
        if os.path.exists(jetto_sin_path):
            self.section_headers_sin = self._clean_jetto_namelist_file(jetto_sin_path)

        # Handle arrays for jetto_tools compatibility
        self.tmp_handle_arrays_open()
        
        # Ensure lookup.json mappings match jetto_tools expectations
        # - Lowercase namelist/field keys (f90nml stores lowercase)
        # - Ensure 'atmi' maps to ImpOptionPanel.impurityMass[0] at nlist4/atmi
        lookup_path = f'{self.path_baserun}/lookup.json'
        try:
            if os.path.exists(lookup_path):
                with open(lookup_path, 'r') as f:
                    lookup_data = json.load(f)
                # Normalize all nml_id namelist/field to lowercase
                updated = False
                for k, v in list(lookup_data.items()):
                    try:
                        nml = v.get('nml_id', {}).get('namelist')
                        fld = v.get('nml_id', {}).get('field')
                        if isinstance(nml, str) and nml != nml.lower():
                            v['nml_id']['namelist'] = nml.lower()
                            updated = True
                        if isinstance(fld, str) and fld != fld.lower():
                            v['nml_id']['field'] = fld.lower()
                            updated = True
                    except Exception:
                        pass

                # Ensure 'atmi' entry exists and has correct mapping
                atmi_entry = lookup_data.get('atmi', {})
                desired_atmi = {
                    'jset_id': 'ImpOptionPanel.impurityMass[0]',
                    'nml_id': {'namelist': 'nlist4', 'field': 'atmi'},
                    'type': 'real',
                    'dimension': 'scalar'
                }
                if (
                    atmi_entry.get('jset_id') != desired_atmi['jset_id'] or
                    atmi_entry.get('nml_id', {}).get('namelist') != 'nlist4' or
                    atmi_entry.get('nml_id', {}).get('field') != 'atmi'
                ):
                    lookup_data['atmi'] = desired_atmi
                    updated = True

                if updated:
                    with open(lookup_path, 'w') as f:
                        json.dump(lookup_data, f, indent=1)
        except Exception as e:
            print(f'WARNING: Could not adjust lookup.json for atmi: {e}')
        # Filter lookup.json to only include parameters that exist in both jset and namelist
        # This is necessary for jetto_tools to accept parameters like dneflfb/dtneflfb
        try:
            jset_path = os.path.join(self.path_baserun, "jetto.jset")
            
            # Load full lookup.json (if not already loaded)
            if not os.path.exists(lookup_path):
                # Try to copy from a reference location
                lookup_source = "/pfs/work/g2mmarin/jetto/runs/lookup.json"
                if os.path.exists(lookup_source):
                    shutil.copyfile(lookup_source, lookup_path)
            
            if os.path.exists(lookup_path):
                with open(lookup_path, 'r') as f:
                    full_lookup = json.load(f)
                
                # Read jset and namelist to check which parameters exist (after cleaning)
                jset = jetto_tools.jset.read(jset_path)
                jetto_nml = jetto_tools.namelist.read(jetto_in_path)
                
                # Filter lookup to only include parameters that exist
                filtered_lookup = {}
                for param_name, param_config in full_lookup.items():
                    if param_config.get("jset_id") is not None:
                        # Has jset_id - check if in jset.extras
                        try:
                            jset.extras.get_row(param_config["nml_id"]["field"])
                            # Also check if in namelist file
                            if jetto_nml.namelist_lookup(param_config["nml_id"]["field"]) is not None:
                                filtered_lookup[param_name] = param_config
                        except (KeyError, IndexError, AttributeError):
                            pass  # Skip this parameter
                    else:
                        # No jset_id - just check if in namelist
                        if jetto_nml.namelist_lookup(param_config["nml_id"]["field"]) is not None:
                            filtered_lookup[param_name] = param_config
                
                # Save filtered lookup
                with open(lookup_path, 'w') as f:
                    json.dump(filtered_lookup, f, indent=2)
                print(f"✓ Filtered lookup.json: {len(filtered_lookup)}/{len(full_lookup)} parameters")
        except Exception as e:
            print(f'WARNING: Could not filter lookup.json: {e}')

        # Load and configure Jetto
        template = jetto_tools.template.from_directory(self.path_baserun)
        config = jetto_tools.config.RunConfig(template)
        
        # Note: modify_jetto_in_via_jetto_tools is NOT called here because:
        # 1. It overwrites the cleaned jetto.in file, breaking section header restoration
        # 2. We're using config.export() which handles all namelist modifications
        # 3. The fallback _modify_jetto_in_direct writes decorative headers that f90nml can't parse
        # If specific namelist modifications are needed, they should be done via config API
        
        # Note: btin and rmj cannot be set via config API (not in namelist)
        # They will be set via modify_jset() direct file manipulation
        
        # Configure timesteps
        try:
            if self.esco_timesteps:
                config.esco_timesteps = self.esco_timesteps
            if self.output_timesteps:
                # Note: profile_timesteps tries to set TPRINT as array, but generator has it as scalar
                # config.profile_timesteps = self.output_timesteps
                config['ntint'] = self.output_timesteps
        except Exception as e:
            print(f'WARNING: Could not set timesteps via config API: {e}')
        
        # Note: start_time and end_time are handled via modify_jetto_in_via_jetto_tools
        # Note: Density feedback is now handled via Namelist API in modify_jetto_in_via_jetto_tools
        # Setting them here causes jetto_tools to try to create TPRINT array which fails
        # config.start_time = self.time_start
        # config.end_time = self.time_end
        
        # Configure run-specific parameters via config API
        # This replaces the old modify_jset() direct file manipulation
        try:
            config['idsIMASDBRunid'] = str(self.run_start)
            config['idsRunid'] = str(self.run_output)
            config['machine'] = self.db
            config['shotNum'] = str(self.shot)
            config['idsIMASDBUser'] = self.username
        except Exception as e:
            print(f'WARNING: Could not set run parameters via config API: {e}')
            print('  Will apply via modify_jset_line() after export')
        
        # Export configuration to temporary directory
        tmp_dir = f'{self.path_baserun}tmp'
        config.export(tmp_dir)
        
        # Copy jset file back
        shutil.copyfile(f'{tmp_dir}/jetto.jset', f'{self.path_baserun}/jetto.jset')
        
        # Copy jetto.in back and restore section headers
        shutil.copyfile(f'{tmp_dir}/jetto.in', jetto_in_path)
        self._restore_section_headers(jetto_in_path, section_headers)
        
        shutil.rmtree(tmp_dir)
        
        # Note: fix_jset_empty_parameters() removed as generator is now pre-fixed
        # If you encounter "Parameter X missing column Y" errors, the generator
        # jetto.jset needs empty parameter cleanup (see git history)
        
        # Restore array formatting
        self.tmp_handle_arrays_close()
        
        # Apply Namelist-level modifications (density feedback arrays, time parameters)
        interpretive_flag = 'interpretive' in self.path_generator
        if self.density_feedback or self.time_start is not None or self.time_end != 100:
            # Clean jetto.in again (remove decorative headers) for jetto_tools to read it
            self._clean_jetto_in_for_namelist_api(jetto_in_path)
            jetto_nml = jetto_tools.namelist.read(jetto_in_path)
            self.modify_jetto_in_via_jetto_tools(jetto_nml, self.path_baserun, 
                                                abs(b0), r0, self.time_start, self.time_end,
                                                self.output_timesteps, None, interpretive_flag)
            # Restore section headers after modification
            self._restore_section_headers(jetto_in_path, section_headers)

    def tmp_handle_arrays_open(self) -> None:
        """
        Temporary workaround: quote array expressions in extranamelist.
        
        jetto_tools has issues with array expressions in parentheses.
        This adds quotes around them before exporting configuration.
        """
        extranamelist = get_extraname_fields(self.path_baserun)
        for key in extranamelist:
            if '(' in extranamelist[key][0] and ')' in extranamelist[key][0]:
                extranamelist[key][0] = f'\'{extranamelist[key][0]}\''
        put_extraname_fields(self.path_baserun, extranamelist)

    def tmp_handle_arrays_close(self) -> None:
        """
        Temporary workaround: restore array expressions in extranamelist.
        
        Removes quotes added by tmp_handle_arrays_open() after configuration
        has been exported by jetto_tools.
        """
        extranamelist = get_extraname_fields(self.path_baserun)
        for key in extranamelist:
            if '(' in extranamelist[key][0] and ')' in extranamelist[key][0]:
                extranamelist[key][0] = extranamelist[key][0].strip('\'')
        put_extraname_fields(self.path_baserun, extranamelist)

    def fix_jset_empty_parameters(self) -> None:
        """
        Fix empty parameters in jetto.jset for jetto_tools 2.1.0+ compatibility.
        
        jetto_tools 2.1.0 has stricter JSET parsing that fails on parameters with
        empty values (lines ending with ':' and no data). This method:
        - Reads jetto.jset
        - Identifies parameters with empty values (ending with ':' and only whitespace)
        - Replaces empty 2D array parameters with '0.0' placeholder
        - Replaces empty 1D array/scalar parameters with '0.0' or 'false' as appropriate
        - Handles ExtraNamelist cell columns (fills empty column 1 with empty string "")
        - Creates backup before modification
        
        This handles generators created with older jetto_tools versions that
        allowed empty parameter values.
        """
        jset_path = f'{self.path_baserun}/jetto.jset'
        backup_path = f'{jset_path}.backup_empty_params'
        
        # Always start from backup if it exists (in case of re-runs)
        # Otherwise create backup first
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, jset_path)
        else:
            shutil.copyfile(jset_path, backup_path)
        
        # Read jset file
        with open(jset_path, 'r') as f:
            lines = f.readlines()
        
        # Process each line
        modified = False
        extra_cell_fixes = 0
        for i, line in enumerate(lines):
            # Check if line has parameter assignment with empty value
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_value = parts[1].strip()
                    
                    # Empty parameter (only whitespace or nothing after ':')
                    if not param_value:
                        # Determine appropriate default based on parameter type
                        if '.cell[' in param_name and param_name.endswith('[1]'):
                            # ExtraNamelist cell column 1 (namelist name, can be empty string)
                            lines[i] = f'{parts[0]}: ""\n'
                            modified = True
                            extra_cell_fixes += 1
                        elif '.select[' in param_name:
                            # Boolean selection parameters
                            lines[i] = f"{parts[0]}: false\n"
                            modified = True
                        elif '.time[' in param_name:
                            # Time array parameters
                            lines[i] = f"{parts[0]}: 0.0\n"
                            modified = True
                        elif '.value[' in param_name and '][' in param_name:
                            # 2D array parameters (e.g., value[0][0])
                            lines[i] = f"{parts[0]}: 0.0\n"
                            modified = True
                        elif '.value[' in param_name:
                            # 1D array parameters (e.g., value[0])
                            lines[i] = f"{parts[0]}: 0.0\n"
                            modified = True
        
        # Write back if modified
        if modified:
            with open(jset_path, 'w') as f:
                f.writelines(lines)
            print(f"Fixed empty parameters in {jset_path}")
            print(f"  - ExtraNamelist cell[*][1] columns fixed: {extra_cell_fixes}")
            print(f"Backup saved to {backup_path}")


    def get_boundary_values_quantities(self) -> None:
        """
        Extract boundary condition values from core profiles.
        
        Reads electron temperature, ion temperature, and electron density at the
        last radial point (boundary) from all core profile time slices.
        Stores in self.boundary_conditions dictionary with times array.
        
        Only extracts if boundary_conditions is empty (not yet loaded).
        """
        if self.boundary_conditions:
            return  # Already loaded
        
        core_profiles = open_and_get_ids(
            self.db, self.shot, self.run_start, 'core_profiles'
        )
        
        self.boundary_conditions['te'] = []
        self.boundary_conditions['ti'] = []
        self.boundary_conditions['ne'] = []
        
        for profile_1d in core_profiles.profiles_1d:
            # Extract boundary values (last radial point)
            self.boundary_conditions['te'].append(
                profile_1d.electrons.temperature[-1]
            )
            self.boundary_conditions['ti'].append(
                profile_1d.ion[0].temperature[-1]
            )
            # Convert density to 10^19 m^-3
            self.boundary_conditions['ne'].append(
                profile_1d.electrons.density[-1] * 1e-6
            )
        
        self.boundary_conditions['times'] = core_profiles.time.tolist()

    def get_feedback_on_density_quantities(self) -> None:
        """
        Extract line-averaged density feedback control data.
        
        Reads density feedback reference values and times from pulse schedule IDS.
        Stores in self.line_ave_density and self.dens_feedback_time for use in
        density feedback control setup.
        """
        pulse_schedule = open_and_get_ids(
            self.db, self.shot, self.run_start, 'pulse_schedule'
        )
        self.dens_feedback_time = pulse_schedule.time
        # Convert memoryview to numpy array for proper handling
        self.line_ave_density = np.asarray(pulse_schedule.density_control.n_e_line.reference.data)

    def setup_boundary_values(self) -> None:
        """
        Setup separatrix boundary conditions (Te, Ti, Ne).
        
        Orchestrates the complete boundary condition setup by:
        1. Loading boundary values from IDS
        2. Configuring jset file with time-dependent boundary values
        3. Configuring jetto.in file with boundary arrays
        """
        if not self.boundary_conditions:
            self.get_boundary_values_quantities()

        # jetto.in is now handled via jetto_tools Namelist API
        self.setup_boundary_values_jset()

    def setup_boundary_values_jetto_in(self) -> None:
        """
        Configure boundary conditions in jetto.in file.
        
        Updates the number of boundary points and the arrays for electron temperature,
        ion temperature, electron density, and their corresponding time arrays.
        Also sets boundary condition options (BCINTRHON, qlk_rhomax).
        """
        run_name = self.path_baserun
        
        # Set number of boundary points
        modify_jettoin_line(run_name, '  NTEB', len(self.boundary_conditions['te']))
        modify_jettoin_line(run_name, '  NTIB', len(self.boundary_conditions['ti']))
        modify_jettoin_line(run_name, '  NDNHB1', len(self.boundary_conditions['ne']))
        
        # Set boundary value arrays
        modify_jettoin_line(run_name, '  TEB', self.boundary_conditions['te'])
        modify_jettoin_line(run_name, '  TIB', self.boundary_conditions['ti'])
        modify_jettoin_line(run_name, '  DNHB1', self.boundary_conditions['ne'])
        
        # Set time arrays for boundary conditions
        modify_jettoin_line(run_name, '  TTEB', self.boundary_conditions['times'])
        modify_jettoin_line(run_name, '  TTIB', self.boundary_conditions['times'])
        modify_jettoin_line(run_name, '  TDNHB1', self.boundary_conditions['times'])
        
        # Set boundary condition options
        modify_jettoin_line(run_name, '  BCINTRHON', 1.0)
        modify_jettoin_line(run_name, '  qlk_rhomax', 0.995)

    def setup_boundary_values_jset(self) -> None:
        """
        Configure boundary conditions in jset file.
        
        Sets up time-dependent boundary condition profiles for electron temperature,
        ion temperature, and electron density. Also configures boundary condition
        parameters like BCINTRHON and qlk_rhomax if available in extranamelist.
        """
        run_name = self.path_baserun
        
        # Setup electron temperature boundary
        modify_jset_time_list(
            run_name, 'BoundCondPanel.eleTemp',
            self.boundary_conditions['times'],
            self.boundary_conditions['te']
        )
        
        # Setup ion temperature boundary
        modify_jset_time_list(
            run_name, 'BoundCondPanel.ionTemp',
            self.boundary_conditions['times'],
            self.boundary_conditions['ti']
        )
        
        # Setup ion density boundary
        modify_jset_time_list(
            run_name, 'BoundCondPanel.ionDens[0]',
            self.boundary_conditions['times'],
            self.boundary_conditions['ne']
        )
        
        # Configure BCINTRHON option if present
        line_start = identify_line_start_extranamelist(run_name, 'BCINTRHON')
        if line_start:
            line_start = line_start.replace('[0]', '[2]')
            modify_jset_line(run_name, line_start, '1.0')
        
        # Configure qlk_rhomax option if present
        line_start = identify_line_start_extranamelist(run_name, 'qlk_rhomax')
        if line_start:
            line_start = line_start.replace('[0]', '[2]')
            modify_jset_line(run_name, line_start, '0.995')

    def add_extra_transport(self) -> None:
        """
        Setup extra transport model (Gaussian formulation).
        
        Configures both jetto.in and jetto.jset files to add an extra transport
        model with Gaussian form. Updates both electron and particle transport.
        """
        run_name = self.path_baserun
        self.add_extra_transport_jettoin(run_name)
        self.add_extra_transport_jettojset(run_name)

    def add_extra_transport_jettoin(self, run_name: str) -> None:
        """
        Configure extra transport in jetto.in file.
        
        Sets form flags and transport parameters for electron and particle transport
        according to the add_extra_transport_flag value.
        """
        modify_jettoin_line(run_name, '  IFORME', 2)
        modify_jettoin_line(run_name, '  IFORMD', 2)
        
        if self.add_extra_transport_flag is True:
            forme_value = [20000.0, 1.0, 0.01]
            formd_value = [20000.0, 1.0, 0.01]
        else:
            forme_value = [self.add_extra_transport_flag, 1.0, 0.01]
            formd_value = [self.add_extra_transport_flag, 1.0, 0.01]
        
        modify_jettoin_line(run_name, '  FORME', forme_value)
        modify_jettoin_line(run_name, '  FORMD', formd_value)

    def add_extra_transport_jettojset(self, run_name: str) -> None:
        """
        Configure extra transport in jetto.jset file.
        
        Sets Gaussian model parameters for both electron thermal and particle
        transport. Updates three parameters for each transport type.
        """
        modify_jset_line(run_name, 'TransportAddFormDialog.model', 'Gaussian')
        
        # Electron thermal transport
        modify_jset_line(run_name, 'TransportAddPanel.FormulaElectronThermal', 'true')
        if self.add_extra_transport_flag is True:
            ele_form_0 = str(20000)
        else:
            ele_form_0 = str(self.add_extra_transport_flag)
        
        modify_jset_line(run_name, 'TransportAddFormDialog.gaussian.eleForm[0]', ele_form_0)
        modify_jset_line(run_name, 'TransportAddFormDialog.gaussian.eleForm[1]', str(1.0))
        modify_jset_line(run_name, 'TransportAddFormDialog.gaussian.eleForm[2]', str(0.01))
        
        # Particle transport
        modify_jset_line(run_name, 'TransportAddPanel.FormulaParticle', 'true')
        if self.add_extra_transport_flag is True:
            par_form_0 = str(20000)
        else:
            par_form_0 = str(self.add_extra_transport_flag)
        
        modify_jset_line(run_name, 'TransportAddFormDialog.gaussian.parForm[0]', par_form_0)
        modify_jset_line(run_name, 'TransportAddFormDialog.gaussian.parForm[1]', str(1.0))
        modify_jset_line(run_name, 'TransportAddFormDialog.gaussian.parForm[2]', str(0.01))

    def setup_feedback_on_density(self) -> None:
        """
        Validate automatic density feedback data is available and consistent.
        
        Note: Arrays are written to jetto.in via jetto_tools Namelist API
        inside modify_jetto_in_via_jetto_tools(). This function performs
        validation only to ensure required data is present.
        
        Raises
        ------
        SystemExit
            If density feedback data is missing or inconsistent
        """
        if self.line_ave_density is None or self.dens_feedback_time is None:
            print('ERROR: No density feedback data available. Call get_feedback_on_density_quantities() first.')
            exit()

        if len(self.line_ave_density) == 0 or len(self.dens_feedback_time) == 0:
            print('ERROR: Density feedback arrays are empty.')
            exit()

        if len(self.line_ave_density) != len(self.dens_feedback_time):
            print('ERROR: Density feedback arrays have different lengths.')
            exit()

        # Optional: ensure strictly non-decreasing time array
        times = self.dens_feedback_time
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            print('ERROR: Density feedback time array must be non-decreasing.')
            exit()

        # All good; actual writing happens in modify_jetto_in_via_jetto_tools()
        print('✓ Density feedback validated; writing handled via jetto_tools.')

    def modify_jset(self, path: str, run_name: str, ids_number: int,
                   ids_output_number: int, b0: float, r0: float) -> None:
        """
        Configure jset file with run-specific parameters (fallback method).
        
        Note: Primary configuration now happens via jetto_tools config API in 
        setup_jetto_simulation(). This method provides fallback direct file 
        modifications for any parameters that couldn't be set via config API.
        
        Parameters
        ----------
        path : str
            Base path for runs directory
        run_name : str
            Name of the baserun directory
        ids_number : int
            Input IDS run number
        ids_output_number : int
            Output IDS run number
        b0 : float
            Toroidal magnetic field [T]
        r0 : float
            Major radius [cm]
        """
        # Map configuration parameters for direct file manipulation
        # Note: btin/rmj are JSET-only parameters (not in namelist) so must be set here
        fallback_config_map = {
            'Creation Name': f'{path}{run_name}/jetto.jset',
            'JobProcessingPanel.runDirNumber': run_name[3:],
            'AdvancedPanel.catMachID': self.db,
            'AdvancedPanel.catMachID_R': self.db,
            'AdvancedPanel.catOwner': self.username,
            'AdvancedPanel.catOwner_R': self.username,
            'AdvancedPanel.catShotID': str(self.shot),
            'AdvancedPanel.catShotID_R': str(self.shot),
            'EquilEscoRefPanel.BField.ConstValue': str(b0),
            'EquilEscoRefPanel.BField ': str(b0),
            'EquilEscoRefPanel.refMajorRadius': str(r0)
        }
        
        # Apply fallback modifications if needed
        for field_name, value in fallback_config_map.items():
            try:
                modify_jset_line(run_name, field_name, value)
            except Exception as e:
                print(f'WARNING: Could not modify jset field {field_name}: {e}')

    def modify_ascot_cntl(self, run_name: str) -> None:
        """
        Configure ascot.cntl file with run path.
        
        Parameters
        ----------
        run_name : str
            Name of the baserun directory
        """
        modify_ascot_cntl_line(run_name, 'Creation Name', f'{run_name}/ascot.cntl')

    def modify_jset_nbi(self, run_name: str, nbi_config_name: str) -> None:
        """
        Configure NBI settings in jset file.
        
        Parameters
        ----------
        run_name : str
            Name of the baserun directory
        nbi_config_name : str
            Path to NBI configuration file
        """
        config_map = {
            'NBIAscotRef.configFileName': nbi_config_name,
            'NBIAscotRef.configPrvDir': '/afs/eufus.eu/user/g/g2ethole/public/tcv_inputs'
        }
        
        for field_name, value in config_map.items():
            modify_jset_line(run_name, field_name, value)

    def modify_jetto_in_via_jetto_tools(self, namelist: jetto_tools.namelist.Namelist, 
                                       run_name: str, r0: float, b0: float, 
                                       time_start: float, time_end: float,
                                       num_times_print: int | None = None,
                                       num_times_eq: int | None = None,
                                       interpretive_flag: bool = False) -> None:
        """
        Modify jetto.in using jetto_tools Namelist API.
        
        Updates namelist fields consistently with jetto_tools data structures.
        
        Parameters
        ----------
        namelist : jetto_tools.namelist.Namelist
            Loaded namelist object from jetto_tools
        run_name : str
            Path to the run directory
        r0 : float
            Major radius (m)
        b0 : float
            Toroidal magnetic field at r0 (T)
        time_start : float
            Simulation start time (s)
        time_end : float
            Simulation end time (s)
        num_times_print : int | None, optional
            Number of output time points
        num_times_eq : int | None, optional
            Number of equilibrium time points
        interpretive_flag : bool, optional
            Whether to run in interpretive mode (default: False)
        """
        try:
            # Update machine parameters (NLIST1) - only if they exist
            if not interpretive_flag:
                try:
                    namelist.set_field('nlist1', 'rmj', r0)
                except:
                    pass  # Parameter doesn't exist in this template
                try:
                    namelist.set_field('nlist1', 'btin', b0)
                except:
                    pass  # Parameter doesn't exist in this template
            
            # Update database and shot (NLIST1)
            try:
                namelist.set_field('nlist1', 'machid', self.db)
            except Exception:
                pass  # Parameter doesn't exist
            try:
                namelist.set_field('nlist1', 'npulse', self.shot)
            except Exception:
                try:
                    # Some templates store NPULSE in INESCO
                    namelist.set_field('inesco', 'npulse', self.shot)
                except Exception:
                    print('WARNING: Could not set NPULSE via Namelist API')

            # Update time range (NLIST1)
            try:
                namelist.set_field('nlist1', 'tbeg', time_start)
            except Exception:
                pass
            try:
                namelist.set_field('nlist1', 'tmax', time_end)
            except:
                pass
            
            # Update output times (NLIST2)
            if num_times_print is not None:
                try:
                    namelist.set_field('nlist2', 'ntint', num_times_print)
                except:
                    pass
                try:
                    namelist.set_field('nlist2', 'ntpr', num_times_print - 2)
                except:
                    pass
            
            # Update equilibrium times (NLIST2)
            if num_times_eq:
                try:
                    dt = (time_end - time_start) / num_times_eq
                    namelist.set_field('nlist2', 'timequ', [time_start, time_end, dt])
                except:
                    pass
            
            # Set density feedback arrays using namelist API
            if self.line_ave_density is not None and len(self.line_ave_density) > 0:
                try:
                    # Convert density from m^-3 to 10^-6 m^-3
                    dneflfb_values = [float(val * 1e-6) for val in self.line_ave_density]
                    # Convert time array to list if needed
                    time_values = self.dens_feedback_time.tolist() if hasattr(self.dens_feedback_time, 'tolist') else list(self.dens_feedback_time)
                    # Set arrays directly in NLIST4
                    namelist.set_array('nlist4', 'dneflfb', dneflfb_values)
                    namelist.set_array('nlist4', 'dtneflfb', time_values)
                    print(f"✓ Set density feedback arrays via Namelist API: {len(dneflfb_values)} points")
                except Exception as e:
                    print(f"WARNING: Could not set density feedback via Namelist API: {e}")
                    import traceback
                    traceback.print_exc()

            # Set separatrix boundary conditions (NLIST1) when enabled
            if self.set_sep_boundaries:
                if not self.boundary_conditions:
                    raise RuntimeError('set_sep_boundaries requested but boundary_conditions is empty')
                bc = self.boundary_conditions
                required_keys = ['te', 'ti', 'ne', 'times']
                if any(k not in bc for k in required_keys):
                    raise RuntimeError('Boundary conditions missing required keys for set_sep_boundaries')
                if not (len(bc['te']) == len(bc['ti']) == len(bc['ne']) == len(bc['times'])):
                    raise RuntimeError('Boundary condition arrays must have matching lengths')

                try:
                    npts = len(bc['te'])
                    namelist.set_field('nlist1', 'nteb', npts)
                    namelist.set_field('nlist1', 'ntib', npts)
                    namelist.set_field('nlist1', 'ndnhb1', npts)

                    namelist.set_array('nlist1', 'teb', bc['te'])
                    namelist.set_array('nlist1', 'tib', bc['ti'])
                    namelist.set_array('nlist1', 'dnhb1', bc['ne'])

                    namelist.set_array('nlist1', 'tteb', bc['times'])
                    namelist.set_array('nlist1', 'ttib', bc['times'])
                    namelist.set_array('nlist1', 'tdnhb1', bc['times'])

                    try:
                        namelist.set_field('nlist1', 'bcintrhon', 1.0)
                    except Exception:
                        pass
                    try:
                        namelist.set_field('nlist1', 'qlk_rhomax', 0.995)
                    except Exception:
                        pass
                    print(f"✓ Set separatrix boundaries via Namelist API: {npts} points")
                except Exception as e:
                    print(f"WARNING: Could not set separatrix boundaries via Namelist API: {e}")
                    traceback.print_exc()
            
            # Set TPRINT array (output time points)
            try:
                existing_tprint = namelist.get_array('nlist2', 'tprint')
            except Exception as e:
                raise RuntimeError('TPRINT missing in generator; expected array') from e

            if not isinstance(existing_tprint, (list, tuple, np.ndarray)):
                raise ValueError('TPRINT in generator must be an array; scalar found')

            if num_times_print is None:
                num_times_print = len(existing_tprint)

            try:
                tprint_values = [
                    time_start + (time_end - time_start) / num_times_print * i
                    for i in range(num_times_print)
                ]
                namelist.set_array('nlist2', 'tprint', tprint_values)
                print(f"✓ Set TPRINT array with {num_times_print} time points")
            except Exception as e:
                print(f"WARNING: Could not set TPRINT array: {e}")
                traceback.print_exc()
            
            # Write modified namelist back to file
            namelist_str = str(namelist)
            with open(f'{run_name}/jetto.in', 'w') as f:
                f.write(namelist_str)
            
            print(f"✓ Modified jetto.in using jetto_tools Namelist API for {run_name}")
            
        except Exception as e:
            if 'TPRINT' in str(e).upper():
                # Do not fall back to direct writes; generator must provide array TPRINT
                raise
            print(f"WARNING: Could not modify jetto.in via jetto_tools: {e}")
            print("  Falling back to direct file modification...")
            # Fall back to direct file modification for density feedback
            self._modify_jetto_in_direct(run_name, r0, b0, time_start, time_end, 
                                        num_times_print, num_times_eq, interpretive_flag)

    def _modify_jetto_in_direct(self, run_name: str, r0: float, b0: float,
                               time_start: float, time_end: float,
                               num_times_print: Optional[int] = None,
                               num_times_eq: Optional[int] = None,
                               interpretive_flag: bool = False) -> None:
        """
        Direct modification of jetto.in file for density feedback (fallback method).
        
        This method directly modifies the Fortran namelist in jetto.in when
        the jetto_tools Namelist API fails. Specifically handles density feedback
        arrays (DNEFLFB and DTNEFLFB) in NLIST4.
        
        Parameters
        ----------
        run_name : str
            Path to the run directory
        r0 : float
            Major radius (m)
        b0 : float
            Toroidal magnetic field at r0 (T)
        time_start : float
            Simulation start time (s)
        time_end : float
            Simulation end time (s)
        num_times_print : int | None, optional
            Number of output time points
        num_times_eq : int | None, optional
            Number of equilibrium time points
        interpretive_flag : bool, optional
            Whether to run in interpretive mode (default: False)
        """
        try:
            jetto_in_path = f'{run_name}/jetto.in'
            
            # Read jetto.in file
            with open(jetto_in_path, 'r') as f:
                read_data = f.readlines()
            
            # Find NLIST4 section
            nlist4_start = None
            nlist4_end = None
            for i, line in enumerate(read_data):
                if line.strip().startswith('&NLIST4'):
                    nlist4_start = i
                if nlist4_start is not None and line.strip().startswith('&END'):
                    nlist4_end = i
                    break
            
            if nlist4_start is None or nlist4_end is None:
                print(f"WARNING: Could not find NLIST4 section in {jetto_in_path}")
                return
            
            # Remove old DNEFLFB and DTNEFLFB arrays if present
            lines_to_remove = []
            for i in range(nlist4_start, nlist4_end):
                if read_data[i].strip().startswith(('DNEFLFB', 'DTNEFLFB')):
                    lines_to_remove.append(i)
            
            # Remove in reverse order to preserve indices
            for i in reversed(lines_to_remove):
                read_data.pop(i)
                nlist4_end -= 1  # Adjust end index
            
            # Add DNEFLFB array if density feedback data available
            if self.line_ave_density is not None and len(self.line_ave_density) > 0:
                # Insert DNEFLFB lines after NLIST4 declaration (after a few lines for readability)
                insert_index = nlist4_start + 10
                
                # Create DNEFLFB lines
                dneflfb_lines = []
                for idx, dens_value in enumerate(self.line_ave_density):
                    # Convert from m^-3 to 10^-6 m^-3
                    dneflfb_value = dens_value * 1e-6
                    dneflfb_line = f'  DNEFLFB({idx+1})=  {dneflfb_value:.6e}    , \n'
                    dneflfb_lines.append(dneflfb_line)
                
                # Insert DNEFLFB lines
                for i, line in enumerate(dneflfb_lines):
                    read_data.insert(insert_index + i, line)
                    nlist4_end += 1
                
                # Insert DTNEFLFB array after DNEFLFB
                insert_index = nlist4_start + 10 + len(dneflfb_lines)
                
                dtneflfb_lines = []
                for idx, time_value in enumerate(self.dens_feedback_time):
                    dtneflfb_line = f'  DTNEFLFB({idx+1})=  {time_value}    , \n'
                    dtneflfb_lines.append(dtneflfb_line)
                
                # Insert DTNEFLFB lines
                for i, line in enumerate(dtneflfb_lines):
                    read_data.insert(insert_index + i, line)
                    nlist4_end += 1
                
                print(f"✓ Added {len(dneflfb_lines)} DNEFLFB and {len(dtneflfb_lines)} DTNEFLFB lines to jetto.in")
            
            # Write modified file back
            with open(jetto_in_path, 'w') as f:
                f.writelines(read_data)
            
            print(f"✓ Modified jetto.in directly (fallback method) for {run_name}")
            
        except Exception as e:
            print(f"ERROR in direct jetto.in modification: {e}")
            import traceback
            traceback.print_exc()

    def setup_feedback_on_density_jset(self) -> None:
        """
        Configure DNEFLFB in jetto.jset OutputExtraNamelist.
        
        Ensures that DNEFLFB variable is present in OutputExtraNamelist
        with proper cell structure:
        - cell[j][0] = 'DNEFLFB'
        - cell[j][1] = '' (empty)
        - cell[j][2] = '(val1, val2, ..., valN)'
        - cell[j][3] = 'true'
        """
        try:
            jset_path = f'{self.path_baserun}/jetto.jset'
            
            # Read jset file
            with open(jset_path, 'r') as f:
                read_data = f.readlines()
            
            if self.line_ave_density is None or len(self.line_ave_density) == 0:
                return
            
            # Check if DNEFLFB already exists in OutputExtraNamelist
            dneflfb_row = None
            for i, line in enumerate(read_data):
                if 'OutputExtraNamelist.selItems.cell' in line and 'DNEFLFB' in line and '[0]' in line:
                    # Extract row number from cell[row][0]
                    import re
                    match = re.search(r'cell\[(\d+)\]\[0\]', line)
                    if match:
                        dneflfb_row = int(match.group(1))
                        break
            
            # If DNEFLFB exists, update it; otherwise add new entry
            if dneflfb_row is not None:
                # Update existing DNEFLFB entry
                print(f"✓ DNEFLFB already in OutputExtraNamelist at row {dneflfb_row}")
                
                # Update cell[dneflfb_row][2] with array values
                for i, line in enumerate(read_data):
                    if f'OutputExtraNamelist.selItems.cell[{dneflfb_row}][2]' in line:
                        # Format array values
                        array_str = ', '.join([f'{d*1e-6:.6e}' for d in self.line_ave_density])
                        read_data[i] = f'OutputExtraNamelist.selItems.cell[{dneflfb_row}][2] : ( {array_str} )\n'
                        print(f"✓ Updated DNEFLFB array values in jetto.jset")
                        break
            else:
                # Find the highest row number in OutputExtraNamelist
                max_row = -1
                for line in read_data:
                    if 'OutputExtraNamelist.selItems.cell' in line:
                        import re
                        match = re.search(r'cell\[(\d+)\]', line)
                        if match:
                            row = int(match.group(1))
                            max_row = max(max_row, row)
                
                new_row = max_row + 1 if max_row >= 0 else 0
                
                # Find position to insert (after last OutputExtraNamelist line)
                insert_pos = None
                for i in range(len(read_data)-1, -1, -1):
                    if 'OutputExtraNamelist.selItems.cell' in read_data[i]:
                        insert_pos = i + 1
                        break
                
                if insert_pos is None:
                    print("WARNING: Could not find OutputExtraNamelist section in jetto.jset")
                    return
                
                # Create new DNEFLFB entry
                array_str = ', '.join([f'{d*1e-6:.6e}' for d in self.line_ave_density])
                new_lines = [
                    f'OutputExtraNamelist.selItems.cell[{new_row}][0] : DNEFLFB\n',
                    f'OutputExtraNamelist.selItems.cell[{new_row}][1] : \n',
                    f'OutputExtraNamelist.selItems.cell[{new_row}][2] : ( {array_str} )\n',
                    f'OutputExtraNamelist.selItems.cell[{new_row}][3] : true\n',
                ]
                
                # Insert new lines
                for i, line in enumerate(new_lines):
                    read_data.insert(insert_pos + i, line)
                
                # Update row count
                for i, line in enumerate(read_data):
                    if 'OutputExtraNamelist.selItems.rows' in line and 'Sanco' not in line:
                        # Extract current row count and increment
                        import re
                        match = re.search(r'rows\s*:\s*(\d+)', line)
                        if match:
                            old_count = int(match.group(1))
                            read_data[i] = re.sub(
                                r'(OutputExtraNamelist\.selItems\.rows\s*:\s*)(\d+)',
                                r'\g<1>' + str(old_count + 1),
                                line
                            )
                            print(f"✓ Added DNEFLFB entry at row {new_row} to OutputExtraNamelist")
                            break
            
            # Write modified jset file
            with open(jset_path, 'w') as f:
                f.writelines(read_data)
            
            print(f"✓ Updated OutputExtraNamelist in jetto.jset for density feedback")
            
        except Exception as e:
            print(f"WARNING: Could not setup DNEFLFB in jetto.jset: {e}")
            import traceback
            traceback.print_exc()

    def _clean_jetto_namelist_file(self, file_path: str) -> dict:
        """
        Clean namelist file by removing decorative headers that f90nml cannot parse.
        Returns section headers to restore later.
        
        Parameters
        ----------
        file_path : str
            Path to the namelist file (jetto.in or jetto.sin)
            
        Returns
        -------
        dict
            Mapping of namelist names to their section header lines
        """
        with open(file_path, 'r') as f:
            original_lines = f.readlines()
        
        # Extract section headers (dashes and "Namelist : XXX" decorations)
        section_headers = {}
        cleaned_lines = []
        in_namelist = False
        found_first_namelist = False
        current_section_header = []
        
        for line in original_lines:
            stripped = line.strip()
            if stripped.startswith('&') and not stripped.startswith('&END'):
                # Start of namelist - save the accumulated section header
                namelist_name = stripped[1:].split()[0].lower()
                section_headers[namelist_name] = current_section_header.copy()
                current_section_header = []
                in_namelist = True
                found_first_namelist = True
                cleaned_lines.append(line)
            elif stripped.startswith('&END'):
                # End of namelist
                in_namelist = False
                cleaned_lines.append(line)
            elif in_namelist:
                # Content within namelist - keep it
                cleaned_lines.append(line)
            elif not found_first_namelist:
                # Line before first namelist
                # Check if it looks like a section header (dashes or "Namelist :")
                if '----' in line or 'Namelist :' in line:
                    # This might be the section header for the first namelist - accumulate it
                    current_section_header.append(line)
                # else: skip global header (jetto_tools will generate it)
            else:
                # Decorative/section header line between namelists
                current_section_header.append(line)
        
        # Write cleaned file
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)
        
        return section_headers

    def _restore_section_headers(self, file_path: str, section_headers: dict) -> None:
        """
        Restore section headers to namelist file and update GIT info.
        
        Parameters
        ----------
        file_path : str
            Path to the namelist file (jetto.in)
        section_headers : dict
            Mapping of namelist names to their section header lines
        """
        with open(file_path, 'r') as f:
            exported_lines = f.readlines()
        
        # Restore section headers and update GIT info
        reconstructed = []
        for line in exported_lines:
            stripped = line.strip()
            if stripped.startswith('&') and not stripped.startswith('&END'):
                # Start of namelist - insert section header if available
                namelist_name = stripped[1:].split()[0].lower()
                if namelist_name in section_headers and section_headers[namelist_name]:
                    reconstructed.extend(section_headers[namelist_name])
                # Now add the namelist line itself
                reconstructed.append(line)
            elif 'CODE INPUT NAMELIST FILE' in line:
                # Fill blank/n.a. field with jetto_tools identifier
                if ':' in line:
                    left, right = line.split(':', 1)
                    if not right.strip() or right.strip().lower() == 'n/a':
                        reconstructed.append(f"{left}: jetto_tools\n")
                    else:
                        reconstructed.append(line)
                else:
                    reconstructed.append(line.rstrip() + ' : jetto_tools\n')
            elif 'Current GIT repository' in line:
                reconstructed.append(self._fill_field_with_jetto_tools(line))
            elif 'Current GIT release tag' in line:
                reconstructed.append(self._fill_field_with_jetto_tools(line))
            elif 'Current GIT branch' in line:
                reconstructed.append(self._fill_field_with_jetto_tools(line))
            elif 'Last commit SHA1-key' in line:
                reconstructed.append(self._fill_field_with_jetto_tools(line))
            elif 'Repository status' in line:
                reconstructed.append(self._fill_field_with_jetto_tools(line))
            else:
                reconstructed.append(line)
        
        with open(file_path, 'w') as f:
            f.writelines(reconstructed)

    @staticmethod
    def _fill_field_with_jetto_tools(line: str) -> str:
        """Ensure metadata fields are set to jetto_tools when empty or n/a."""
        if ':' in line:
            left, right = line.split(':', 1)
            if not right.strip() or right.strip().lower() == 'n/a':
                return f"{left}: jetto_tools\n"
            return line if line.endswith('\n') else line + '\n'
        # No colon present, append it
        return line.rstrip() + ' : jetto_tools\n'

    def modify_jetto_in_impurities(self, imp_data_list: List[List[Any]]) -> None:
        """
        Configure impurity composition in jetto.in file.
        
        Updates the jetto.in file with impurity data (density, mass, charge state,
        super-states) extracted from IDS. Supports up to 7 impurities.
        
        Parameters
        ----------
        imp_data_list : list
            List of impurity data from select_impurities_from_ids()
        """
        # Initialize impurity data array (up to 7 impurities)
        imp_datas = [[0.0, 0.0, 0, 0.0] for _ in range(7)]
        
        for index, imp_data in enumerate(imp_data_list):
            imp_datas[index] = imp_data
        
        # Read jetto.in file
        file_path = f'{self.path_baserun}/jetto.in'
        with open(file_path) as f:
            read_data = f.readlines()
        
        # Update NIMP count if impurities present
        if imp_data_list:
            for index, line in enumerate(read_data):
                if line.startswith('  NIMP'):
                    read_data[index] = f'{line[:14]}{len(imp_data_list)}    , \n'
        
        # Write updated file
        with open(file_path, 'w') as f:
            f.writelines(read_data)

    def setup_time_polygon(self) -> None:
        """
        Setup time-dependent maximum timestep polygon.
        
        Configures both jset and jetto.in files with time-dependent timestep
        constraints that vary through different phases of the simulation.
        """
        modify_jset_time_polygon(
            self.path + self.baserun_name,
            self.time_start, self.time_end
        )
        modify_jettoin_time_polygon(
            self.path + self.baserun_name + '/jetto.in',
            self.time_start, self.time_end
        )

    def change_impurity_puff(self) -> None:
        """
        Modify impurity puff value based on plasma parameters.
        
        Scales the impurity puff value according to line-averaged density and Zeff
        using a predefined scaling. Updates both jset and jetto.sin files.
        """
        # Read current puff value
        self.puff_value = read_puff_jettosin(
            f'{self.path}{self.baserun_name}/jetto.sin'
        )
        
        # Load plasma parameters
        pulse_schedule = open_and_get_ids(
            self.db, self.shot, self.run_start, 'pulse_schedule'
        )
        core_profiles = open_and_get_ids(
            self.db, self.shot, self.run_start, 'core_profiles'
        )
        
        # Scale puff if flag is a numeric value
        if isinstance(self.change_impurity_puff_flag, (int, float)) and \
           not isinstance(self.change_impurity_puff_flag, bool) and \
           self.puff_value is not None:
            self.puff_value = self.puff_value * self.change_impurity_puff_flag
        
        # Get feedback quantities
        self.dens_feedback_time = pulse_schedule.time
        # Convert memoryview to numpy array for proper handling
        self.line_ave_density = np.asarray(pulse_schedule.density_control.n_e_line.reference.data)
        
        # Calculate average Zeff in time window
        zeff_times = []
        for profile_1d, time in zip(core_profiles.profiles_1d, core_profiles.time):
            if time > 0.2:
                zeff_times.append(np.average(profile_1d.zeff))
        zeff = np.average(np.asarray(zeff_times)) if zeff_times else 1.0
        
        # Get average density in simulation time window
        line_ave_times = [
            dens for dens, time in zip(self.line_ave_density, self.dens_feedback_time)
            if self.time_start < time < self.time_end
        ]
        line_ave_density = np.average(np.asarray(line_ave_times)) if line_ave_times else np.average(self.line_ave_density)
        
        # Scale puff value
        self.puff_value = calculate_impurity_puff(self.puff_value, zeff, line_ave_density)
        
        # Update jetto.sin via Namelist API first, fallback to direct if needed
        sin_path = f'{self.path}{self.baserun_name}/jetto.sin'
        api_ok = False
        tmp_sin = None
        try:
            # Clean jetto.sin (keep only namelist blocks) to satisfy jetto_tools parser
            with open(sin_path, 'r') as f:
                lines = f.readlines()
            cleaned = []
            in_nml = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('&') and not stripped.startswith('&END'):
                    in_nml = True
                    cleaned.append(line)
                elif stripped.startswith('&END'):
                    cleaned.append(line)
                    in_nml = False
                elif in_nml:
                    cleaned.append(line)

            tmp = tempfile.NamedTemporaryFile('w', delete=False, suffix='.sin', prefix='clean_', dir=os.path.dirname(sin_path))
            tmp.writelines(cleaned)
            tmp_sin = tmp.name
            tmp.close()

            sin_nml = jetto_tools.namelist.read(tmp_sin)

            # Update the array-style SPEFLX: set only non-zero elements to the new value
            try:
                current = sin_nml.get_field('physic', 'speflx')
            except Exception:
                current = None

            if isinstance(current, list) and current:
                # Set only non-zero array elements to the new puff value
                updated = [float(self.puff_value) if val != 0 else 0.0 for val in current]
                try:
                    sin_nml.set_array('physic', 'speflx', updated)
                except Exception:
                    sin_nml.set_field('physic', 'speflx', updated)
            else:
                sin_nml.set_field('physic', 'speflx', [float(self.puff_value)])

            with open(sin_path, 'w') as f:
                f.write(str(sin_nml))
            print(f"✓ Set SPEFLX via Namelist API (cleaned): {self.puff_value}")
            # Restore decorative headers now that namelist write is done
            try:
                if getattr(self, 'section_headers_sin', None):
                    self._restore_section_headers(sin_path, self.section_headers_sin)
            except Exception as restore_err:
                print(f"WARNING: Could not restore jetto.sin headers: {restore_err}")
            api_ok = True
        except Exception as e:
            print(f"WARNING: Could not set SPEFLX via Namelist API: {e}")
            traceback.print_exc()
        finally:
            if tmp_sin and os.path.exists(tmp_sin):
                try:
                    os.unlink(tmp_sin)
                except Exception:
                    pass
        
        modify_jset_line(
            self.path + self.baserun_name,
            'SancoBCPanel.Species1NeutralInflux.tpoly.value[0]',
            str(self.puff_value)
        )
        
        if not api_ok:
            modify_jettosin_time_polygon_single(
                sin_path,
                ['  SPEFLX'],
                [self.puff_value]
            )
            try:
                if getattr(self, 'section_headers_sin', None):
                    self._restore_section_headers(sin_path, self.section_headers_sin)
            except Exception as restore_err:
                print(f"WARNING: Could not restore jetto.sin headers after SPEFLX fallback: {restore_err}")

    def setup_time_polygon_impurity_puff(self) -> None:
        """
        Setup time-dependent impurity puff polygon.
        
        Configures time-dependent impurity puff values in both jset and jetto.sin
        files using a polynomial (time-dependent) representation.
        """
        self.puff_value = read_puff_jettosin(
            f'{self.path}{self.baserun_name}/jetto.sin'
        )
        
        modify_jset_time_polygon_puff(
            f'{self.path}{self.baserun_name}',
            self.time_start, self.time_end,
            self.puff_value
        )
        modify_jettosin_time_polygon(
            f'{self.path}{self.baserun_name}/jetto.sin',
            self.time_start, self.time_end,
            self.puff_value
        )

        try:
            if getattr(self, 'section_headers_sin', None):
                self._restore_section_headers(f'{self.path}{self.baserun_name}/jetto.sin', self.section_headers_sin)
        except Exception as restore_err:
            print(f"WARNING: Could not restore jetto.sin headers after puff polygon: {restore_err}")



# ============================================================================
# UTILITY FUNCTIONS - File I/O and Configuration Management
# ============================================================================


def _modify_file_line_generic(
    file_path: str,
    line_prefix: str,
    new_content: Any,
    *,
    position: int | None = None,
    value_separator: str = ':',
    formatter: Optional[Callable[[Any], str]] = None,
) -> None:
    """Generic line modifier used by jset/jetto.in/jetto.sin helpers.

    Reads the file, finds the first line starting with ``line_prefix`` and
    rewrites the value portion using either a fixed character position
    (``position``) or a separator-based replacement (``value_separator``).

    Parameters
    ----------
    file_path : str
        Path to the file to update.
    line_prefix : str
        Leading text to match a target line.
    new_content : Any
        Value to write; formatted by ``formatter`` when provided.
    position : int | None, optional
        If set, replace content starting at this character index.
        If None, fall back to separator-based replacement.
    value_separator : str, optional
        Separator token for key/value lines (default ``:``).
    formatter : callable | None, optional
        Optional callable to convert ``new_content`` into a string. If not
        provided, default formatting rules for float/int/list/str are used.
    """
    if not os.path.exists(file_path):
        print(f"WARNING: file not found: {file_path}")
        return

    try:
        lines = read_file(file_path)
    except Exception as exc:  # file read errors should not crash workflow
        print(f"WARNING: could not read {file_path}: {exc}")
        return

    user_formatter = formatter is not None

    def _format_content(value: Any) -> str:
        if formatter:
            return formatter(value)
        if isinstance(value, float):
            return f"{value:.3E}"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, (list, np.ndarray)):
            return '  '.join(f"{v:.3E}" for v in value)
        return str(value)

    updated = False
    prefix_len = len(line_prefix)

    for idx, line in enumerate(lines):
        if line.startswith(line_prefix):
            if position is not None:
                # Position-based replacement (e.g., jset column-aligned fields)
                lines[idx] = line[:position] + _format_content(new_content) + "\n"
            else:
                # Separator-based replacement (e.g., jetto.in with '=')
                if value_separator in line:
                    left, _sep, _right = line.partition(value_separator)
                    content = _format_content(new_content)
                    if not user_formatter:
                        if isinstance(new_content, (list, np.ndarray)):
                            # Preserve trailing comma/space style used in jetto.in
                            content = '  ' + content.replace('  ', '  ') + ' , '
                        elif isinstance(new_content, (float, int)):
                            content = f'  {content} , '
                        else:
                            content = f' {content}'
                    lines[idx] = f"{left}{value_separator}{content}\n"
                else:
                    lines[idx] = f"{line_prefix}{value_separator} {_format_content(new_content)}\n"
            updated = True
            break

    if updated:
        try:
            write_file(file_path, lines)
        except Exception as exc:
            print(f"WARNING: could not write {file_path}: {exc}")


def copy_files(source_folder: str, destination_folder: str) -> None:
    """
    Copy all files from source to destination folder.
    
    Parameters
    ----------
    source_folder : str
        Path to source directory
    destination_folder : str
        Path to destination directory
    """
    for file_name in os.listdir(source_folder):
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copy2(source_file, destination_file)


# ============================================================================
# NBI Configuration Management
# ============================================================================

def change_line_nbi(line: str, start: str, values: list) -> str:
    """
    Format and modify a line from NBI configuration file.
    
    Formats values according to type (energy, frequency, particle number) and
    constructs a properly spaced configuration line.
    
    Parameters
    ----------
    line : str
        Original configuration line
    start : str
        Line identifier/label (e.g., 'E1 [keV]', 'f1', 'ANum')
    values : list
        List of [value1, value2, value3] to insert
        
    Returns
    -------
    str
        Reformatted line with new values
    """
    values_str = []
    len_values_str = []
    new_line = line
    
    # Format values based on parameter type
    for value in values:
        if start == 'E1 [keV]':
            # Energy in keV (convert from eV)
            values_str.append(f'{value * 1e-3:.2f}')
        elif start in ('f1', 'f2', 'f3'):
            # Frequency parameters
            values_str.append(f'{value:.2f}')
        elif start in ('ANum', 'ZNum'):
            # Atomic/nuclear number
            values_str.append(f'{value:.0f}')
        
        len_values_str.append(len(values_str[-1]))
    
    # Reconstruct line with proper spacing
    if line.startswith(start):
        num_spaces_start = 22 - len(start) - len_values_str[0]
        num_spaces1 = 8 - len_values_str[1]
        num_spaces2 = 8 - len_values_str[2]
        
        new_line = (f"{start}{' ' * num_spaces_start}{values_str[0]}"
                   f"{' ' * num_spaces1}{values_str[1]}"
                   f"{' ' * num_spaces2}{values_str[2]}\n")
    
    return new_line


def add_line_power(lines: list, time: float, powers: list) -> list:
    """
    Add a power profile line to NBI configuration.
    
    Formats time and power values and appends a properly formatted line to
    the power profile configuration.
    
    Parameters
    ----------
    lines : list
        List of configuration lines to append to
    time : float
        Time point (seconds)
    powers : list
        List of [power1, power2, power3] in watts
        
    Returns
    -------
    list
        Updated lines list with new power profile entry
    """
    power_str = []
    len_power_str = []
    
    # Format power values (convert to MW)
    for power in powers:
        power_str.append(f'{power * 1e-6:.3f}')
        len_power_str.append(len(power_str[-1]))
    
    # Format time with appropriate precision
    if time > 1:
        num_digits = int(np.floor(np.log10(abs(time))))
    else:
        num_digits = 0
    
    time_str = f'{time:.{4-num_digits}f}'
    time_str = ' ' + time_str
    
    # Build new line with proper spacing
    num_spaces_start = 15 - len_power_str[0]
    num_spaces1 = 8 - len_power_str[1]
    num_spaces2 = 8 - len_power_str[2]
    
    new_line = (f"{time_str}{' ' * num_spaces_start}{power_str[0]}"
               f"{' ' * num_spaces1}{power_str[1]}"
               f"{' ' * num_spaces2}{power_str[2]}\n")
    
    lines.append(new_line)
    
    return lines


def read_file(path: str) -> List[str]:
    """
    Read all lines from a file.
    
    Parameters
    ----------
    path : str
        File path
        
    Returns
    -------
    list
        List of lines (with newlines preserved)
    """
    with open(path) as f:
        return f.readlines()


def write_file(path: str, lines: List[str]) -> None:
    """
    Write lines to a file.
    
    Parameters
    ----------
    path : str
        File path
    lines : list
        List of lines to write
    """
    with open(path, 'w') as f:
        f.writelines(lines)


# ============================================================================
# JETTO.JSET FILE UTILITIES
# ============================================================================


def read_jset_line(run_name: str, line_start: str) -> Optional[str]:
    """
    Read a value from jset file.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    line_start : str
        Line prefix to search for
        
    Returns
    -------
    str or None
        Value found after the colon, or None if not found
    """
    with open(f'{run_name}/jetto.jset') as f:
        for line in f:
            if line.startswith(line_start):
                value = line.split(':')[1].strip()
                return value
    return None


def identify_line_start_extranamelist(run_name: str, variable_name: str) -> Optional[str]:
    """
    Find the line prefix for a variable in extranamelist.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    variable_name : str
        Variable name to search for
        
    Returns
    -------
    str or None
        Line prefix (first 60 chars) if found, None otherwise
    """
    with open(f'{run_name}/jetto.jset') as f:
        for line in f:
            if variable_name in line:
                return line[:60]
    return None


def modify_jset_line(run_name: str, line_start: str, new_content: Any) -> None:
    """
    Modify a single line in jset file.
    
    Finds the line starting with line_start and replaces content from position 62 onwards.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    line_start : str
        Line prefix to search for
    new_content : str, int, float
        New content to write
    """
    file_path = f'{run_name}/jetto.jset'
    _modify_file_line_generic(
        file_path,
        line_start,
        new_content,
        position=62,
        value_separator=':'
    )


def delete_jset_value_in_line(run_name: str, line_start: str) -> None:
    """
    Delete or clear jset line values.
    
    Clears the value part of lines starting with line_start, or sets 'select' lines to false.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    line_start : str
        Line prefix to search for
    """
    file_path = f'{run_name}/jetto.jset'
    read_data = read_file(file_path)
    
    updated = []
    for line in read_data:
        if line.startswith(line_start):
            if 'select' in line:
                updated.append(line.replace('true', 'false'))
            else:
                updated.append(f'{line.split(":")[0]}: \n')
        else:
            updated.append(line)
    
    write_file(file_path, updated)


def insert_jset_line(run_name: str, previous_line_start: str, content: str) -> None:
    """
    Insert a new line in jset file after a matching line.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    previous_line_start : str
        Prefix of line to insert after
    content : str
        Content to insert
    """
    file_path = f'{run_name}/jetto.jset'
    read_data = read_file(file_path)
    
    for index, line in enumerate(read_data):
        if line.startswith(previous_line_start):
            read_data.insert(index, content)
            break
    
    write_file(file_path, read_data)


def create_jset_time_list(run_name: str, panel_name: str,
                         times: List[float], values: List[float]) -> Tuple[List[str], List[str]]:
    """
    Create jset field names and values for time-dependent configuration.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    panel_name : str
        Panel name prefix (e.g., 'BoundCondPanel.eleTemp')
    times : list
        Time values
    values : list
        Values at each time
        
    Returns
    -------
    tuple
        (field_names, field_values) lists
    """
    field_names = [f'{panel_name}.option', f'{panel_name}.ConstValue']
    field_values = ['Time Dependent', str(values[0])]
    
    # Add select flags
    for i in range(len(times)):
        field_names.append(f'{panel_name}.tpoly.select[{i}]')
    field_values.extend(['true'] * len(times))
    
    # Add times
    for t in times:
        field_names.append(f'{panel_name}.tpoly.time[{len(field_names) - len(field_values)}]')
    field_values.extend([str(t) for t in times])
    
    # Add values
    for v in values:
        field_names.append(f'{panel_name}.tpoly.value[{len(field_names) - len(field_values)}][0]')
    field_values.extend([str(v) for v in values])
    
    return field_names, field_values


def modify_jset_time_list(run_name: str, panel_name: str,
                         times: List[float], values: List[float]) -> None:
    """
    Modify time-dependent configuration in jset file.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    panel_name : str
        Panel name prefix
    times : list
        Time values
    values : list
        Values at each time
    """
    delete_jset_value_in_line(run_name, panel_name)
    field_names, field_values = create_jset_time_list(run_name, panel_name, times, values)
    
    for field_name, field_value in zip(field_names, field_values):
        modify_jset_line(run_name, field_name, field_value)


# ============================================================================
# JETTO.IN FILE UTILITIES
# ============================================================================


def read_jettoin_line(path_jetto_in: str, line_start: str) -> Optional[List[float]]:
    """
    Read numeric values from jetto.in line.
    
    Parameters
    ----------
    path_jetto_in : str
        Path to jetto.in file
    line_start : str
        Line prefix to search for
        
    Returns
    -------
    list or None
        List of numeric values, or None if not found
    """
    with open(path_jetto_in) as f:
        for line in f:
            if line.startswith(line_start):
                values_str = line.split('=')[1]
                values = values_str.split(',')
                try:
                    return [float(v) for v in values[:-1]]  # Skip last empty element
                except ValueError:
                    return None
    return None


def modify_jettoin_line(run_name: str, line_start: str, new_content: Any) -> None:
    """
    Modify a line in jetto.in file.
    
    Handles formatting for float, int, list, and string values.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    line_start : str
        Line prefix to search for
    new_content : float, int, list, str, np.ndarray
        New content (will be formatted appropriately)
    """
    file_path = f'{run_name}/jetto.in'

    # Ensure line_start ends with '=' and spacing consistent with existing format
    if not line_start.endswith('='):
        num_spaces = len(line_start)
        line_start = line_start + (11 - num_spaces) * ' ' + '='

    def _format_nml(value: Any) -> str:
        if isinstance(value, float):
            return f'  {value:.3E} , '
        if isinstance(value, int):
            return f'  {value} , '
        if isinstance(value, (list, np.ndarray)):
            return '  ' + '  '.join(f'{v:.3E} ,' for v in value) + ' '
        if isinstance(value, str):
            return f' {value}' if value != '\n' else value
        return f' {value}'

    _modify_file_line_generic(
        file_path,
        line_start,
        new_content,
        position=None,
        value_separator='=',
        formatter=_format_nml,
    )


def modify_jettoin_time_polygon(path_jetto_in: str, time_start: float, time_end: float) -> None:
    """
    Setup time-dependent maximum timestep polygon in jetto.in.
    
    Parameters
    ----------
    path_jetto_in : str
        Path to jetto.in file
    time_start : float
        Simulation start time
    time_end : float
        Simulation end time
    """
    fields_single = ['  DTMAX', '  NDTMAX']
    fields_array = ['  PDTMAX', '  TDTMAX']
    
    numbers_single = [2.0e-6, 5]
    times = [time_start, time_start + 0.001, time_start + 0.01, time_start + 0.02, time_end]
    numbers_array = [[1.0, 2.0, 100.0, 500.0, 500.0], times]
    
    modify_jettoin_time_polygon_single(path_jetto_in, fields_single, numbers_single)
    modify_jettoin_time_polygon_array(path_jetto_in, fields_array, numbers_array)


def modify_jettoin_time_polygon_single(path_jetto_in: str, fields: List[str], numbers: List[float]) -> None:
    """
    Setup simple (non-array) time-dependent fields in jetto.in.
    
    Parameters
    ----------
    path_jetto_in : str
        Path to jetto.in file
    fields : list
        Field names to modify
    numbers : list
        Values for each field
    """
    run_name = path_jetto_in.rsplit('/', 1)[0]  # Extract directory
    for field, number in zip(fields, numbers):
        modify_jettoin_line(run_name, field, number)


def modify_jettoin_time_polygon_array(path_jetto_in: str, fields: List[str], numbers: List[List[float]]) -> None:
    """
    Setup array time-dependent fields in jetto.in.
    
    Parameters
    ----------
    path_jetto_in : str
        Path to jetto.in file
    fields : list
        Field names to modify
    numbers : list
        Array values for each field
    """
    run_name = path_jetto_in.rsplit('/', 1)[0]
    for field, values in zip(fields, numbers):
        modify_jettoin_line(run_name, field, values)


# ============================================================================
# ASCOT AND JINTRAC FILE UTILITIES
# ============================================================================


def modify_ascot_cntl_line(run_name: str, line_start: str, new_content: str) -> None:
    """
    Modify a line in ascot.cntl file.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    line_start : str
        Line prefix to search for
    new_content : str
        New content to write
    """
    file_path = f'{run_name}/ascot.cntl'
    _modify_file_line_generic(
        file_path,
        line_start,
        new_content,
        position=42,
        value_separator=':'
    )


def modify_jintrac_launch(run_name: str, generator_name: str, generator_username: str,
                         db: str, shot: int, time_begin: float, time_end: float) -> None:
    """
    Configure jintrac.launch file with run-specific parameters.
    
    Adjusts time boundaries using bit-shifting to match jams conventions.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    generator_name : str
        Generator run name to replace
    generator_username : str
        Generator username to replace
    db : str
        Database name
    shot : int
        Shot number
    time_begin : float
        Simulation start time
    time_end : float
        Simulation end time
    """
    # Adjust times using bit-shift operations (jams convention)
    time_begin = (1/2**24) * (time_begin // (1/2**24)) + (1/2**24)
    time_end = (1/2**24) * (time_end // (1/2**24)) + (1/2**24)
    
    file_path = f'{run_name}/jintrac.launch'
    read_data = read_file(file_path)

    # First replace generator/run identifiers and usernames
    current_username = getpass.getuser()
    replacements = {
        generator_name: run_name,
        generator_username: current_username,
    }

    updated = []
    for line in read_data:
        modified_line = line
        for pattern, replacement in replacements.items():
            modified_line = modified_line.replace(pattern, replacement)
        updated.append(modified_line)

    write_file(file_path, updated)

    # Then enforce scalar fields via consolidated helper (avoids ad-hoc string slicing)
    def _plain(val: Any) -> str:
        return f' {val}'

    for prefix, value in [
        ('  shot_in', shot),
        ('  shot_out', shot),
        ('  machine_in', db),
        ('  machine_out', db),
        ('  tstart', time_begin),
        ('  tend', time_end),
    ]:
        _modify_file_line_generic(
            file_path,
            prefix,
            value,
            position=None,
            value_separator=':',
            formatter=_plain,
        )


def modify_llcmd(run_name: str, baserun_name: str, generator_username: str) -> None:
    """
    Configure .llcmd batch script for the run.
    
    Replaces generator run name and username with new values.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    baserun_name : str
        Name to replace generator name with
    generator_username : str
        Generator username to replace
    """
    file_path = f'{run_name}/.llcmd'
    read_data = read_file(file_path)
    
    current_username = getpass.getuser()
    updated = []
    for line in read_data:
        modified_line = line.replace(baserun_name, run_name)
        if generator_username:
            modified_line = modified_line.replace(generator_username, current_username)
        updated.append(modified_line)
    
    write_file(file_path, updated)


def modify_hfps_launch(
    baserun_dir: str,
    run_name: str,
    shot: int,
    machine: str,
    time_start: float,
    time_end: float,
    generator_name: str
) -> None:
    """
    Modify hfps.launch YAML file with updated run parameters.
    
    Updates the following parameters in hfps.launch:
    - run_name: Job name in models.jetto.args and executable path
    - shot_out: Output shot number
    - shot_in: Input shot number (same as shot_out)
    - machine_out: Output machine/database name
    - tstart: Simulation start time
    - tend: Simulation end time
    
    Preserves unchanged parameters:
    - user_out, run_in, run_out
    
    Parameters
    ----------
    baserun_dir : str
        Path to baserun directory
    run_name : str
        New run name to set in the configuration
    shot : int
        Shot number for both shot_in and shot_out
    machine : str
        Machine/database name for machine_out
    time_start : float
        Simulation start time (tstart)
    time_end : float
        Simulation end time (tend)
    generator_name : str
        Generator run name to replace in paths
    """
    file_path = os.path.join(baserun_dir, 'hfps.launch')
    
    if not os.path.exists(file_path):
        print(f"WARNING: hfps.launch file not found at {file_path}")
        return
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse YAML content line by line
        lines = content.split('\n')
        updated_lines = []
        
        for line in lines:
            # Update run_name in models.jetto.args
            if line.strip().startswith('args:'):
                # The args line format is: args: -S -I0 -p -xmpi -x64 <generator_name> <path_to_jetto>
                # We need to replace <generator_name> (which starts with 'rungenerator') with run_name
                indent = len(line) - len(line.lstrip())
                parts = line.split()
                
                # Find the index of the generator name (contains 'rungenerator')
                updated_parts = []
                for i, part in enumerate(parts):
                    if 'rungenerator' in part.lower():
                        # Replace this token with run_name
                        updated_parts.append(run_name)
                    else:
                        updated_parts.append(part)
                
                updated_lines.append(' ' * indent + ' '.join(updated_parts))
            # Update executable path to point to new run directory
            elif line.strip().startswith('executable:'):
                # Replace generator_name with run_name in the path
                indent = len(line) - len(line.lstrip())
                updated_line = line.replace(generator_name, run_name)
                updated_lines.append(updated_line)
            # Update shot_out
            elif line.strip().startswith('shot_out:'):
                indent = len(line) - len(line.lstrip())
                updated_lines.append(f"{' ' * indent}shot_out: {shot}")
            # Update shot_in
            elif line.strip().startswith('shot_in:'):
                indent = len(line) - len(line.lstrip())
                updated_lines.append(f"{' ' * indent}shot_in: {shot}")
            # Update machine_out
            elif line.strip().startswith('machine_out:'):
                indent = len(line) - len(line.lstrip())
                updated_lines.append(f"{' ' * indent}machine_out: {machine}")
            # Update tstart
            elif line.strip().startswith('tstart:'):
                indent = len(line) - len(line.lstrip())
                updated_lines.append(f"{' ' * indent}tstart: {time_start}")
            # Update tend
            elif line.strip().startswith('tend:'):
                indent = len(line) - len(line.lstrip())
                updated_lines.append(f"{' ' * indent}tend: {time_end}")
            else:
                updated_lines.append(line)
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print(
            f"✓ Modified hfps.launch with new parameters: "
            f"run_name={run_name}, shot={shot}, machine={machine}, "
            f"tstart={time_start}, tend={time_end}"
        )
        
    except Exception as e:
        print(f"ERROR modifying hfps.launch: {e}")


# ============================================================================
# JETTO.SIN AND JETTO.JSET TIME-DEPENDENT UTILITIES
# ============================================================================


def read_puff_jettosin(path_jetto_sin: str) -> Optional[float]:
    """
    Read impurity puff value from jetto.sin file.
    
    Parameters
    ----------
    path_jetto_sin : str
        Path to jetto.sin file
        
    Returns
    -------
    float or None
        Puff value if found, None otherwise
    """
    with open(path_jetto_sin) as f:
        for line in f:
            if line.startswith('  SPEFLX'):
                try:
                    return float(line[14:].split(',')[0])
                except (ValueError, IndexError):
                    return None
    return None


def adapt_to_jettosin(values: List[float]) -> List[float]:
    """
    Adapt array for jettosin format (add zeros for multi-component fields).
    
    Parameters
    ----------
    values : list
        Input values
        
    Returns
    -------
    list
        Expanded values with zeros
    """
    result = []
    for value in values:
        result.append(value)
        result.extend([0.0] * 6)
    return result


def reshape_array(array: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Reshape array into chunks.
    
    Parameters
    ----------
    array : list
        Input array
    chunk_size : int
        Size of each chunk
        
    Returns
    -------
    list
        List of chunks
    """
    return [array[i:i+chunk_size] for i in range(0, len(array), chunk_size)]


def create_single_line_jettosin(line_start: str, numbers_line: List[float]) -> str:
    """
    Create a formatted line for jetto.sin file.
    
    Parameters
    ----------
    line_start : str
        Line prefix
    numbers_line : list
        Numbers to format
        
    Returns
    -------
    str
        Formatted line
    """
    line = line_start
    for number in numbers_line:
        line += f'  {number:.3E}  ,'
    line += '\n'
    return line


def find_index_start(start: str, read_data: List[str]) -> Optional[int]:
    """
    Find line index after a marker line.
    
    Parameters
    ----------
    start : str
        Marker string
    read_data : list
        Lines to search
        
    Returns
    -------
    int or None
        Index of first line after marker, or None if not found
    """
    for i, line in enumerate(read_data):
        if line.startswith(start):
            return i + 1
    return None


def remove_lines_after_marker(marker: str, read_data: List[str]) -> List[str]:
    """
    Remove continuation lines after a marker line.
    
    Parameters
    ----------
    marker : str
        Marker string
    read_data : list
        Lines to process
        
    Returns
    -------
    list
        Processed lines
    """
    index_start = find_index_start(marker, read_data)
    
    if not index_start:
        return read_data
    
    # Find end of continuation lines (lines starting with spaces)
    index_end = 0
    for i, line in enumerate(read_data[index_start:]):
        if not line.startswith('        '):
            index_end = i
            break
    
    index_end += index_start
    return read_data[:index_start] + read_data[index_end:]


def insert_lines_jettosin(marker: str, read_data: List[str], line_arrays: List[List[float]]) -> List[str]:
    """
    Insert formatted lines in jetto.sin file.
    
    Parameters
    ----------
    marker : str
        Field marker
    read_data : list
        File lines
    line_arrays : list
        Arrays of values to insert
        
    Returns
    -------
    list
        Updated file lines
    """
    new_read_data = read_data[:]
    
    for index, line in enumerate(read_data):
        if line.startswith(marker):
            new_read_data[index] = create_single_line_jettosin(f'{marker}  = ', line_arrays[0])
            for jndex, line_array in enumerate(line_arrays[1:]):
                new_read_data.insert(index + jndex + 1,
                                    create_single_line_jettosin('        ', line_array))
    
    return new_read_data


def modify_jettosin_multiline(path_jetto_sin: str, fields_array: List[str], numbers_array: List[List[float]]) -> None:
    """
    Modify multi-line fields in jetto.sin file.
    
    Parameters
    ----------
    path_jetto_sin : str
        Path to jetto.sin file
    fields_array : list
        Field markers to update
    numbers_array : list
        Arrays for each field
    """
    values, times = numbers_array[0], numbers_array[1]
    
    values_jettosin = adapt_to_jettosin(values)
    times_jettosin = adapt_to_jettosin(times)
    
    line_arrays_values = reshape_array(values_jettosin, 5)
    line_arrays_times = reshape_array(times_jettosin, 5)
    
    read_data = read_file(path_jetto_sin)
    read_data = remove_lines_after_marker('  SPEFLX', read_data)
    read_data = remove_lines_after_marker('  TINFLX', read_data)
    read_data = insert_lines_jettosin('  SPEFLX', read_data, line_arrays_values)
    read_data = insert_lines_jettosin('  TINFLX', read_data, line_arrays_times)
    
    write_file(path_jetto_sin, read_data)


def modify_jettosin_time_polygon(path_jetto_sin: str, time_start: float, time_end: float, puff_value: float) -> None:
    """
    Setup time-dependent impurity puff polygon in jetto.sin.
    
    Parameters
    ----------
    path_jetto_sin : str
        Path to jetto.sin file
    time_start : float
        Simulation start time
    time_end : float
        Simulation end time
    puff_value : float
        Reference puff value
    """
    times = [time_start, time_start + 0.05, time_end]
    values = [0.0, puff_value, puff_value]
    
    modify_jettosin_multiline(path_jetto_sin, ['  SPEFLX', '  TINFLX'], [values, times])


def modify_jettosin_time_polygon_single(path_jetto_sin: str, fields: List[str], numbers: List[float]) -> None:
    """
    Modify simple fields in jetto.sin file.
    
    Parameters
    ----------
    path_jetto_sin : str
        Path to jetto.sin file
    fields : list
        Field names
    numbers : list
        Field values
    """
    lines = read_file(path_jetto_sin)
    
    for field, number in zip(fields, numbers):
        for index, line in enumerate(lines):
            if line.startswith(field):
                numbers_line = [number] + [0.0] * 4
                lines[index] = create_single_line_jettosin(f'{field}   =', numbers_line)
    
    write_file(path_jetto_sin, lines)


def _apply_jset_time_polygon(run_name: str, prefix: str, times: List[float], values: List[float]) -> None:
    """Shared helper to write time-polygon structures into jetto.jset."""
    config_fields = [f'{prefix}.tpoly.option']
    config_fields.extend([f'{prefix}.tpoly.select[{i}]' for i in range(len(times))])
    config_fields.extend([f'{prefix}.tpoly.time[{i}]' for i in range(len(times))])
    config_fields.extend([f'{prefix}.tpoly.value[{i}]' for i in range(len(times))])

    config_values = ['Time Dependent']
    config_values.extend(['true'] * len(times))
    config_values.extend([str(t) for t in times])
    config_values.extend([str(v) for v in values])

    for field, value in zip(config_fields, config_values):
        modify_jset_line(run_name, field, value)


def modify_jset_time_polygon(run_name: str, time_start: float, time_end: float) -> None:
    """
    Setup time-dependent maximum timestep polygon in jset.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    time_start : float
        Simulation start time
    time_end : float
        Simulation end time
    """
    times = [time_start, time_start + 0.001, time_start + 0.01, time_start + 0.02, time_end]
    values = [2.0e-6, 4.0e-6, 2.0e-4, 1.0e-3, 1.0e-3]

    _apply_jset_time_polygon(run_name, 'SetUpPanel.maxTimeStep', times, values)


def modify_jset_time_polygon_puff(run_name: str, time_start: float, time_end: float, puff_value: float) -> None:
    """
    Setup time-dependent impurity puff polygon in jset.
    
    Parameters
    ----------
    run_name : str
        Baserun directory name
    time_start : float
        Simulation start time
    time_end : float
        Simulation end time
    puff_value : float
        Reference puff value
    """
    times = [time_start, time_start + 0.1, time_end]
    values = [0.0, puff_value, puff_value]

    _apply_jset_time_polygon(run_name, 'SancoBCPanel.Species1NeutralInflux', times, values)


# ============================================================================
# EXTRANAMELIST UTILITIES
# ============================================================================


def get_extraname_fields(path: str) -> Dict[str, List[str]]:
    """
    Extract all extranamelist fields from jset file.
    
    Parameters
    ----------
    path : str
        Baserun directory path
        
    Returns
    -------
    dict
        Dictionary mapping field names to value lists
    """
    jset_path = f'{path}/jetto.jset'
    
    # Read jset file and find extranamelist entries
    with open(jset_path) as f:
        lines = f.readlines()
    
    # Find number of active extra elements
    num_extra_elements = 0
    for line in lines:
        if 'OutputExtraNamelist.selItems.rows' in line:
            try:
                num_extra_elements = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
    
    # Extract entries
    extranamelist = {}
    for line in lines:
        if 'OutputExtraNamelist.selItems.cell' not in line:
            continue
        
        try:
            # Parse: OutputExtraNamelist.selItems.cell[i][j] : value
            parts = line.split('[')
            idx1 = int(parts[1].split(']')[0])
            idx2 = int(parts[2].split(']')[0])
            value = line.split(':')[1].strip()
            
            if idx1 > num_extra_elements:
                continue
            
            if idx2 == 0:  # Field name
                key = value
                extranamelist[key] = []
            elif idx2 == 2:  # Value
                if key in extranamelist:
                    extranamelist[key].append(value)
            elif idx2 == 3:  # Active flag
                if value == 'false' and key in extranamelist:
                    del extranamelist[key]
        except (ValueError, IndexError):
            continue
    
    return extranamelist


def add_extraname_fields(extranamelist: Dict[str, List[str]], key: str, values: List[str]) -> Dict[str, List[str]]:
    """
    Add or update an extranamelist field.
    
    Parameters
    ----------
    extranamelist : dict
        Existing extranamelist dictionary
    key : str
        Field name
    values : list
        Field values (list of strings)
        
    Returns
    -------
    dict
        Updated extranamelist (sorted by key)
    """
    extranamelist[key] = values
    return {k: v for k, v in sorted(extranamelist.items())}


def put_extraname_fields(path: str, extranamelist: Dict[str, List[str]]) -> None:
    """
    Write extranamelist fields to jset file.
    
    Parameters
    ----------
    path : str
        Baserun directory path
    extranamelist : dict
        Extranamelist dictionary to write
    """
    jset_path = f'{path}/jetto.jset'
    read_data = read_file(jset_path)
    
    # Remove old extranamelist entries (only OutputExtraNamelist, not Sanco!)
    filtered = [line for line in read_data if not ('OutputExtraNamelist.selItems.cell' in line and 'Sanco' not in line)]
    
    # Find insertion point (before NeutralSourcePanel)
    insert_index = 0
    for i, line in enumerate(filtered):
        if 'NeutralSourcePanel' in line:
            insert_index = i + 1
            break
    
    # Create new extranamelist entries
    new_entries = []
    namelist_start = 'OutputExtraNamelist.selItems.cell'
    
    for ilist, (key, vals) in enumerate(extranamelist.items()):
        for icol in range(4):
            field = f'{namelist_start}[{ilist}][{icol}]'
            spaces = ' ' * (60 - len(field))
            
            if icol == 0:  # Field name
                new_entries.append(f'{field}{spaces}: {key}\n')
            elif icol == 1:  # (unused)
                new_entries.append(f'{field}{spaces}: \n')
            elif icol == 2:  # Values
                if len(vals) == 1:
                    new_entries.append(f'{field}{spaces}: {vals[0]}\n')
                else:
                    val_str = '(' + ' ,'.join(vals) + ' ) \n'
                    new_entries.append(f'{field}{spaces}: {val_str}')
            elif icol == 3:  # Active flag
                new_entries.append(f'{field}{spaces}: true \n')
    
    # Insert new entries
    filtered.insert(insert_index, ''.join(new_entries))
    
    # Update row count (only for OutputExtraNamelist, not SancoOutputExtraNamelist!)
    for i, line in enumerate(filtered):
        if 'OutputExtraNamelist.selItems.rows' in line and 'Sanco' not in line:
            filtered[i] = line.split(':')[0] + f': {len(extranamelist)}\n'
    
    write_file(jset_path, filtered)


# ============================================================================
# PHYSICS CALCULATION UTILITIES
# ============================================================================


def calculate_impurity_puff(impurity_puff_ref: float, zeff: float, line_ave_density: float) -> float:
    """
    Calculate scaled impurity puff value.
    
    Uses empirical scaling based on line-averaged density and Zeff.
    
    Parameters
    ----------
    impurity_puff_ref : float
        Reference puff value
    zeff : float
        Effective charge
    line_ave_density : float
        Line-averaged density [10^19 m^-3]
        
    Returns
    -------
    float
        Scaled puff value
    """
    line_ave_density_ref = 1.0e19
    zeff_ref = 1.0
    
    scale_len_ave = line_ave_density / line_ave_density_ref
    scale_zeff = (zeff - zeff_ref) * 10
    
    return scale_len_ave * scale_zeff * impurity_puff_ref


# ============================================================================
# IMAS INTERFACE UTILITIES
# ============================================================================


def get_backend(db: str, shot: int, run: int, username: Optional[str] = None) -> int:
    """
    Determine IMAS backend for database entry.
    
    Tries HDF5 first, falls back to MDSPlus if not available.
    
    Parameters
    ----------
    db : str
        Database name
    shot : int
        Shot number
    run : int
        Run number
    username : str, optional
        IMAS user name (default: current user)
        
    Returns
    -------
    int
        IMAS backend constant (HDF5_BACKEND or MDSPLUS_BACKEND)
    """
    if imas is None:
        raise ImportError("IMAS module is not available")
    
    if username is None:
        username = getpass.getuser()
    
    imas_backend = imasdef.HDF5_BACKEND
    data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)
    
    op = data_entry.open()
    if op[0] < 0:
        imas_backend = imasdef.MDSPLUS_BACKEND
    
    data_entry.close()
    
    # Verify backend works
    data_entry = imas.DBEntry(imas_backend, db, shot, run, user_name=username)
    op = data_entry.open()
    if op[0] < 0:
        print(f'ERROR: IDS {db}/{shot}/{run} not found in either backend')
        exit()
    
    data_entry.close()
    return imas_backend


def open_and_get_ids(db: str, shot: int, run: int, ids_name: str,
                    username: Optional[str] = None, backend: Optional[int] = None) -> Any:
    """
    Open IMAS database and get specified IDS.
    
    Parameters
    ----------
    db : str
        Database name
    shot : int
        Shot number
    run : int
        Run number
    ids_name : str
        IDS name (e.g., 'equilibrium', 'core_profiles')
    username : str, optional
        IMAS user name
    backend : int, optional
        IMAS backend constant
        
    Returns
    -------
    IDS object
        The requested IDS
    """
    if backend is None:
        backend = get_backend(db, shot, run)
    if username is None:
        username = getpass.getuser()
    
    data_entry = imas.DBEntry(backend, db, shot, run, user_name=username)
    op = data_entry.open()
    
    if op[0] < 0:
        cp = data_entry.create()
        if cp[0] == 0:
            print("Data entry created")
    else:
        print("Data entry opened")
    
    ids_obj = data_entry.get(ids_name)
    data_entry.close()
    
    return ids_obj


def fit_and_substitute(x_old: np.ndarray, x_new: np.ndarray, data_old: np.ndarray) -> np.ndarray:
    """
    Interpolate data to new x coordinates using spline.
    
    Parameters
    ----------
    x_old : np.ndarray
        Original x coordinates
    x_new : np.ndarray
        New x coordinates
    data_old : np.ndarray
        Data values at old x
        
    Returns
    -------
    np.ndarray
        Interpolated data at new x (clipped to reasonable values)
    """
    f = interp1d(x_old, data_old, fill_value='extrapolate')
    interpolated = np.array(f(x_new))
    
    # Remove unreasonable extrapolated values
    interpolated[interpolated > 1.0e25] = 0
    
    return interpolated


# ============================================================================
# Deuterium/Tritium Isotope Management
# ============================================================================

def correct_isotope_composition(run_name: str) -> None:
    """
    Correct deuterium/tritium isotope fraction composition to sum to 1.0.
    
    Reads the first isotope fraction from JSET, calculates the second fraction
    as the complement, and writes both to JETO and JSET files.
    
    Parameters
    ----------
    run_name : str
        Path to the run directory
        
    Returns
    -------
    None
    """
    line_start_jettoin = '  DNFRAC ='
    line_start_jset = 'EquationsPanel.ionDens[0].fraction'
    
    # Read first isotope fraction from JSET
    value = read_jset_line(run_name, line_start_jset)
    values = [value, str(1 - float(value))]
    
    # Write corrected second isotope fraction to JSET
    line_start_jset = 'EquationsPanel.ionDens[1].fraction'
    modify_jset_line(run_name, line_start_jset, values[1])
    
    # Write both fractions to JETTO.IN
    values = [float(value) for value in values]
    modify_jettoin_line(run_name, line_start_jettoin, values)


def correct_isotope_composition_all(folder_path: str) -> None:
    """
    Correct isotope composition for all runs in a directory.
    
    Recursively processes all subdirectories starting with 'run_' and applies
    isotope composition correction to each.
    
    Parameters
    ----------
    folder_path : str
        Path to the directory containing run folders
        
    Returns
    -------
    None
    """
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    for subfolder in subfolders:
        last_level = subfolder.split('/')[-1]
        if last_level.startswith('run_'):
            correct_isotope_composition(subfolder)


# ============================================================================
# Configuration Management (Lookup JSON)
# ============================================================================

def add_item_lookup(name: str, name_jset: str, namelist: str, 
                   name_type: str, name_dim: str, path: str) -> None:
    """
    Add a new item entry to lookup.json configuration file.
    
    Inserts a new parameter definition into the lookup.json file that maps
    parameter names to JSET IDs and namelist fields.
    
    Parameters
    ----------
    name : str
        Parameter name
    name_jset : str
        JSET ID (or 'null' if not applicable)
    namelist : str
        Namelist name containing the field
    name_type : str
        Data type ('float', 'int', etc.)
    name_dim : str
        Parameter dimension specification
    path : str
        Path to directory containing lookup.json
        
    Returns
    -------
    None
    """
    with open(f"{path}/lookup.json", 'r') as f:
        lines = f.readlines()
    
    # Build new lookup entry
    jset_value = name_jset if name_jset == "null" else f'"{name_jset}"'
    new_item = [
        f' "{name}": {{ \n',
        f'  "jset_id": {jset_value},\n',
        '  "nml_id": { \n',
        f'   "namelist": "{namelist}",\n',
        f'   "field":  "{name.upper()}" \n',
        '  }, \n',
        f'  "type": "{name_type}",\n',
        f'  "dimension": "{name_dim}" \n',
        ' }, \n'
    ]
    
    # Insert after first line
    lines.insert(1, new_item)
    
    with open(f"{path}/lookup.json", 'w') as f:
        for line in lines:
            if isinstance(line, list):
                f.writelines(line)
            else:
                f.write(line)


# ============================================================================
# Utility Functions (Miscellaneous)
# ============================================================================

def get_put_namelist(path: str) -> None:
    """
    Test utility: Read, modify and write extranamelist configuration.
    
    This function is for testing purposes. It demonstrates reading an 
    extranamelist, adding/modifying fields, and writing back to file.
    
    Parameters
    ----------
    path : str
        Path to the run directory
        
    Returns
    -------
    None
    """
    # Read current extranamelist
    extranamelist = get_extraname_fields(path)
    
    # Add or modify fields
    extranamelist = add_extraname_fields(extranamelist, 'DNEFLFB', ['1e13', '2e13'])
    extranamelist = add_extraname_fields(extranamelist, 'DTNEFLFB', ['1', '2'])
    
    # Write back to file
    put_extraname_fields(path, extranamelist)


# ============================================================================
# Multi-Shot Campaign Runner
# ============================================================================

def run_all_shots(
    json_input: dict,
    instructions_list: list,
    shot_numbers: list,
    runs_input: list,
    runs_start: list,
    times: list,
    first_number: int,
    generator_name: str,
    db: str = 'tcv',
    misallignements: list | None = None,
    setup_time_polygon_flag: bool = True,
    esco_timesteps: int = 100,
    set_sep_boundaries: bool = False,
    boundary_conditions: dict | None = None,
    run_name_end: str = 'hfps',
    change_impurity_puff_flag: bool = False,
    select_impurities_from_ids_flag: bool = True,
    add_extra_transport_flag: bool = False,
    density_feedback: bool = True
) -> None:
    """
    Execute integrated modelling runs for multiple experimental shots.
    
    Loops through a list of shots and creates/configures IntegratedModellingRuns
    instances for each, handling errors gracefully for missing data.
    
    Parameters
    ----------
    json_input : dict
        JSON configuration dictionary with misalignment schema
    instructions_list : list
        List of instruction dictionaries
    shot_numbers : list
        List of shot IDs to process
    runs_input : list
        List of run input configurations (one per shot)
    runs_start : list
        List of run start parameters (one per shot)
    times : list
        List of [time_start, time_end] pairs (one per shot)
    first_number : int
        Starting run number
    generator_name : str
        Name of the run generator
    db : str, optional
        Database name (default: 'tcv')
    misallignements : list | None, optional
        List of [x, y, z] misalignment factors (default: all 1.0)
    setup_time_polygon_flag : bool, optional
        Whether to setup time polygon (default: True)
    esco_timesteps : int, optional
        Number of ESCO timesteps (default: 100)
    set_sep_boundaries : bool, optional
        Whether to set separatrix boundaries (default: False)
    boundary_conditions : dict | None, optional
        Dictionary of boundary condition overrides (default: {})
    run_name_end : str, optional
        Suffix for run names (default: 'hfps')
    change_impurity_puff_flag : bool, optional
        Whether to change impurity puff (default: False)
    select_impurities_from_ids_flag : bool, optional
        Whether to select impurities from IDS (default: True)
    add_extra_transport_flag : bool, optional
        Whether to add extra transport (default: False)
    density_feedback : bool, optional
        Whether to enable density feedback (default: True)
        
    Returns
    -------
    None
    """
    if boundary_conditions is None:
        boundary_conditions = {}
    
    if not misallignements:
        misallignements = [[1, 1, 1]] * len(shot_numbers)
    
    run_number = first_number
    
    for shot_number, run_input, run_start, time, misallignment in zip(
        shot_numbers, runs_input, runs_start, times, misallignements
    ):
        # Format run name with padding
        if run_number < 100:
            run_name = f'run0{run_number}_{shot_number}_{run_name_end}'
        else:
            run_name = f'run{run_number}_{shot_number}_{run_name_end}'
        
        run_number += 1
        
        # Update misalignment in config
        json_input['misalignment']['schema'] = misallignment
        
        # Create and configure run
        run_test = IntegratedModellingRuns(
            shot_number,
            instructions_list,
            generator_name,
            run_name,
            run_input=run_input,
            run_start=run_start,
            json_input=json_input,
            db=db,
            esco_timesteps=esco_timesteps,
            output_timesteps=100,
            time_start=time[0],
            time_end=time[1],
            setup_time_polygon_flag=setup_time_polygon_flag,
            change_impurity_puff_flag=change_impurity_puff_flag,
            setup_time_polygon_impurities_flag=True,
            add_extra_transport_flag=add_extra_transport_flag,
            select_impurities_from_ids_flag=select_impurities_from_ids_flag,
            density_feedback=density_feedback,
            force_run=True,
            force_input_overwrite=True,
            set_sep_boundaries=set_sep_boundaries,
            boundary_conditions=boundary_conditions
        )
        
        try:
            run_test.setup_create_compare()
        except Exception as mde:  # Catch MissingDataError or other exceptions
            print(f'Caught an exception: {mde}')
            print(f'No experimental data available to set run for shot {shot_number}')
            continue
    
    # Explicitly clear to prevent survival outside function
    boundary_conditions.clear()


if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Use: from prepare_im_runs import IntegratedModellingRuns")

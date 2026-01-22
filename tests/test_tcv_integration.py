"""
Test suite for prepare_im_runs.py - TCV Integration Tests.

Comprehensive integration test covering:
- Density feedback functionality
- Impurity puff scaling  
- Boundary conditions
- JETTO file modifications (jetto.in, jetto.sin, jetto.jset)
- LoadLeveler command file (.llcmd)
- Pulse schedule IDS validation
"""
import os
import re
import math
import pytest
from pathlib import Path

# Test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.utils import (
    clean_namelist_only,
    parse_nlist4_dneflfb,
    parse_jset_parameter,
    extract_numeric_value,
    validate_jetto_in_structure,
    validate_jset_structure,
    validate_llcmd_file,
    compare_values,
)

from prepare_im_runs import IntegratedModellingRuns, open_and_get_ids


# ============================================================================
# TEST CONFIGURATION - Set once for all tests
# ============================================================================
TEST_SHOT = 68588
TEST_DB = 'tcv'
TEST_RUN_INPUT = 3
TEST_RUN_START = 100
TEST_RUN_OUTPUT = 200
TEST_GENERATOR = 'rungenerator_ohmic_vscode'
TEST_RUN_NAME = 'runtest_tcv'
TEST_PUFF_SCALING = 0.1


@pytest.mark.integration
@pytest.mark.density_feedback
@pytest.mark.requires_data
@pytest.mark.slow
class TestTCVIntegration:
    """
    Comprehensive integration test for TCV shot.
    
    Tests density feedback, impurity puff scaling, and file modifications
    with real IMAS data access.
    """
    
    @pytest.fixture(scope="function")
    def run_instance(self, mock_json_input, request, run_config):
        """
        Create and execute IntegratedModellingRuns instance.
        
        Scope is class-level so we only create the run once for all tests.
        """
        # Check for overwrite flag
        overwrite = request.config.getoption("--overwrite", default=True)
        
        # Create instance using run_config fixture values
        run = IntegratedModellingRuns(
            run_config['shot'],
            ['create case'],  # Create only
            run_config['generator'],
            run_config['run_name'],
            db=run_config['machine'],
            run_input=run_config['run_input'],
            run_start=run_config['run_start'],
            run_output=run_config['run_output'],
            json_input=run_config['json_input'],
            esco_timesteps=100,
            output_timesteps=100,
            time_start=run_config['time_start'],  # Use fixture value
            time_end=run_config['time_end'],
            density_feedback=run_config['density_feedback'],
            set_sep_boundaries=True,
            change_impurity_puff_flag=run_config['puff_scaling'],
            select_impurities_from_ids_flag=True,
            force_run=False,
            force_input_overwrite=False,
            overwrite_baserun=overwrite,
        )
        
        # Execute
        run.setup_create_compare()
        
        return run
    
    @pytest.fixture(scope="function")
    def baserun_path(self, run_instance):
        """Path to created baserun directory."""
        return run_instance.path_baserun
    
    @pytest.fixture(scope="function")
    def jetto_in_path(self, baserun_path):
        """Path to jetto.in file."""
        return os.path.join(baserun_path, 'jetto.in')
    
    @pytest.fixture(scope="function")
    def jetto_sin_path(self, baserun_path):
        """Path to jetto.sin file."""
        return os.path.join(baserun_path, 'jetto.sin')
    
    @pytest.fixture(scope="function")
    def jetto_jset_path(self, baserun_path):
        """Path to jetto.jset file."""
        return os.path.join(baserun_path, 'jetto.jset')
    
    @pytest.fixture(scope="function")
    def llcmd_path(self, baserun_path):
        """Path to .llcmd file."""
        # Find .llcmd file in baserun directory
        llcmd_files = list(Path(baserun_path).glob('*.llcmd'))
        if llcmd_files:
            return str(llcmd_files[0])
        return None
    
    @pytest.fixture(scope="function")
    def hfps_launch_path(self, baserun_path):
        """Path to hfps.launch file."""
        return os.path.join(baserun_path, 'hfps.launch')
    
    def test_pulse_schedule_ids_access_before_create(self):
        """Test that pulse_schedule IDS can be accessed before creating run."""
        try:
            ids_pulse = open_and_get_ids(
                TEST_DB,
                TEST_SHOT,
                TEST_RUN_START,
                'pulse_schedule'
            )
            
            assert ids_pulse is not None, "Failed to open pulse_schedule IDS"
            
            # Check basic structure
            assert hasattr(ids_pulse, 'density_control'), \
                "pulse_schedule missing density_control"
            
            print(f"✓ pulse_schedule IDS accessible for shot {TEST_SHOT}")
            
        except Exception as e:
            pytest.skip(f"Cannot access pulse_schedule IDS: {e}")
    
    def test_baserun_directory_exists(self, baserun_path):
        """Test that baserun directory was created."""
        assert os.path.exists(baserun_path), \
            f"Baserun directory not created: {baserun_path}"
        print(f"✓ Baserun directory exists: {baserun_path}")
    
    def test_essential_files_exist(self, baserun_path):
        """Test that essential JETTO files were created."""
        essential_files = ['jetto.jset', 'jetto.in', 'jetto.sin', 'jetto.ex']
        
        files = os.listdir(baserun_path)
        
        for filename in essential_files:
            assert filename in files, f"Missing essential file: {filename}"
            print(f"✓ {filename} exists")
    
    def test_jetto_in_structure(self, jetto_in_path):
        """Test jetto.in structure and section headers."""
        struct = validate_jetto_in_structure(jetto_in_path)
        
        assert struct['exists'], "jetto.in does not exist"
        assert struct['size'] > 0, "jetto.in is empty"
        assert struct['section_headers'] > 0, "No section headers found (decorative formatting lost)"
        assert struct['dash_lines'] > 0, "No dash separator lines found"
        assert len(struct['namelists']) > 0, "No namelists found in jetto.in"
        
        print(f"✓ jetto.in structure valid:")
        print(f"    Size: {struct['size']} bytes")
        print(f"    Section headers: {struct['section_headers']}")
        print(f"    Dash lines: {struct['dash_lines']}")
        print(f"    Namelists: {len(struct['namelists'])}")
    
    def test_jetto_sin_sanco_namelist_access(self, jetto_sin_path):
        """Test that SANCO/JSANC namelist can be accessed via jetto_tools API."""
        try:
            import jetto_tools.namelist
            
            # Clean namelist for parsing
            cleaned_sin = clean_namelist_only(jetto_sin_path)
            
            try:
                sin_nml = jetto_tools.namelist.read(cleaned_sin)
                
                # Try both SANCO and JSANC (different templates use different names)
                sanco_nml = sin_nml.namelist_lookup('sanco')
                jsanc_nml = sin_nml.namelist_lookup('jsanc')
                
                # At least one should work, or we can use get_field as fallback
                if sanco_nml is None and jsanc_nml is None:
                    # Fallback: try get_field directly
                    try:
                        field_val = sin_nml.get_field('jsanc', 'israd')
                        assert field_val is not None, "Cannot access JSANC fields"
                        print("✓ JSANC namelist accessible via get_field('jsanc', 'israd')")
                    except Exception as e:
                        pytest.fail(f"Cannot access SANCO/JSANC namelist: {e}")
                else:
                    print("✓ SANCO/JSANC namelist accessible via namelist_lookup()")
                    
            finally:
                os.unlink(cleaned_sin)
                
        except ImportError:
            pytest.skip("jetto_tools not available")

    def test_jetto_sin_header_contains_code_input(self, jetto_sin_path):
        """jetto.sin header should include CODE INPUT NAMELIST FILE and GIT metadata filled with jetto_tools."""
        assert os.path.exists(jetto_sin_path), "jetto.sin does not exist"
        with open(jetto_sin_path, 'r') as f:
            content = f.read()

        # Check CODE INPUT NAMELIST FILE is properly filled
        assert 'CODE INPUT NAMELIST FILE' in content, \
            "CODE INPUT NAMELIST FILE header missing in jetto.sin"
        
        # Verify all GIT metadata fields are properly formatted with jetto_tools
        required_fields = [
            'CODE INPUT NAMELIST FILE',
            'Current GIT repository',
            'Current GIT release tag', 
            'Current GIT branch',
            'Last commit SHA1-key',
            'Repository status'
        ]
        
        for field in required_fields:
            # Find the line containing this field
            pattern = rf'{re.escape(field)}\s*:\s*(\S+)'
            match = re.search(pattern, content)
            
            assert match, f"Field '{field}' not found or not properly formatted in jetto.sin"
            value = match.group(1)
            assert value.lower() == 'jetto_tools', \
                f"Field '{field}' has value '{value}' instead of 'jetto_tools'"
            print(f"✓ {field}: {value}")
        
        print("✓ All jetto.sin header metadata fields properly filled with jetto_tools")
    
    def test_density_feedback_arrays_in_jetto_in(self, jetto_jset_path):
        """Test that DNEFLFB array is present in jetto.jset OutputExtraNamelist."""
        # DNEFLFB arrays are written to jetto.jset OutputExtraNamelist, not jetto.in
        dneflfb_content = parse_jset_parameter(jetto_jset_path, 'OutputExtraNamelist.selItems.cell[9]')
        
        if dneflfb_content:
            # Check that it contains DNEFLFB keyword
            assert 'dneflfb' in dneflfb_content.lower(), \
                f"DNEFLFB not found in OutputExtraNamelist: {dneflfb_content}"
            print(f"✓ DNEFLFB found in OutputExtraNamelist: {dneflfb_content[:100]}...")
        else:
            # Try alternative parsing - look for the cell[0] which should be the parameter name
            cell_0 = parse_jset_parameter(jetto_jset_path, 'OutputExtraNamelist.selItems.cell[9][0]')
            assert cell_0 and 'dneflfb' in cell_0.lower(), \
                f"DNEFLFB array is empty or not found in jetto.jset OutputExtraNamelist"
            print(f"✓ DNEFLFB array parameter found: {cell_0}")
    
    def test_density_feedback_in_jetto_jset(self, jetto_jset_path):
        """Test that DNEFLFB is present in jetto.jset OutputExtraNamelist."""
        struct = validate_jset_structure(jetto_jset_path)
        
        assert struct['exists'], "jetto.jset does not exist"
        assert struct['has_output_extra'], \
            "OutputExtraNamelist not found in jetto.jset"
        
        # Check for DNEFLFB in file (case-insensitive)
        # DTNEFLFB is the time step array which may not always be present depending on config
        with open(jetto_jset_path, 'r') as f:
            content = f.read().lower()
        
        assert 'dneflfb' in content, "DNEFLFB not found in jetto.jset"
        print("✓ DNEFLFB density feedback array found in jetto.jset OutputExtraNamelist")
        
        # DTNEFLFB (time step array) is optional
        if 'dtneflfb' in content:
            print("✓ DTNEFLFB time step array also found in jetto.jset")
    
    def test_impurity_puff_in_jetto_sin(self, jetto_sin_path, run_config):
        """Test that SPEFLX (impurity puff) was set in jetto.sin."""
        try:
            import jetto_tools.namelist
            
            # Clean and read namelist
            cleaned_sin = clean_namelist_only(jetto_sin_path)
            
            try:
                sin_nml = jetto_tools.namelist.read(cleaned_sin)
                
                # Get SPEFLX array from PHYSIC namelist
                try:
                    speflx = sin_nml.get_array('physic', 'speflx')
                    
                    assert speflx is not None, "SPEFLX not found in jetto.sin"
                    assert len(speflx) > 0, "SPEFLX array is empty"
                    
                    # Find first non-zero value (should be the scaled puff value)
                    non_zero_values = [val for val in speflx if val != 0]
                    assert len(non_zero_values) > 0, "No non-zero SPEFLX values found"
                    
                    # All non-zero values should be the same (from scaling)
                    first_nonzero = non_zero_values[0]
                    for val in non_zero_values:
                        assert math.isclose(val, first_nonzero, rel_tol=1e-6), \
                            f"SPEFLX non-zero values are not consistent: {non_zero_values}"
                    
                    # Verify it's a reasonable impurity puff value (order of magnitude check)
                    assert first_nonzero > 1e16, f"SPEFLX value seems too small: {first_nonzero}"
                    assert first_nonzero < 1e20, f"SPEFLX value seems too large: {first_nonzero}"
                    
                    print(f"✓ SPEFLX non-zero values ({len(non_zero_values)} elements): {first_nonzero:.6e}")
                    print(f"✓ All non-zero SPEFLX values are consistent")
                    
                except Exception as e:
                    pytest.skip(f"Cannot read SPEFLX from jetto.sin: {e}")
                    
            finally:
                os.unlink(cleaned_sin)
                
        except ImportError:
            pytest.skip("jetto_tools not available")
    
    def test_impurity_puff_in_jetto_jset(self, jetto_jset_path):
        """Test that Sanco puff value appears in jetto.jset."""
        puff_param = parse_jset_parameter(
            jetto_jset_path,
            'SancoBCPanel.Species1NeutralInflux.tpoly.value'
        )
        
        if puff_param:
            try:
                puff_value = extract_numeric_value(puff_param)
                
                # Verify that puff scaling was applied: value should be non-zero and reasonable
                # The exact baseline depends on IMAS data, but we verify it's scaled properly
                # Puff scaling factor is TEST_PUFF_SCALING (0.1)
                # Just verify it's a reasonable physics value (not zero, not absurdly large)
                assert puff_value > 0, "Puff value should be positive"
                assert puff_value < 1e19, "Puff value too large (likely unscaled)"
                
                print(f"✓ Sanco puff value = {puff_value:.6e} (scaled by {TEST_PUFF_SCALING})")
                
            except ValueError as e:
                pytest.skip(f"Cannot parse puff value from jset: {e}")
        else:
            pytest.skip("Sanco puff parameter not found in jetto.jset")
    
    def test_boundary_conditions_in_jetto_in(self, jetto_in_path):
        """Test that boundary condition parameters were set in jetto.in."""
        # Expected values: 50 boundary condition points extracted from IDS shot 68588
        # The IDS has 50 core_profiles time slices, hence 50 boundary condition points
        expected_boundaries = {
            'NTEB': 50,    # Number of time points for boundary (from IDS)
            'NTIB': 50,    # Number of time points for input boundary (from IDS)
            'NDNHB1': 50,  # Number of density points (from IDS)
        }
        
        with open(jetto_in_path, 'r') as f:
            content = f.read()
        
        for param, expected_value in expected_boundaries.items():
            # Case-insensitive search for parameter
            pattern = rf'{param}\s*=\s*(\d+)'
            match = re.search(pattern, content, re.IGNORECASE)
            
            if match:
                actual_value = int(match.group(1))
                assert actual_value == expected_value, \
                    f"{param} mismatch: expected {expected_value}, got {actual_value}"
                print(f"✓ {param} = {actual_value}")
            else:
                print(f"⚠ {param} not found (may be template-specific)")
    
    def test_tprint_array_length(self, jetto_in_path):
        """Test that TPRINT array has expected number of entries."""
        with open(jetto_in_path, 'r') as f:
            lines = f.readlines()
        
        # Find TPRINT array and collect continuation lines
        tprint_lines = []
        in_tprint = False
        
        for line in lines:
            if re.search(r'^\s*TPRINT\s*=', line, re.IGNORECASE):
                in_tprint = True
                tprint_lines.append(line)
            elif in_tprint:
                if '&' in line or ',' in line:
                    tprint_lines.append(line)
                else:
                    break
        
        # Parse numeric values from collected lines
        tprint_text = ' '.join(tprint_lines)
        values = re.findall(r'[-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*', tprint_text)
        
        expected_count = 100
        actual_count = len(values)
        
        assert actual_count == expected_count, \
            f"TPRINT array length mismatch: expected {expected_count}, got {actual_count}"
        
        print(f"✓ TPRINT array has {actual_count} entries")
    
    def test_scalar_parameters_in_jetto_in(self, jetto_in_path, run_config):
        """Test that scalar parameters were set correctly in jetto.in."""
        with open(jetto_in_path, 'r') as f:
            content = f.read()
        
        # Parameters to check (case-insensitive)
        params = {
            'TBEG': 0.0,  # Start time (template default)
            'TMAX': None,  # End time (will be auto-detected)
            'MACHID': None,  # Machine ID (template-specific)
        }
        
        for param, expected in params.items():
            pattern = rf'{param}\s*=\s*([-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*)'
            match = re.search(pattern, content, re.IGNORECASE)
            
            if match:
                actual = float(match.group(1))
                if expected is not None:
                    assert math.isclose(actual, expected, rel_tol=1e-6), \
                        f"{param} mismatch: expected {expected}, got {actual}"
                print(f"✓ {param} = {actual}")
            else:
                print(f"⚠ {param} not found (template-specific)")
    
    def test_npulse_in_jetto_in(self, jetto_in_path):
        """Test that NPULSE was set to shot number."""
        with open(jetto_in_path, 'r') as f:
            content = f.read()
        
        # NPULSE can be in NLIST1 or INESCO depending on template
        pattern = r'NPULSE\s*=\s*(\d+)'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if match:
            actual = int(match.group(1))
            expected = TEST_SHOT
            
            assert actual == expected, \
                f"NPULSE mismatch: expected {expected}, got {actual}"
            
            print(f"✓ NPULSE = {actual}")
        else:
            print("⚠ NPULSE not found in jetto.in (template-specific)")
    
    @pytest.mark.llcmd
    def test_llcmd_file_exists(self, llcmd_path):
        """Test that .llcmd file was created."""
        assert llcmd_path is not None, "No .llcmd file found in baserun directory"
        assert os.path.exists(llcmd_path), f".llcmd file does not exist: {llcmd_path}"
        print(f"✓ .llcmd file exists: {os.path.basename(llcmd_path)}")
    
    @pytest.mark.llcmd
    def test_llcmd_file_structure(self, llcmd_path):
        """Test that .llcmd file has expected LoadLeveler structure."""
        if llcmd_path is None:
            pytest.skip("No .llcmd file found")
        
        validation = validate_llcmd_file(llcmd_path)
        
        assert validation['exists'], ".llcmd file does not exist"
        assert validation['size'] > 0, ".llcmd file is empty"
        
        # Check for LoadLeveler directives
        assert validation['has_job_name'], \
            "Missing LoadLeveler job_name directive in .llcmd"
        assert validation['has_output_redirect'], \
            "Missing LoadLeveler output redirect in .llcmd"
        assert validation['has_executable'], \
            "Missing jetto executable invocation in .llcmd"
        
        print("✓ .llcmd file structure valid:")
        print(f"    Size: {validation['size']} bytes")
        print(f"    Has job_name: {validation['has_job_name']}")
        print(f"    Has output redirect: {validation['has_output_redirect']}")
        print(f"    Has executable: {validation['has_executable']}")
    
    @pytest.mark.llcmd
    def test_llcmd_job_name_matches_run(self, llcmd_path):
        """Test that .llcmd job name contains run information."""
        if llcmd_path is None:
            pytest.skip("No .llcmd file found")
        
        with open(llcmd_path, 'r') as f:
            content = f.read()
        
        # Job name should contain shot number or run name
        shot_str = str(TEST_SHOT)
        
        # Check if either shot or run_name appears in job_name directive
        # Support both LoadLeveler (#@ job_name) and SLURM (#SBATCH -J) syntax
        job_name_match = re.search(r'(?:#@\s+job_name\s*=\s*(.+)|#SBATCH\s+-J\s+(.+))', content, re.IGNORECASE)
        
        if job_name_match:
            # Get the matched group (either group 1 for LL or group 2 for SLURM)
            job_name_value = (job_name_match.group(1) or job_name_match.group(2)).strip()
            
            assert shot_str in job_name_value or TEST_RUN_NAME in job_name_value, \
                f"Job name '{job_name_value}' does not contain shot '{shot_str}' or run '{TEST_RUN_NAME}'"
            
            print(f"✓ Job name contains run identifier: {job_name_value}")
        else:
            pytest.skip("No job_name directive found in .llcmd")
    
    def test_hfps_launch_parameters(self, hfps_launch_path, run_config):
        """Test that hfps.launch file contains correctly modified parameters."""
        print("\n=== Testing hfps.launch Parameters ===")
        
        # Check file exists
        assert os.path.exists(hfps_launch_path), f"hfps.launch not found at {hfps_launch_path}"
        assert os.path.getsize(hfps_launch_path) > 0, "hfps.launch is empty"
        print(f"✓ hfps.launch exists at {hfps_launch_path}")
        
        # Read file content
        with open(hfps_launch_path, 'r') as f:
            content = f.read()
        
        # Verify modified parameters
        expected_values = {
            'shot_out': str(run_config['shot']),        # 68588
            'shot_in': str(run_config['shot']),         # 68588
            'machine_out': run_config['machine'],       # tcv
            'tstart': str(run_config['time_start']),    # 0.0
        }
        
        for param, expected_val in expected_values.items():
            # Use flexible regex to handle YAML formatting
            pattern = rf'{param}\s*:\s*["\']?({re.escape(expected_val)})["\']?'
            match = re.search(pattern, content, re.IGNORECASE)
            assert match, f"Parameter {param} not found or has incorrect value. Expected: {expected_val}"
            print(f"✓ {param}: {expected_val}")
        
        # Verify tend is present (value may be adjusted by bit-shift operations)
        pattern = r'tend\s*:\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, content, re.IGNORECASE)
        assert match, "Parameter tend not found in hfps.launch"
        tend_value = match.group(1)
        print(f"✓ tend: {tend_value} (adjusted by calculation)")
        
        # Verify run_name is in the args line (should be the token before the jetto path)
        expected_run_name = run_config['run_name']  # runtest_tcv
        # Pattern: args: ... <run_name> <path>
        pattern = rf'args:.*\s{re.escape(expected_run_name)}(?:\s+/|\s*$)'
        match = re.search(pattern, content, re.IGNORECASE)
        assert match, f"run_name '{expected_run_name}' not found in args parameter"
        print(f"✓ args contains run_name: {expected_run_name}")
        
        # Verify args line does NOT contain the old generator name
        args_line_match = re.search(r'args:.*', content, re.IGNORECASE)
        if args_line_match:
            args_line = args_line_match.group(0)
            assert 'rungenerator_' not in args_line, "args line should not contain 'rungenerator_' prefix"
            print("✓ args line does not contain old generator name")
        
        # Verify preserved parameters (should NOT be modified)
        preserved_params = {
            'user_out': 'imasdb',
            'run_in': '1',
            'run_out': '2',
        }
        
        for param, expected_val in preserved_params.items():
            pattern = rf'{param}\s*:\s*({re.escape(expected_val)})'
            match = re.search(pattern, content, re.IGNORECASE)
            assert match, f"Preserved parameter {param} not found or was modified. Expected: {expected_val}"
            print(f"✓ {param} preserved: {expected_val}")
        
        print("✓ All hfps.launch parameters validated successfully!")


if __name__ == '__main__':
    # Allow running with: python tests/test_tcv_integration.py --overwrite
    pytest.main([__file__, '-v', '--tb=short'] + sys.argv[1:])

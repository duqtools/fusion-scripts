"""
Test suite for prepare_im_runs.py - JETTO Setup and Header Restoration.

Tests for:
- setup_jetto_simulation() functionality
- Section header restoration in jetto.in
- Basic workflow testing (setup/create/full modes)
"""
import os
import re
import pytest
from pathlib import Path

# Test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.utils import validate_jetto_in_structure

from prepare_im_runs import IntegratedModellingRuns


# ============================================================================
# TEST CONFIGURATION - Set once for all tests
# ============================================================================
TEST_SHOT = 64958
TEST_DB = 'tcv'
TEST_RUN_INPUT = 5
TEST_RUN_START = 1010
TEST_GENERATOR = 'rungenerator_ohmic_vscode'
TEST_RUN_NAME = 'runtest_64958'
TEST_TIME_START = 1.0
TEST_TIME_END = 1.5


@pytest.mark.headers
@pytest.mark.integration
class TestJETTOSetupAndHeaders:
    """
    Test JETTO simulation setup and section header restoration.
    
    Validates that setup_jetto_simulation() properly:
    - Creates jetto.in with correct structure
    - Preserves decorative section headers
    - Maintains dash separator lines
    """
    
    @pytest.fixture(scope="class")
    def run_instance(self, mock_json_input, request):
        """Create and execute IntegratedModellingRuns instance."""
        overwrite = request.config.getoption("--overwrite", default=False)
        
        run = IntegratedModellingRuns(
            TEST_SHOT,
            ['create case'],
            TEST_GENERATOR,
            TEST_RUN_NAME,
            run_input=TEST_RUN_INPUT,
            run_start=TEST_RUN_START,
            json_input=mock_json_input,
            db=TEST_DB,
            esco_timesteps=100,
            output_timesteps=100,
            time_start=TEST_TIME_START,
            time_end=TEST_TIME_END,
            overwrite_baserun=overwrite,
            # Disable all feature flags for minimal test
            density_feedback=False,
            select_impurities_from_ids_flag=False,
            setup_nbi_flag=False,
            setup_time_polygon_flag=False,
            change_impurity_puff_flag=False,
            add_extra_transport_flag=False,
            setup_time_polygon_impurities_flag=False,
            set_sep_boundaries=False
        )
        
        # Execute
        run.setup_create_compare(verbose=True)
        
        return run
    
    @pytest.fixture(scope="class")
    def baserun_path(self, run_instance):
        """Path to created baserun directory."""
        return run_instance.path_baserun
    
    @pytest.fixture(scope="class")
    def jetto_in_path(self, baserun_path):
        """Path to jetto.in file."""
        return os.path.join(baserun_path, 'jetto.in')
    
    def test_jetto_in_exists(self, jetto_in_path):
        """Test that jetto.in was created."""
        assert os.path.exists(jetto_in_path), \
            f"jetto.in not created: {jetto_in_path}"
        print(f"✓ jetto.in exists: {jetto_in_path}")
    
    def test_jetto_in_not_empty(self, jetto_in_path):
        """Test that jetto.in has content."""
        size = os.path.getsize(jetto_in_path)
        assert size > 0, "jetto.in is empty"
        print(f"✓ jetto.in size: {size} bytes")
    
    def test_section_headers_preserved(self, jetto_in_path):
        """Test that decorative 'Namelist :' headers are preserved."""
        struct = validate_jetto_in_structure(jetto_in_path)
        
        assert struct['section_headers'] > 0, \
            "No 'Namelist :' section headers found - decorative formatting lost"
        
        print(f"✓ Found {struct['section_headers']} section headers")
    
    def test_dash_separators_preserved(self, jetto_in_path):
        """Test that dash separator lines are preserved."""
        struct = validate_jetto_in_structure(jetto_in_path)
        
        assert struct['dash_lines'] > 0, \
            "No dash separator lines found - decorative formatting lost"
        
        print(f"✓ Found {struct['dash_lines']} dash separator lines")
    
    def test_namelists_present(self, jetto_in_path):
        """Test that Fortran namelists are present in jetto.in."""
        struct = validate_jetto_in_structure(jetto_in_path)
        
        assert len(struct['namelists']) > 0, \
            "No namelists found in jetto.in"
        
        print(f"✓ Found {len(struct['namelists'])} namelists")
        
        # Common namelists that should be present
        common_namelists = ['NLIST0', 'NLIST1', 'NLIST4']
        found_common = [nml for nml in struct['namelists'] 
                        if nml.upper() in [c.upper() for c in common_namelists]]
        
        if found_common:
            print(f"  Common namelists found: {', '.join(found_common)}")
    
    def test_first_lines_structure(self, jetto_in_path):
        """Test that first lines of jetto.in have expected structure."""
        with open(jetto_in_path, 'r') as f:
            first_lines = f.readlines()[:50]
        
        # Should have at least one namelist header in first 50 lines
        has_header = any(re.search(r'Namelist\s*:', line, re.IGNORECASE) 
                         for line in first_lines)
        assert has_header, "No 'Namelist :' header in first 50 lines"
        
        # Should have at least one dash line in first 50 lines
        has_dashes = any(re.match(r'^-{10,}', line.strip()) 
                         for line in first_lines)
        assert has_dashes, "No dash separator in first 50 lines"
        
        # Should have at least one namelist start in first 50 lines
        has_namelist = any(re.match(r'^\s*&\w+', line) 
                           for line in first_lines)
        assert has_namelist, "No namelist start (&NAME) in first 50 lines"
        
        print("✓ First 50 lines have expected structure:")
        print("  - Has section header")
        print("  - Has dash separators")
        print("  - Has namelist declarations")
    
    def test_detailed_line_inspection(self, jetto_in_path):
        """Detailed inspection of first 100 lines for debugging."""
        with open(jetto_in_path, 'r') as f:
            lines = f.readlines()[:100]
        
        header_count = 0
        dash_count = 0
        namelist_count = 0
        
        for i, line in enumerate(lines, 1):
            if re.search(r'Namelist\s*:', line, re.IGNORECASE):
                header_count += 1
            if re.match(r'^-{10,}', line.strip()):
                dash_count += 1
            if re.match(r'^\s*&\w+', line):
                namelist_count += 1
        
        print(f"\n✓ First 100 lines analysis:")
        print(f"    Section headers: {header_count}")
        print(f"    Dash lines: {dash_count}")
        print(f"    Namelist starts: {namelist_count}")
        
        # Print a sample of lines for manual inspection if needed
        if header_count == 0 or dash_count == 0:
            print("\n⚠ Potentially missing formatting - first 20 lines:")
            for i, line in enumerate(lines[:20], 1):
                print(f"    {i:3d}: {line.rstrip()}")


@pytest.mark.unit
class TestWorkflowModes:
    """
    Test different workflow modes (setup, create, full).
    
    Validates that instructions_list controls workflow correctly.
    """
    
    @pytest.mark.parametrize("instructions,expected_behavior", [
        (['setup case'], "setup_only"),
        (['create case'], "create_only"),
        (['setup case', 'create case'], "full_workflow"),
    ])
    def test_instruction_modes(self, mock_json_input, instructions, expected_behavior, request):
        """Test that different instruction modes work correctly."""
        overwrite = request.config.getoption("--overwrite", default=False)
        
        # Create instance with specified instructions using module-level constants
        run = IntegratedModellingRuns(
            TEST_SHOT,
            instructions,
            TEST_GENERATOR,
            f"{TEST_RUN_NAME}_{expected_behavior}",
            run_input=TEST_RUN_INPUT,
            run_start=TEST_RUN_START + hash(expected_behavior) % 100,  # Unique run_start
            json_input=mock_json_input,
            db=TEST_DB,
            esco_timesteps=100,
            output_timesteps=100,
            time_start=TEST_TIME_START,
            time_end=TEST_TIME_END,
            overwrite_baserun=overwrite,
            density_feedback=False,
        )
        
        # Verify instructions were set correctly
        # instructions_dict is a dict with True/False for each operation
        assert isinstance(run.instructions, dict), \
            f"Instructions should be a dict, got {type(run.instructions)}"
        
        # Check that the expected instructions are enabled
        for instr in instructions:
            assert run.instructions[instr] == True, \
                f"Expected instruction '{instr}' to be enabled"
        
        # Check that other instructions are disabled
        for key in run.instructions:
            if key not in instructions:
                assert run.instructions[key] == False, \
                    f"Expected instruction '{key}' to be disabled"
        
        print(f"✓ Mode '{expected_behavior}' configured with instructions: {instructions}")
        
        # Note: Not executing to keep test fast - this is a unit test
        # Integration tests will execute the full workflow


if __name__ == '__main__':
    # Allow running with: python tests/test_jetto_setup.py --overwrite
    pytest.main([__file__, '-v', '--tb=short'] + sys.argv[1:])

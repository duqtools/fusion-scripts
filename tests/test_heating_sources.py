"""
Test suite for prepare_im_runs.py - Heating Sources (NBI, EC).

Tests for:
- NBI heating setup and validation
- EC heating setup and validation
- Combined NBI+EC core sources
- jetto.jset parameter validation for heating sources
"""
import os
import re
import pytest
from pathlib import Path

# Test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.utils import parse_jset_parameter, validate_jset_structure

from prepare_im_runs import IntegratedModellingRuns


# ============================================================================
# TEST CONFIGURATION - Set once for all tests
# ============================================================================
# NBI tests
TEST_NBI_SHOT = 73388
TEST_NBI_DB = 'jet'
TEST_NBI_RUN_INPUT = 1
TEST_NBI_RUN_START = 1270

# EC tests  
TEST_EC_SHOT = 80599
TEST_EC_DB = 'jet'
TEST_EC_RUN_INPUT = 1
TEST_EC_RUN_START = 1300

# Combined tests
TEST_COMBINED_SHOT = 80604
TEST_COMBINED_DB = 'jet'
TEST_COMBINED_RUN_INPUT = 1
TEST_COMBINED_RUN_START = 1350


@pytest.mark.heating
@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.slow
class TestNBIHeating:
    """
    Test NBI heating source setup.
    
    Validates that NBI heating instructions properly configure
    jetto.jset with NBI parameters.
    """
    
    @pytest.fixture(scope="class")
    def nbi_instructions(self):
        """JSON instructions for NBI heating."""
        return [
            {
                "Heating": "NBI",
                "name": "NBI1",
                "power": 1.0e6,
                "energy": 50.0e3,
            }
        ]
    
    @pytest.fixture(scope="class")
    def run_instance(self, mock_json_input, nbi_instructions, request):
        """Create IntegratedModellingRuns instance with NBI heating."""
        overwrite = request.config.getoption("--overwrite", default=False)
        
        # Update JSON with NBI instructions
        json_config = mock_json_input.copy()
        json_config['heating_sources'] = nbi_instructions
        
        run = IntegratedModellingRuns(
            TEST_NBI_SHOT,
            ['create case'],
            'rungenerator_nbi_test',
            'runtest_nbi',
            db=TEST_NBI_DB,
            run_input=TEST_NBI_RUN_INPUT,
            run_start=TEST_NBI_RUN_START,
            json_input=json_config,
            esco_timesteps=100,
            output_timesteps=100,
            time_start=0.03,
            time_end=0.33,
            density_feedback=False,
            setup_nbi_flag=True,
            force_run=False,
            force_input_overwrite=False,
            overwrite_baserun=overwrite,
        )
        
        # Execute
        run.setup_create_compare()
        
        return run
    
    @pytest.fixture(scope="class")
    def jetto_jset_path(self, run_instance):
        """Path to jetto.jset file."""
        return os.path.join(run_instance.path_baserun, 'jetto.jset')
    
    @pytest.mark.skip(reason="Requires NBI-specific IMAS data and validation implementation")
    def test_jset_has_nbi_parameters(self, jetto_jset_path):
        """Test that jetto.jset contains NBI heating parameters."""
        struct = validate_jset_structure(jetto_jset_path)
        
        assert struct['exists'], "jetto.jset does not exist"
        
        # Check for NBI-related parameters in jset
        with open(jetto_jset_path, 'r') as f:
            content = f.read()
        
        # Look for NBI panel or heating parameters
        # (Exact parameter names depend on JETTO template structure)
        has_nbi = any(keyword in content.lower() for keyword in 
                      ['nbi', 'neutral beam', 'heating'])
        
        assert has_nbi, "No NBI-related parameters found in jetto.jset"
        
        print("✓ NBI parameters found in jetto.jset")


@pytest.mark.heating
@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.slow
class TestECHeating:
    """
    Test EC (Electron Cyclotron) heating source setup.
    
    Validates that EC heating instructions properly configure
    jetto.jset with EC parameters.
    """
    
    @pytest.fixture(scope="class")
    def ec_instructions(self):
        """JSON instructions for EC heating."""
        return [
            {
                "Heating": "EC",
                "name": "EC1",
                "power": 5.0e5,
                "frequency": 140.0e9,
            }
        ]
    
    @pytest.fixture(scope="class")
    def run_instance(self, mock_json_input, ec_instructions, request):
        """Create IntegratedModellingRuns instance with EC heating."""
        overwrite = request.config.getoption("--overwrite", default=False)
        
        # Update JSON with EC instructions
        json_config = mock_json_input.copy()
        json_config['heating_sources'] = ec_instructions
        
        run = IntegratedModellingRuns(
            TEST_EC_SHOT,
            ['create case'],
            'rungenerator_ec_test',
            'runtest_ec',
            db=TEST_EC_DB,
            run_input=TEST_EC_RUN_INPUT,
            run_start=TEST_EC_RUN_START,
            json_input=json_config,
            esco_timesteps=100,
            output_timesteps=100,
            time_start=0.03,
            time_end=0.33,
            density_feedback=False,
            force_run=False,
            force_input_overwrite=False,
            overwrite_baserun=overwrite,
        )
        
        # Execute
        run.setup_create_compare()
        
        return run
    
    @pytest.fixture(scope="class")
    def jetto_jset_path(self, run_instance):
        """Path to jetto.jset file."""
        return os.path.join(run_instance.path_baserun, 'jetto.jset')
    
    @pytest.mark.skip(reason="Requires EC-specific IMAS data and validation implementation")
    def test_jset_has_ec_parameters(self, jetto_jset_path):
        """Test that jetto.jset contains EC heating parameters."""
        struct = validate_jset_structure(jetto_jset_path)
        
        assert struct['exists'], "jetto.jset does not exist"
        
        # Check for EC-related parameters in jset
        with open(jetto_jset_path, 'r') as f:
            content = f.read()
        
        # Look for EC panel or heating parameters
        has_ec = any(keyword in content.lower() for keyword in 
                     ['ec', 'electron cyclotron', 'ecrh'])
        
        assert has_ec, "No EC-related parameters found in jetto.jset"
        
        print("✓ EC parameters found in jetto.jset")


@pytest.mark.heating
@pytest.mark.integration
@pytest.mark.requires_data
@pytest.mark.slow
class TestCombinedHeatingSources:
    """
    Test combined heating sources (NBI + EC).
    
    Validates that multiple heating sources can be configured
    simultaneously in core_sources setup.
    """
    
    @pytest.fixture(scope="class")
    def combined_instructions(self):
        """JSON instructions for NBI + EC heating."""
        return [
            {
                "Heating": "NBI",
                "name": "NBI1",
                "power": 1.0e6,
                "energy": 50.0e3,
            },
            {
                "Heating": "EC",
                "name": "EC1",
                "power": 5.0e5,
                "frequency": 140.0e9,
            }
        ]
    
    @pytest.fixture(scope="class")
    def run_instance(self, mock_json_input, combined_instructions, request):
        """Create IntegratedModellingRuns instance with combined heating."""
        overwrite = request.config.getoption("--overwrite", default=False)
        
        # Update JSON with combined instructions
        json_config = mock_json_input.copy()
        json_config['heating_sources'] = combined_instructions
        
        run = IntegratedModellingRuns(
            TEST_COMBINED_SHOT,
            ['create case'],
            'rungenerator_combined_test',
            'runtest_combined_heating',
            db=TEST_COMBINED_DB,
            run_input=TEST_COMBINED_RUN_INPUT,
            run_start=TEST_COMBINED_RUN_START,
            json_input=json_config,
            esco_timesteps=100,
            output_timesteps=100,
            time_start=0.03,
            time_end=0.33,
            density_feedback=False,
            force_run=False,
            force_input_overwrite=False,
            overwrite_baserun=overwrite,
        )
        
        # Execute
        run.setup_create_compare()
        
        return run
    
    @pytest.fixture(scope="class")
    def jetto_jset_path(self, run_instance):
        """Path to jetto.jset file."""
        return os.path.join(run_instance.path_baserun, 'jetto.jset')
    
    @pytest.mark.skip(reason="Requires heating source validation implementation")
    def test_jset_has_both_heating_sources(self, jetto_jset_path):
        """Test that jetto.jset contains both NBI and EC parameters."""
        struct = validate_jset_structure(jetto_jset_path)
        
        assert struct['exists'], "jetto.jset does not exist"
        
        with open(jetto_jset_path, 'r') as f:
            content = f.read().lower()
        
        # Check for both NBI and EC
        has_nbi = any(keyword in content for keyword in ['nbi', 'neutral beam'])
        has_ec = any(keyword in content for keyword in ['ec', 'electron cyclotron', 'ecrh'])
        
        assert has_nbi, "No NBI parameters found in jetto.jset"
        assert has_ec, "No EC parameters found in jetto.jset"
        
        print("✓ Both NBI and EC parameters found in jetto.jset")


if __name__ == '__main__':
    # Allow running with: python tests/test_heating_sources.py --overwrite
    pytest.main([__file__, '-v', '--tb=short'] + sys.argv[1:])

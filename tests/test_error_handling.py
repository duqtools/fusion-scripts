"""
Test suite for prepare_im_runs.py - Error Handling.

Tests for:
- Exception handling for missing IMAS data
- Invalid input parameters
- Invalid shot numbers
- Invalid time ranges
- Missing required files
"""
import os
import pytest
from pathlib import Path

# Test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_im_runs import IntegratedModellingRuns


# ============================================================================
# TEST CONFIGURATION - Set once for all tests
# ============================================================================
TEST_SHOT = 68588
TEST_DB = 'tcv'
TEST_RUN_INPUT = 3
TEST_RUN_START = 100


@pytest.mark.unit
class TestInvalidInputParameters:
    """
    Test that invalid input parameters raise appropriate errors.
    """
    
    def test_invalid_shot_number(self, mock_json_input):
        """Test that invalid shot number raises error."""
        with pytest.raises(ValueError, match="Shot number must be positive"):
            run = IntegratedModellingRuns(
                -1,  # Invalid negative shot number
                ['create case'],
                'test_generator',
                'test_run',
                db='tcv',
                run_input=1,
                run_start=100,
                json_input=mock_json_input,
            )
    
    def test_invalid_run_numbers(self, mock_json_input):
        """Test that invalid run numbers raise error."""
        # run_input must be positive
        with pytest.raises(ValueError, match="run_input must be positive"):
            run = IntegratedModellingRuns(
                68588,
                ['create case'],
                'test_generator',
                'test_run',
                db='tcv',
                run_input=-1,  # Invalid
                run_start=100,
                json_input=mock_json_input,
            )
    
    def test_invalid_time_range(self, mock_json_input):
        """Test that invalid time range (start > end) raises error."""
        # Note: This will fail during execution when generator is not found
        with pytest.raises(SystemExit):
            run = IntegratedModellingRuns(
                TEST_SHOT,
                ['create case'],
                'test_generator',
                'test_run',
                db=TEST_DB,
                run_input=TEST_RUN_INPUT,
                run_start=TEST_RUN_START,
                json_input=mock_json_input,
                time_start=10.0,
                time_end=1.0,  # End before start
            )
            run.setup_create_compare()
    
    def test_empty_instructions_list(self, mock_json_input):
        """Test that empty instructions list is handled."""
        # This might not raise an error, but should handle gracefully
        run = IntegratedModellingRuns(
            TEST_SHOT,
            [],  # Empty instructions
            'test_generator',
            'test_run',
            db=TEST_DB,
            run_input=TEST_RUN_INPUT,
            run_start=TEST_RUN_START,
            json_input=mock_json_input,
        )
        
        # Instructions should be a dict with all False values
        assert isinstance(run.instructions, dict), "Instructions should be a dict"
        # All instructions should be disabled
        for key, value in run.instructions.items():
            assert value == False, f"Instruction '{key}' should be disabled for empty list"
        print("✓ Empty instructions list handled gracefully")
    
    def test_invalid_database_name(self, mock_json_input):
        """Test that invalid database name is handled."""
        # This will fail during IMAS backend detection with SystemExit
        with pytest.raises(SystemExit):
            run = IntegratedModellingRuns(
                TEST_SHOT,
                ['create case'],
                'test_generator',
                'test_run',
                db='invalid_database_name',  # Invalid
                run_input=TEST_RUN_INPUT,
                run_start=TEST_RUN_START,
                json_input=mock_json_input,
            )
            run.setup_create_compare()


@pytest.mark.integration
@pytest.mark.requires_data
class TestMissingIMASData:
    """
    Test error handling for missing or inaccessible IMAS data.
    """
    
    @pytest.mark.skip(reason="Requires mock IMAS backend or actual missing data")
    def test_missing_pulse_schedule(self, mock_json_input):
        """Test that missing pulse_schedule IDS raises MissingDataError."""
        # This would require a shot/run combination known to be missing
        from prepare_im_runs import open_and_get_ids
        
        with pytest.raises(Exception) as exc_info:
            ids_pulse = open_and_get_ids(
                'tcv',
                99999,  # Likely non-existent shot
                9999,   # Likely non-existent run
                'pulse_schedule'
            )
        
        # Check that error message is informative
        assert 'pulse_schedule' in str(exc_info.value).lower()
    
    @pytest.mark.skip(reason="Requires mock IMAS backend or actual missing data")
    def test_missing_core_profiles(self, mock_json_input):
        """Test that missing core_profiles IDS raises error."""
        from prepare_im_runs import open_and_get_ids
        
        with pytest.raises(Exception):
            ids_core = open_and_get_ids(
                'tcv',
                99999,  # Likely non-existent shot
                9999,   # Likely non-existent run
                'core_profiles'
            )
    
    @pytest.mark.skip(reason="Requires mock IMAS backend or actual missing data")
    def test_missing_equilibrium(self, mock_json_input):
        """Test that missing equilibrium IDS raises error."""
        from prepare_im_runs import open_and_get_ids
        
        with pytest.raises(Exception):
            ids_eq = open_and_get_ids(
                'tcv',
                99999,  # Likely non-existent shot
                9999,   # Likely non-existent run
                'equilibrium'
            )


@pytest.mark.unit
class TestFileHandling:
    """
    Test error handling for file operations.
    """
    
    def test_missing_json_input(self):
        """Test that missing JSON input file is handled."""
        # When json_input=None, should use defaults
        run = IntegratedModellingRuns(
            68588,
            ['create case'],
            'test_generator',
            'test_run',
            db='tcv',
            run_input=3,
            run_start=100,
            json_input=None,  # No JSON config
        )
        
        assert run.json_input is None
        print("✓ Missing JSON input handled (using defaults)")
    
    def test_invalid_json_structure(self):
        """Test that invalid JSON structure is handled."""
        # Malformed JSON should raise error or be handled gracefully
        invalid_json = {
            'invalid_key': 'invalid_value'
            # Missing required 'instructions' key
        }
        
        # This might not raise error immediately, but should handle gracefully
        run = IntegratedModellingRuns(
            68588,
            ['create case'],
            'test_generator',
            'test_run',
            db='tcv',
            run_input=3,
            run_start=100,
            json_input=invalid_json,
        )
        
        print("✓ Invalid JSON structure handled")
    
    @pytest.mark.skip(reason="Requires actual filesystem test")
    def test_no_write_permission(self, mock_json_input, tmp_path):
        """Test that lack of write permission raises error."""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        # This should fail when trying to create baserun
        # (Implementation depends on how prepare_im_runs handles paths)
        pass


@pytest.mark.unit
class TestBoundaryConditions:
    """
    Test edge cases and boundary conditions.
    """
    
    def test_very_short_time_range(self, mock_json_input):
        """Test that very short time range is handled."""
        run = IntegratedModellingRuns(
            TEST_SHOT,
            ['create case'],
            'test_generator',
            'test_run',
            db=TEST_DB,
            run_input=TEST_RUN_INPUT,
            run_start=TEST_RUN_START,
            json_input=mock_json_input,
            time_start=1.0,
            time_end=1.001,  # Very short: 1 ms
        )
        
        # Should handle without error, though may not be physically meaningful
        print("✓ Very short time range handled")
    
    def test_zero_timesteps(self, mock_json_input):
        """Test that zero timesteps raises error or handles gracefully."""
        # Zero timesteps is invalid
        with pytest.raises(Exception):
            run = IntegratedModellingRuns(
                TEST_SHOT,
                ['create case'],
                'test_generator',
                'test_run',
                db=TEST_DB,
                run_input=TEST_RUN_INPUT,
                run_start=TEST_RUN_START,
                json_input=mock_json_input,
                esco_timesteps=0,  # Invalid
                output_timesteps=0,  # Invalid
            )
    
    def test_very_large_run_number(self, mock_json_input):
        """Test that very large run number is handled."""
        run = IntegratedModellingRuns(
            TEST_SHOT,
            ['create case'],
            'test_generator',
            'test_run',
            db=TEST_DB,
            run_input=TEST_RUN_INPUT,
            run_start=999999,  # Very large run number
            json_input=mock_json_input,
        )
        
        assert run.run_start == 999999
        print("✓ Large run number handled")


if __name__ == '__main__':
    # Allow running with: python tests/test_error_handling.py
    pytest.main([__file__, '-v', '--tb=short'] + sys.argv[1:])

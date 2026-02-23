"""
Pytest configuration and fixtures for prepare_im_runs.py test suite.
"""
import os
import sys
import pytest
import tempfile
import shutil

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Disable IMAS validation globally for all tests
os.environ['IMAS_AL_DISABLE_VALIDATE'] = '1'


@pytest.fixture(scope="session")
def workspace_root():
    """Path to workspace root directory."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def template_json_path(workspace_root):
    """Path to template JSON configuration file."""
    return os.path.join(workspace_root, 'template_prepare_input.json')


@pytest.fixture
def temp_test_dir():
    """
    Create temporary directory for test outputs.
    
    Yields the temporary directory path and cleans up after test.
    """
    temp_dir = tempfile.mkdtemp(prefix='test_prepare_im_')
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def overwrite_flag(request):
    """
    Check for --overwrite flag in pytest command line.
    
    Usage in test: @pytest.mark.parametrize("overwrite_flag", [True], indirect=True)
    Or check from command line: pytest --overwrite
    """
    return request.config.getoption("--overwrite", default=False)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing test run directories"
    )
    parser.addoption(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Keep temporary test directories after tests complete"
    )


@pytest.fixture(scope="session")
def common_test_config():
    """
    Common configuration values used across tests.
    
    Returns
    -------
    dict
        Configuration dictionary with common test parameters
    """
    return {
        'esco_timesteps': 100,
        'output_timesteps': 100,
        'default_db': 'tcv',
        'test_shots': {
            'tcv': [68588, 64958, 58760, 64965],
            'jet': [73388, 80604, 80599]
        }
    }


@pytest.fixture(scope="class")
def mock_json_input(template_json_path):
    """
    Load and return template JSON configuration.
    
    Creates a copy that can be modified per test without affecting others.
    """
    import json
    
    with open(template_json_path, 'r') as f:
        json_data = json.load(f)
    
    # Set common defaults for testing
    json_data['instructions']['rebase'] = True
    json_data['instructions']['set boundaries'] = True
    json_data['instructions']['add early profiles'] = True
    
    return json_data


def pytest_configure(config):
    """Configure pytest environment."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "integration: integration tests requiring IMAS data"
    )
    config.addinivalue_line(
        "markers", "unit: unit tests with no external dependencies"
    )
    config.addinivalue_line(
        "markers", "density_feedback: tests for density feedback feature"
    )
    config.addinivalue_line(
        "markers", "heating: tests for heating sources (NBI, EC)"
    )
    config.addinivalue_line(
        "markers", "headers: tests for JETTO section header restoration"
    )
    config.addinivalue_line(
        "markers", "llcmd: tests for LoadLeveler command file generation"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "requires_data: tests requiring specific shot data"
    )
    config.addinivalue_line(
        "markers", "interpolation: tests for IDS interpolation functionality"
    )
    config.addinivalue_line(
        "markers", "lazy_load: tests for lazy loading functionality"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Auto-mark based on test name patterns
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        if "density_feedback" in item.nodeid.lower():
            item.add_marker(pytest.mark.density_feedback)
        if any(x in item.nodeid.lower() for x in ["nbi", "ec", "heating"]):
            item.add_marker(pytest.mark.heating)
        if "header" in item.nodeid.lower():
            item.add_marker(pytest.mark.headers)
        if "llcmd" in item.nodeid.lower():
            item.add_marker(pytest.mark.llcmd)


@pytest.fixture
def run_config(mock_json_input):
    """Configuration for TCV shot 68588 tests."""
    return {
        'shot': 68588,
        'machine': 'tcv',
        'run_input': 3,
        'run_start': 100,
        'run_output': 200,
        'generator': 'rungenerator_ohmic_vscode',
        'run_name': 'runtest_tcv',
        'json_input': mock_json_input,
        'time_start': 0.0,
        'time_end': 100,  # End time for simulation
        'density_feedback': True,
        'puff_scaling': 0.1
    }


@pytest.fixture
def shot_64958_config(mock_json_input):
    """Configuration for TCV shot 64958 tests."""
    return {
        'shot': 64958,
        'machine': 'tcv',
        'run_input': 5,
        'run_start': 1010,
        'generator': 'rungenerator_ohmic_vscode',
        'run_name': 'runtest_64958',
        'json_input': mock_json_input,
        'time_start': 1.0,
        'time_end': 1.5,
        'density_feedback': False,
    }

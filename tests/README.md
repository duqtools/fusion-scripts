# prepare_im_runs.py Test Suite

Pytest-based test suite for validating prepare_im_runs.py functionality.

## Default Behavior

- Integration tests run in create-only mode by default. The setup phase (input IDS preparation) is not executed unless a specific test opts into it. This keeps runs fast and avoids hangs from heavy IMAS preprocessing.
- Unit tests have no external dependencies and run quickly.
- Heating tests (NBI/EC) are marked slow and require additional data; they are excluded from typical runs.

## Prerequisites

- Valid SIMDB token for IMAS access:
	- Run: `simdbtoken`
- Generator template(s) available (e.g., `/pfs/work/<user>/jetto/runs/rungenerator_ohmic_vscode`).
- IMAS site installation available on the system (module is forced by prepare_im_runs.py).
- Optional: `jetto_tools` in PYTHONPATH for namelist operations (tests skip gracefully if unavailable where applicable).

## Structure

```
tests/
├── utils/
│   ├── __init__.py
│   └── validation_helpers.py       # Reusable validation functions
├── test_tcv_integration.py         # Comprehensive TCV integration test
├── test_jetto_setup.py             # JETTO setup and header restoration tests
├── test_heating_sources.py         # NBI/EC heating source tests
└── test_error_handling.py          # Error handling and edge cases
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_tcv_integration.py
```

### Run with overwrite flag (recreate test runs)
```bash
pytest --overwrite
```

### Run only fast tests (exclude slow integration tests)
```bash
pytest -m "not slow"
```

### Run only integration tests
```bash
pytest -m integration
```

Note: Integration tests default to create-only workflows (no setup step) to avoid long-running preprocessing. This is enforced in the integration fixtures in this suite.

### Run specific test markers
```bash
# Density feedback tests only
pytest -m density_feedback

# Heating source tests only  
pytest -m heating

# Header restoration tests only
pytest -m headers

# LoadLeveler command file tests only
pytest -m llcmd
```

### Verbose output
```bash
pytest -v
```

### Show print statements
```bash
pytest -s
```

## Test Markers

- `integration`: Full integration tests requiring IMAS data (slow)
- `unit`: Unit tests with no external dependencies (fast)
- `density_feedback`: Tests for density feedback functionality
- `heating`: Tests for heating sources (NBI, EC)
- `headers`: Tests for JETTO section header restoration
- `llcmd`: Tests for LoadLeveler command file generation
- `slow`: Tests that take a long time to run
- `requires_data`: Tests requiring specific shot data

## Test Coverage

### test_tcv_integration.py
Comprehensive integration test for TCV shot:
- ✓ Pulse schedule IDS access before and after
- ✓ Baserun directory and file creation
- ✓ jetto.in structure and section headers
- ✓ SANCO/JSANC namelist API access
- ✓ Density feedback arrays (DNEFLFB/DTNEFLFB)
- ✓ Impurity puff scaling (SPEFLX)
- ✓ Boundary conditions (NTEB/NTIB/NDNHB1)
- ✓ TPRINT array length
- ✓ Scalar parameters (TBEG/TMAX/NPULSE)
- ✓ LoadLeveler .llcmd file creation and structure
- ✓ Job name matching run configuration

### test_jetto_setup.py
JETTO setup and header restoration (create-only by default):
- ✓ jetto.in creation and structure validation
- ✓ Decorative section header preservation
- ✓ Dash separator line preservation
- ✓ Fortran namelist presence
- ✓ First-lines structure validation
- ✓ Detailed line-by-line inspection (debugging)
- ✓ Workflow mode testing (setup/create/full)

### test_heating_sources.py
Heating source setup (NBI, EC, combined):
- ⚠ NBI heating parameter validation (skipped - requires implementation)
- ⚠ EC heating parameter validation (skipped - requires implementation)
- ⚠ Combined NBI+EC validation (skipped - requires implementation)

### test_error_handling.py
Error handling and edge cases:
- ✓ Invalid shot numbers
- ✓ Invalid run numbers
- ✓ Invalid time ranges
- ✓ Empty instructions list
- ✓ Invalid database names
- ✓ Missing JSON input
- ✓ Invalid JSON structure
- ✓ Very short time ranges
- ✓ Zero timesteps
- ✓ Very large run numbers
- ⚠ Missing IMAS data (skipped - requires mock backend)
- ⚠ Missing files (skipped - requires filesystem setup)

## Configuration

### pytest.ini
- Test discovery patterns
- Output formatting
- Test markers
- Warning filters

### conftest.py
Shared fixtures:
- `workspace_root`: Workspace root directory path
- `template_json_path`: Template JSON configuration path
- `temp_test_dir`: Temporary directory for test outputs
- `overwrite_flag`: --overwrite command line flag
- `common_test_config`: Common configuration values
- `mock_json_input`: Template JSON configuration copy
- `shot_68588_config`: TCV shot 68588 configuration
- `shot_64958_config`: TCV shot 64958 configuration

Notes:
- The template JSON includes both `nbi heating` and `ec heating` keys under `instructions`. Integration fixtures rely on this shape.

## Troubleshooting

- Tests hang or exit early:
	- Ensure SIMDB token is active: `simdbtoken`
	- Verify generator directory exists (e.g., `/pfs/work/<user>/jetto/runs/rungenerator_ohmic_vscode`).
- Missing `jetto_tools`:
	- Some namelist parsing tests will skip gracefully; functionality still validated via alternative checks.
- Heating tests are slow and may require additional IMAS core_sources data:
	- Run them explicitly when ready: `pytest -m heating -v`

### Validation Helpers
Located in `tests/utils/validation_helpers.py`:
- `clean_namelist_only()`: Clean decorative headers from namelists
- `parse_fortran_namelist_array()`: Extract Fortran array values
- `parse_nlist4_dneflfb()`: Extract DNEFLFB/DTNEFLFB arrays
- `parse_jset_cell_array()`: Extract jset cell arrays
- `parse_jset_outputextralist()`: Extract OutputExtraNamelist entries
- `parse_jset_parameter()`: Extract specific jset parameter
- `extract_numeric_value()`: Extract numeric value from jset string
- `validate_jetto_in_structure()`: Validate jetto.in structure
- `validate_jset_structure()`: Validate jetto.jset structure
- `validate_llcmd_file()`: Validate .llcmd file structure
- `compare_values()`: Compare numeric values with tolerance

## Example Usage

### Run comprehensive integration test with overwrite
```bash
pytest tests/test_tcv_68588_integration.py --overwrite -v
```

### Run only fast unit tests
```bash
pytest -m "unit and not slow" -v
```

### Run all tests except those requiring specific data
```bash
pytest -m "not requires_data" -v
```

### Run with detailed output and show prints
```bash
pytest -v -s
```

### Generate HTML coverage report
```bash
pytest --cov=prepare_im_runs --cov-report=html
open htmlcov/index.html
```

## Migrated from Old Tests

### Consolidated Tests
- `test_68588.py` → `tests/test_tcv_68588_integration.py`
- `test_prepare_im_setup_jetto.py` + `test_jetto_headers_debug.py` → `tests/test_jetto_setup.py`
- `test_nbi.py` + `test_ec.py` + `test_core_sources.py` → `tests/test_heating_sources.py`

### Moved to Examples
- `test.py` → `examples/multi_shot_batch.py` (production script, not a test)

### Deprecated Tests
The following old test files can be deleted after verification:
- `test_68588.py` (replaced by test_tcv_68588_integration.py)
- `test_prepare_im_setup_jetto.py` (merged into test_jetto_setup.py)
- `test_jetto_headers_debug.py` (merged into test_jetto_setup.py)
- `test_create_only.py` (covered by test_jetto_setup.py workflow modes)
- `test_nbi.py` (replaced by test_heating_sources.py)
- `test_ec.py` (replaced by test_heating_sources.py)
- `test_core_sources.py` (replaced by test_heating_sources.py)

## Development

### Adding New Tests
1. Create test file in `tests/` directory
2. Import validation helpers: `from tests.utils import ...`
3. Use pytest fixtures from conftest.py
4. Add appropriate markers (`@pytest.mark.integration`, etc.)
5. Follow naming convention: `test_*.py` for files, `test_*()` for functions
6. Use class-based organization for related tests: `class TestFeature:`

### Adding New Validation Helpers
1. Add function to `tests/utils/validation_helpers.py`
2. Export in `tests/utils/__init__.py`
3. Document with docstring including parameters, returns, examples
4. Use consistent naming: `parse_*()`, `validate_*()`, `extract_*()`

### Best Practices
- Use fixtures for shared setup
- Mark slow tests with `@pytest.mark.slow`
- Mark tests requiring data with `@pytest.mark.requires_data`
- Use `pytest.skip()` for tests that can't run in current environment
- Use descriptive test names: `test_feature_behavior_expected_outcome()`
- Print informative messages for manual verification
- Use assertions with helpful failure messages

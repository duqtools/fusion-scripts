"""
Shared validation helper functions for prepare_im_runs.py tests.

Extracted from test_refactored.py and test_68588.py to enable code reuse
across the test suite.
"""
import os
import re
import tempfile


def clean_namelist_only(src_path: str) -> str:
    """
    Strip decorative lines and keep only namelist blocks; write to temp file.
    
    Used for cleaning JETTO namelist files (jetto.sin, jetto.in) to remove
    decorative headers that prevent jetto_tools.namelist.read() from parsing.
    
    Parameters
    ----------
    src_path : str
        Path to source namelist file with decorative headers
        
    Returns
    -------
    str
        Path to temporary cleaned file (caller must delete after use)
        
    Examples
    --------
    >>> tmp_path = clean_namelist_only('jetto.sin')
    >>> nml = jetto_tools.namelist.read(tmp_path)
    >>> os.unlink(tmp_path)
    """
    with open(src_path, 'r') as f:
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

    tmp = tempfile.NamedTemporaryFile(
        'w', delete=False, suffix='.sin', 
        prefix='clean_', dir=os.path.dirname(src_path)
    )
    tmp.writelines(cleaned)
    tmp_path = tmp.name
    tmp.close()
    return tmp_path


def parse_fortran_namelist_array(jetto_in_path: str, array_name: str, 
                                  namelist_section: str = None) -> dict:
    """
    Extract Fortran namelist array values from jetto.in.
    
    Parses array assignments in format:
      VARNAME(1) = 3.0e13 , VARNAME(2) = 3.5e13 , ...
    
    Parameters
    ----------
    jetto_in_path : str
        Path to jetto.in file
    array_name : str
        Name of array variable (case-insensitive, e.g., 'DNEFLFB')
    namelist_section : str, optional
        Namelist section name to limit search (e.g., 'NLIST4')
        
    Returns
    -------
    dict
        {index: value} mapping (e.g., {1: 3.0e13, 2: 3.5e13})
        
    Examples
    --------
    >>> values = parse_fortran_namelist_array('jetto.in', 'DNEFLFB')
    >>> print(values)
    {1: 3.0e13, 2: 3.5e13, ...}
    """
    array_values = {}
    
    try:
        with open(jetto_in_path, 'r') as f:
            content = f.read()
        
        # Pattern: VARNAME(index) = value or VARNAME = value (scalar)
        pattern = rf'{array_name}\s*(\([0-9]+\))?\s*=\s*([-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*)'
        
        for match in re.finditer(pattern, content, re.IGNORECASE):
            index_part = match.group(1)  # e.g., "(1)" or None
            value_str = match.group(2)
            
            if index_part:
                # Extract integer from "(1)" format
                index = int(re.search(r'\d+', index_part).group())
            else:
                # Scalar case - use index 1
                index = 1
            
            array_values[index] = float(value_str)
            
    except FileNotFoundError:
        print(f"ERROR: File not found: {jetto_in_path}")
    except Exception as e:
        print(f"ERROR parsing {array_name} from jetto.in: {e}")
    
    return array_values


def parse_nlist4_dneflfb(jetto_in_path: str) -> dict:
    """
    Extract DNEFLFB and DTNEFLFB arrays from jetto.in NLIST4.
    
    Convenience wrapper for parse_fortran_namelist_array that extracts
    both density feedback arrays commonly used together.
    
    Parameters
    ----------
    jetto_in_path : str
        Path to jetto.in file
        
    Returns
    -------
    dict
        {'dneflfb': {1: 3.0e13, 2: 3.5e13, ...}, 
         'dtneflfb': {1: 1.0, 2: 1.1, ...}}
    """
    return {
        'dneflfb': parse_fortran_namelist_array(jetto_in_path, 'DNEFLFB'),
        'dtneflfb': parse_fortran_namelist_array(jetto_in_path, 'DTNEFLFB')
    }


def parse_jset_cell_array(jetto_jset_path: str, cell_prefix: str) -> dict:
    """
    Extract cell array entries from jetto.jset.
    
    Parses cell structure in jset format:
      SomeStruct.cell[j][k] : value
    
    Parameters
    ----------
    jetto_jset_path : str
        Path to jetto.jset file
    cell_prefix : str
        Cell array prefix (e.g., 'OutputExtraNamelist.selItems')
        
    Returns
    -------
    dict
        {row_index: {column_index: value}} nested mapping
        
    Examples
    --------
    >>> cells = parse_jset_cell_array('jetto.jset', 'OutputExtraNamelist.selItems')
    >>> print(cells[0])
    {0: 'DNEFLFB', 1: '', 2: '( 3.0e13 , 3.5e13 )', 3: 'true'}
    """
    cells = {}
    
    try:
        with open(jetto_jset_path, 'r') as f:
            lines = f.readlines()
        
        # Pattern: Prefix.cell[j][k] : value
        pattern = rf'{re.escape(cell_prefix)}\.cell\[(\d+)\]\[(\d+)\]\s*:\s*(.*)'
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                row = int(match.group(1))
                col = int(match.group(2))
                value = match.group(3).strip()
                
                if row not in cells:
                    cells[row] = {}
                cells[row][col] = value
                
    except FileNotFoundError:
        print(f"ERROR: File not found: {jetto_jset_path}")
    except Exception as e:
        print(f"ERROR parsing jset cells: {e}")
    
    return cells


def parse_jset_outputextralist(jetto_jset_path: str) -> dict:
    """
    Extract OutputExtraNamelist entries from jetto.jset for DNEFLFB.
    
    jset format for OutputExtraNamelist.selItems.cell[j][k]:
      cell[0][0] : DNEFLFB      (variable name)
      cell[0][1] :              (column label, usually empty)
      cell[0][2] : ( val1 , val2 , ... )  (values array)
      cell[0][3] : true/false   (active flag)
    
    Parameters
    ----------
    jetto_jset_path : str
        Path to jetto.jset file
        
    Returns
    -------
    dict
        {row_index: {'name': 'DNEFLFB', 'column_1': '', 
                     'values': '(...)', 'active': 'true/false'}}
    """
    cells = parse_jset_cell_array(jetto_jset_path, 'OutputExtraNamelist.selItems')
    
    entries = {}
    for row, cols in cells.items():
        entries[row] = {
            'name': cols.get(0, ''),
            'column_1': cols.get(1, ''),
            'values': cols.get(2, ''),
            'active': cols.get(3, 'false')
        }
    
    return entries


def validate_jetto_in_structure(jetto_in_path: str) -> dict:
    """
    Validate basic jetto.in structure and extract key metadata.
    
    Parameters
    ----------
    jetto_in_path : str
        Path to jetto.in file
        
    Returns
    -------
    dict
        {'exists': bool, 'size': int, 'namelists': list, 
         'section_headers': int, 'dash_lines': int}
    """
    result = {
        'exists': os.path.exists(jetto_in_path),
        'size': 0,
        'namelists': [],
        'section_headers': 0,
        'dash_lines': 0
    }
    
    if not result['exists']:
        return result
    
    result['size'] = os.path.getsize(jetto_in_path)
    
    try:
        with open(jetto_in_path, 'r') as f:
            for line in f:
                # Count section headers (decorative namelist labels)
                if re.search(r'Namelist\s*:', line, re.IGNORECASE):
                    result['section_headers'] += 1
                
                # Count dash separator lines
                if re.match(r'^-{10,}', line.strip()):
                    result['dash_lines'] += 1
                
                # Extract namelist names
                namelist_match = re.match(r'&(\w+)', line.strip())
                if namelist_match:
                    result['namelists'].append(namelist_match.group(1))
                    
    except Exception as e:
        print(f"ERROR validating jetto.in structure: {e}")
    
    return result


def validate_jset_structure(jetto_jset_path: str) -> dict:
    """
    Validate basic jetto.jset structure and extract key metadata.
    
    Parameters
    ----------
    jetto_jset_path : str
        Path to jetto.jset file
        
    Returns
    -------
    dict
        {'exists': bool, 'size': int, 'line_count': int,
         'has_output_extra': bool}
    """
    result = {
        'exists': os.path.exists(jetto_jset_path),
        'size': 0,
        'line_count': 0,
        'has_output_extra': False
    }
    
    if not result['exists']:
        return result
    
    result['size'] = os.path.getsize(jetto_jset_path)
    
    try:
        with open(jetto_jset_path, 'r') as f:
            lines = f.readlines()
            result['line_count'] = len(lines)
            
            # Check for OutputExtraNamelist entries
            for line in lines:
                if 'OutputExtraNamelist' in line:
                    result['has_output_extra'] = True
                    break
                    
    except Exception as e:
        print(f"ERROR validating jset structure: {e}")
    
    return result


def parse_jset_parameter(jetto_jset_path: str, parameter_path: str) -> str:
    """
    Extract a specific parameter value from jetto.jset.
    
    Parameters
    ----------
    jetto_jset_path : str
        Path to jetto.jset file
    parameter_path : str
        Parameter path (e.g., 'SancoBCPanel.Species1NeutralInflux.tpoly.value')
        
    Returns
    -------
    str
        Parameter value (empty string if not found)
        
    Examples
    --------
    >>> value = parse_jset_parameter('jetto.jset', 
    ...                               'SancoBCPanel.Species1NeutralInflux.tpoly.value')
    >>> print(value)
    8.299290e+18[0][0]
    """
    try:
        with open(jetto_jset_path, 'r') as f:
            for line in f:
                if parameter_path in line:
                    # Format: "parameter_path : value"
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        return parts[1].strip()
    except FileNotFoundError:
        print(f"ERROR: File not found: {jetto_jset_path}")
    except Exception as e:
        print(f"ERROR parsing jset parameter {parameter_path}: {e}")
    
    return ''


def extract_numeric_value(value_str: str) -> float:
    """
    Extract numeric value from jset parameter string.
    
    Handles formats like:
      - "8.299290e+18"
      - "8.299290e+18[0]"
      - "8.299290e+18[0][0]"
      - "( 3.0e13 , 3.5e13 )"
    
    Parameters
    ----------
    value_str : str
        Value string from jset file
        
    Returns
    -------
    float
        Extracted numeric value (first value if array)
        
    Raises
    ------
    ValueError
        If no numeric value found
    """
    # Remove array indices like [0] or [0][0]
    clean = re.sub(r'\[\d+\]', '', value_str)
    
    # Remove parentheses for array format
    clean = clean.replace('(', '').replace(')', '')
    
    # Extract first numeric value
    match = re.search(r'([-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*)', clean)
    if match:
        return float(match.group(1))
    
    raise ValueError(f"No numeric value found in: {value_str}")


def validate_llcmd_file(llcmd_path: str) -> dict:
    """
    Validate .llcmd batch script structure and extract key metadata.
    
    Parameters
    ----------
    llcmd_path : str
        Path to .llcmd file
        
    Returns
    -------
    dict
        {'exists': bool, 'size': int, 'has_job_name': bool,
         'has_output_redirect': bool, 'has_executable': bool}
    """
    result = {
        'exists': os.path.exists(llcmd_path),
        'size': 0,
        'has_job_name': False,
        'has_output_redirect': False,
        'has_executable': False
    }
    
    if not result['exists']:
        return result
    
    result['size'] = os.path.getsize(llcmd_path)
    
    try:
        with open(llcmd_path, 'r') as f:
            content = f.read()
            
            # Check for SLURM or LoadLeveler directives (support both)
            # SLURM uses #SBATCH, LoadLeveler uses #@
            if re.search(r'#SBATCH\s+-J', content, re.IGNORECASE) or \
               re.search(r'#@\s+job_name', content, re.IGNORECASE):
                result['has_job_name'] = True
            
            if re.search(r'#SBATCH\s+-o', content, re.IGNORECASE) or \
               re.search(r'#@\s+output', content, re.IGNORECASE):
                result['has_output_redirect'] = True
            
            # Check for executable invocation
            if 'hfps' in content.lower() or 'jetto' in content.lower():
                result['has_executable'] = True
                
    except Exception as e:
        print(f"ERROR validating .llcmd file: {e}")
    
    return result


def compare_values(expected, actual, tolerance=1e-9, description="Value"):
    """
    Compare two numeric values with tolerance and print result.
    
    Parameters
    ----------
    expected : float
        Expected value
    actual : float
        Actual value
    tolerance : float, optional
        Relative tolerance for comparison (default: 1e-9)
    description : str, optional
        Description for output message
        
    Returns
    -------
    bool
        True if values match within tolerance
    """
    import math
    
    match = math.isclose(expected, actual, rel_tol=tolerance)
    status = "✓" if match else "✗"
    
    print(f"{status} {description}: expected={expected:.6e}, actual={actual:.6e}")
    
    return match

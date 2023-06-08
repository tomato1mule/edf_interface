from typing import Tuple, Union
from beartype import beartype

def str_to_units(x: str) -> Tuple[float, str]:
    val, unit = x.split(' ')
    assert unit[0] == '[' and unit[-1]==']', f"Wrong unit format {x}"
    try:
        val = float(val)
    except ValueError:
        raise ValueError(f"Wrong unit format {x}")
    
    return val, unit

def units_to_str(val: Union[float, int], unit: str) -> str:
    assert unit[0] == '[' and unit[-1]==']', f"Wrong unit format {unit}"
    return str(val) + ' ' + unit
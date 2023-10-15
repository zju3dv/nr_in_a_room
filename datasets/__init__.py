from .habitat_base import HabitatBase
from .habitat_single import HabitatSingle
from .igobject_single import IGObject_Single
from .igbackground_single import IGBackground_Single

dataset_dict = {
    "habitat_base": HabitatBase,
    "habitat_single": HabitatSingle,
    "igibson_object_single": IGObject_Single,
    "igibson_background_single": IGBackground_Single,
}

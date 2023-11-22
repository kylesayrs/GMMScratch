from enum import Enum, auto


class MixtureFamily(Enum):
    FULL = auto()              # fully expressive eigenvalues
    DIAGONAL = auto()          # eigenvalues align with data axes
    ISOTROPIC = auto()         # same variance for all directions
    SHARED_ISOTROPIC = auto()  # same variance for all directions and components
    CONSTANT = auto()          # sigma is not learned


FAMILY_NAMES = {
    "full": MixtureFamily.FULL,
    "diagonal": MixtureFamily.DIAGONAL,
    "isotropic": MixtureFamily.ISOTROPIC,
    "shared_isotropic": MixtureFamily.SHARED_ISOTROPIC,
    "constant": MixtureFamily.CONSTANT
}


def get_mixture_family_from_str(family_name: str):
    if family_name in FAMILY_NAMES:
        return FAMILY_NAMES[family_name]

    raise ValueError(
        f"Unknown mixture family {family_name}. "
        f"Please select from {FAMILY_NAMES.keys()}"
    )

from enum import Enum, auto


class MixtureFamily:
    FULL = auto()              # fully expressive eigenvalues
    DIAGONAL = auto()          # eigenvalues align with data axes
    ISOTROPIC = auto()         # same variance for all directions
    SHARED_ISOTROPIC = auto()  # same variance for all directions and components
    CONSTANT = auto()          # sigma is not learned

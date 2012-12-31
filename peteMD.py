#!/usr/bin/env python
from mass import MASS
import numpy as np

"""peteMD.py -- MD simulations in python."""

class Atom(object):
    """Describes an atom with a mass.
    Functions described herein allow the Atom position to evolve through time
    subject to Newtonian forces.

    [include description of verlet integrator]

    """

    def __init__(self, element, position=np.zeros(3), name=None):
        """Read in element symbol. Mass is assigned from a lookup table."""
        self.index = 0  # index can be assigned from some wrapper class
        self.name = name if name else element
        self.position_history = []
        self._element = element
        self._mass = MASS[element]
        self.set_position(position)
        # scaled a, b, and c are the fractional positions of the atom
        # in a box.
        self._scaled_a = 0.
        self._scaled_b = 0.
        self._scaled_c = 0.

    @property
    def mass(self):
        return self._mass

    def set_position(self, position):
        """Assigns the scaled position to the atom, based on the dimensions
        of the cell.  
        Each cartesian coordinate is stored as a separate private variable,
        which makes for easier readability.

        """
        if self.position_history:
            raise ValueError("Cannot assign positions once \
                              MD simulation has begun.")
        self._x = position[0]
        self._y = position[1]
        self._z = position[2]

    def get_position(self):
        """Returns the position as a numpy array."""
        return np.array([self._x, self._y, self._z])

    position = property(get_position, set_position)

class System(object):
    """Contains atoms and system-wide thermodynamic data such as Temperature
    and periodic boundaries, etc..

    """

    def __init__(self, temperature, boundaries):
        """Initialize the MD system with a temperature and periodic boundaries.
        
        """
        self.temperature = temperature
        self.boundaries = boundaries


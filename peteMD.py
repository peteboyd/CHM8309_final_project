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

    def __init__(self, element, position=np.zeros(3), name=None,
                 dimensions=np.matrix(np.identity(3))):
        """Read in element symbol. Mass is assigned from a lookup table."""
        self.index = 0  # index can be assigned from some wrapper class
        self.name = name if name else element
        self.position_history = []
        # dimensions are the system cell dimensions, initially set to 
        # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self._dimensions = np.matrix(dimensions)
        self._element = element
        self._mass = MASS[element]
        self.set_position(position)

    @property
    def mass(self):
        return self._mass

    def get_position(self):
        """Returns the cartesian position of the atom."""
        return np.array(np.dot(self.dimensions, self._position)).flatten()

    def set_position(self, position):
        """Assigns the scaled position to the atom, based on the dimensions
        of the cell. Raises a ValueError if assignment is attempted after
        initialization of MD simulation.

        """
        if self.position_history:
            raise ValueError("Cannot assign positions once \
                              MD simulation has begun.")
        position = np.array(position)
        scaled_position = np.array(np.dot(self._dimensions.I, position)).flatten()
        self._position = scaled_position - np.floor(scaled_position)

    position = property(get_position, set_position)

    def set_dimensions(self, dimensions):
        """Assigns the matrix dimensions and converts the position to internal
        coordinates.

        """
        curr_pos = self.get_position()
        self._dimensions = np.matrix(dimensions)
        self.set_position(curr_pos)

    def get_dimensions(self):
        return self._dimensions

    dimensions = property(get_dimensions, set_dimensions)


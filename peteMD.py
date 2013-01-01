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
        self._position

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
        self._position = np.array(position)

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
        self.boundaries = np.matrix(boundaries)
        self.atoms = [] # array of atoms in the system

    def get_temperature(self):
        """Determines the global temperature based on the velocities of 
        the atoms in the system.

        """
        pass

class LeapFrog(Integrator):
    """Integration scheme for propagating an object with mass through time."""

    def __init__(self):
        """Information required to increment position. I haven't decided
        how to interface the integrator class with Atom.

        Ideally all these variables in __init__ should be taken from the 
        parent without having to pass it to the class each time the
        position is updated.
        """
        self._mass = 0.
        self._position = 0.
        self._velocity = 0.
        self._force = 0.
        self._dt = 0.

    def update_position(self):
        """Leap-frog integration to propagate the object through time."""
        return (self._position + (self._velocity*self._dt))

    def update_velocity(self):
        """Note, Leap-frog velocities are set at integer + 1/2 timesteps."""
        return (self._velocity + (self._force*self._dt))

class Verlet(Integrator):
    """Move an object through time with the verlet integrator."""


class Integrator(object):
    """Object currently contains two integration schemes, the verlet and
    leap frog algorithms.
    More can be included in this class at a later date.

    """
    def __init__(self, mass, position, velocity, force, dt):
        """Information required to increment position. I haven't decided
        how to interface the integrator class with Atom.

        """
        self._mass = mass 
        self._position = np.array(position)
        self._velocity = np.array(velocity)
        self._force = np.array(force)
        self._dt = dt


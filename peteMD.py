#!/usr/bin/env python
from mass import MASS
import numpy as np

"""peteMD.py -- MD simulations in python."""

class LeapFrog:
    """Integration scheme for propagating an object with mass through time."""

    # global variable for the time step.  This should not change through
    # a simulation.
    dt = 1.

    @classmethod
    def update_position(cls, atom):
        """Leap-frog integration to propagate the object through time.
        x(t+dt) = x(t) + v(t-dt/2)*dt

        """
        atom.position_history.append(atom._position)
        atom._position = (atom._position + (atom._velocity*cls.dt))

    @classmethod
    def update_velocity(cls, atom):
        """Note, Leap-frog velocities are set at half-integer timesteps.
        v(t+dt/2) = v(t-dt/2) + a(t)*dt

        """
        atom.force_history.append(atom._force)
        atom.velocity_history.append(atom._velocity)
        atom._velocity = (atom._velocity + (atom._force/atom._mass)*cls.dt)


class VelocityVerlet:
    """Move an object through time with the velocity verlet integrator."""

    # global variable for the time step.  This should not change through
    # a simulation.
    dt = 1.

    @classmethod
    def update_position(cls):
        """Velocity verlet integration
        x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2

        """
        atom.position_history.append(atom._position)
        atom.position = (atom._position + (atom._velocity*cls.dt) + 
                (atom._force/(2.*atom._mass)*cls.dt*cls.dt))
        # set the force to None for the current time-step.  This is to
        # ensure that the calculation of velocity is done correctly
        atom.force_history.append(atom._force)
        atom._force = None

    @classmethod
    def update_velocity(cls):
        """Update velocities, this requires computed forces at the current
        time step.
        v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt

        """
        if atom._force is None:
            raise Error("Cannot update velocity in Velocity Verlet scheme"+
                "unless forces are re-calculated at the current time step!")
        atom.velocity_history.append(atom._velocity)
        atom._velocity = (atom._velocity + 
                (atom.force_history[-1] + atom._force)/(2.*atom._mass)*
                cls.dt)

class Atom(object):
    """Describes an atom with a mass.

    """
    def __init__(self, element, position=np.zeros(3), name=None):
        """Read in element symbol. Mass is assigned from a lookup table."""
        self.index = 0  # index can be assigned from some wrapper class
        self.element = element
        self.name = name if name else element
        self.position_history = []
        self.velocity_history = []
        self.force_history = []
        self._mass = MASS[element]
        self._position = np.array(position)
        self._velocity = np.zeros(3)
        self._force = np.zeros(3)


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
        # NOTE only lennard-jones parameters available at the moment.
        self.lr_param = {'lj':{}}

    def get_temperature(self):
        """Determines the global temperature based on the velocities of 
        the atoms in the system.

        """
        pass

    def mix_pair_potentials(self):
        """Returns the pair potential mixing.

        """
        elements = [atom.element for atom in self.atoms]
        atom_pairs = list(itertools.combinations(elements, 2))
        for pair in atom_pairs:
            # mix the parameters
            pair.sort()
            eps = math.sqrt(epsilon[pair[0]]*epsilon[pair[1]])
            sig = (sigma[pair[0]] + sigma[pair[1]]) / 2.
            self.lr_param['lj'].setdefault(pair, {'epsilon':eps, 'sigma':sig})

def lennardjones(epsilon, sigma, vector):

    r = np.linalg.norm(vector)
    direction = vector / r 

    def energy():
        """determine the pairwise energy.
        E = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]

        """
        return (4 * epsilon * ((sigma/r)**12 - (sigma/r)**6))

    def force():
        """Returns the force vector based on a pairwise
        lennard-jones function.
        
        F = -24*epsilon*[2*sigma^12/r^13 - sigma^6/r^7]

        """

        return direction * (-24 * epsilon * (2*sigma**12/r**13 - sigma**6/r**7))

def buckingham(A, B, C, vector):
   
    r = np.linalg.norm(vector)
    direction = vector / r

    def energy():
        """calculate the pairwise energy.
        E = A*exp(-B*r) - C/r^6

        """

        return (A * exp(-B*r) - C/(r**6))
    def force():
        """returns a vector parallel to the original
        vector, with direction and magnitude determined
        by the force calculation based on the buckingham
        potential.

        F = -B*A*exp(-B*r) + 6*C/r^7

        """

        return direction * (-B * A * exp(-B*r) + 6 * C / (r**7))

def main():
    atm = Atom("Cu")

if __name__ == "__main__":
    main()

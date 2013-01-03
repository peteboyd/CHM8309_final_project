#!/usr/bin/env python
from element import MASS, epsilon, sigma
import numpy as np
import math
import itertools
from numpy import asarray, asmatrix

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
        atom.velocity_history.append(atom._velocity)
        atom._velocity = (atom._velocity + (atom._force/atom._mass)*cls.dt)


class VelocityVerlet:
    """Move an object through time with the velocity verlet integrator."""

    # global variable for the time step.  This should not change through
    # a simulation.
    dt = 1.

    @classmethod
    def update_position(cls, atom):
        """Velocity verlet integration
        x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2

        """
        atom.position_history.append(atom._position)
        atom.position = (atom._position + (atom._velocity*cls.dt) + 
                (atom._force/(2.*atom._mass)*cls.dt*cls.dt))

    @classmethod
    def update_velocity(cls, atom):
        """Update velocities, this requires computed forces at the current
        time step.
        v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt

        """
        atom.velocity_history.append(atom._velocity)
        if atom.force_history:
            prev_force = atom.force_history[-1]
        else:
            prev_force = np.zeros(3)
        atom._velocity = (atom._velocity + 
                (prev_force + atom._force)/(2.*atom._mass)*
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

    def get_position(self):
        """Returns the position of the atom."""
        return self._position

    def set_position(self, position):
        """Sets the position of the atom to the external variable."""
        self._position = position

    position = property(get_position, set_position)

    def store_force(self):
        """Stores the current value for the force and resets to zero."""
        self.force_history.append(self._force)
        self._force = np.zeros(3)


class System(object):
    """Contains atoms and system-wide thermodynamic data such as Temperature
    and periodic boundaries, etc..

    """

    def __init__(self, temperature, boundaries):
        """Initialize the MD system with a temperature and periodic boundaries.
        
        """
        # total system energy
        self.energy = 0.
        # total number of timesteps
        self.timesteps = 0
        self.temperature = temperature
        self.boundaries = asarray(boundaries)
        # gross way to invert the boundaries
        self.inverted_bounds = asarray(asmatrix(self.boundaries).I)
        self.atoms = [] # array of atoms in the system
        # NOTE only lennard-jones parameters available at the moment.
        self.lr_param = {'lj':{}}

    def get_temperature(self):
        """Determines the global temperature based on the velocities of 
        the atoms in the system.

        """
        pass

    def mix_pair_potentials(self):
        """Returns the pair potential parameter mixing.

        """
        elements = [atom.element for atom in self.atoms]
        atom_pairs = list(itertools.combinations(elements, 2))
        # eliminate duplicates, dictionary style
        elim_dupes = {}
        for pair in atom_pairs:
            # convert to list to sort
            pair = tuple(sorted(list(pair)))
            elim_dupes[pair] = 1
        atom_pairs = elim_dupes.keys()

        for pair in atom_pairs:
            # mix the parameters
            pair = tuple(sorted(list(pair)))
            eps = math.sqrt(epsilon[pair[0]]*epsilon[pair[1]])
            sig = (sigma[pair[0]] + sigma[pair[1]]) / 2.
            self.lr_param['lj'].setdefault(pair, {'epsilon':eps, 'sigma':sig})

    def add_atom(self, atom):
        """Lame function to include an atom to the simulation.  Also shifts
        coordinates to within the periodic bounds.
        
        """
        # get fractional coordinates according to the simulation box
        atom.position = np.dot(asarray(self.inverted_bounds), atom.position)
        # shift to within the periodic boundaries
        atom.position -= np.floor(atom.position)
        # convert back to cartesian coordinates
        atom.position = np.dot(self.boundaries, atom.position)
        atom.index = len(self.atoms)
        self.atoms.append(atom)

    def calculate_forces(self):
        """Calculate the forces acting on each atom using a pairwise
        dispersion calculation.

        """
        # reset all atom forces
        for atom in self.atoms:
            atom.store_force()

        indices = range(len(self.atoms))
        # generate all atom pairs, no neighbour lists here!
        atom_pairs = itertools.combinations(indices, 2)
        # calculate all the pairwise interactions for each atom.
        for pair in atom_pairs:
            ind1, ind2 = pair
            atom1 = self.atoms[ind1]
            atom2 = self.atoms[ind2]
            # determine the distance vector between the pair of atoms
            vector = atom1.position - atom2.position
            # shift the vector to within half-unit-cell length.
            vector = self.boxshift(vector)
            # determine the interaction potential
            elements = [atom1.element, atom2.element]
            elements = tuple(sorted(elements))
            # get the parameters for the calculation
            eps = self.lr_param['lj'][elements]['epsilon']
            sig = self.lr_param['lj'][elements]['sigma']
            energy, force = lennardjones(eps, sig, vector)
            # update the atomic forces 
            atom1._force += force()
            # opposite sign for atom2, since the direction vector
            # was originally pointing in the direction of atom1
            atom2._force -= force()
            # update the total energy
            self.energy += energy()

    def increment_time(self):
        """Main MD part, this will increment the system in time by one
        time step.

        """
        self.timesteps += 1
        for atom in self.atoms:
            # update velocities for the current time step
            VelocityVerlet.update_velocity(atom)
            VelocityVerlet.update_position(atom)

    def boxshift(self, vector):
        """Shift a vector to within the bounds of the periodic box."""
        # determine fractional magnitude of the vector
        fvect = np.dot(self.inverted_bounds, vector)
        # reduce vector to within a half-unit-cell length.
        fvect -= np.around(fvect)
        return np.dot(fvect, self.boundaries)

def lennardjones(eps, sig, vector):

    r = np.linalg.norm(vector)
    direction = vector / r

    def energy():
        """determine the pairwise energy.
        E = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]

        """
        return (4 * eps * ((sig/r)**12 - (sig/r)**6))

    def force():
        """Returns the force vector based on a pairwise
        lennard-jones function.
        
        F = -24*epsilon*[2*sigma^12/r^13 - sigma^6/r^7]

        """
        return direction * (-24 * eps * (2*sig**12/r**13 - sig**6/r**7))
    
    return energy, force

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

    return energy, force

def main():
    sim_box = 10.*np.identity(3)
    md = System(298.15, sim_box)
    r = np.random
    atom1 = Atom("Ar", position=r.rand(3)*r.randint(0, 8))
    atom2 = Atom("Kr", position=r.rand(3)*r.randint(0, 20))
    md.add_atom(atom1)
    md.add_atom(atom2)
    md.mix_pair_potentials()
    md.calculate_forces()
    md.increment_time()

if __name__ == "__main__":
    main()

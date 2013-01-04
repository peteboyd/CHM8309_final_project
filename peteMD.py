#!/usr/bin/env python
from element import MASS, epsilon, sigma
import numpy as np
import math
import itertools
import sys
import ConfigParser
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
        # energy history
        self.energy_history = []
        # total number of timesteps
        self.timesteps = 0
        self.temperature = temperature
        self.boundaries = asarray(boundaries)
        # gross way to invert the boundaries
        self.inverted_bounds = asarray(asmatrix(self.boundaries).I)
        self.atoms = [] # array of atoms in the system
        # NOTE only lennard-jones parameters available at the moment.
        self.lr_param = {'lj':{}}
        # initiate the history file which will be appended to
        # throughout the simulation
        hisfile = open("his.xyz", "w")
        hisfile.close()

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

    def calculate_velocities(self):
        """Determine velocities for all the atoms in the system."""

        for atom in self.atoms:
            VelocityVerlet.update_velocity(atom)

    def increment_time(self):
        """Main MD part, this will increment the system in time by one
        time step.

        """
        self.timesteps += 1
        for atom in self.atoms:
            VelocityVerlet.update_position(atom)
            # shift atom to within the boundaries
            atom.position = self.shift_position(atom.position)

        # reset energy
        self.energy_history.append(self.energy)
        self.energy = 0.

    def boxshift(self, vector):
        """Shift a vector to within the bounds of the periodic box."""
        # determine fractional magnitude of the vector
        fvect = np.dot(self.inverted_bounds, vector)
        # reduce vector to within a half-unit-cell length.
        fvect -= np.around(fvect)
        return np.dot(fvect, self.boundaries)

    def shift_position(self, position):
        """Shift a coordinate position to within the bounds of the periodic
        box.

        """
        fpos = np.dot(self.inverted_bounds, position)
        fpos -= np.floor(fpos)
        return np.dot(fpos, self.boundaries)

    def append_history(self):
        """Write atomic coordinates to a history file called his.xyz"""
        natoms = len(self.atoms)
        # first write the vectors corresponding to the bounding box
        box = "%-10s%9.3f%9.3f%9.3f%9.3f%9.3f%9.3f"%("bbox_xyz",
                0., self.boundaries[0][0], 0., self.boundaries[1][1],
                0., self.boundaries[2][2])

        atom_lines = ""
        for atom in self.atoms:
            atom_pos = "%-6s %9.3f %9.3f %9.3f"%(atom.element, 
                                              atom.position[0],
                                              atom.position[1], 
                                              atom.position[2])
            atom_vel = "%-14s %9.3f %9.3f %9.3f"%("atom_vector", 
                                               atom._velocity[0],
                                               atom._velocity[1],
                                               atom._velocity[2])
            atom_frc = "%-14s %9.3f %9.3f %9.3f"%("atom_vector", 
                                               atom._force[0],
                                               atom._force[1],
                                               atom._force[2])

            if atom.index == 0:
                atom_pos += " %s"%(box)

            atom_lines += "%s %s %s\n"%(atom_pos, atom_vel, atom_frc) 

        header = "%i\nstep # %i energy: %f\n"%(natoms, self.timesteps, 
                                             self.energy)
        hisfile = open("his.xyz", "a")
        hisfile.writelines(header + atom_lines)
        hisfile.close()

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
        return (A * math.exp(-B*r) - C/(r**6))

    def force():
        """returns a vector parallel to the original
        vector, with direction and magnitude determined
        by the force calculation based on the buckingham
        potential.

        F = -B*A*exp(-B*r) + 6*C/r^7

        """
        return direction * (-B * A * math.exp(-B*r) + 6 * C / (r**7))

    return energy, force

class Input(object):
    """Class to parse the input file and return important values."""

    def __init__(self, filename="petemd.inp"):
        """initiate parsing, default file is petemd.inp"""
        self.filename = filename
        self.parser = ConfigParser.SafeConfigParser()
        self.parser.read(self.filename)
        
    def return_timestep(self):
        """get the timestep in picoseconds"""
        return self.parser.getfloat("parameters", "timestep")

    def return_atoms(self):
        """Returns a list of atoms to be used in an md system."""
        atom_list = []
        for atm in self.parser.items("atoms"):
            atom = atm[1].split()
            element = atom[0]
            coordinates = np.array([float(i) for i in atom[1:]])
            atom_list.append(Atom(element, position=coordinates))
        return atom_list

    def return_simulation_box(self):
        """Return the periodic boundaries in a 3x3 numpy array."""
        box = np.zeros((3,3))
        for vector in self.parser.items("dimensions"):
            values = vector[1].split()
            if vector[0] == "vect1":
                box[0] = np.array([float(i) for i in values]) 
            elif vector[0] == "vect2":
                box[1] = np.array([float(i) for i in values])
            elif vector[0] == "vect3":
                box[2] = np.array([float(i) for i in values])

        return box

    def return_temperature(self):
        """Get the temperature from the input file."""
        return self.parser.getfloat("parameters", "temperature")

    def return_nsteps(self):
        """Return the number of steps in the md simulation."""
        return self.parser.getint("parameters", "nsteps")

def main():
    # get info from input file
    input = Input()
    box = input.return_simulation_box()
    atoms = input.return_atoms()
    temp = input.return_temperature()
    nsteps = input.return_nsteps()
    timestep = input.return_timestep()

    # generate md system
    md = System(temp, box)
    for atom in atoms:
        md.add_atom(atom)
    md.mix_pair_potentials()

    # set the timesteps for the integrators
    VelocityVerlet.dt = timestep
    LeapFrog.dt = timestep

    for t in range(nsteps):

        # determine forces for the current time step
        md.calculate_forces()
        # determine atomic velocities for the current timestep
        md.calculate_velocities()
        # store the atomic positions, velocities and forces in a history file
        md.append_history()
        # advance the atom to t+dt
        md.increment_time()

if __name__ == "__main__":
    main()

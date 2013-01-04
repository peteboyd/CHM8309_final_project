#!/usr/bin/env python
"""
Unit testing of peteMD.py

"""
import unittest
import peteMD
from peteMD import Atom, LeapFrog, VelocityVerlet, Input
import numpy as np
import os
import math

class TestAtom(unittest.TestCase):

    def test_wrong_element(self):
        self.assertRaises(KeyError, Atom, "HotCarl")

    def test_mass_assignment(self):
        copper = Atom("Cu")
        self.assertEqual(copper._mass, 63.546)

    def test_name_assignment(self):
        tin = Atom("Sn")
        self.assertEqual(tin.name, "Sn")


class TestSystem(unittest.TestCase):
    def setUp(self):
        """Instance a new MD simulation."""
        box = np.identity(3) * 10.
        self.md_system = peteMD.System(298.15, box)

    def test_atom_placement(self):
        """Test the functionality of the periodic boundary conditions."""
        krypton = Atom("Kr", position=np.array([15., 15., 15.]))
        self.md_system.add_atom(krypton)
        self.assertEqual(self.md_system.atoms[0].position.any(),
                         np.array([5.,5.,5.]).any())

    def test_mixing_params(self):
        """Test mixing of lennard-jones parameters."""
        carbon = Atom("C", position=np.zeros(3))
        oxygen = Atom("O", position=np.array([5., 5., 5.]))
        self.md_system.add_atom(carbon)
        self.md_system.add_atom(oxygen)
        self.md_system.mix_pair_potentials()
        pair = ("C", "O")
        # correct values for epsilon and sigma
        eps = math.sqrt(0.1050*0.0600)
        sig = (3.4309 + 3.1181) / 2
        params = self.md_system.lr_param['lj'][pair]
        self.assertEqual(params['epsilon'], eps)
        self.assertEqual(params['sigma'], sig)

    def test_vv_position_advance(self):
        """Testing the increment of position with a timstep of 1 ps
        using a velocity verlet integrator.

        """
        nitrogen = Atom("N", position=np.zeros(3))
        nitrogen._force = np.zeros(3)
        nitrogen._velocity = np.array([0.5, 0.5, 0.5])
        VelocityVerlet.update_position(nitrogen)
        self.assertEqual(nitrogen.position.any(), 
                          np.array([0.5, 0.5, 0.5]).any())

    def test_vv_velocity_calculation(self):
        """Test the calculation of velocity subject to a force of 
        [1, 1, 1] using a velocity verlet integrator.

        """
        iron = Atom("Fe", position=np.zeros(3))
        iron._force = np.array([1., 1., 1.])
        # artificially change mass to make velocity calc easy on me.
        iron._mass = 10.
        iron._velocity = np.zeros(3)
        VelocityVerlet.update_velocity(iron)
        self.assertEqual(iron._velocity.any(),
                         np.array([0.05, 0.05, 0.05]).any())

    def test_lf_position_advance(self):
        """Testing the increment of position with a timstep of 1 ps
        using a leap frog integrator.

        """
        nitrogen = Atom("N", position=np.zeros(3))
        nitrogen._force = np.zeros(3)
        nitrogen._velocity = np.array([0.5, 0.5, 0.5])
        LeapFrog.update_position(nitrogen)
        self.assertEqual(nitrogen.position.any(), 
                          np.array([0.5, 0.5, 0.5]).any())

    def test_lf_velocity_calculation(self):
        """Test the calculation of velocity subject to a force of 
        [1, 1, 1] using a leap frog integrator.
        Note, the velocity will be different from that calculated
        with the velocity verlet integrator because the leapfrog
        velocity is calculated at half-integer timesteps.

        """
        iron = Atom("Fe", position=np.zeros(3))
        iron._force = np.array([1., 1., 1.])
        # artificially change mass to make velocity calc easy on me.
        iron._mass = 10.
        iron._velocity = np.zeros(3)
        LeapFrog.update_velocity(iron)
        self.assertEqual(iron._velocity.any(), 
                         np.array([0.1, 0.1, 0.1]).any())

    def test_box_shift(self):
        """Determine if periodic vector adjustments are done
        correctly.

        """
        test_vect = self.md_system.boxshift(np.array([6., 6., 6.]))
        self.assertEqual(test_vect.any(), 
                         np.array([-4., -4., -4.]).any())

    def test_shift_position(self):
        """Ensure that coordinates are kept within the bounding box."""
        
        out_of_range_pos = np.array([25., 25., 25.])
        in_range_pos = self.md_system.shift_position(out_of_range_pos)
        self.assertEqual(in_range_pos.any(), np.array([5., 5., 5.]).any()) 

class TestEnergy(unittest.TestCase):
    """Ensure energy calculations are performed correctly."""

    def test_lennard_jones(self):
        """Ensure the lennard-jones potential gives the right energy, force."""
        energy, force = peteMD.lennardjones(1, 1, np.array([1., 0., 0.]))
        self.assertEqual(energy(), 0.)
        self.assertEqual(force().any(), np.array([-24., 0., 0.]).any())

    def test_buckingham(self):
        """Ensure the buckingham potential gives the right energy, force."""
        vector = np.array([1., 0., 0.])
        energy, force = peteMD.buckingham(1, 1, 1, vector) 
        self.assertEqual(energy(), (math.exp(-1) - 1))
        newvect = (-1*math.exp(-1) + 6) * vector
        self.assertEqual(force().any(), newvect.any()) 


class TestInput(unittest.TestCase):
    """Make sure the input file is parsed correctly"""

    def setUp(self):
        """read in input file."""
        lines = "[parameters]\n"
        lines += "timestep = 1.\n"
        lines += "temperature = 0.\n"
        lines += "nsteps = 500\n"

        lines += "[atoms]\n"
        lines += "atm1: K  0. 5. 0.\n"
        lines += "[dimensions]\n"
        lines += "vect1 = 5. 0. 0.\n"
        lines += "vect2 = 0. 5. 0.\n"
        lines += "vect3 = 0. 0. 5."
        tempfile = open(".test.inp", "w")
        tempfile.writelines(lines)
        tempfile.close()
        self.inp = Input(filename=".test.inp")
        os.remove(".test.inp")

    def test_atoms(self):
        """Determine if atom is assigned properly."""
        atoms = self.inp.return_atoms()
        self.assertEqual(atoms[0].element, "K")

    def test_timestep(self):
        """Check if timestep is assigned properly."""
        self.assertEqual(self.inp.return_timestep(), 1.)

    def test_sim_box(self):
        """Ensure the simulation box is parsed correctly."""
        box = self.inp.return_simulation_box()
        test = np.identity(3) * 5.
        self.assertEqual(box.any(), test.any())

    def test_temp(self):
        """Test temperature assignment."""
        self.assertEqual(self.inp.return_temperature(), 0.)

    def test_nsteps(self):
        """Test number of simulation steps."""
        self.assertEqual(self.inp.return_nsteps(), 500)

atom_suite = unittest.TestLoader().loadTestsFromTestCase(TestAtom)
system_suite = unittest.TestLoader().loadTestsFromTestCase(TestSystem)
energy_suite = unittest.TestLoader().loadTestsFromTestCase(TestEnergy)
input_suite = unittest.TestLoader().loadTestsFromTestCase(TestInput)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(atom_suite)
    unittest.TextTestRunner(verbosity=2).run(system_suite)
    unittest.TextTestRunner(verbosity=2).run(energy_suite)
    unittest.TextTestRunner(verbosity=2).run(input_suite)
    unittest.TextTestRunner(verbosity=2).run(energy_suite)

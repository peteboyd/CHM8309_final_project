#!/usr/bin/env python
"""
Unit testing of peteMD.py

"""
import unittest
from peteMD import Atom
import numpy as np

class TestAtom(unittest.TestCase):

    def test_wrong_element(self):
        self.assertRaises(KeyError, Atom, "HotCarl")

    def test_mass_assignment(self):
        copper = Atom("Cu")
        self.assertEqual(copper.mass, 63.546)

    def test_name_assignment(self):
        tin = Atom("Sn")
        self.assertEqual(tin.name, "Sn")

    def test_position(self):
        """Test assignment of internal coordinates to the Atom."""
        mercury = Atom("Hg", position=[1., 1., 1.], 
                       dimensions=2.*np.identity(3))
        self.assertTrue(np.allclose(mercury._position, 
                                    np.array([0.5, 0.5, 0.5])))

atom_suite = unittest.TestLoader().loadTestsFromTestCase(TestAtom)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(atom_suite)

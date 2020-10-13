import unittest
import math as m
from scipy.special import sph_harm

from FastCaSH import SphericalHarmonic


class TestLadderOperators(unittest.TestCase):
    def test_positive_base_cases(self) -> None:
        """Test cases not reliant on ladder operators: Y(l, l)"""
        # a non-specific test point
        x_pt = 0.2; y_pt = 0.5; z_pt = 0.8

        # cycle over a number of lower order spherical harmonics
        for l_ord in range(3):
            self.auto_test(l_ord, l_ord, x_pt, y_pt, z_pt)

    def test_negative_base_cases(self):
        """Test cases not reliant on ladder operators: Y(l, -l)"""
        # another non-specific test point
        x_pt = 5; y_pt = 3.5; z_pt = 0.2

        for l_ord in range(3, 7):
            self.auto_test(l_ord, -l_ord, x_pt, y_pt, z_pt)

    def test_Y2m(self):
        """Test l=2 case: Y(2, m) for all m"""
        x_pt = 2.2; y_pt = 4.8; z_pt = 1.4

        l_ord = 2
        for m_ord in range(-l_ord, l_ord+1):
            self.auto_test(l_ord, m_ord, x_pt, y_pt, z_pt)

    def test_Y5m(self):
        """Test l=5 case: Y(5, m) for all m"""
        x_pt = 2.2; y_pt = 4.8; z_pt = 1.4

        l_ord = 5
        for m_ord in range(-l_ord, l_ord+1):
            self.auto_test(l_ord, m_ord, x_pt, y_pt, z_pt)

    def auto_test(self, l_ord, m_ord, x_pt, y_pt, z_pt, acc_thresh=12):
        """Automate the process of test a spherical harmonic polynomial against scipy function"""
        # spherical coordinates for scipy function
        r_pt, th_pt, phi_pt = cart2sph(x_pt, y_pt, z_pt)

        # obtain polynomial expression
        Ylm = SphericalHarmonic(l_ord, m_ord)
        Ylm.init_sym_deriv()
        # evaluate polynomial expression
        val_lad = Ylm.eval_sym(x_pt, y_pt, z_pt)
        # scipy evaluation: note differences in scipy variable names
        val_sci = sph_harm(m_ord, l_ord, phi_pt, th_pt)

        # test real and imaginary parts
        self.assertAlmostEqual(val_lad.real, val_sci.real, places=acc_thresh)
        self.assertAlmostEqual(val_lad.imag, val_sci.imag, places=acc_thresh)


def cart2sph(x_pt: float, y_pt: float, z_pt: float) -> tuple:
    """Converts Cartesion inputs to spherical coordinates"""
    r_pt = m.sqrt(x_pt**2 + y_pt**2 + z_pt**2)
    # physics convention: theta is polar angle and phi is azimuthal
    th_pt = m.atan2(m.sqrt(x_pt ** 2 + y_pt ** 2), z_pt)
    phi_pt = m.atan2(y_pt, x_pt)
    return r_pt, th_pt, phi_pt


if __name__ == '__main__':
    unittest.main()

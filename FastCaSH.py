import sympy as sym


class SphericalHarmonic:
    """A class for evaluating spherical harmonics in Cartesian coordinates"""

    def __init__(self, l_ord, m_ord):
        self.l_ord, self.m_ord = l_ord, m_ord

        # initialize expression for Y(l, m) as Y(l, -l)
        self.x_sym, self.y_sym, self.z_sym = sym.symbols('x, y, z')
        self.sph_harm = (self.x_sym - sym.I * self.y_sym) ** self.l_ord
        # apply Condon-Shortley phase
        self.sph_harm = self.sph_harm / 2 ** self.l_ord / sym.factorial(self.l_ord) \
            * sym.sqrt(sym.factorial(2 * self.l_ord + 1) / 4 / sym.pi)
        self.sph_harm = sym.simplify(self.sph_harm)

        # apply raising operator to obtain expression for Y(l, m)
        self.cur_ord = -self.l_ord  # counter for current order

    def raising_op(self):
        # apply operator
        deriv = -(self.x_sym + sym.I * self.y_sym) * sym.Derivative(self.sph_harm, self.z_sym) \
                + self.z_sym * (sym.Derivative(self.sph_harm, self.x_sym)
                                + sym.I * sym.Derivative(self.sph_harm, self.y_sym))
        # scaling constant
        self.sph_harm = deriv.doit() / sym.sqrt(self.l_ord * (self.l_ord + 1) - self.cur_ord * (self.cur_ord + 1))
        self.sph_harm = sym.simplify(self.sph_harm)
        # increment counter
        self.cur_ord += 1

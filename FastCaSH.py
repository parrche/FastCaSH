import sympy as sym


class SphericalHarmonic:
    """A class for evaluating spherical harmonics in Cartesian coordinates"""

    def __init__(self, l_ord: int, m_ord: int):
        self.l_ord, self.m_ord = l_ord, m_ord

    def init_sym_deriv(self):
        """Generate a polynomial in Cartesion form for given spherical harmonic"""
        self.x_sym, self.y_sym, self.z_sym = sym.symbols('x, y, z')

        # decide whether to iterate by raising or lowering operators
        if self.m_ord < 0:
            self.cur_m_ord = -self.l_ord  # counter for current order
            # initialize expression for Y(l, m) as Y(l, -l)
            self.sym_form = (self.x_sym - sym.I * self.y_sym) ** self.l_ord
        else:
            self.cur_m_ord = self.l_ord  # counter for current order
            # initialize expression for Y(l, m) as Y(l, l)
            self.sym_form = (-1)**self.l_ord * (self.x_sym + sym.I * self.y_sym) ** self.l_ord

        # apply Condon-Shortley phase
        self.sym_form = self.sym_form / 2 ** self.l_ord / sym.factorial(self.l_ord) \
                        * sym.sqrt(sym.factorial(2 * self.l_ord + 1) / 4 / sym.pi)

        # apply ladder operators to obtain expression for Y(l, m)
        for _ in range(self.l_ord - abs(self.m_ord)):
            if self.m_ord < 0:
                self._raising_op()
            else:
                self._lowering_op()

    def eval_sym(self, x_pt: float, y_pt: float, z_pt: float) -> complex:
        """Evaluate spherical harmonic polynemial using user specified x,y,z values"""
        dist_sqr = x_pt**2 + y_pt**2 + z_pt**2
        # substitute values and normalize x, y, z inputs by distance
        return complex(self.sym_form.subs({self.x_sym: x_pt, self.y_sym:y_pt, self.z_sym:z_pt})) / dist_sqr**(self.l_ord/2)

    def simplify(self):
        """Use sympy to simplify polynomial expression"""
        self.sym_form = sym.simplify(self.sym_form)

    def _raising_op(self):
        """Applies raising operator once"""
        # apply operator
        deriv = -(self.x_sym + sym.I * self.y_sym) * sym.Derivative(self.sym_form, self.z_sym) \
                + self.z_sym * (sym.Derivative(self.sym_form, self.x_sym)
                                + sym.I * sym.Derivative(self.sym_form, self.y_sym))
        # scaling constant
        self.sym_form = deriv.doit() / sym.sqrt(self.l_ord * (self.l_ord + 1) - self.cur_m_ord * (self.cur_m_ord + 1))
        # increment counter
        self.cur_m_ord += 1

    def _lowering_op(self):
        """Applies lower operator once"""
        # apply operator
        deriv = (self.x_sym - sym.I * self.y_sym) * sym.Derivative(self.sym_form, self.z_sym) \
                + self.z_sym * (-sym.Derivative(self.sym_form, self.x_sym)
                                + sym.I * sym.Derivative(self.sym_form, self.y_sym))
        # scaling constant
        self.sym_form = deriv.doit() / sym.sqrt(self.l_ord * (self.l_ord + 1) - self.cur_m_ord * (self.cur_m_ord - 1))
        # decrement counter
        self.cur_m_ord -= 1

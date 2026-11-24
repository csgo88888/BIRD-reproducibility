# -*- coding: utf-8 -*-
"""
Certification script for Corollary A.6 (slice safety certificate): monotonicity
of c(x) on the scalar slice
==================================================================================
Statement certified: in the canonical configuration (mu_c = sigma_c = s = 1, E = 4,
m_bar = 0.8, delta = r*(1,-1,1,-1)/2), the product c(x) := lambda*(x) * x is strictly
decreasing in x = r^2 on the interval [0.001, 0.31]. This is the certificate invoked
in Appendix A.4 (Corollary A.6) of the paper.

Method: rigorous interval arithmetic (mpmath.iv) with dual-number forward propagation
of dc/dx and d^2c/dx^2. For each sub-interval [a, b] we verify

    c'(a) + max|c''|([a,b]) * (b - a) < 0,

which by the mean value theorem certifies c' < 0 throughout [a, b]. Sub-intervals on
which the enclosure is too coarse are bisected recursively (depth cap 6); every
sub-interval of the uniform partition below is certified, so the claim holds on the
whole interval.

Two algebraic rewrites are used to keep the interval enclosures tight:
  1. Lagrange's identity  A = E*sum(beta_e^2) - (sum beta_e)^2 = sum_{e<f} (beta_e - beta_f)^2,
     which eliminates the catastrophic cancellation in A near x = 0;
  2. with G := A/x (bounded away from 0), the factor x cancels analytically:
         c(x) = (B + sqrt(B^2 + 4*x*G*a0^2)) / (2*G).

Requirements: mpmath (pip install mpmath). Runtime: a few seconds.
"""
from mpmath import iv
from collections import deque
import math

iv.dps = 20
IVT = type(iv.mpf(0))

D_DELTA = (0.5, -0.5, 0.5, -0.5)          # delta / r
PAIRS = [(e, f) for e in range(4) for f in range(e + 1, 4)]


class D1:  # first-order dual numbers over intervals: (value, derivative)
    def __init__(s, v, d=0.0):
        s.v = v if isinstance(v, IVT) else iv.mpf(v)
        s.d = d if isinstance(d, IVT) else iv.mpf(d)

    def __add__(a, b):
        b = b if isinstance(b, D1) else D1(b)
        return D1(a.v + b.v, a.d + b.d)
    __radd__ = __add__

    def __sub__(a, b):
        b = b if isinstance(b, D1) else D1(b)
        return D1(a.v - b.v, a.d - b.d)

    def __rsub__(a, b):
        return D1(b) - a

    def __mul__(a, b):
        b = b if isinstance(b, D1) else D1(b)
        return D1(a.v * b.v, a.d * b.v + a.v * b.d)
    __rmul__ = __mul__

    def __truediv__(a, b):
        b = b if isinstance(b, D1) else D1(b)
        return D1(a.v / b.v, (a.d * b.v - a.v * b.d) / b.v ** 2)

    def __rtruediv__(a, b):
        return D1(b) / a

    def __pow__(a, n):
        return D1(a.v ** n, n * a.v ** (n - 1) * a.d)

    def sqrt(a):
        sv = iv.sqrt(a.v)
        return D1(sv, a.d / (2 * sv))


class D2:  # second-order dual numbers over intervals: (value, first, second)
    def __init__(s, v, d=0.0, d2=0.0):
        cv = lambda t: t if isinstance(t, IVT) else iv.mpf(t)
        s.v, s.d, s.d2 = cv(v), cv(d), cv(d2)

    def __add__(a, b):
        b = b if isinstance(b, D2) else D2(b)
        return D2(a.v + b.v, a.d + b.d, a.d2 + b.d2)
    __radd__ = __add__

    def __sub__(a, b):
        b = b if isinstance(b, D2) else D2(b)
        return D2(a.v - b.v, a.d - b.d, a.d2 - b.d2)

    def __rsub__(a, b):
        return D2(b) - a

    def __mul__(a, b):
        b = b if isinstance(b, D2) else D2(b)
        return D2(a.v * b.v, a.d * b.v + a.v * b.d,
                  a.d2 * b.v + 2 * a.d * b.d + a.v * b.d2)
    __rmul__ = __mul__

    def inv(a):
        v = 1 / a.v
        return D2(v, -a.d * v ** 2, 2 * a.d ** 2 * v ** 3 - a.d2 * v ** 2)

    def __truediv__(a, b):
        b = b if isinstance(b, D2) else D2(b)
        return a * b.inv()

    def __rtruediv__(a, b):
        return D2(b).inv() * a

    def __pow__(a, n):
        return D2(a.v ** n, n * a.v ** (n - 1) * a.d,
                  n * (n - 1) * a.v ** (n - 2) * a.d ** 2 + n * a.v ** (n - 1) * a.d2)

    def sqrt(a):
        sv = iv.sqrt(a.v)
        return D2(sv, a.d / (2 * sv), a.d2 / (2 * sv) - a.d ** 2 / (4 * sv ** 3))


def c_dual(xv, D):
    """c(x) = lambda*(x)*x in the cancellation-free form, with derivatives w.r.t. x."""
    xx = D(xv, 1.0, 0.0) if D is D2 else D(xv, 1.0)
    rr = xx.sqrt()                                   # r = sqrt(x)
    mb, Rc = D(iv.mpf('0.8')), D(iv.mpf('0.5'))      # m_bar, R_c = sigma_c^2/(mu_c^2+sigma_c^2)
    m = [mb + rr * di for di in D_DELTA]             # environment means
    beta = [mi * Rc / (mi ** 2 + 1) for mi in m]     # per-environment optima beta_e*
    a0 = sum(mi * Rc for mi in m)
    a1 = sum(beta)
    S_ = sum(bi * bi for bi in beta)
    C = sum(mi * mi + 1 for mi in m)
    B = 2 * a0 * a1 - S_ * C
    G = sum((Rc * (D_DELTA[e] - D_DELTA[f]) * (1 - m[e] * m[f])
             / ((m[e] ** 2 + 1) * (m[f] ** 2 + 1))) ** 2 for e, f in PAIRS)   # G = A/x
    return (B + (B * B + 4 * xx * G * a0 * a0).sqrt()) / (2 * G)


def segment_certified(lo, hi, depth=0):
    """Certify c' < 0 on [lo, hi] via c'(lo) + max|c''|([lo,hi]) * (hi-lo) < 0."""
    try:
        d_pt = c_dual(iv.mpf(float(lo)), D1).d
        d2_seg = c_dual(iv.mpf((float(lo), float(hi))), D2).d2
        M2 = float(max(abs(d2_seg.a), abs(d2_seg.b)))
        if M2 == float('inf'):
            raise ValueError
        if d_pt.b + M2 * (hi - lo) < 0:
            return True
        if d_pt.a > 0:
            return False
    except Exception:
        pass
    if depth >= 6:
        return None
    mid = (lo + hi) / 2
    r1, r2 = segment_certified(lo, mid, depth + 1), segment_certified(mid, hi, depth + 1)
    if r1 is False or r2 is False:
        return False
    if r1 is None or r2 is None:
        return None
    return True


if __name__ == '__main__':
    LO, HI, N_SEG = 0.001, 0.31, 155
    edges = [LO + (HI - LO) * i / N_SEG for i in range(N_SEG + 1)]
    bad = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if segment_certified(a, b) is not True]
    if bad:
        print('NOT certified on:', bad)
    else:
        print(f'Certified: c(x) = lambda*(x)*x is strictly decreasing on '
              f'[{LO}, {HI}] ({N_SEG} segments, interval arithmetic).')

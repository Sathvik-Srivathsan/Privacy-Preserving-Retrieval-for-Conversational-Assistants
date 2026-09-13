import numpy as np
import ipfe as P
from ipfe import IPFEScheme, get_bsgs_table

rng = np.random.default_rng(7)
n, prec = 4, 3
x = rng.normal(size=n); v = rng.normal(size=n)
x = x / np.linalg.norm(x); v = v / np.linalg.norm(v)
if float(np.dot(x, v)) < 0:
    v = -v

sche = IPFEScheme.setup(n, quant_bits=prec)
qqu = sche.quant(x); vqu = sche.quant(v)
expected = sum(a * b for a, b in zip(qqu, vqu))
print("expected", expected, "range", sche._range)
gE = pow(sche.g, expected, sche.p)

# first: standalone ratio on a FRESH Ct/key
def diag(sche, x, v):
    Ct, _ = sche.encrypt(x)
    sk = sche.key_derive(v)
    num = 1
    for ci, vi in zip(Ct[1:], vqu):
        num = (num * sche._power(ci, vi)) % sche.p
    denom = sche._power(Ct[0], sk)
    ratio = (num * pow(denom, -1, sche.p)) % sche.p
    return ratio, ratio == gE

r1, ok1 = diag(sche, x, v)
print("standalone ratio == g^exp ?", ok1)

# now: run the ACTUAL engine inner_product, spying on its ratio via dlog wrapper
orig = P.BSGSTable.dlog
seen = {}
def spy(self, h, x_range):
    seen.setdefault('h', h)
    return orig(self, h, x_range)
P.BSGSTable.dlog = spy
Ct, _ = sche.encrypt(x)
try:
    val = sche.inner_product(Ct, sche.key_derive(v), v)
except Exception as e:
    print("inner_product FAIL", type(e).__name__)
h = seen.get('h')
print("engine sent to dlog: h == g^expected ?", h == gE)
if h is not None and h != gE:
    inv = pow(h, -1, sche.p)
    print("engine h == g^-expected ?", inv == pow(gE, -1, sche.p))
    print("engine h g^? sample:", pow(sche.g, expected - 1, sche.p) == h)
P.BSGSTable.dlog = orig

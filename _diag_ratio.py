import numpy as np
from ipfe import IPFEScheme, get_bsgs_table, DiscreteLogNotFound

rng = np.random.default_rng(7)
n, prec = 4, 3
x = rng.normal(size=n); v = rng.normal(size=n)
x = x / np.linalg.norm(x); v = v / np.linalg.norm(v)
if float(np.dot(x, v)) < 0:
    v = -v

sch = IPFEScheme.setup(n, quant_bits=prec)
qqu = sch.model_source_quant(sch.g, sch.p, sch._range)  # called later
qq = sch.quant(x); vv = sch.quant(v)
expected = sum(a * b for a, b in zip(qq, vv))
print("expected", expected, "range", sch._range, "B", 10**prec)
print("g", sch.g, "p_bits", sch.p.bit_length())

# Capture the EXACT ratio inner_product will compute, using the SAME public
# params (Ct from same setup, sk_fe from same setup) so nothing can differ.
Ct, qq2 = sch.encrypt(x)
print("encrypt returned", len(Ct), "ciphertext ints")

sk_fe = sch.key_derive(v)

num = 1
for ci, vi in zip(Ct[1:], vv):
    num = (num * sch._power(ci, vi)) % sch.p
denom = sch._power(Ct[0], sk_fe)
ratio = (num * pow(denom, -1, sch.p)) % sch.p
g_exp = pow(sch.g, expected, sch.p)
print("ratio == g^expected ?", ratio == g_exp)
print("ratio == g^-expected ?", ratio == pow(g_exp, -1, sch.p))

tbl = get_bsgs_table(sch.g, sch.p, sch._range)
print("table entries", len(tbl.table), "step", tbl.step)
try:
    xt = tbl.dlog(ratio, sch._range)
    print("dlog(ratio) ->", xt)
except Exception as e:
    print("dlog(ratio) FAIL", type(e).__name__)
    try:
        xt = tbl.dlog(pow(ratio, -1, sch.p), sch._range)
        print("dlog(inv ratio) ->", xt)
    except Exception as e2:
        print("dlog(inv ratio) FAIL", type(e2).__name__)

# Now perform the engine's own inner_product and observe its trace.
try:
    val = sch.inner_product(Ct, sk_fe, v)
    print("inner_product ->", val)
except Exception as e:
    print("inner_product FAIL", type(e).__name__, "->", e)

import pathlib, sys
sys.path.insert(0, r"C:\Users\Sathvik Srivathsan\PycharmProjects\UVCE Project\Privacy Preserving Retrieval for Conversational Assistants\src")
import ipfe as P
from ipfe import BSGSTable, get_bsgs_table

LOG = []
_orig = BSGSTable.dlog

def spy(self, h, x_range):
    LOG.append((h, x_range))
    return _orig(self, h, x_range)

BSGSTable.dlog = spy

try:
    import tests.test_ipfe_engine as T
    raise SystemExit(0)
except BaseException as e:
    print("test exit:", type(e).__name__)

print("dlog calls seen:", len(LOG))
sche = None
# classify: which integers h got sent, and do they equal g^expected for that group?
for idx, (h, xr) in enumerate(LOG):
    print(f"[{idx}] x_range={xr} h_first6={str(h)[:6]}")

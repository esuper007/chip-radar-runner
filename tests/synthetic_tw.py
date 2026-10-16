"""Generate synthetic TW store data to smoke-test factors -> scores -> forecast -> output."""
import numpy as np, pandas as pd, os, sys
os.environ["CHIP_DATA_DIR"] = sys.argv[1] if len(sys.argv) > 1 else "/tmp/chipdata"
from pipeline.store import Store
rng = np.random.default_rng(1)
st = Store("tw")
cal = pd.bdate_range("2019-01-01", "2026-10-14")
codes = [str(1101 + i) for i in range(250)]
N, T = len(codes), len(cal)
flow = rng.normal(0, 1, (T, N)).cumsum(0) * 0.02 + rng.normal(0, 1, (T, N))   # persistent foreign flow
ret = 0.0003 + 0.012 * rng.normal(size=(T, N)) + 0.0006 * np.roll(flow, 1, 0) / 3   # flow predicts returns a bit
px = 50 * np.exp(np.cumsum(ret, 0))
shares = rng.uniform(1e8, 5e9, N)
vol = shares * rng.uniform(0.002, 0.01, (T, N))
long = lambda a, nm: pd.DataFrame(a, index=cal, columns=codes).stack().rename(nm)
idx = pd.MultiIndex.from_product([cal, codes], names=["date", "code"])
def frame(**cols):
    df = pd.concat([long(v, k) for k, v in cols.items()], axis=1).reset_index()
    df.columns = ["date", "code"] + list(cols)
    return df
st.append("price", frame(adj_close=px, close=px, volume=vol))
st.append("quote", frame(close=px, volume=vol, value=px * vol).assign(mkt="TWSE", name="X"))
fo = flow * shares * 1e-4
st.append("inst", frame(foreign=fo, foreign_dealer=fo * 0, trust=fo * 0.3 + rng.normal(0, 1, (T, N)) * shares * 1e-5,
                        dealer_self=rng.normal(0, 1, (T, N)) * shares * 1e-5, dealer_hedge=fo * 0, inst_total=fo))
mb = np.abs(rng.normal(0, 1, (T, N)).cumsum(0)) * 100 + 1000
st.append("margin", frame(margin_bal=mb, margin_limit=shares / 4000 + 0 * mb, short_bal=mb * 0.1))
st.append("sbl", frame(sbl_bal=np.abs(rng.normal(0, 1, (T, N)).cumsum(0)) * 1e5, sbl_sell=vol * 0.01))
st.append("qfii", frame(shares=np.tile(shares, (T, 1)), foreign_ratio=np.clip(30 + flow.cumsum(0) * 0.001, 0, 100),
                        foreign_cap=np.full((T, N), 100.0)))
st.append("daytrade", frame(daytrade_shares=vol * 0.2))
wk = cal[cal.weekday == 4]
rows = []
for d in wk:
    b4 = rng.uniform(40, 80, N); rows.append(pd.DataFrame(dict(date=d, code=codes, big400_pct=b4, big1000_pct=b4 * .7,
        retail_pct=100 - b4 - 10, mid_pct=10, holders=rng.integers(1000, 90000, N), avg_lots_per_holder=rng.uniform(5, 50, N),
        big1000_holders=50, retail_holders=1000)))
st.append("tdcc", pd.concat(rows))
st.append("universe", pd.DataFrame(dict(date=cal[-1], code=codes, name=[f"公司{c}" for c in codes], industry="電子", mkt="TWSE")))
print("synthetic store written")

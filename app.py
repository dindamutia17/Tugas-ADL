"""
Aggregate Planning Lab
Aplikasi pembelajaran berbasis Chapter 14 "Aggregate Planning"
(buku: Operations Management, Pressbooks) sebagai pengantar modul Manufacturing pada ERP.
"""
from __future__ import annotations

# ======================================================================
# BAGIAN 1 — LOGIKA PERHITUNGAN
# ======================================================================
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

COST_KEYS = [
    "Produksi reguler",
    "Lembur",
    "Subkontrak",
    "Rekrut / PHK",
    "Persediaan",
    "Backorder",
]


@dataclass
class Params:
    demand: list[float]
    inv0: float = 0.0               # persediaan awal
    workforce0: float = 50.0        # tenaga kerja awal
    reg_cap: float = 1000.0         # kapasitas reguler (unit/bulan) pada tenaga kerja awal
    reg_cost: float = 10.0          # biaya produksi reguler per unit
    ot_cap: float = 0.0             # kapasitas lembur (unit/bulan)
    ot_cost: float = 15.0
    sub_cap: float = 0.0            # kapasitas subkontrak (unit/bulan)
    sub_cost: float = 12.0
    hire_cost: float = 200.0
    layoff_cost: float = 250.0
    hold_cost: float = 2.0          # per unit per bulan
    backorder_cost: float = 0.0     # per unit per bulan

    @property
    def n(self) -> int:
        return len(self.demand)

    @property
    def units_per_worker(self) -> float:
        return self.reg_cap / self.workforce0


@dataclass
class PlanResult:
    name: str
    table: pd.DataFrame
    costs: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    feasible: bool = True

    def __post_init__(self):
        self.costs = {k: float(v) for k, v in self.costs.items()}

    @property
    def total(self) -> float:
        return float(sum(self.costs.values()))


def _empty_costs() -> dict:
    return {k: 0.0 for k in COST_KEYS}


def _ceil(x: float) -> int:
    return int(math.ceil(x - 1e-9))


# ---------------------------------------------------------------- Chase
def chase_strategy(p: Params) -> PlanResult:
    """Produksi = permintaan; tenaga kerja naik-turun mengikuti permintaan."""
    rows, costs = [], _empty_costs()
    inv, w_prev = p.inv0, p.workforce0
    for t, d in enumerate(p.demand, start=1):
        prod = max(0.0, d - inv)
        inv = inv + prod - d
        w = _ceil(prod / p.units_per_worker)
        hire, lay = max(0, w - w_prev), max(0, w_prev - w)
        w_prev = w
        costs["Produksi reguler"] += prod * p.reg_cost
        costs["Rekrut / PHK"] += hire * p.hire_cost + lay * p.layoff_cost
        costs["Persediaan"] += inv * p.hold_cost
        rows.append(dict(Bulan=t, Permintaan=d, Produksi=prod, Lembur=0.0, Subkontrak=0.0,
                         **{"Tenaga kerja": w, "Rekrut": hire, "PHK": lay,
                            "Persediaan akhir": inv, "Backorder": 0.0}))
    return PlanResult("Chase", pd.DataFrame(rows), costs)


# ---------------------------------------------------------------- Level
def level_strategy(p: Params, mode: str = "min_stockout", count_initial_adjust: bool = True) -> PlanResult:
    """
    Produksi konstan tiap bulan; persediaan menyerap fluktuasi.
    mode = "avg"          -> laju = rata-rata permintaan (seperti contoh buku), kekurangan jadi backorder
    mode = "min_stockout" -> laju terkecil yang tidak pernah menyebabkan kekurangan
    """
    D = np.array(p.demand, dtype=float)
    n = len(D)
    if mode == "avg":
        rate = _ceil(max(0.0, D.sum() - p.inv0) / n)
    else:
        cum = np.cumsum(D) - p.inv0
        rate = _ceil(max(0.0, max(cum[t] / (t + 1) for t in range(n))))
    w = _ceil(rate / p.units_per_worker)

    rows, costs = [], _empty_costs()
    net, w_prev = p.inv0, p.workforce0
    for t, d in enumerate(D, start=1):
        net = net + rate - d
        inv, back = max(net, 0.0), max(-net, 0.0)
        hire = lay = 0
        if t == 1 and count_initial_adjust:
            hire, lay = max(0, w - w_prev), max(0, w_prev - w)
        costs["Produksi reguler"] += rate * p.reg_cost
        costs["Rekrut / PHK"] += hire * p.hire_cost + lay * p.layoff_cost
        costs["Persediaan"] += inv * p.hold_cost
        costs["Backorder"] += back * p.backorder_cost
        rows.append(dict(Bulan=t, Permintaan=d, Produksi=float(rate), Lembur=0.0, Subkontrak=0.0,
                         **{"Tenaga kerja": w, "Rekrut": hire, "PHK": lay,
                            "Persediaan akhir": inv, "Backorder": back}))
    notes = [f"Laju produksi konstan: {rate:,.0f} unit/bulan (≈ {w} pekerja)."]
    if rate > p.reg_cap:
        notes.append(f"Laju ini melebihi kapasitas reguler awal ({p.reg_cap:,.0f}); "
                     "butuh tambahan pekerja di awal periode.")
    return PlanResult("Level", pd.DataFrame(rows), costs, notes)


# ---------------------------------------------------------------- Transportation / Mixed
def transport_plan(p: Params, name: str = "Mixed (Transportasi)"):
    """
    Model transportasi: sumber = persediaan awal, reguler, lembur, subkontrak tiap bulan;
    tujuan = permintaan tiap bulan. Biaya sel = biaya unit + biaya simpan x selisih bulan.
    Tenaga kerja dijaga konstan (tanpa rekrut/PHK).
    Mengembalikan (PlanResult, matriks_alokasi DataFrame, matriks_biaya DataFrame).
    """
    n = p.n
    sources = [("Persediaan awal", 0, 0.0, p.inv0)]
    for t in range(1, n + 1):
        sources.append((f"Reguler B{t}", t, p.reg_cost, p.reg_cap))
        if p.ot_cap > 0:
            sources.append((f"Lembur B{t}", t, p.ot_cost, p.ot_cap))
        if p.sub_cap > 0:
            sources.append((f"Subkontrak B{t}", t, p.sub_cost, p.sub_cap))

    S = len(sources)
    cost_mat = np.full((S, n), np.nan)
    var_idx = {}
    c = []
    for s, (_, per, uc, _cap) in enumerate(sources):
        for j in range(1, n + 1):
            if per == 0:
                cell = uc + p.hold_cost * (j - 1)
            elif j >= per:
                cell = uc + p.hold_cost * (j - per)
            else:
                continue  # tidak boleh memproduksi untuk bulan yang sudah lewat (tanpa backorder)
            cost_mat[s, j - 1] = cell
            var_idx[(s, j - 1)] = len(c)
            c.append(cell)

    A_eq = np.zeros((n, len(c)))
    b_eq = np.array(p.demand, dtype=float)
    A_ub = np.zeros((S, len(c)))
    b_ub = np.array([s[3] for s in sources], dtype=float)
    for (s, j), k in var_idx.items():
        A_eq[j, k] = 1
        A_ub[s, k] = 1

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    labels = [s[0] for s in sources]
    cols = [f"Permintaan B{j}" for j in range(1, n + 1)]
    cost_df = pd.DataFrame(cost_mat, index=labels, columns=cols)
    if not res.success:
        empty = PlanResult(name, pd.DataFrame(), _empty_costs(), feasible=False,
                           notes=["Tidak ada solusi layak: total kapasitas (reguler + lembur + subkontrak + "
                                  "persediaan awal) belum cukup memenuhi permintaan tepat waktu."])
        return empty, pd.DataFrame(), cost_df

    x = np.zeros((S, n))
    for (s, j), k in var_idx.items():
        x[s, j] = res.x[k]
    x[np.abs(x) < 1e-7] = 0
    alloc = pd.DataFrame(x, index=labels, columns=cols)
    alloc["Terpakai"] = x.sum(axis=1)
    alloc["Kapasitas"] = b_ub
    alloc["Sisa"] = b_ub - x.sum(axis=1)

    costs = _empty_costs()
    rows = []
    for t in range(1, n + 1):
        reg = ot = sub = 0.0
        for s, (_, per, uc, _cap) in enumerate(sources):
            if per != t:
                continue
            used = x[s].sum()
            if labels[s].startswith("Reguler"):
                reg = used
            elif labels[s].startswith("Lembur"):
                ot = used
            else:
                sub = used
        # persediaan akhir bulan t = unit dari sumber periode <= t yang dipakai untuk bulan > t
        inv_end = 0.0
        for s, (_, per, _uc, _cap) in enumerate(sources):
            if per <= t:
                inv_end += x[s, t:].sum()
        rows.append(dict(Bulan=t, Permintaan=p.demand[t - 1], Produksi=reg, Lembur=ot, Subkontrak=sub,
                         **{"Tenaga kerja": p.workforce0, "Rekrut": 0, "PHK": 0,
                            "Persediaan akhir": inv_end, "Backorder": 0.0}))
    table = pd.DataFrame(rows)
    costs["Produksi reguler"] = float(table["Produksi"].sum() * p.reg_cost)
    costs["Lembur"] = float(table["Lembur"].sum() * p.ot_cost)
    costs["Subkontrak"] = float(table["Subkontrak"].sum() * p.sub_cost)
    # biaya simpan dari alokasi (sisa persediaan awal yang tidak terpakai tidak dikenai biaya)
    hold = 0.0
    for (s, j), k in var_idx.items():
        per = sources[s][1]
        months_held = (j + 1 - 1) if per == 0 else (j + 1 - per)
        hold += x[s, j] * p.hold_cost * months_held
    costs["Persediaan"] = float(hold)
    return PlanResult(name, table, costs), alloc, cost_df


# ---------------------------------------------------------------- Linear programming (2 produk)
def lp_two_products(cost_a, cost_b, capacity, dem_a, dem_b, hold_cost=0.0):
    """
    Minimalkan biaya produksi produk A dan B selama n bulan.
    hold_cost = 0 : permintaan menjadi batas bawah produksi tiap bulan (formulasi buku).
    hold_cost > 0 : produksi boleh lebih awal; permintaan dipenuhi secara kumulatif dan
                    biaya simpan = hold x unit x bulan simpan.
    """
    n = len(capacity)
    nv = 2 * n  # x1..xn (produk A), y1..yn (produk B)
    c = np.array([cost_a] * n + [cost_b] * n, dtype=float)
    A_ub, b_ub = [], []
    for t in range(n):
        row = np.zeros(nv)
        row[t] = 1
        row[n + t] = 1
        A_ub.append(row)
        b_ub.append(capacity[t])
    if hold_cost > 0:
        for t in range(n):
            ra, rb = np.zeros(nv), np.zeros(nv)
            ra[: t + 1] = -1
            rb[n: n + t + 1] = -1
            A_ub.append(ra)
            b_ub.append(-sum(dem_a[: t + 1]))
            A_ub.append(rb)
            b_ub.append(-sum(dem_b[: t + 1]))
        for i in range(n):  # unit bulan i+1 disimpan sampai akhir bulan n-1 jika tidak dipakai
            c[i] += hold_cost * (n - 1 - i)
            c[n + i] += hold_cost * (n - 1 - i)
        bounds = [(0, None)] * nv
    else:
        bounds = [(dem_a[t], None) for t in range(n)] + [(dem_b[t], None) for t in range(n)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if res.success and hold_cost > 0:
        # koreksi konstanta: biaya simpan = hold x sum_t (produksi kumulatif - permintaan kumulatif)
        cum_dem = np.cumsum(dem_a) + np.cumsum(dem_b)
        res.fun = float(res.fun - hold_cost * cum_dem[: n - 1].sum())
    return res


# ---------------------------------------------------------------- Management coefficient
def fit_regression(demand, workforce):
    x, y = np.array(demand, float), np.array(workforce, float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(intercept), float(slope), r2


# ---------------------------------------------------------------- Disagregasi ke MPS
def split_preserving_total(total: float, shares: list[float]) -> list[int]:
    """Bagi total ke beberapa bagian sesuai proporsi, hasil bilangan bulat dan jumlahnya tetap (largest remainder)."""
    shares = np.array(shares, float)
    if shares.sum() <= 0:
        return [0] * len(shares)
    raw = total * shares / shares.sum()
    base = np.floor(raw).astype(int)
    remainder = int(round(total)) - int(base.sum())
    order = np.argsort(-(raw - base))
    for i in range(max(0, remainder)):
        base[order[i % len(base)]] += 1
    return base.tolist()


# ======================================================================
# BAGIAN 2 — ANTARMUKA STREAMLIT
# ======================================================================
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Aggregate Planning Lab", page_icon="🏭", layout="wide")

# ------------------------------------------------------------------ gaya
C_CHASE, C_LEVEL, C_MIXED = "#1F5C7A", "#C0762B", "#4E8A5B"
C_REG, C_OT, C_SUB = "#1F5C7A", "#D9A441", "#8E5B8E"
C_DEMAND, C_INV, C_BACK = "#222222", "#6B7B83", "#B3392F"
COST_COLORS = {
    "Produksi reguler": "#1F5C7A",
    "Lembur": "#D9A441",
    "Subkontrak": "#8E5B8E",
    "Rekrut / PHK": "#B3392F",
    "Persediaan": "#6B7B83",
    "Backorder": "#2B2B2B",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1250px;}
      div[data-testid="stMetric"] {border: 1px solid #D5DCE0; border-radius: 6px; padding: 10px 14px;}
      h1 {letter-spacing: -0.5px;}
      .kecil {color:#5A6870; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helper
def money(x: float) -> str:
    return f"${x:,.0f}"


def parse_series(text: str):
    """'1000, 1200; 1500' -> [1000.0, 1200.0, 1500.0]; None jika tidak valid."""
    parts = [s.strip() for s in re.split(r"[;,\n]+", text) if s.strip()]
    try:
        vals = [float(s) for s in parts]
    except ValueError:
        return None
    return vals if vals else None


def num(label, key, value, step=1.0, fmt="%.0f", help=None, min_value=0.0):
    return st.number_input(label, min_value=min_value, value=float(value), step=float(step),
                           format=fmt, key=key, help=help)


def _hl(v):
    return "background-color: rgba(31,92,122,0.20); font-weight: 600" if isinstance(v, (int, float)) and v > 0 else ""


def style_matrix(df: pd.DataFrame, cols, highlight=True):
    sty = df.style.format("{:,.0f}", subset=cols, na_rep="—")
    if highlight:
        sty = sty.map(_hl, subset=cols) if hasattr(sty, "map") else sty.applymap(_hl, subset=cols)
    return sty


def plan_chart(res, title=""):
    t = res.table
    fig = go.Figure()
    fig.add_bar(x=t["Bulan"], y=t["Produksi"], name="Produksi reguler", marker_color=C_REG)
    if t["Lembur"].sum() > 0:
        fig.add_bar(x=t["Bulan"], y=t["Lembur"], name="Lembur", marker_color=C_OT)
    if t["Subkontrak"].sum() > 0:
        fig.add_bar(x=t["Bulan"], y=t["Subkontrak"], name="Subkontrak", marker_color=C_SUB)
    fig.add_scatter(x=t["Bulan"], y=t["Permintaan"], name="Permintaan", mode="lines+markers",
                    line=dict(color=C_DEMAND, width=2.5))
    fig.add_scatter(x=t["Bulan"], y=t["Persediaan akhir"], name="Persediaan akhir", mode="lines+markers",
                    line=dict(color=C_INV, width=2, dash="dash"))
    if t["Backorder"].sum() > 0:
        fig.add_scatter(x=t["Bulan"], y=t["Backorder"], name="Backorder", mode="lines+markers",
                        line=dict(color=C_BACK, width=2, dash="dot"))
    fig.update_layout(barmode="stack", height=380, title=title, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(title="Bulan", tickmode="linear", dtick=1),
                      yaxis=dict(title="Unit"),
                      legend=dict(orientation="h", y=-0.22))
    return fig


def show_plan(res, p: Params):
    """Metrik, grafik, dan tabel untuk satu hasil strategi."""
    c = res.costs
    prod_cost = c["Produksi reguler"] + c["Lembur"] + c["Subkontrak"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total biaya", money(res.total))
    m2.metric("Biaya produksi", money(prod_cost))
    m3.metric("Rekrut / PHK", money(c["Rekrut / PHK"]))
    m4.metric("Persediaan + backorder", money(c["Persediaan"] + c["Backorder"]))
    for n in res.notes:
        st.caption(n)
    st.plotly_chart(plan_chart(res), width="stretch")
    t = res.table.copy()
    fmt_cols = ["Permintaan", "Produksi", "Lembur", "Subkontrak", "Tenaga kerja", "Rekrut", "PHK",
                "Persediaan akhir", "Backorder"]
    st.dataframe(t.style.format({k: "{:,.0f}" for k in fmt_cols}), hide_index=True, width="stretch")
    if t["Backorder"].sum() > 0:
        st.warning(f"Terjadi backorder total {t['Backorder'].sum():,.0f} unit-bulan. "
                   "Pelanggan menunggu — ini risiko layanan yang tidak tertangkap oleh biaya produksi saja.")


# ------------------------------------------------------------------ header
st.title("Aggregate Planning Lab")
st.markdown(
    "Simulator interaktif untuk **Chapter 14: Aggregate Planning** (*Operations Management*). "
    "Chapter ini dipilih karena rencana agregat adalah titik awal alur perencanaan di ERP: "
    "S&OP → MPS → MRP."
)

with st.sidebar:
    st.header("Tentang aplikasi")
    st.markdown(
        "Dibuat untuk persiapan mata kuliah **ERP — modul Manufacturing**.\n\n"
        "Setiap tab memetakan satu bagian chapter 14 menjadi kalkulator yang bisa diubah "
        "angkanya, lalu tab terakhir menghubungkannya ke konsep MPS di ERP."
    )
    st.markdown("**Cara belajar yang disarankan**")
    st.markdown(
        "1. Baca tab *Konsep & ERP*\n"
        "2. Jalankan contoh buku di tab *Strategi*, cocokkan hasilnya\n"
        "3. Ubah parameter, amati strategi mana yang berubah\n"
        "4. Selesaikan tab *Dari agregat ke MPS*"
    )
    st.caption("Satuan mata uang mengikuti buku (USD).")

tab_konsep, tab_strategi, tab_lp, tab_mc, tab_jasa, tab_mps = st.tabs([
    "Konsep & ERP",
    "Strategi agregat",
    "Linear programming",
    "Management coefficient",
    "Jasa & yield management",
    "Dari agregat ke MPS",
])

# ================================================================== 1. KONSEP
with tab_konsep:
    st.subheader("Apa itu aggregate planning?")
    st.markdown(
        "Aggregate planning menentukan **berapa banyak yang diproduksi, dengan berapa pekerja, dan berapa "
        "persediaan yang ditahan** untuk satu *keluarga produk* (bukan item satuan) pada horizon menengah. "
        "Tujuannya memenuhi permintaan dengan biaya terendah, sambil menjaga layanan ke pelanggan dan "
        "stabilitas tenaga kerja."
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Tiga horizon perencanaan**")
        st.dataframe(pd.DataFrame({
            "Horizon": ["Jangka panjang (strategis)", "Jangka menengah (taktis)", "Jangka pendek (operasional)"],
            "Rentang": ["1–5 tahun", "3–18 bulan", "Harian – 6 bulan"],
            "Contoh keputusan": ["Ekspansi kapasitas, produk baru", "Rencana agregat, S&OP, anggaran",
                                 "Penjadwalan, urutan pekerjaan"],
        }), hide_index=True, width="stretch")

        st.markdown("**Input yang dibutuhkan rencana agregat**")
        st.markdown(
            "- Ramalan permintaan (Chapter 3)\n- Persediaan awal\n- Ketersediaan dan biaya tenaga kerja\n"
            "- Kapasitas reguler, lembur, subkontrak\n- Biaya simpan, rekrut, PHK, dan kekurangan stok"
        )
    with col_b:
        st.markdown("**Alur S&OP menjadi rencana produksi**")
        st.graphviz_chart(
            """
            digraph G {
              rankdir=LR; bgcolor="transparent";
              node [shape=box, style="rounded,filled", fillcolor="#EEF2F4", color="#9FB0B8",
                    fontname="Helvetica", fontsize=11];
              edge [color="#6B7B83"];
              A [label="Kumpulkan data\\npenjualan"]; B [label="Demand\\nplanning"];
              C [label="Supply\\nplanning"]; D [label="Rekonsiliasi\\nkeuangan"];
              E [label="Rapat\\neksekutif S&OP", fillcolor="#C9DCE6"];
              A -> B -> C -> D -> E;
            }
            """,
            width="stretch",
        )
        st.caption("Lima langkah proses S&OP menurut chapter 14.1.")

    st.divider()
    st.subheader("Tiga strategi dasar")
    st.dataframe(pd.DataFrame({
        "Strategi": ["Chase", "Level", "Mixed"],
        "Cara kerja": [
            "Produksi mengikuti permintaan; pekerja direkrut dan di-PHK",
            "Produksi dan pekerja konstan; persediaan menyerap fluktuasi",
            "Gabungan: tenaga kerja tetap + lembur, subkontrak, dan persediaan",
        ],
        "Biaya yang dominan": ["Rekrut / PHK", "Persediaan (dan backorder)", "Lembur + subkontrak"],
        "Risiko": ["Moral dan keterampilan pekerja turun", "Stok menumpuk, barang usang",
                   "Koordinasi dengan vendor, kualitas subkontrak"],
    }), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Peta istilah: buku ke ERP manufaktur")
    st.dataframe(pd.DataFrame({
        "Konsep di buku": [
            "Ramalan permintaan (Bab 3)",
            "Rencana agregat per keluarga produk",
            "Pemecahan ke produk dan minggu",
            "Kapasitas reguler, lembur, subkontrak",
            "Persediaan awal dan biaya simpan",
            "Rekrut / PHK / jam kerja",
            "Kebutuhan material (lanjutan)",
        ],
        "Padanan di ERP": [
            "Forecast / demand planning",
            "Sales & Operations Planning (S&OP)",
            "Master Production Schedule (MPS)",
            "Kapasitas work center, rough-cut capacity planning, PO subkontrak",
            "Stok on-hand, safety stock, valuasi persediaan",
            "Kalender shift dan ketersediaan sumber daya",
            "MRP: kebutuhan bahan dari BOM, lalu purchase request dan work order",
        ],
    }), hide_index=True, width="stretch")
    st.caption("Nama menu dan transaksi berbeda di tiap ERP (SAP, Odoo, Oracle, dll.), tetapi urutan logikanya sama.")

# ================================================================== 2. STRATEGI
with tab_strategi:
    PRESETS = {
        "Contoh buku 14.3 — enam bulan": dict(
            demand="1000, 1200, 1500, 900, 1100, 1300", inv0=0, workforce0=50, reg_cap=1000, reg_cost=10,
            ot_cap=200, ot_cost=15, sub_cap=300, sub_cost=12, hire=200, layoff=250, hold=2,
            note="Buku tidak menyebut batas lembur dan subkontrak. Di sini diasumsikan 200 dan 300 unit/bulan "
                 "supaya strategi *mixed* dapat dihitung. Total chase di buku: $86,500.",
        ),
        "Contoh buku 14.5 — lantai kayu oak (model transportasi)": dict(
            demand="5000, 15000, 20000, 5200", inv0=500, workforce0=25, reg_cap=5000, reg_cost=40,
            ot_cap=5000, ot_cost=75, sub_cap=4000, sub_cost=85, hire=500, layoff=600, hold=2,
            note="Jumlah pekerja (25), biaya rekrut ($500), dan biaya PHK ($600) tidak ada di buku; "
                 "diasumsikan hanya untuk strategi chase dan level. Fokus contoh ini ada di tab *Mixed*.",
        ),
        "Soal latihan no. 1": dict(
            demand="1200, 1500, 1800, 1600, 1400, 1300", inv0=0, workforce0=70, reg_cap=1400, reg_cost=12,
            ot_cap=280, ot_cost=18, sub_cap=0, sub_cost=0, hire=300, layoff=400, hold=3,
            note="Soal tidak menyebut jumlah pekerja (diasumsikan 70, yaitu 20 unit/pekerja) dan batas lembur "
                 "(diasumsikan 280 unit/bulan = 20% kapasitas). Tidak ada subkontrak.",
        ),
    }
    names = list(PRESETS)
    sel = st.selectbox("Data yang dipakai", names, key="preset_sel")
    pk = names.index(sel)
    P = PRESETS[sel]
    st.info(P["note"])

    with st.expander("Parameter (bisa diubah)", expanded=True):
        demand_txt = st.text_input("Permintaan per bulan (pisahkan dengan koma)", P["demand"], key=f"{pk}_demand")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Kapasitas dan awal periode**")
            inv0 = num("Persediaan awal (unit)", f"{pk}_inv0", P["inv0"])
            workforce0 = num("Pekerja awal", f"{pk}_w0", P["workforce0"], min_value=1.0)
            reg_cap = num("Kapasitas reguler (unit/bulan)", f"{pk}_regcap", P["reg_cap"], min_value=1.0)
            ot_cap = num("Batas lembur (unit/bulan)", f"{pk}_otcap", P["ot_cap"], help="0 = tidak ada lembur")
            sub_cap = num("Batas subkontrak (unit/bulan)", f"{pk}_subcap", P["sub_cap"], help="0 = tidak ada subkontrak")
        with c2:
            st.markdown("**Biaya produksi (per unit)**")
            reg_cost = num("Reguler", f"{pk}_regc", P["reg_cost"], fmt="%.2f")
            ot_cost = num("Lembur", f"{pk}_otc", P["ot_cost"], fmt="%.2f")
            sub_cost = num("Subkontrak", f"{pk}_subc", P["sub_cost"], fmt="%.2f")
        with c3:
            st.markdown("**Biaya tenaga kerja dan persediaan**")
            hire = num("Rekrut (per pekerja)", f"{pk}_hire", P["hire"], fmt="%.2f")
            layoff = num("PHK (per pekerja)", f"{pk}_lay", P["layoff"], fmt="%.2f")
            hold = num("Simpan (per unit per bulan)", f"{pk}_hold", P["hold"], fmt="%.2f")

    demand = parse_series(demand_txt)
    if demand is None or len(demand) < 2:
        st.error("Isi permintaan dengan minimal dua angka dipisahkan koma, tanpa titik ribuan. Contoh: 1000, 1200, 1500")
        st.stop()
    p = Params(demand=demand, inv0=inv0, workforce0=workforce0, reg_cap=reg_cap, reg_cost=reg_cost,
               ot_cap=ot_cap, ot_cost=ot_cost, sub_cap=sub_cap, sub_cost=sub_cost,
               hire_cost=hire, layoff_cost=layoff, hold_cost=hold)

    st.caption(f"Produktivitas: {p.units_per_worker:,.1f} unit per pekerja per bulan. "
               f"Total permintaan {sum(demand):,.0f} unit dalam {p.n} bulan.")

    r_chase = chase_strategy(p)
    r_mixed, alloc, cost_mat = transport_plan(p)

    t_chase, t_level, t_mixed, t_cmp = st.tabs(["Chase", "Level", "Mixed (transportasi)", "Perbandingan"])

    # ---------------- Chase
    with t_chase:
        st.markdown(
            "**Chase**: produksi bulan *t* = permintaan bulan *t*. Jumlah pekerja diubah setiap bulan, "
            "sehingga persediaan hampir nol tetapi biaya rekrut dan PHK muncul."
        )
        show_plan(r_chase, p)
        with st.expander("Rumus yang dipakai"):
            st.latex(r"\text{Pekerja}_t = \left\lceil \frac{\text{Produksi}_t}{\text{unit per pekerja}} \right\rceil")
            st.latex(r"\text{Biaya} = \sum_t \big(c_{reg} \cdot Q_t + c_{hire}\cdot H_t + c_{layoff}\cdot L_t + h\cdot I_t\big)")

    # ---------------- Level
    with t_level:
        st.markdown(
            "**Level**: produksi dan jumlah pekerja konstan. Saat produksi melebihi permintaan, "
            "selisihnya masuk persediaan; saat kurang, persediaan dipakai."
        )
        lc1, lc2 = st.columns([2, 1])
        with lc1:
            mode_label = st.radio(
                "Penentuan laju produksi",
                ["Minimum tanpa kekurangan stok", "Rata-rata permintaan (seperti contoh buku)"],
                key=f"{pk}_levelmode",
                help="Rata-rata permintaan bisa menyebabkan kekurangan stok di bulan permintaan tinggi.",
            )
        with lc2:
            adjust0 = st.checkbox("Hitung biaya rekrut/PHK di awal", value=True, key=f"{pk}_adjust0",
                                  help="Jika laju produksi konstan berbeda dari kapasitas pekerja awal.")
        mode = "avg" if mode_label.startswith("Rata") else "min_stockout"
        if mode == "avg":
            p.backorder_cost = num("Biaya backorder (per unit per bulan) — asumsi, tidak ada di buku",
                                   f"{pk}_backc", 5.0, fmt="%.2f")
        r_level = level_strategy(p, mode, adjust0)
        show_plan(r_level, p)
        st.caption(
            "Catatan: dengan laju = rata-rata permintaan (1,167 unit pada contoh 14.3), kumulatif produksi bulan 3 "
            "masih di bawah kumulatif permintaan. Buku tidak membahas ini; aplikasi menandainya sebagai backorder."
        )

    # ---------------- Mixed
    with t_mixed:
        st.markdown(
            "**Mixed**: pekerja tetap pada kapasitas reguler; kekurangan ditutup lembur, subkontrak, dan "
            "persediaan. Dipecahkan sebagai **model transportasi**: sumber = (reguler, lembur, subkontrak) tiap bulan "
            "+ persediaan awal; tujuan = permintaan tiap bulan. Biaya sel = biaya unit + biaya simpan × jumlah bulan disimpan."
        )
        if not r_mixed.feasible:
            for n_ in r_mixed.notes:
                st.error(n_)
        else:
            show_plan(r_mixed, p)
            st.markdown("**Matriks biaya per unit** (sel kosong = tidak boleh, karena tidak ada backorder)")
            dem_cols = [c for c in cost_mat.columns]
            st.dataframe(style_matrix(cost_mat, dem_cols, highlight=False), width="stretch")
            st.markdown("**Alokasi optimal** (sel berwarna = unit yang dikirim dari sumber ke bulan permintaan)")
            st.dataframe(style_matrix(alloc, list(alloc.columns)), width="stretch")
            if sel.startswith("Contoh buku 14.5"):
                st.warning(
                    "**Selisih dengan buku.** Buku melaporkan total $2,186,000, tetapi angka itu lebih kecil dari "
                    "batas bawah teoretis: 20,000 m² reguler × $40 + 24,700 m² sisa kebutuhan × $75 = $2,652,500 "
                    "(belum termasuk biaya simpan dan selisih harga subkontrak). Solver LP di aplikasi ini "
                    f"menghasilkan {money(r_mixed.total)}. Rincian perhitungan buku pada subbab 14.5 tampaknya keliru; "
                    "gunakan hasil di atas dan catat selisihnya saat belajar."
                )
            with st.expander("Cara membaca tabel transportasi"):
                st.markdown(
                    "- Baca per **kolom**: tiap kolom adalah satu bulan permintaan, dan jumlah selnya = permintaan bulan itu.\n"
                    "- Baca per **baris**: jumlah sel tidak boleh melebihi kapasitas sumber (kolom *Kapasitas*).\n"
                    "- Unit yang diproduksi bulan sebelumnya untuk dipakai bulan berikutnya otomatis dikenai biaya simpan.\n"
                    "- Kolom *Sisa* menunjukkan kapasitas yang tidak terpakai."
                )

    # ---------------- Perbandingan
    with t_cmp:
        results = [("Chase", r_chase, C_CHASE), ("Level", r_level, C_LEVEL)]
        if r_mixed.feasible:
            results.append(("Mixed", r_mixed, C_MIXED))

        fig = go.Figure()
        for k in COST_KEYS:
            ys = [r.costs[k] for _, r, _ in results]
            if sum(ys) > 0:
                fig.add_bar(name=k, x=[n_ for n_, _, _ in results], y=ys, marker_color=COST_COLORS[k])
        fig.update_layout(barmode="stack", height=400, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis=dict(title="Biaya (USD)"), legend=dict(orientation="h", y=-0.15))
        for n_, r, _ in results:
            fig.add_annotation(x=n_, y=r.total, text=f"<b>{money(r.total)}</b>", showarrow=False, yshift=14)
        st.plotly_chart(fig, width="stretch")

        rows = []
        for n_, r, _ in results:
            t = r.table
            rows.append({
                "Strategi": n_,
                "Total biaya": r.total,
                "Pekerja direkrut + di-PHK": float(t["Rekrut"].sum() + t["PHK"].sum()),
                "Persediaan maks (unit)": float(t["Persediaan akhir"].max()),
                "Lembur + subkontrak (unit)": float(t["Lembur"].sum() + t["Subkontrak"].sum()),
                "Backorder (unit-bulan)": float(t["Backorder"].sum()),
            })
        cmp_df = pd.DataFrame(rows)
        st.dataframe(
            cmp_df.style.format({c: "{:,.0f}" for c in cmp_df.columns if c != "Strategi"}),
            hide_index=True, width="stretch",
        )

        best = cmp_df.loc[cmp_df["Total biaya"].idxmin()]
        worst = cmp_df["Total biaya"].max()
        saving = (worst - best["Total biaya"]) / worst * 100 if worst else 0
        st.success(f"Biaya terendah: **{best['Strategi']}** ({money(best['Total biaya'])}), "
                   f"{saving:.1f}% lebih murah daripada yang termahal.")
        st.markdown(
            "**Jangan memilih hanya dari biaya.** Buku menekankan faktor lain: stabilitas pekerja "
            "(chase paling berat bagi pekerja), risiko stok menumpuk (level), dan ketergantungan pada vendor "
            "(mixed). Coba ubah biaya rekrut, biaya simpan, atau batas lembur di atas, lalu lihat kapan peringkat berubah."
        )

# ================================================================== 3. LP
with tab_lp:
    st.subheader("Linear programming untuk rencana produksi")
    st.markdown(
        "Dua produk khusus (A dan B) harus dipenuhi dalam tiga bulan dengan kapasitas terbatas. "
        "Tujuan: **meminimalkan biaya produksi** (contoh buku 14.5)."
    )
    lc1, lc2 = st.columns([1, 2])
    with lc1:
        cost_a = num("Biaya produksi A (per unit)", "lp_ca", 50, fmt="%.2f")
        cost_b = num("Biaya produksi B (per unit)", "lp_cb", 70, fmt="%.2f")
        allow_early = st.checkbox("Boleh produksi lebih awal (ada biaya simpan)", value=False, key="lp_early")
        lp_hold = num("Biaya simpan (per unit per bulan)", "lp_hold", 2, fmt="%.2f") if allow_early else 0.0
    with lc2:
        lp_df = st.data_editor(
            pd.DataFrame({"Bulan": [1, 2, 3], "Kapasitas": [100.0, 120.0, 110.0],
                          "Permintaan A": [20.0, 40.0, 50.0], "Permintaan B": [20.0, 30.0, 40.0]}),
            hide_index=True, disabled=["Bulan"], key="lp_table", width="stretch",
        )
    cap, da, db = lp_df["Kapasitas"].tolist(), lp_df["Permintaan A"].tolist(), lp_df["Permintaan B"].tolist()
    n_lp = len(cap)

    st.markdown("**Formulasi model**")
    st.latex(rf"\min Z = {cost_a:g}\sum_{{t=1}}^{{{n_lp}}} x_t + {cost_b:g}\sum_{{t=1}}^{{{n_lp}}} y_t"
             + (rf" + {lp_hold:g}\sum_t I_t" if allow_early else ""))
    st.markdown(
        "dengan $x_t$ dan $y_t$ = unit produk A dan B yang diproduksi bulan $t$.\n\n"
        + ("- Kapasitas: $x_t + y_t \\le K_t$\n"
           + ("- Permintaan kumulatif: $\\sum_{i\\le t} x_i \\ge \\sum_{i\\le t} d^A_i$ (juga untuk B)\n"
              if allow_early else "- Permintaan: $x_t \\ge d^A_t,\\; y_t \\ge d^B_t$\n")
           + "- Non-negatif: $x_t, y_t \\ge 0$")
    )

    res = lp_two_products(cost_a, cost_b, cap, da, db, lp_hold)
    if not res.success:
        st.error("Model tidak layak: permintaan melebihi kapasitas. Naikkan kapasitas atau turunkan permintaan.")
    else:
        x, y = res.x[:n_lp], res.x[n_lp:]
        out = pd.DataFrame({"Bulan": range(1, n_lp + 1), "Produk A": x, "Produk B": y,
                            "Total": x + y, "Kapasitas": cap, "Utilisasi": (x + y) / np.array(cap)})
        m1, m2 = st.columns(2)
        m1.metric("Biaya minimum (Z)", money(res.fun))
        m2.metric("Utilisasi kapasitas rata-rata", f"{out['Utilisasi'].mean() * 100:.0f}%")
        fig = go.Figure()
        fig.add_bar(x=out["Bulan"], y=out["Produk A"], name="Produk A", marker_color=C_REG)
        fig.add_bar(x=out["Bulan"], y=out["Produk B"], name="Produk B", marker_color=C_OT)
        fig.add_scatter(x=out["Bulan"], y=out["Kapasitas"], name="Kapasitas", mode="lines+markers",
                        line=dict(color=C_BACK, dash="dash"))
        fig.update_layout(barmode="stack", height=340, margin=dict(l=10, r=10, t=20, b=10),
                          xaxis=dict(tickmode="linear", dtick=1, title="Bulan"), yaxis_title="Unit",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(out.style.format({"Produk A": "{:,.0f}", "Produk B": "{:,.0f}", "Total": "{:,.0f}",
                                       "Kapasitas": "{:,.0f}", "Utilisasi": "{:.0%}"}),
                     hide_index=True, width="stretch")
        if not allow_early:
            st.caption(
                "Pada formulasi buku, permintaan berfungsi sebagai batas bawah produksi per bulan dan biaya unit "
                "sama tiap bulan, sehingga solusi optimalnya adalah memproduksi tepat sebesar permintaan. "
                "Aktifkan *produksi lebih awal* lalu kecilkan kapasitas bulan 3 (misal 60) untuk melihat LP "
                "memindahkan produksi ke bulan sebelumnya."
            )

# ================================================================== 4. MANAGEMENT COEFFICIENT
with tab_mc:
    st.subheader("Management coefficient model (Bowman)")
    st.markdown(
        "Regresi dari data historis menghasilkan hubungan **pekerja = a + b × permintaan**. Manajer kemudian "
        "boleh menyesuaikan koefisien berdasarkan informasi yang belum terlihat di data (misalnya otomasi baru)."
    )
    MC = {
        "Contoh buku 14.6": dict(
            hist=pd.DataFrame({"Bulan": range(1, 7), "Permintaan": [10000, 12000, 11000, 13000, 14000, 12500],
                               "Pekerja": [100, 110, 105, 115, 120, 112]}),
            fc=[13500, 14000, 15000, 14500, 13000, 12500], a=50.0, b=0.004),
        "Soal latihan no. 5–6": dict(
            hist=pd.DataFrame({"Bulan": range(1, 7), "Permintaan": [1000, 1200, 1500, 1300, 1100, 900],
                               "Pekerja": [60, 70, 80, 75, 65, 55]}),
            fc=[1400, 1600, 1300, 1100], a=None, b=None),
    }
    mc_sel = st.selectbox("Data", list(MC), key="mc_sel")
    M = MC[mc_sel]
    mk = list(MC).index(mc_sel)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Data historis**")
        hist = st.data_editor(M["hist"].astype(float), hide_index=True, disabled=["Bulan"], key=f"mc_hist_{mk}",
                              width="stretch")
    with cc2:
        st.markdown("**Ramalan permintaan ke depan**")
        fc_df = st.data_editor(
            pd.DataFrame({"Bulan": range(len(M["hist"]) + 1, len(M["hist"]) + 1 + len(M["fc"])),
                          "Permintaan": [float(v) for v in M["fc"]]}),
            hide_index=True, disabled=["Bulan"], key=f"mc_fc_{mk}", width="stretch")

    a_fit, b_fit, r2 = fit_regression(hist["Permintaan"], hist["Pekerja"])
    st.markdown(f"Hasil regresi: **Pekerja = {a_fit:.2f} + {b_fit:.5f} × Permintaan**  (R² = {r2:.3f})")

    ac1, ac2 = st.columns(2)
    a_def = M["a"] if M["a"] is not None else round(a_fit, 2)
    b_def = M["b"] if M["b"] is not None else round(b_fit, 5)
    with ac1:
        a_adj = st.number_input("Konstanta (a) setelah penyesuaian manajer", value=float(a_def), step=0.5,
                                format="%.2f", key=f"mc_a_{mk}")
    with ac2:
        b_adj = st.number_input("Koefisien (b) setelah penyesuaian manajer", value=float(b_def), step=0.0005,
                                format="%.5f", key=f"mc_b_{mk}",
                                help="Contoh buku: manajer menurunkan 0.005 menjadi 0.004 karena efisiensi meningkat.")

    out = fc_df.copy()
    out["Pekerja (regresi murni)"] = a_fit + b_fit * out["Permintaan"]
    out["Pekerja (disesuaikan manajer)"] = a_adj + b_adj * out["Permintaan"]
    out["Selisih"] = out["Pekerja (regresi murni)"] - out["Pekerja (disesuaikan manajer)"]

    fig = go.Figure()
    fig.add_scatter(x=hist["Permintaan"], y=hist["Pekerja"], mode="markers", name="Data historis",
                    marker=dict(color=C_DEMAND, size=9))
    xs = np.linspace(min(hist["Permintaan"].min(), out["Permintaan"].min()),
                     max(hist["Permintaan"].max(), out["Permintaan"].max()), 50)
    fig.add_scatter(x=xs, y=a_fit + b_fit * xs, mode="lines", name="Regresi", line=dict(color=C_REG, width=2.5))
    fig.add_scatter(x=xs, y=a_adj + b_adj * xs, mode="lines", name="Disesuaikan manajer",
                    line=dict(color=C_OT, width=2.5, dash="dash"))
    fig.add_scatter(x=out["Permintaan"], y=out["Pekerja (disesuaikan manajer)"], mode="markers", name="Rencana pekerja",
                    marker=dict(color=C_OT, size=11, symbol="diamond"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Permintaan (unit)",
                      yaxis_title="Pekerja", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(out.style.format({"Permintaan": "{:,.0f}", "Pekerja (regresi murni)": "{:,.1f}",
                                   "Pekerja (disesuaikan manajer)": "{:,.1f}", "Selisih": "{:,.1f}"}),
                 hide_index=True, width="stretch")
    st.caption("Buku membulatkan hasil ke bilangan bulat (misal 104 pekerja). Perbedaan kecil dengan buku muncul "
               "karena aplikasi menghitung regresi langsung dari data, sedangkan buku memakai persamaan yang sudah dibulatkan.")

# ================================================================== 5. JASA & YIELD
with tab_jasa:
    st.subheader("Aggregate planning di jasa")
    st.markdown(
        "Jasa **tidak bisa disimpan** dan sering tidak bisa di-backorder, jadi kapasitas bulan sepi tidak bisa "
        "menutup bulan sibuk. Contoh: firma akuntansi dengan permintaan jam tertagih per bulan."
    )
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        sv_dem = parse_series(st.text_input("Permintaan (jam) per bulan", "2000, 3500, 4000, 2500", key="sv_dem")) or [0.0]
        staff = num("Akuntan tetap", "sv_staff", 10)
        hrs = num("Jam tertagih per akuntan per bulan", "sv_hrs", 160)
        temp_cost = num("Biaya staf sementara ($/jam)", "sv_temp", 50, fmt="%.2f")
    cap_m = staff * hrs
    sv = pd.DataFrame({"Bulan": range(1, len(sv_dem) + 1), "Permintaan": sv_dem, "Kapasitas": cap_m})
    sv["Kekurangan"] = (sv["Permintaan"] - sv["Kapasitas"]).clip(lower=0)
    sv["Kapasitas menganggur"] = (sv["Kapasitas"] - sv["Permintaan"]).clip(lower=0)
    naive = max(0.0, sum(sv_dem) - cap_m * len(sv_dem))
    with sc2:
        fig = go.Figure()
        fig.add_bar(x=sv["Bulan"], y=sv["Permintaan"], name="Permintaan", marker_color=C_REG)
        fig.add_scatter(x=sv["Bulan"], y=sv["Kapasitas"], name="Kapasitas tetap", mode="lines",
                        line=dict(color=C_BACK, dash="dash", width=2.5))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(tickmode="linear", dtick=1, title="Bulan"),
                          yaxis_title="Jam", legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")
    k1, k2, k3 = st.columns(3)
    k1.metric("Kekurangan per bulan (dijumlah)", f"{sv['Kekurangan'].sum():,.0f} jam")
    k2.metric("Biaya staf sementara", money(sv["Kekurangan"].sum() * temp_cost))
    k3.metric("Kapasitas menganggur", f"{sv['Kapasitas menganggur'].sum():,.0f} jam")
    if abs(naive - sv["Kekurangan"].sum()) > 1e-6:
        st.warning(f"Perhitungan total kasar (permintaan total − kapasitas total) hanya menunjukkan {naive:,.0f} jam. "
                   "Itu meremehkan kebutuhan karena kapasitas menganggur di bulan sepi tidak bisa dipindahkan.")
    else:
        st.caption("Pada data ini setiap bulan permintaan melebihi kapasitas, sehingga hasilnya sama dengan "
                   "perhitungan total di buku (5,600 jam × $50 = $280,000). Ubah salah satu bulan menjadi rendah untuk melihat bedanya.")

    st.divider()
    st.subheader("Yield management")
    st.markdown(
        "Untuk kapasitas yang kedaluwarsa (kamar hotel, kursi pesawat), tujuannya **memaksimalkan pendapatan** lewat "
        "segmentasi pelanggan dan harga dinamis. Contoh hotel di buku:"
    )
    ycap = num("Jumlah kamar", "y_cap", 100)
    ycost = num("Biaya per kamar terisi ($)", "y_cost", 50, fmt="%.2f")
    yc1, yc2 = st.columns(2)
    scen = {}
    for col, label, vals in [(yc1, "Tanpa yield management", (60, 200, 20, 150)),
                             (yc2, "Dengan yield management", (70, 220, 30, 140))]:
        with col:
            st.markdown(f"**{label}**")
            kk = label[:4]
            bq = num("Kamar bisnis terjual", f"y_bq_{kk}", vals[0])
            bp = num("Tarif bisnis ($)", f"y_bp_{kk}", vals[1], fmt="%.2f")
            lq = num("Kamar wisata terjual", f"y_lq_{kk}", vals[2])
            lp_ = num("Tarif wisata ($)", f"y_lp_{kk}", vals[3], fmt="%.2f")
            sold = bq + lq
            rev = bq * bp + lq * lp_
            profit = rev - sold * ycost
            scen[label] = dict(sold=sold, rev=rev, profit=profit)
            if sold > ycap:
                st.error(f"Kamar terjual ({sold:.0f}) melebihi kapasitas ({ycap:.0f}).")
            st.metric("Pendapatan", money(rev))
            st.metric("Laba", money(profit))
            st.caption(f"Okupansi {sold / ycap * 100:.0f}% · RevPAR {money(rev / ycap)} · tarif rata-rata {money(rev / sold) if sold else '–'}")
    a_, b_ = scen["Tanpa yield management"], scen["Dengan yield management"]
    if a_["profit"] > 0:
        st.success(f"Kenaikan laba: {money(b_['profit'] - a_['profit'])} ({(b_['profit'] / a_['profit'] - 1) * 100:.1f}%). "
                   "Di buku: $3,600 atau sekitar 32.7%.")

# ================================================================== 6. MPS
with tab_mps:
    st.subheader("Dari rencana agregat ke Master Production Schedule")
    st.markdown(
        "Di ERP, rencana agregat per keluarga produk dipecah menjadi **jadwal produksi per produk per minggu** (MPS), "
        "lalu dicek kecukupan kapasitasnya (*rough-cut capacity planning*) sebelum MRP menghitung kebutuhan bahan. "
        "Tab ini mensimulasikan langkah itu."
    )
    g1, g2 = st.columns([1, 1])
    with g1:
        agg_txt = st.text_input("Rencana produksi agregat per bulan (unit, keluarga produk)",
                                "1200, 1500, 1000, 1300", key="mps_agg",
                                help="Bisa diambil dari kolom Produksi + Lembur + Subkontrak pada tab Strategi agregat.")
        weekly_cap = num("Kapasitas reguler per minggu (jam mesin/orang)", "mps_cap", 300)
        wk = st.text_input("Pola produksi per minggu (%)", "25, 25, 25, 25", key="mps_wk",
                           help="Empat angka untuk minggu 1–4. Contoh 20, 20, 30, 30 jika produksi condong ke akhir bulan.")
    with g2:
        mix = st.data_editor(
            pd.DataFrame({"Produk": ["Produk A", "Produk B", "Produk C"], "Porsi (%)": [50.0, 30.0, 20.0],
                          "Jam per unit": [0.6, 0.9, 1.2]}),
            hide_index=True, key="mps_mix", num_rows="dynamic", width="stretch",
        )
    agg = parse_series(agg_txt)
    wk_shares = parse_series(wk)
    mix = mix.dropna()
    if agg is None or wk_shares is None or len(wk_shares) != 4 or mix.empty or mix["Porsi (%)"].sum() <= 0:
        st.error("Pastikan rencana agregat berisi angka, pola minggu berisi tepat 4 angka, dan tabel produk terisi.")
    else:
        rows = []
        for mi, total in enumerate(agg, start=1):
            per_prod = split_preserving_total(total, mix["Porsi (%)"].tolist())
            for prod_name, qty, hpu in zip(mix["Produk"], per_prod, mix["Jam per unit"]):
                weekly = split_preserving_total(qty, wk_shares)
                rows.append(dict(Bulan=mi, Produk=prod_name, Total=qty, **{f"M{w + 1}": weekly[w] for w in range(4)},
                                 _hpu=hpu))
        mps = pd.DataFrame(rows)
        st.markdown("**MPS: unit yang dijadwalkan per minggu**")
        st.dataframe(mps.drop(columns="_hpu").style.format({c: "{:,.0f}" for c in ["Total", "M1", "M2", "M3", "M4"]}),
                     hide_index=True, width="stretch")

        # RCCP
        labels, util = [], []
        for mi in range(1, len(agg) + 1):
            sub = mps[mps["Bulan"] == mi]
            for w in range(1, 5):
                hours = float((sub[f"M{w}"] * sub["_hpu"]).sum())
                labels.append(f"B{mi}-M{w}")
                util.append(hours / weekly_cap if weekly_cap else np.nan)
        colors = [C_BACK if u > 1 else C_REG for u in util]
        fig = go.Figure(go.Bar(x=labels, y=[u * 100 for u in util], marker_color=colors,
                               text=[f"{u * 100:.0f}%" for u in util], textposition="outside"))
        fig.add_hline(y=100, line_dash="dash", line_color=C_BACK)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Beban terhadap kapasitas (%)",
                          xaxis_title="Bulan–minggu")
        st.markdown("**Rough-cut capacity: beban jam per minggu dibanding kapasitas**")
        st.plotly_chart(fig, width="stretch")
        over = [l for l, u in zip(labels, util) if u > 1]
        if over:
            st.warning(f"Minggu yang melebihi kapasitas: {', '.join(over)}. Pilihannya sama dengan chapter 14: geser produksi "
                       "ke minggu sebelumnya (persediaan), tambah lembur, atau subkontrak.")
        else:
            st.success("Semua minggu masih dalam kapasitas reguler.")

        st.markdown("**Langkah berikutnya di ERP**")
        st.markdown(
            "MPS yang sudah disetujui menjadi input **MRP**. MRP mengurai tiap produk lewat *bill of materials*, "
            "mengurangi stok on-hand, lalu menghasilkan purchase request untuk bahan beli dan work order untuk "
            "komponen buatan sendiri. Itulah materi modul Manufacturing berikutnya."
        )

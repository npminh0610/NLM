import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import tkinter.font as tkfont

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Arc, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

#---- Màu giao diện----
C = dict(
    header="#16231f", header_sub="#9fb5ae", header_btn_bg="#22332e",
    page="#eeefec", card="#ffffff", border="#e1e4df",
    teal="#0f6b5f", teal_soft="#dff3ec", mint="#8ee3cf",
    text="#16231f", muted="#5b6661",
    entry_border="#d9dcd7", code_bg="#efefed",
    plot_bg="#f7f8f6", grid="#d9dcd7", dim="#8a948f",
    link2="#5b4bd6", link3="#12936f", link4="#16231f",
    force="#c2610c", m3="#6d4fd6", m4="#c2307a", danger="#b3261e",
    phi="#e11d1d",
)
#---Phông chữ---#
FONT = "Segoe UI"
MONO = "Consolas"

#--Lịch sử thông số đã lưu----
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "lich_su_thong_so.json")
HISTORY_MAX = 50


#---Các phương án cho trước---#

PRESETS = {
    "PA 1": dict(L2=5.0, L3=4.5, lBM=2.0, lME=1.0, L4=6.0, L1=3.0, phi2=10.0,
                 omega2=5.0, omega2_dir="cùng", eps2=10.0, eps2_dir="cùng",
                 P3=210.0, alpha=10.0, M3=30.0, M4=150.0,
                 m3=2.0, Js3=2.5, m4=1.0),
    "PA 2": dict(L2=5.0, L3=4.0, lBM=2.5, lME=1.5, L4=6.0, L1=3.0, phi2=20.0,
                 omega2=5.0, omega2_dir="cùng", eps2=20.0, eps2_dir="ngược",
                 P3=240.0, alpha=20.0, M3=60.0, M4=200.0,
                 m3=3.0, Js3=3.0, m4=2.0),
}
DEFAULT = PRESETS["PA 1"]

FIELDS = [
    ("L2", "L₂ = lAB  (m)"), ("L3", "L₃ = lBC  (m)"),
    ("lBM", "lBM  (m)"), ("lME", "lME  (m)"),
    ("L4", "L₄ = lCD  (m)"), ("L1", "L₁ = lAD  (m)"),
    ("phi2", "Góc vị trí φ₂  (độ)"),
    ("omega2", "ω₂  (rad/s)"), ("eps2", "ε₂  (rad/s²)"),
    ("P3", "P₃  (N)"), ("alpha", "Góc α của P₃  (độ)"),
    ("M3", "M₃  (N·m)"), ("M4", "M₄  (N·m)"),
    ("m3", "m₃  (kg)"), ("Js3", "Jₛ₃  (kg·m²)"), ("m4", "m₄  (kg)"),
]
LABELS = dict(FIELDS)
DIR_FIELDS = [
    ("omega2_dir", "Chiều ω₂ so với kim đồng hồ"),
    ("eps2_dir", "Chiều ε₂ so với kim đồng hồ"),
]
LABELS.update(dict(DIR_FIELDS))
ALL_KEYS = [k for k, _ in FIELDS] + [k for k, _ in DIR_FIELDS]


# ----------------------------------------------------------------------
# 2. HÌNH HỌC
# ----------------------------------------------------------------------
def circle_intersect(p0, r0, p1, r1, branch="upper", prev=None):
    """Giao điểm 2 đường tròn. Nếu có `prev` thì chọn nghiệm gần prev nhất
    (giữ liên tục nhánh lắp ráp khi chạy hoạt hình), ngược lại chọn theo `branch`."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    d = np.linalg.norm(p1 - p0)
    if d < 1e-9 or d > r0 + r1 or d < abs(r0 - r1):
        return None
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(r0 ** 2 - a ** 2, 0.0))
    p2 = p0 + a * (p1 - p0) / d
    ux, uy = (p1 - p0) / d
    s1 = np.array([p2[0] + h * uy, p2[1] - h * ux])
    s2 = np.array([p2[0] - h * uy, p2[1] + h * ux])
    if prev is not None:
        prev = np.asarray(prev, dtype=float)
        return s1 if np.linalg.norm(s1 - prev) <= np.linalg.norm(s2 - prev) else s2
    if branch == "upper":
        return s1 if s1[1] >= s2[1] else s2
    return s2 if s1[1] >= s2[1] else s1


def solve_positions(p, prev_C=None):
    A = np.array([0.0, 0.0])
    D = np.array([p["L1"], 0.0])
    phi2 = np.radians(p["phi2"])
    B = A + p["L2"] * np.array([np.cos(phi2), np.sin(phi2)])
    Cc = circle_intersect(B, p["L3"], D, p["L4"], "upper", prev_C)
    if Cc is None:
        raise ValueError("Không dựng được cơ cấu với bộ thông số này "
                         "(vòng kín BCD không tồn tại). Kiểm tra L2, L3, L4, L1, φ2.")
    BC = np.linalg.norm(Cc - B)
    M = B + (p["lBM"] / BC) * (Cc - B)
    d = (Cc - B) / BC
    E = M + p["lME"] * np.array([d[1], -d[0]])
    return dict(A=A, B=B, C=Cc, D=D, M=M, E=E)


def compute_view_limits(p):
    xs, ys = [], []
    prev = None
    for a in range(0, 360, 4):
        try:
            pts = solve_positions(dict(p, phi2=a), prev)
        except ValueError:
            continue
        prev = pts["C"]
        for q in pts.values():
            xs.append(q[0]); ys.append(q[1])
    if not xs:
        return None
    return (min(xs) - 1.6, max(xs) + 2.0, min(ys) - 1.6, max(ys) + 1.4)

def can_full_rotate(p, step=1):
    """True nếu khâu dẫn quay trọn 360° liên tục, không văng nhánh lắp ráp."""
    prev, first = None, None
    for a in range(0, 360 + step, step):
        try:
            pts = solve_positions(dict(p, phi2=a), prev)
        except ValueError:
            return False
        if first is None:
            first = pts["C"]
        prev = pts["C"]
    return bool(np.allclose(prev, first, atol=1e-6))


# ----------------------------------------------------------------------
# 2b. PHÂN TÍCH VẬN TỐC
#     Quy ước: góc quay dương = NGƯỢC chiều kim đồng hồ (hệ Oxy, y hướng lên).
#     v_P = v_Q + ω × (P - Q), với  ω × r = ω·(-ry, rx)
#     vB = ω2 × AB ;  vC = vB + ω3 × BC = vD + ω4 × DC  (vD = 0)
# ----------------------------------------------------------------------
def perp(r):
    """Vectơ r quay +90° (tức ω × r với ω = 1)."""
    return np.array([-r[1], r[0]], dtype=float)


def solve_velocities(p, pts):
    A, B, Cc, D, M, E = (pts[k] for k in "ABCDME")
    w2 = -p["omega2"] if p["omega2_dir"] == "cùng" else p["omega2"]  # dấu: CCW dương
    vB = w2 * perp(B - A)

    # vB + w3·perp(C-B) = w4·perp(C-D)   ->   [perp(C-B) | -perp(C-D)]·[w3, w4]ᵀ = -vB
    Amat = np.column_stack([perp(Cc - B), -perp(Cc - D)])
    if abs(np.linalg.det(Amat)) < 1e-9:
        raise ValueError("BC // CD (vị trí đặc biệt): không xác định được ω₃, ω₄ từ họa đồ vận tốc.")
    w3, w4 = np.linalg.solve(Amat, -vB)

    vC = w4 * perp(Cc - D)
    vM = vB + w3 * perp(M - B)
    vE = vB + w3 * perp(E - B)
    return dict(w2=w2, w3=float(w3), w4=float(w4),
                A=np.zeros(2), D=np.zeros(2), B=vB, C=vC, M=vM, E=vE)


def rot_text(w, sym="ω"):
    """Chiều quay (so với kim đồng hồ) của đại lượng góc có dấu w (CCW dương)."""
    if abs(w) < 1e-9:
        return f"bằng 0 ({sym} = 0)" if sym != "ω" else "đứng yên (ω = 0)"
    return "↺ ngược chiều kim đồng hồ" if w > 0 else "↻ cùng chiều kim đồng hồ"


def vec_polar(v):
    mag = float(np.linalg.norm(v))
    ang = float(np.degrees(np.arctan2(v[1], v[0])) % 360) if mag > 1e-9 else None
    return mag, ang


def velocity_table_text(vel):
    rows = [f"{'Điểm':<6}{'|v| (m/s)':>12}{'Hướng so với Ox':>20}",
            "-" * 38]
    for name in "BCME":
        mag, ang = vec_polar(vel[name])
        a = f"{ang:.1f}°" if ang is not None else "—"
        rows.append(f"v{name:<5}{mag:>12.4f}{a:>20}")
    return "\n".join(rows)


def velocity_text(p, vel):
    w3, w4 = vel["w3"], vel["w4"]
    return (
        "PHÂN TÍCH VẬN TỐC\n" + "-" * 50 + "\n"
        "Phương trình vận tốc:\n"
        "  vB = ω2 x AB                (vB ⊥ AB)\n"
        "  vC = vB + vCB = vD + vCD    (vD = 0; vCB ⊥ BC, vCD ⊥ CD)\n"
        "  vM = vB + vMB, vE = vB + vEB (M, E thuộc khâu 3)\n\n"
        f"- ω2 = {abs(vel['w2']):.4f} rad/s, {rot_text(vel['w2'])}\n"
        f"- ω3 = {abs(w3):.4f} rad/s, {rot_text(w3)}\n"
        f"- ω4 = {abs(w4):.4f} rad/s, {rot_text(w4)}\n\n"
        "Vận tốc các điểm (hướng đo từ trục Ox, ngược chiều kim đồng hồ):\n"
        + velocity_table_text(vel) + "\n"
        "(Họa đồ vận tốc được lưu kèm trong file ảnh *_hoa_do_van_toc)\n"
    )


# ----------------------------------------------------------------------
# 2c. PHÂN TÍCH GIA TỐC
#     a_P = a_Q + ε × (P - Q) - ω²·(P - Q)      (ε × r = ε·(-ry, rx))
#     aB = ε2 × AB - ω2²·AB
#     aC = aB + ε3 × BC - ω3²·BC = aD + ε4 × DC - ω4²·DC   (aD = 0)
# ----------------------------------------------------------------------
def solve_accelerations(p, pts, vel):
    A, B, Cc, D, M, E = (pts[k] for k in "ABCDME")
    w2, w3, w4 = vel["w2"], vel["w3"], vel["w4"]
    e2 = -p["eps2"] if p["eps2_dir"] == "cùng" else p["eps2"]   # CCW dương
    aB = e2 * perp(B - A) - w2 ** 2 * (B - A)

    Amat = np.column_stack([perp(Cc - B), -perp(Cc - D)])
    if abs(np.linalg.det(Amat)) < 1e-9:
        raise ValueError("BC // CD (vị trí đặc biệt): không xác định được ε₃, ε₄.")
    rhs = -aB + w3 ** 2 * (Cc - B) - w4 ** 2 * (Cc - D)
    e3, e4 = np.linalg.solve(Amat, rhs)

    aC = e4 * perp(Cc - D) - w4 ** 2 * (Cc - D)
    aM = aB + e3 * perp(M - B) - w3 ** 2 * (M - B)
    aE = aB + e3 * perp(E - B) - w3 ** 2 * (E - B)
    return dict(e2=e2, e3=float(e3), e4=float(e4),
                A=np.zeros(2), D=np.zeros(2), B=aB, C=aC, M=aM, E=aE)


def accel_table_text(acc):
    rows = [f"{'Điểm':<6}{'|a| (m/s²)':>12}{'Hướng so với Ox':>20}",
            "-" * 38]
    for name in "BCME":
        mag, ang = vec_polar(acc[name])
        a = f"{ang:.1f}°" if ang is not None else "—"
        rows.append(f"a{name:<5}{mag:>12.4f}{a:>20}")
    return "\n".join(rows)


def accel_text(p, acc):
    e3, e4 = acc["e3"], acc["e4"]
    return (
        "PHÂN TÍCH GIA TỐC\n" + "-" * 50 + "\n"
        "Phương trình gia tốc:\n"
        "  aB = aBn + aBt         (aBn = ω2²·AB hướng B→A, aBt = ε2·AB ⊥ AB)\n"
        "  aC = aB + aCBn + aCBt = aD + aCDn + aCDt   (aD = 0)\n"
        "  aM = aB + aMB, aE = aB + aEB (M, E thuộc khâu 3)\n\n"
        f"- ε2 = {abs(acc['e2']):.4f} rad/s², {rot_text(acc['e2'], 'ε')}\n"
        f"- ε3 = {abs(e3):.4f} rad/s², {rot_text(e3, 'ε')}\n"
        f"- ε4 = {abs(e4):.4f} rad/s², {rot_text(e4, 'ε')}\n\n"
        "Gia tốc các điểm (hướng đo từ trục Ox, ngược chiều kim đồng hồ):\n"
        + accel_table_text(acc) + "\n"
        "(Họa đồ gia tốc được lưu kèm trong file ảnh *_hoa_do_gia_toc)\n"
    )


# ----------------------------------------------------------------------
# 3. VẼ LƯỢC ĐỒ
# ----------------------------------------------------------------------
def draw_mechanism(ax, pts, p, lim=None):
    ax.clear()
    ax.set_facecolor(C["plot_bg"])
    A, B, Cc, D, M, E = (pts[k] for k in "ABCDME")

    # giá + đường chuẩn AD
    ax.plot([A[0], D[0]], [A[1], D[1]], "--", color=C["muted"], lw=1, alpha=.7, zorder=1)
    for pt in (A, D):
        for k in range(-3, 4):
            ax.plot([pt[0] - .15 + .05 * k, pt[0] - .25 + .05 * k],
                    [pt[1] - .05, pt[1] - .30], color=C["muted"], lw=.8, alpha=.8, zorder=1)
        ax.plot([pt[0] - .35, pt[0] + .35], [pt[1] - .05] * 2, color=C["muted"], lw=1.4, zorder=1)

    # các khâu
    ax.plot(*zip(A, B), color=C["link2"], lw=4, zorder=2, solid_capstyle="round")
    ax.plot(*zip(B, Cc), color=C["link3"], lw=4, zorder=2, solid_capstyle="round")
    ax.plot(*zip(Cc, D), color=C["link4"], lw=4, zorder=2, solid_capstyle="round")
    ax.plot(*zip(M, E), "--", color=C["muted"], lw=1, alpha=.8, zorder=2)

    # ký hiệu vuông góc tại M
    u = (Cc - B) / np.linalg.norm(Cc - B)
    n = np.array([u[1], -u[0]])
    s = 0.12
    ax.plot(*zip(M + s * u, M + s * u + s * n, M + s * n), color=C["muted"], lw=.9, zorder=2)

    # khớp bản lề
    for pt, name in [(A, "A"), (B, "B"), (Cc, "C"), (D, "D")]:
        ax.add_patch(Circle(pt, .08, facecolor="white", edgecolor=C["text"], lw=2, zorder=4))
        ax.annotate(name, pt, textcoords="offset points", xytext=(8, 8),
                    fontsize=12, fontweight="bold", color=C["text"], zorder=5)
    ax.plot(*M, "o", color=C["text"], ms=4, zorder=4)
    ax.plot(*E, "o", color=C["text"], ms=5, zorder=4)
    ax.annotate(r"$M\equiv S_3$", M, textcoords="offset points", xytext=(8, 6),
                fontsize=10, fontweight="bold", color=C["text"], zorder=5)
    ax.annotate("E", E, textcoords="offset points", xytext=(6, -14),
                fontsize=10, fontweight="bold", color=C["text"], zorder=5)

    # góc φ2 (chuẩn hóa về [0, 360) để cung đúng cả khi φ2 > 180°)
    ang = np.degrees(np.arctan2(B[1] - A[1], B[0] - A[0])) % 360
    rad = .6
    if ang > 1e-6:
        ax.add_patch(Arc(A, 2 * rad, 2 * rad, theta1=0, theta2=ang,
                         color=C["phi"], lw=1.4, zorder=3))
    mid = np.radians(ang / 2)
    ax.annotate(r"$\varphi_2$", A + rad * 1.35 * np.array([np.cos(mid), np.sin(mid)]),
                color=C["phi"], fontsize=11, zorder=5)

    # lực P3 tại E
    al = np.radians(p["alpha"])
    Ftip = E + 1.1 * np.array([np.cos(al), np.sin(al)])
    ax.plot([E[0], E[0] + 1.3], [E[1]] * 2, ":", color=C["force"], lw=1, alpha=.7, zorder=2)
    ax.add_patch(FancyArrowPatch(E, Ftip, color=C["force"], lw=2.4, arrowstyle="-|>",
                                 mutation_scale=16, zorder=5))
    ax.annotate(r"$P_3$", Ftip, textcoords="offset points", xytext=(6, 4),
                color=C["force"], fontsize=11, fontweight="bold")
    ax.annotate(r"$\alpha$", E + 0.75 * np.array([1, 0]), textcoords="offset points",
                xytext=(0, -14), color=C["force"], fontsize=10)

    # mô men M3, M4: giá trị dương -> ngược chiều kim đồng hồ, âm -> cùng chiều kim đồng hồ
    def moment_arc(center, radius, label, color, value):
        ax.add_patch(Arc(center, 2 * radius, 2 * radius, theta1=30, theta2=300,
                         color=color, lw=1.8, zorder=3))
        if value >= 0:      # CCW: mũi tên ở đầu 300°, tiếp tuyến theo chiều tăng góc
            t = np.radians(300)
            tan = np.array([-np.sin(t), np.cos(t)])
        else:               # CW: mũi tên ở đầu 30°, tiếp tuyến theo chiều giảm góc
            t = np.radians(30)
            tan = np.array([np.sin(t), -np.cos(t)])
        tip = center + radius * np.array([np.cos(t), np.sin(t)])
        ax.add_patch(FancyArrowPatch(tip - .12 * tan, tip, color=color, lw=1.8,
                                     arrowstyle="-|>", mutation_scale=11, zorder=3,
                                     shrinkA=0, shrinkB=0))
        ax.annotate(label, center, textcoords="offset points", xytext=(-30, 0),
                    color=color, fontsize=11, fontweight="bold")
        return center

    c3 = moment_arc(M + .55 * np.array([-1, .3]), .32, r"$M_3$", C["m3"], p["M3"])
    c4 = moment_arc((Cc + D) / 2 + .5 * np.array([.6, .2]), .32, r"$M_4$", C["m4"], p["M4"])

    # nhãn chiều dài
    def lab(P1, P2, name, val, off=(0, 6)):
        ax.annotate(rf"$l_{{{name}}}$ = {val:.2f}", (P1 + P2) / 2, textcoords="offset points",
                    xytext=off, fontsize=9, ha="center", color=C["dim"], family="monospace")
    lab(A, B, "AB", p["L2"])
    lab(B, Cc, "BC", p["L3"], (26, -6))
    lab(Cc, D, "CD", p["L4"], (-30, 0))
    lab(A, D, "AD", p["L1"], (0, -22))
    lab(M, E, "ME", p["lME"], (16, -14))

    ax.set_aspect("equal", adjustable="datalim")
    allp = list(pts.values()) + [Ftip, c3, c4]
    xs = [q[0] for q in allp]
    ys = [q[1] for q in allp]
    if lim is not None:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    else:
        ax.set_xlim(min(xs) - 1.4, max(xs) + 1.6)
        ax.set_ylim(min(ys) - 1.4, max(ys) + 1.0)
    ax.grid(True, ls=":", lw=.7, color=C["grid"], alpha=.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ----------------------------------------------------------------------
# 3b. VẼ HỌA ĐỒ VẬN TỐC / GIA TỐC
#     Cực p (≡ a ≡ d, vận tốc = 0);  pb = vB, pc = vC, bc = vCB,
#     m nằm trên bc (theo tỉ lệ lBM/lBC), pm = vM, pe = vE, me = vEM.
#     Các vectơ vẽ đúng phương, chiều thực (không xoay 90°).
# ----------------------------------------------------------------------
def draw_vector_polygon(ax, d, sym, title, pole, sfx, pole_tex):
    ax.clear()
    ax.set_facecolor(C["plot_bg"])
    o = np.zeros(2)
    b, c, m, e = d["B"], d["C"], d["M"], d["E"]

    def arrow(p0, p1, color, lw=2.0, ls="-"):
        if np.linalg.norm(p1 - p0) < 1e-9:
            return
        ax.add_patch(FancyArrowPatch(p0, p1, color=color, lw=lw, arrowstyle="-|>",
                                     mutation_scale=12, linestyle=ls, zorder=3,
                                     shrinkA=0, shrinkB=0))

    arrow(o, b, C["link2"])
    arrow(o, c, C["link4"])
    arrow(b, c, C["link3"])
    arrow(o, m, C["m3"], lw=1.6)
    arrow(o, e, C["force"], lw=1.6)
    arrow(m, e, C["muted"], lw=1.2, ls="--")

    ax.plot(*o, "o", color=C["text"], ms=5, zorder=4)
    ax.annotate(pole_tex, o, textcoords="offset points", xytext=(-8, -16),
                fontsize=10, fontweight="bold", color=C["text"], ha="right")
    for pt, name, off in [(b, "b", (-10, 6)), (c, "c", (7, 6)),
                          (m, "m", (7, -14)), (e, "e", (7, 6))]:
        ax.plot(*pt, "o", color=C["text"], ms=4, zorder=4)
        ax.annotate(name + sfx, pt, textcoords="offset points", xytext=off,
                    fontsize=11, fontweight="bold", color=C["text"], zorder=5)

    def vec(sub):
        return r"$\vec{" + sym + r"}_{" + sub + r"}$"
    handles = [
        Line2D([0], [0], color=C["link2"], lw=2, label=f"{vec('B')} = {pole}b{sfx}"),
        Line2D([0], [0], color=C["link4"], lw=2, label=f"{vec('C')} = {pole}c{sfx}"),
        Line2D([0], [0], color=C["link3"], lw=2, label=f"{vec('CB')} = b{sfx}c{sfx}"),
        Line2D([0], [0], color=C["m3"], lw=2, label=f"{vec('M')} = {pole}m{sfx}"),
        Line2D([0], [0], color=C["force"], lw=2, label=f"{vec('E')} = {pole}e{sfx}"),
        Line2D([0], [0], color=C["muted"], lw=1.4, ls="--",
               label=f"{vec('EM')} = m{sfx}e{sfx}"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -0.08), ncol=3,
              fontsize=8, frameon=False)

    allp = np.array([o, b, c, m, e])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    if span < 1e-6:
        span = 1.0
        lo, hi = lo - .5, hi + .5
    mg = 0.18 * span
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(lo[0] - mg, hi[0] + mg)
    ax.set_ylim(lo[1] - mg, hi[1] + mg)
    ax.grid(True, ls=":", lw=.7, color=C["grid"], alpha=.8)
    ax.tick_params(labelsize=8, colors=C["muted"])
    for sp in ax.spines.values():
        sp.set_color(C["border"])
    ax.set_title(title, fontsize=10, color=C["text"], fontweight="bold")


def draw_velocity_polygon(ax, vel):
    draw_vector_polygon(ax, vel, "v", "Họa đồ vận tốc (đơn vị m/s)", "p", "",
                        r"$p\equiv a\equiv d$")


def draw_accel_polygon(ax, acc):
    draw_vector_polygon(ax, acc, "a", "Họa đồ gia tốc (đơn vị m/s²)", "π", "'",
                        r"$\pi\equiv a\equiv d$")


# ----------------------------------------------------------------------
# 4. PHÂN TÍCH CẤU TRÚC
# ----------------------------------------------------------------------
def structure_numbers():
    n, p5, p4 = 3, 4, 0
    return n, p5, p4, 3 * n - 2 * p5 - p4


def structural_analysis_text(p):
    n, p5, p4, W = structure_numbers()
    dm = {"cùng": "cùng chiều kim đồng hồ", "ngược": "ngược chiều kim đồng hồ"}
    return (
        "PHÂN TÍCH CẤU TRÚC CƠ CẤU\n" + "-" * 50 + "\n"
        f"- Số khâu động n = {n} (khâu 2: AB, khâu 3: BC, khâu 4: CD)\n"
        f"- Số khớp loại 5 p5 = {p5} (A, B, C, D);  số khớp loại 4 p4 = {p4}\n"
        f"- Bậc tự do: W = 3n - 2p5 - p4 = 3.{n} - 2.{p5} - {p4} = {W}\n\n"
        "TÁCH NHÓM ASSUR\n" + "-" * 50 + "\n"
        "- Khâu dẫn: khâu 2 (AB) nối giá bằng khớp quay tại A -> cơ cấu loại I (hạng 1).\n"
        "- Còn lại khâu 3 (BC) và khâu 4 (CD), nối nhau và nối giá qua 3 khớp quay B, C, D\n"
        "  (dạng RRR). Bậc tự do nhóm: Wn = 3.2 - 2.3 = 0 -> nhóm tĩnh định hợp lệ.\n\n"
        "XẾP LOẠI CƠ CẤU\n" + "-" * 50 + "\n"
        "Cơ cấu = 1 khâu dẫn loại I ghép nối tiếp 1 nhóm Assur loại 2.\n"
        "=> CƠ CẤU LOẠI 2 (HẠNG II).\n\n"
        "THÔNG SỐ ĐỘNG HỌC / LỰC ĐÃ NHẬP\n" + "-" * 50 + "\n"
        f"- ω2 = {p['omega2']} rad/s, {dm.get(p['omega2_dir'], p['omega2_dir'])}\n"
        f"- ε2 = {p['eps2']} rad/s², {dm.get(p['eps2_dir'], p['eps2_dir'])}\n"
        f"- P3 = {p['P3']} N, hợp phương ngang góc α = {p['alpha']}° (tại E)\n"
        f"- M3 = {p['M3']} N·m (tại M), M4 = {p['M4']} N·m (trên khâu 4)\n"
        f"- m3 = {p['m3']} kg, Js3 = {p['Js3']} kg·m², m4 = {p['m4']} kg\n"
    )


# ----------------------------------------------------------------------
# 5. LỊCH SỬ THÔNG SỐ
# ----------------------------------------------------------------------
def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # chỉ nhận các dòng có đủ khóa thông số (tránh lỗi với file lịch sử cũ/thiếu)
            return [h for h in data
                    if isinstance(h, dict) and isinstance(h.get("params"), dict)
                    and all(k in h["params"] for k in ALL_KEYS)]
    except (OSError, ValueError):
        pass
    return []


def save_history(hist):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def history_label(h):
    p = h["params"]
    return (f"{h.get('time', '')}  φ₂={p['phi2']:g}°  "
            f"L₂={p['L2']:g} L₃={p['L3']:g} L₄={p['L4']:g} L₁={p['L1']:g}")


# ----------------------------------------------------------------------
# 6. GIAO DIỆN
# ----------------------------------------------------------------------
def card(parent, **kw):
    return tk.Frame(parent, bg=C["card"], highlightthickness=1,
                    highlightbackground=C["border"], **kw)


def pick_font(candidates, fallback):
    have = set(tkfont.families())
    for f in candidates:
        if f in have:
            return f
    return fallback


class MechanismHMI(tk.Tk):
    def __init__(self):
        
        super().__init__()
        self.title("Lược đồ cơ cấu 4 khâu bản lề")
        self.geometry("1600x900")
        self.minsize(1300, 720)
        self.configure(bg=C["page"])
        self.entries, self.dir_vars = {}, {}
        self.last_params = None
        self.last_vel = None
        self.last_acc = None
        self._after = None
        self._loading = False
        self.history = load_history()
        self._full_rot = False
        global FONT, MONO
        FONT = pick_font(["Segoe UI", "Inter", "IBM Plex Sans", "Noto Sans",
                          "DejaVu Sans", "Arial"], "TkDefaultFont")
        MONO = pick_font(["Consolas", "IBM Plex Mono", "DejaVu Sans Mono",
                          "Menlo", "Courier New"], "TkFixedFont")
        self.animating = False
        self.anim_dir = 1
        self._anim_job = None
        self._anim_lim = None
        self._prev_C = None          # vị trí C của khung hình trước (giữ liên tục nhánh)
        self._style()
        self._header()
        body = tk.Frame(self, bg=C["page"])
        body.pack(fill="both", expand=True, padx=16, pady=16)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0)
        body.rowconfigure(0, weight=1)
        self._left(body)
        self._middle(body)
        self._right(body)

        for v in list(self.entries.values()) + list(self.dir_vars.values()):
            v.trace_add("write", self._schedule)

        # khởi tạo: nạp bộ thông số gần nhất trong lịch sử, nếu chưa có thì dùng mặc định
        self._refresh_history_list()
        start = self.history[0]["params"] if self.history else DEFAULT
        self.load_params(start)

    # ---------- style ----------
    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TCombobox", fieldbackground="white", background="white",
                     foreground=C["text"], bordercolor=C["entry_border"],
                     lightcolor="white", darkcolor="white", arrowcolor=C["teal"],
                     padding=6, font=(FONT, 11))
        st.map("TCombobox", fieldbackground=[("readonly", "white")],
               foreground=[("readonly", C["text"])],
               selectbackground=[("readonly", "white")],
               selectforeground=[("readonly", C["text"])])
        st.configure("TNotebook", background=C["card"], borderwidth=0)
        st.configure("TNotebook.Tab", padding=(18, 8), font=(FONT, 11, "bold"),
                     background=C["code_bg"], foreground=C["muted"], borderwidth=0)
        st.map("TNotebook.Tab",
               background=[("selected", C["teal"])],
               foreground=[("selected", "white")])

    # ---------- header ----------
    def _header(self):
        h = tk.Frame(self, bg=C["header"])
        h.pack(fill="x")
        inner = tk.Frame(h, bg=C["header"])
        inner.pack(fill="x", padx=28, pady=16)

        ic = tk.Canvas(inner, width=58, height=58, bg=C["header"], highlightthickness=0)
        ic.pack(side="left", padx=(0, 16))
        pts = [(10, 48), (20, 20), (46, 14), (42, 46)]
        ic.create_line(*pts[0], *pts[1], *pts[2], fill="white", width=2)
        ic.create_line(*pts[2], *pts[3], fill="white", width=2)
        for x, y in pts:
            ic.create_oval(x - 6, y - 6, x + 6, y + 6, fill=C["header"], outline="white", width=2)

        t = tk.Frame(inner, bg=C["header"])
        t.pack(side="left")
        tk.Label(t, text="Lược đồ cơ cấu 4 khâu bản lề", bg=C["header"], fg="white",
                 font=(FONT, 20, "bold")).pack(anchor="w")
        tk.Label(t, text="Nhập thông số · vẽ lược đồ · phân tích cấu trúc · vận tốc · gia tốc",
                 bg=C["header"], fg=C["header_sub"], font=(FONT, 11)).pack(anchor="w")

    # ---------- cột trái: nhập liệu + lịch sử ----------
    def _left(self, body):
        wrap = tk.Frame(body, bg=C["page"], width=420)
        wrap.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        wrap.grid_propagate(False)
        cv = tk.Canvas(wrap, bg=C["page"], highlightthickness=0, width=400)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cv, bg=C["page"])
        win = cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))

        def wheel(e):
            cv.yview_scroll(-1 if (e.num == 4 or e.delta > 0) else 1, "units")

        def on(_):
            cv.bind_all("<MouseWheel>", wheel)
            cv.bind_all("<Button-4>", wheel)
            cv.bind_all("<Button-5>", wheel)

        def off(_):
            cv.unbind_all("<MouseWheel>"); cv.unbind_all("<Button-4>"); cv.unbind_all("<Button-5>")
        cv.bind("<Enter>", on); cv.bind("<Leave>", off)

        def section(title):
            c = card(inner)
            c.pack(fill="x", pady=(0, 14))
            tk.Label(c, text=title, bg=C["card"], fg=C["teal"],
                     font=(FONT, 12, "bold")).pack(anchor="w", padx=18, pady=(16, 6))
            g = tk.Frame(c, bg=C["card"])
            g.pack(fill="x", padx=18, pady=(0, 16))
            return g

        def field(g, key, r, col, span=1, combo=False):
            g.columnconfigure(col, weight=1, uniform="f")
            tk.Label(g, text=LABELS[key], bg=C["card"], fg=C["muted"], font=(FONT, 10),
                     justify="left", anchor="w", wraplength=165 * span
                     ).grid(row=r * 2, column=col, columnspan=span, sticky="w",
                            padx=(0 if col == 0 else 8, 0), pady=(8, 3))
            var = tk.StringVar()
            if combo:
                w = ttk.Combobox(g, textvariable=var, state="readonly",
                                 values=["cùng", "ngược"], font=(FONT, 11))
                self.dir_vars[key] = var
            else:
                w = tk.Entry(g, textvariable=var, font=(MONO, 12), fg=C["text"], bg="white",
                             relief="flat", highlightthickness=1,
                             highlightbackground=C["entry_border"], highlightcolor=C["teal"],
                             insertbackground=C["text"])
                self.entries[key] = var
            w.grid(row=r * 2 + 1, column=col, columnspan=span, sticky="ew",
                   padx=(0 if col == 0 else 8, 0), ipady=7 if not combo else 2)

        # ---- chọn phương án cho trước ----
        pc_ = card(inner)
        pc_.pack(fill="x", pady=(0, 14))
        tk.Label(pc_, text="0 · PHƯƠNG ÁN CHO TRƯỚC (BẢNG 1)", bg=C["card"], fg=C["teal"],
                 font=(FONT, 12, "bold")).pack(anchor="w", padx=18, pady=(16, 8))
        prow = tk.Frame(pc_, bg=C["card"])
        prow.pack(fill="x", padx=18, pady=(0, 16))
        for i, name in enumerate(PRESETS):
            prow.columnconfigure(i, weight=1, uniform="pa")
            b = tk.Label(prow, text=name, bg=C["teal_soft"], fg=C["teal"],
                         font=(FONT, 11, "bold"), pady=9, cursor="hand2")
            b.bind("<Button-1>", lambda e, n=name: self.on_preset(n))
            b.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))

        g = section("1 · KÍCH THƯỚC KHÂU")
        for i, (a, b) in enumerate([("L2", "L3"), ("lBM", "lME"), ("L4", "L1")]):
            field(g, a, i, 0); field(g, b, i, 1)
        field(g, "phi2", 3, 0, span=2)

        g = section("2 · ĐỘNG HỌC KHÂU DẪN")
        field(g, "omega2", 0, 0); field(g, "omega2_dir", 0, 1, combo=True)
        field(g, "eps2", 1, 0); field(g, "eps2_dir", 1, 1, combo=True)

        g = section("3 · LỰC VÀ MÔMEN")
        field(g, "P3", 0, 0); field(g, "alpha", 0, 1)
        field(g, "M3", 1, 0); field(g, "M4", 1, 1)

        g = section("4 · KHỐI LƯỢNG VÀ QUÁN TÍNH")
        for i, k in enumerate(("m3", "Js3", "m4")):
            field(g, k, 0, i)

        def mkbtn(parent, text, bg, fg, cmd, pack_kw=None, font_size=11, pady=11):
            b = tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, font_size, "bold"),
                         pady=pady, cursor="hand2")
            b.bind("<Button-1>", lambda e: cmd())
            b.pack(**(pack_kw or dict(fill="x", pady=(0, 8))))
            return b
        mkbtn(inner, "▶  Vẽ lược đồ & Phân tích (cấu trúc + vận tốc + gia tốc)",
              C["teal"], "white", lambda: self.on_draw(False))
        mkbtn(inner, "💾  Lưu kết quả (ảnh + họa đồ + thông số)", C["teal_soft"], C["teal"],
              self.on_save)

        # ---- lịch sử ----
        hc = card(inner)
        hc.pack(fill="x", pady=(6, 14))
        tk.Label(hc, text="5 · LỊCH SỬ THÔNG SỐ ĐÃ NHẬP", bg=C["card"], fg=C["teal"],
                 font=(FONT, 12, "bold")).pack(anchor="w", padx=18, pady=(16, 2))
        tk.Label(hc, text="Mỗi lần bấm “Vẽ lược đồ” sẽ lưu một dòng. Nhấp đúp để nạp lại.",
                 bg=C["card"], fg=C["muted"], font=(FONT, 9), wraplength=340,
                 justify="left").pack(anchor="w", padx=18, pady=(0, 8))
        lbf = tk.Frame(hc, bg=C["card"])
        lbf.pack(fill="x", padx=18)
        self.hist_lb = tk.Listbox(lbf, height=7, font=(MONO, 9), activestyle="none",
                                  exportselection=False, relief="flat",
                                  highlightthickness=1, highlightbackground=C["entry_border"],
                                  selectbackground=C["teal"], selectforeground="white",
                                  fg=C["text"], bg="white")
        hsb = ttk.Scrollbar(lbf, orient="vertical", command=self.hist_lb.yview)
        self.hist_lb.configure(yscrollcommand=hsb.set)
        hsb.pack(side="right", fill="y")
        self.hist_lb.pack(side="left", fill="x", expand=True)
        self.hist_lb.bind("<Double-Button-1>", lambda e: self.on_history_load())

        br = tk.Frame(hc, bg=C["card"])
        br.pack(fill="x", padx=18, pady=(10, 16))
        for i in range(3):
            br.columnconfigure(i, weight=1, uniform="hb")
        for i, (txt, bg, fg, cmd) in enumerate([
                ("Nạp lại", C["teal_soft"], C["teal"], self.on_history_load),
                ("Xóa dòng", C["code_bg"], C["text"], self.on_history_delete),
                ("Xóa hết", C["code_bg"], C["danger"], self.on_history_clear)]):
            b = tk.Label(br, text=txt, bg=bg, fg=fg, font=(FONT, 10, "bold"),
                         pady=8, cursor="hand2")
            b.bind("<Button-1>", lambda e, f=cmd: f())
            b.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))

    # ---------- cột giữa: lược đồ ----------
    def _middle(self, body):
        mid = tk.Frame(body, bg=C["page"])
        mid.grid(row=0, column=1, sticky="nsew")
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)

        pc = card(mid)
        pc.grid(row=0, column=0, sticky="nsew")
        top = tk.Frame(pc, bg=C["card"])
        top.pack(fill="x", padx=22, pady=(20, 8))
        tk.Label(top, text="Lược đồ cơ cấu", bg=C["card"], fg=C["text"],
                 font=(FONT, 15, "bold")).pack(side="left")

        self.speed = tk.Scale(top, from_=1, to=10, orient="horizontal", length=120,
                              label="Tốc độ", bg=C["card"], highlightthickness=0,
                              troughcolor=C["teal_soft"], font=(FONT, 9))
        self.speed.set(3)
        self.speed.pack(side="right")
        self.btn_anim = tk.Label(top, text="▶  Chạy chuyển động", bg=C["teal"], fg="white",
                                 font=(FONT, 10, "bold"), padx=14, pady=8, cursor="hand2")
        self.btn_anim.bind("<Button-1>", lambda e: self.on_anim_toggle())
        self.btn_anim.pack(side="right", padx=(0, 14))

        self.fig = Figure(figsize=(6.6, 5.0), facecolor="white")
        self.fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=pc)
        self.canvas.get_tk_widget().configure(bg="white", highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=22, pady=(0, 6))

        lg = tk.Frame(pc, bg=C["card"], highlightthickness=0)
        lg.pack(fill="x", padx=22, pady=(4, 16))
        tk.Frame(lg, bg=C["border"], height=1).pack(fill="x", pady=(0, 10))
        row = tk.Frame(lg, bg=C["card"])
        row.pack(anchor="w")
        for color, text in [(C["link2"], "Khâu 2 (AB) – dẫn"), (C["link3"], "Khâu 3 (BC) – truyền"),
                            (C["link4"], "Khâu 4 (CD)"), (C["force"], "Lực P₃"), (C["phi"], "Góc φ₂"),
                            (C["m3"], "M₃"), (C["m4"], "M₄")]:
            tk.Frame(row, bg=color, width=22, height=5).pack(side="left", padx=(0, 6))
            tk.Label(row, text=text, bg=C["card"], fg=C["muted"],
                     font=(FONT, 10)).pack(side="left", padx=(0, 16))

    # ---------- cột phải: tab Cấu trúc + Vận tốc + Gia tốc ----------
    TAB_STRIP = "#ffffff"   # nền thanh tab (trắng)

    def _right(self, body):
        rc = card(body, width=480)
        rc.grid(row=0, column=2, sticky="ns", padx=(16, 0))
        rc.pack_propagate(False)
        self.rc = rc
        self.tab_active = "struct"
        self.tab_rects = {}
        self.tab_bar = tk.Canvas(rc, height=50, bg=self.TAB_STRIP, highlightthickness=0,
                                 cursor="hand2")
        self.tab_bar.pack(fill="x")
        self.tab_bar.bind("<Configure>", lambda e: self._draw_tabs())
        self.tab_bar.bind("<Button-1>", self._tab_click)
        content = tk.Frame(rc, bg=C["card"])
        content.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        self.sec = {k: dict(body=tk.Frame(content, bg=C["card"])) for k in ("struct", "vel", "acc")}
        self._tab_structure(self.sec["struct"]["body"])
        self._tab_velocity(self.sec["vel"]["body"])
        self._tab_acceleration(self.sec["acc"]["body"])
        self._set_section("struct")

    def _draw_tabs(self):
        c = self.tab_bar
        c.delete("all")
        w, H = c.winfo_width(), 50
        if w < 40:
            return
        pad, r = 8, 12
        avail = w - 2 * pad
        sel_w = int(avail * 0.40)
        oth_w = (avail - sel_w) // 2
        x = pad
        for key, title in (("struct", "CẤU TRÚC"), ("vel", "VẬN TỐC"), ("acc", "GIA TỐC")):
            on = (key == self.tab_active)
            tw = sel_w if on else oth_w
            x0, x1, y0 = x, x + tw, (6 if on else 15)
            self.tab_rects[key] = (x0, x1)
            top = [x0, y0 + 2 * r, x0, y0 + r, x0, y0, x0 + r, y0,
                   x1 - r, y0, x1, y0, x1, y0 + r, x1, y0 + 2 * r]
            if on:      # tab đang chọn: xanh teal, chữ trắng
                c.create_polygon(*top, smooth=True, fill=C["teal"], outline="")
                c.create_rectangle(x0, y0 + r, x1, H, fill=C["teal"], outline="")
            else:       # tab còn lại: trắng, viền mảnh
                c.create_polygon(*top, smooth=True, fill="white", outline="")
                c.create_rectangle(x0, y0 + r, x1, H, fill="white", outline="")
                c.create_line(x0, H, *top, x1, H, smooth=True, fill=C["border"], width=1)
            c.create_text((x0 + x1) / 2, (y0 + H) / 2 + 1, text=title,
                          fill="white" if on else C["muted"],
                          font=(FONT, 13 if on else 11, "bold"))
            x = x1

    def _tab_click(self, e):
        for k, (x0, x1) in self.tab_rects.items():
            if x0 <= e.x <= x1:
                self._set_section(k)
                return

    def _set_section(self, active):
        self.tab_active = active
        self._draw_tabs()
        for k, s in self.sec.items():
            if k == active:
                s["body"].pack(fill="both", expand=True)
            else:
                s["body"].pack_forget()

    def _tab_structure(self, rc):
        n, p5, p4, W = structure_numbers()
        sr = tk.Frame(rc, bg=C["card"])
        sr.pack(fill="x", padx=6, pady=(12, 0))
        for i in range(4):
            sr.columnconfigure(i, weight=1, uniform="s")
        for i, (cap, val, hl) in enumerate([
                ("Số khâu động", f"n = {n}", False),
                ("Khớp loại 5", f"p₅ = {p5}", False),
                ("Khớp loại 4", f"p₄ = {p4}", False),
                ("Bậc tự do", f"W = {W}", True)]):
            bg = C["teal"] if hl else C["card"]
            cc = tk.Frame(sr, bg=bg, highlightthickness=1,
                          highlightbackground=C["teal"] if hl else C["border"])
            cc.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            tk.Label(cc, text=cap, bg=bg, fg="white" if hl else C["muted"],
                     font=(FONT, 9), anchor="w").pack(anchor="w", padx=8, pady=(8, 0))
            tk.Label(cc, text=val, bg=bg, fg="white" if hl else C["text"],
                     font=(MONO, 14, "bold")).pack(anchor="w", padx=8, pady=(0, 8))
        tk.Label(rc, text=(f"W = 3n − 2p₅ − p₄\n"
                           f"  = 3·{n} − 2·{p5} − {p4} = {W}"),
                 bg=C["code_bg"], fg=C["text"], font=(MONO, 12), justify="left", anchor="w",
                 padx=18, pady=14).pack(fill="x", padx=6, pady=(12, 14))

        steps = [
            ("01", "Khâu dẫn",
             "Khâu 2 (AB) nối với giá bằng khớp quay tại A. Khâu dẫn cùng giá tạo thành cơ cấu loại I (hạng 1)."),
            ("02", "Tách nhóm tĩnh định",
             "Còn lại khâu 3 (BC) và khâu 4 (CD), nối nhau và nối giá qua 3 khớp quay B, C, D (dạng RRR). "
             "Bậc tự do nhóm: Wₙ = 3·2 − 2·3 = 0, là nhóm tĩnh định hợp lệ."),
            ("03", "Xếp loại",
             "Cơ cấu gồm 1 khâu dẫn loại I ghép nối tiếp với 1 nhóm tĩnh định loại 2. "
             "Nhóm có bậc cao nhất là loại 2."),
        ]
        for num, title, txt in steps:
            r = tk.Frame(rc, bg=C["card"])
            r.pack(fill="x", padx=16, pady=(0, 12))
            cv = tk.Canvas(r, width=44, height=44, bg=C["card"], highlightthickness=0)
            cv.create_oval(2, 2, 42, 42, fill=C["teal_soft"], outline="")
            cv.create_text(22, 22, text=num, fill=C["teal"], font=(MONO, 12, "bold"))
            cv.pack(side="left", anchor="n", padx=(0, 14))
            col = tk.Frame(r, bg=C["card"])
            col.pack(side="left", fill="x", expand=True)
            tk.Label(col, text=title, bg=C["card"], fg=C["text"],
                     font=(FONT, 12, "bold")).pack(anchor="w")
            tk.Label(col, text=txt, bg=C["card"], fg=C["muted"], font=(FONT, 11),
                     wraplength=340, justify="left").pack(anchor="w", pady=(3, 0))

        kl = tk.Frame(rc, bg=C["header"])
        kl.pack(side="bottom", fill="x", padx=16, pady=16)
        tk.Label(kl, text="KẾT LUẬN", bg=C["header"], fg=C["mint"],
                 font=(FONT, 10, "bold")).pack(anchor="w", padx=20, pady=(18, 2))
        tk.Label(kl, text="Cơ cấu loại 2 (hạng II)", bg=C["header"], fg="white",
                 font=(FONT, 18, "bold")).pack(anchor="w", padx=20)
        tk.Label(kl, text="Cơ cấu 4 khâu bản lề dạng tay quay – thanh truyền – con lắc, "
                          "tùy điều kiện Grashof của L₁, L₂, L₃, L₄.",
                 bg=C["header"], fg="#c7d6d1", font=(FONT, 11), wraplength=370,
                 justify="left").pack(anchor="w", padx=20, pady=(6, 18))

    def _tab_velocity(self, rc):
        tk.Label(rc, text="vB = ω₂×AB ;  vC = vB + vCB = vD + vCD  (vD = 0)",
                 bg=C["card"], fg=C["muted"], font=(MONO, 9)).pack(anchor="w", padx=6, pady=(12, 10))

        wr = tk.Frame(rc, bg=C["card"])
        wr.pack(fill="x", padx=16)
        wr.columnconfigure(0, weight=1, uniform="w")
        wr.columnconfigure(1, weight=1, uniform="w")
        self.w_val, self.w_dir = {}, {}
        for i, (key, cap) in enumerate([("w3", "Vận tốc góc khâu 3 (ω₃)"),
                                        ("w4", "Vận tốc góc khâu 4 (ω₄)")]):
            c = tk.Frame(wr, bg=C["teal_soft"])
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            tk.Label(c, text=cap, bg=C["teal_soft"], fg=C["muted"], font=(FONT, 9),
                     anchor="w").pack(anchor="w", padx=12, pady=(10, 0))
            self.w_val[key] = tk.Label(c, text="—", bg=C["teal_soft"], fg=C["teal"],
                                       font=(MONO, 15, "bold"), anchor="w")
            self.w_val[key].pack(anchor="w", padx=12)
            self.w_dir[key] = tk.Label(c, text="", bg=C["teal_soft"], fg=C["text"],
                                       font=(FONT, 9), wraplength=170, justify="left", anchor="w")
            self.w_dir[key].pack(anchor="w", padx=12, pady=(0, 10))

        self.v_txt = tk.Label(rc, text="", bg=C["code_bg"], fg=C["text"], font=(MONO, 10),
                              justify="left", anchor="w", padx=14, pady=10, wraplength=420)
        self.v_txt.pack(fill="x", padx=16, pady=(10, 8))

        self.vfig = Figure(figsize=(4.4, 3.0), facecolor="white")
        self.vfig.subplots_adjust(left=.13, right=.96, top=.90, bottom=.22)
        self.vax = self.vfig.add_subplot(111)
        self.vcanvas = FigureCanvasTkAgg(self.vfig, master=rc)
        self.vcanvas.get_tk_widget().configure(bg="white", highlightthickness=0)
        self.vcanvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def _update_velocity(self, vel, err=None):
        if vel is None:
            for k in ("w3", "w4"):
                self.w_val[k].config(text="—")
                self.w_dir[k].config(text="")
            self.v_txt.config(text=err or "Chưa có dữ liệu.", fg=C["danger"])
            self.vax.clear()
            self.vax.set_facecolor(C["plot_bg"])
            self.vax.text(.5, .5, "Chưa có họa đồ vận tốc", ha="center", va="center",
                          transform=self.vax.transAxes, color=C["muted"], fontsize=10)
            self.vax.set_xticks([]); self.vax.set_yticks([])
            self.vcanvas.draw_idle()
            return
        for k in ("w3", "w4"):
            w = vel[k]
            self.w_val[k].config(text=f"{abs(w):.4f} rad/s")
            self.w_dir[k].config(text=rot_text(w))
        self.v_txt.config(text=velocity_table_text(vel), fg=C["text"])
        draw_velocity_polygon(self.vax, vel)
        self.vcanvas.draw_idle()

    def _tab_acceleration(self, rc):
        tk.Label(rc, text="aB = ε₂×AB − ω₂²·AB ;  aC = aB + aCB = aD + aCD  (aD = 0)",
                 bg=C["card"], fg=C["muted"], font=(MONO, 9), wraplength=430,
                 justify="left").pack(anchor="w", padx=6, pady=(12, 10))

        wr = tk.Frame(rc, bg=C["card"])
        wr.pack(fill="x", padx=16)
        wr.columnconfigure(0, weight=1, uniform="e")
        wr.columnconfigure(1, weight=1, uniform="e")
        self.e_val, self.e_dir = {}, {}
        for i, (key, cap) in enumerate([("e3", "Gia tốc góc khâu 3 (ε₃)"),
                                        ("e4", "Gia tốc góc khâu 4 (ε₄)")]):
            c = tk.Frame(wr, bg=C["teal_soft"])
            c.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            tk.Label(c, text=cap, bg=C["teal_soft"], fg=C["muted"], font=(FONT, 9),
                     anchor="w").pack(anchor="w", padx=12, pady=(10, 0))
            self.e_val[key] = tk.Label(c, text="—", bg=C["teal_soft"], fg=C["teal"],
                                       font=(MONO, 13, "bold"), anchor="w")
            self.e_val[key].pack(anchor="w", padx=12)
            self.e_dir[key] = tk.Label(c, text="", bg=C["teal_soft"], fg=C["text"],
                                       font=(FONT, 9), wraplength=170, justify="left", anchor="w")
            self.e_dir[key].pack(anchor="w", padx=12, pady=(0, 10))

        self.a_txt = tk.Label(rc, text="", bg=C["code_bg"], fg=C["text"], font=(MONO, 10),
                              justify="left", anchor="w", padx=14, pady=10, wraplength=420)
        self.a_txt.pack(fill="x", padx=16, pady=(10, 8))

        self.afig = Figure(figsize=(4.4, 3.0), facecolor="white")
        self.afig.subplots_adjust(left=.13, right=.96, top=.90, bottom=.22)
        self.aax = self.afig.add_subplot(111)
        self.acanvas = FigureCanvasTkAgg(self.afig, master=rc)
        self.acanvas.get_tk_widget().configure(bg="white", highlightthickness=0)
        self.acanvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def _update_acceleration(self, acc, err=None):
        if acc is None:
            for k in ("e3", "e4"):
                self.e_val[k].config(text="—")
                self.e_dir[k].config(text="")
            self.a_txt.config(text=err or "Chưa có dữ liệu.", fg=C["danger"])
            self.aax.clear()
            self.aax.set_facecolor(C["plot_bg"])
            self.aax.text(.5, .5, "Chưa có họa đồ gia tốc", ha="center", va="center",
                          transform=self.aax.transAxes, color=C["muted"], fontsize=10)
            self.aax.set_xticks([]); self.aax.set_yticks([])
            self.acanvas.draw_idle()
            return
        for k in ("e3", "e4"):
            self.e_val[k].config(text=f"{abs(acc[k]):.4f} rad/s²")
            self.e_dir[k].config(text=rot_text(acc[k], "ε"))
        self.a_txt.config(text=accel_table_text(acc), fg=C["text"])
        draw_accel_polygon(self.aax, acc)
        self.acanvas.draw_idle()

    # ---------- lịch sử ----------
    def _refresh_history_list(self):
        self.hist_lb.delete(0, "end")
        for h in self.history:
            self.hist_lb.insert("end", history_label(h))

    def _add_history(self, p):
        if self.history and self.history[0]["params"] == p:
            self.history[0]["time"] = datetime.now().strftime("%d/%m %H:%M:%S")
        else:
            self.history.insert(0, dict(time=datetime.now().strftime("%d/%m %H:%M:%S"),
                                        params=dict(p)))
            del self.history[HISTORY_MAX:]
        save_history(self.history)
        self._refresh_history_list()

    def _selected_index(self):
        sel = self.hist_lb.curselection()
        if not sel:
            messagebox.showinfo("Lịch sử", "Hãy chọn một dòng trong lịch sử.")
            return None
        return sel[0]

    def on_history_load(self):
        i = self._selected_index()
        if i is not None:
            self.load_params(self.history[i]["params"])

    def on_history_delete(self):
        i = self._selected_index()
        if i is not None:
            del self.history[i]
            save_history(self.history)
            self._refresh_history_list()

    def on_history_clear(self):
        if self.history and messagebox.askyesno("Xóa lịch sử",
                                                "Xóa toàn bộ lịch sử thông số đã nhập?"):
            self.history = []
            save_history(self.history)
            self._refresh_history_list()

    # ---------- logic ----------
    def on_preset(self, name):
        self.load_params(PRESETS[name])   # nạp vào các ô nhập và vẽ
        self.on_draw(False)               # ghi vào lịch sử như khi bấm "Vẽ"

    def load_params(self, p):
        self._loading = True
        for k, v in p.items():
            if k in self.dir_vars:
                self.dir_vars[k].set(v)
            elif k in self.entries:
                self.entries[k].set(f"{v:g}" if isinstance(v, (int, float)) else str(v))
        self._loading = False
        self.on_draw(True)

    def _schedule(self, *_):
        if self._loading:
            return
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(450, lambda: self.on_draw(True))

    def _read_params(self):
        p = {}
        for key, label in FIELDS:
            raw = self.entries[key].get().strip().replace(",", ".")
            if raw == "":
                raise ValueError(f"Chưa nhập '{label}'.")
            try:
                p[key] = float(raw)
            except ValueError:
                raise ValueError(f"Giá trị không hợp lệ tại '{label}': '{raw}'")
            if not np.isfinite(p[key]):
                raise ValueError(f"Giá trị không hợp lệ tại '{label}': '{raw}'")
        for key, label in DIR_FIELDS:
            v = self.dir_vars[key].get()
            if not v:
                raise ValueError(f"Chưa chọn '{label}'.")
            p[key] = v
        if min(p["L1"], p["L2"], p["L3"], p["L4"]) <= 0:
            raise ValueError("Chiều dài các khâu (L1, L2, L3, L4) phải > 0.")
        if p["lBM"] < 0 or p["lME"] < 0:
            raise ValueError("lBM và lME phải ≥ 0.")
        if p["lBM"] > p["L3"]:
            raise ValueError("lBM phải ≤ L3 = lBC (M nằm trên đoạn BC).")
        return p

    def _show_message(self, msg):
        self.ax.clear()
        self.ax.set_facecolor(C["plot_bg"])
        self.ax.text(.5, .5, msg, ha="center", va="center", transform=self.ax.transAxes,
                     color=C["muted"], fontsize=11, wrap=True)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.canvas.draw_idle()

    def on_draw(self, silent=False):
        try:
            p = self._read_params()
            pts = solve_positions(p, self._prev_C if self.animating else None)
        except ValueError as ex:
            # dữ liệu lỗi: xóa kết quả cũ để không lưu nhầm thông số cũ
            self.last_params = None
            self.last_vel = None
            self.last_acc = None
            self._update_velocity(None, str(ex))
            self._update_acceleration(None, str(ex))
            if silent:
                self._show_message(str(ex))
            else:
                messagebox.showerror("Lỗi dữ liệu", str(ex))
            return
        if self.animating:
            self._prev_C = pts["C"]
        draw_mechanism(self.ax, pts, p, self._anim_lim)
        self.canvas.draw_idle()
        self.last_params = p

        # vận tốc, gia tốc (nếu gặp vị trí đặc biệt thì vẫn vẽ lược đồ, chỉ báo lỗi ở các tab)
        try:
            self.last_vel = solve_velocities(p, pts)
            self._update_velocity(self.last_vel)
            self.last_acc = solve_accelerations(p, pts, self.last_vel)
            self._update_acceleration(self.last_acc)
        except ValueError as ex:
            self.last_vel = None
            self.last_acc = None
            self._update_velocity(None, str(ex))
            self._update_acceleration(None, str(ex))

        if not silent:           # chỉ lưu lịch sử khi người dùng chủ động bấm Vẽ
            self._add_history(p)

    def on_anim_toggle(self):
        if self.animating:
            self.animating = False
            self._anim_lim = None
            self._prev_C = None
            self.btn_anim.config(text="▶  Chạy chuyển động", bg=C["teal"])
            if self._anim_job:
                self.after_cancel(self._anim_job)
                self._anim_job = None
            return
        try:
            p = self._read_params()
            self._prev_C = solve_positions(p)["C"]
        except ValueError as ex:
            messagebox.showerror("Lỗi dữ liệu", str(ex))
            return
        self._anim_lim = compute_view_limits(p)
        self._full_rot = can_full_rotate(p)
        self.animating = True
        self.anim_dir = 1
        self.btn_anim.config(text="⏸  Dừng", bg=C["danger"])
        self._anim_tick()

    def _anim_tick(self):
        if not self.animating:
            return
        try:
            p = self._read_params()
        except ValueError:
            self.on_anim_toggle()
            return
        step = float(self.speed.get())
        base = -1 if p["omega2_dir"] == "cùng" else 1   # cùng chiều kim đồng hồ -> φ2 giảm
        new = p["phi2"] + base * self.anim_dir * step
        try:
            solve_positions(dict(p, phi2=new), self._prev_C)
        except ValueError:
            if self._full_rot:
                # quay trọn vòng được mà vẫn lỗi (bước quá lớn): thử bước nhỏ, không đảo chiều
                new = p["phi2"] + base * self.anim_dir * 1.0
                try:
                    solve_positions(dict(p, phi2=new), self._prev_C)
                except ValueError:
                    self.on_anim_toggle()
                    return
            else:
                # tay quay-lắc: chỉ lắc trong một khoảng, đảo chiều là đúng
                self.anim_dir *= -1
                new = p["phi2"] + base * self.anim_dir * step
                try:
                    solve_positions(dict(p, phi2=new), self._prev_C)
                except ValueError:
                    self.on_anim_toggle()
                    return
        self._loading = True                             # không kích hoạt vẽ trễ 450 ms
        self.entries["phi2"].set(f"{new % 360:.1f}")
        self._loading = False
        self.on_draw(True)
        if self.animating:
            self._anim_job = self.after(40, self._anim_tick)
        
    def on_save(self):
        if self.last_params is None:
            messagebox.showwarning("Chưa có kết quả", "Hãy nhập thông số hợp lệ trước khi lưu.")
            return
        default = "coCau_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="Lưu kết quả (tạo ảnh lược đồ, ảnh họa đồ vận tốc, gia tốc và .txt)",
            defaultextension=".png",
            initialfile=default, filetypes=[("Ảnh PNG", "*.png"), ("Ảnh PDF", "*.pdf")])
        if not path:
            return
        base, ext = os.path.splitext(path)
        ext = ext if ext.lower() in (".png", ".pdf") else ".png"
        img = base + ext
        vimg = base + "_hoa_do_van_toc" + ext
        aimg = base + "_hoa_do_gia_toc" + ext
        txt = base + ".txt"
        saved = [img]
        try:
            self.fig.savefig(img, dpi=200, bbox_inches="tight", facecolor=self.fig.get_facecolor())
            if self.last_vel is not None:
                self.vfig.savefig(vimg, dpi=200, bbox_inches="tight",
                                  facecolor=self.vfig.get_facecolor())
                saved.append(vimg)
            if self.last_acc is not None:
                self.afig.savefig(aimg, dpi=200, bbox_inches="tight",
                                  facecolor=self.afig.get_facecolor())
                saved.append(aimg)
            p = self.last_params
            lines = ["KẾT QUẢ CƠ CẤU 4 KHÂU BẢN LỀ",
                     "Thời gian lưu: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                     "=" * 50, "THÔNG SỐ ĐẦU VÀO", "-" * 50]
            for key, label in FIELDS + DIR_FIELDS:
                lines.append(f"{label:<32}= {p[key]}")
            lines += ["", "=" * 50, structural_analysis_text(p)]
            if self.last_vel is not None:
                lines += ["=" * 50, velocity_text(p, self.last_vel)]
            else:
                lines += ["=" * 50, "PHÂN TÍCH VẬN TỐC: không xác định được ở vị trí này."]
            if self.last_acc is not None:
                lines += ["=" * 50, accel_text(p, self.last_acc)]
            else:
                lines += ["=" * 50, "PHÂN TÍCH GIA TỐC: không xác định được ở vị trí này."]
            with open(txt, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            saved.append(txt)
        except OSError as ex:
            messagebox.showerror("Lỗi lưu file", str(ex))
            return
        messagebox.showinfo("Đã lưu", "Đã lưu:\n" + "\n".join("- " + s for s in saved))


if __name__ == "__main__":
    MechanismHMI().mainloop()

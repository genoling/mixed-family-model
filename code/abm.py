# -*- coding: utf-8 -*-
"""混合家庭模式 ABM —— 可复现实验脚本（论文第五章的数据来源）。

U = R*S*(1-C)；S = 1-exp(-beta*(s_ext + kappa*同辈数))
  s_ext>0 修正原稿 S_A=0 -> U_A=0 的硬错误（独生子女仍有学校/社区同伴）
  共居 R = (1-lam)*(E/n) + lam*((E+E_p)/(n+n_p))  （lam=0 即各自独立，修正原式在 lam=0 时仍平摊的错）
  共居需 A 角色(n=1) 与 B 角色(n>=2) 配对；供子需 A 家接收；献子需 B 家接收；未匹配者退回纯模式（记失败率）
用法: python abm.py --mode baseline|phase|all
输出: results/*.csv, figures/*.png
"""
from __future__ import annotations
import argparse, csv, os
from dataclasses import dataclass, replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAMES = ["pureA", "pureB", "foster", "dedicate", "cohabit"]
LABELS = ["纯A(独生)", "纯B(多子)", "供子", "献子", "共居(A+B)"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class P:
    n_agents: int = 300
    generations: int = 80
    alpha: float = 0.15      # 资源稀释
    beta: float = 0.6        # 社会化转化
    s_ext: float = 2.0       # 家庭外同伴等效数
    kappa_f: float = 0.6     # 跨家庭关系折损
    kappa_c: float = 1.0     # 共居互动系数
    theta: float = 0.8       # A 家资源接纳度
    delta: float = 0.6       # B 家资源分配系数
    lam: float = 0.5         # 共居资源共享比例
    c_foster: float = 0.22
    c_cohabit: float = 0.08
    temp: float = 0.05       # softmax 温度
    eps: float = 0.05        # 探索率
    inherit_p: float = 0.7
    income_tol: float = 0.35
    o_min: float = 0.30
    seed: int = 20260930


def dilution(E, n, alpha):
    """人均资源（含稀释）：(E/n)·[1-alpha(n-1)]"""
    return (E / n) * (1.0 - alpha * (n - 1))


def soc(beta, s_ext, peers):
    """社会化：1-exp(-beta*(s_ext+peers))"""
    return 1.0 - np.exp(-beta * (s_ext + peers))


def dev(m, E, n, E_h, n_h, p: P):
    """某家庭内子女的平均发展水平 U（m: 模式码; 收养/献子时 E_h/n_h 为对方家庭）"""
    if m == 0:                                   # 纯A
        return E * soc(p.beta, p.s_ext, 0.0), 1.0
    if m == 1:                                   # 纯B
        R = dilution(E, n, p.alpha)
        return R * soc(p.beta, p.s_ext, n - 1), float(n)
    if m == 2:                                   # 供子：送一子到 A 家（宿主家 n_h 个亲子女）
        u_out = (p.theta * (E_h / (n_h + 1))) * soc(p.beta, p.s_ext, p.kappa_f * n_h) * (1 - p.c_foster)
        k = max(n - 1, 0)
        if k <= 0:
            return u_out, 1.0
        u_in = dilution(E, k, p.alpha) * soc(p.beta, p.s_ext, k - 1)
        return (u_out + k * u_in) / (k + 1), n
    if m == 3:                                   # 献子：独子到 B 家（宿主家 n_h>=2）
        u = (p.delta * (E_h / (n_h + 1))) * soc(p.beta, p.s_ext, p.kappa_f * (n_h - 1)) * (1 - p.c_foster)
        return u, 1.0
    if m == 4:                                   # 共居
        R = (1 - p.lam) * (E / n) + p.lam * ((E + E_h) / (n + n_h))
        return R * soc(p.beta, p.s_ext, p.kappa_c * (n + n_h - 1)) * (1 - p.c_cohabit), float(n)
    raise ValueError(m)


def host_dev(p: P, E_h, n_h):
    """宿主家亲子女：分母 +1 被稀释，但获得 1 名跨家庭同伴 → 返回 (接收后, 接收前)"""
    return ((E_h / (n_h + 1)) * soc(p.beta, p.s_ext, p.kappa_c),
            (E_h / n_h) * soc(p.beta, p.s_ext, n_h - 1))


def match(mode, role, E, o, p, rng):
    """返回 partner(-1 未匹配)。共居双向配对；供子/献子需对方家庭接收。"""
    N = len(mode)
    partner = np.full(N, -1, dtype=int)
    cap = np.zeros(N, dtype=int)

    def greedy(seek, hosts):
        order = rng.permutation(len(seek))
        for i in order:
            s = int(seek[i])
            cand = [h for h in hosts if h != s and cap[h] > 0 and o[h] >= p.o_min and abs(E[h] - E[s]) <= p.income_tol]
            if not cand:
                continue
            h = min(cand, key=lambda x: abs(E[x] - E[s]))
            partner[s] = h
            cap[h] -= 1

    a_role = np.where(role == 0)[0]
    b_role = np.where(role == 1)[0]
    # 共居：A 角色 与 B 角色 双向需要
    ca = np.array([i for i in a_role if mode[i] == 4], dtype=int)
    cb = np.array([i for i in b_role if mode[i] == 4], dtype=int)
    cap[ca] = 1
    cap[cb] = 1
    greedy(cb, list(ca))
    for s in cb:                                  # A 侧写回
        if partner[s] >= 0:
            partner[partner[s]] = s
    # 供子：B 家求 A 家接收（n_h==1）
    host_a = [i for i in a_role if mode[i] != 4]
    cap[host_a] = 1
    greedy(np.array([i for i in b_role if mode[i] == 2], dtype=int), host_a)
    # 献子：A 家求 B 家接收（n_h>=2）
    host_b = [i for i in b_role if mode[i] != 4]
    cap[host_b] = 1
    greedy(np.array([i for i in a_role if mode[i] == 3], dtype=int), host_b)
    return partner


def run(p: P):
    rng = np.random.default_rng(p.seed)
    N = p.n_agents
    E = rng.beta(2, 5, N) * 0.9 + 0.1
    o = rng.random(N)
    n = rng.integers(1, 5, N).astype(float)
    role = (n >= 2).astype(int)
    mode = rng.choice(5, N, p=[0.35, 0.35, 0.10, 0.10, 0.10])
    # 角色与模式对齐（纯A/献子须 n=1；纯B/供子须 n>=2）
    bad0 = (mode == 0) & (role == 1); mode[bad0] = 1
    bad1 = (mode == 1) & (role == 0); mode[bad1] = 0
    bad2 = (mode == 2) & (role == 0); mode[bad2] = 0
    bad3 = (mode == 3) & (role == 1); mode[bad3] = 1

    shares, devs, mix, fail = [], [], [], []
    for _ in range(p.generations):
        # 内生子女数（B 角色）
        bi = np.where(role == 1)[0]
        for i in bi:
            best = max(((dilution(np.array([E[i]]), np.array([c]), p.alpha)[0]
                         * soc(p.beta, p.s_ext, c - 1), c) for c in (2, 3, 4)))
            n[i] = float(best[1])
        # 匹配 + 退回
        partner = match(mode, role, E, o, p, rng)
        f = 0
        for i in range(N):
            if mode[i] in (2, 3, 4) and partner[i] < 0:
                mode[i] = 1 if role[i] == 1 else 0
                f += 1
        # 发展水平
        u = np.zeros(N)
        for i in range(N):
            h = partner[i] if partner[i] >= 0 else i
            u[i], _ = dev(mode[i], E[i], n[i], E[h], n[h], p)
        # 宿主家庭的亲子女：资源被 +1 稀释，换得 1 名跨家庭同伴；按开放度 o 作利他加权
        for s in range(N):
            h = partner[s]
            if h >= 0 and mode[s] in (2, 3):
                u1, u0 = host_dev(p, E[h], n[h])
                u[h] = (1 - o[h]) * u0 + o[h] * u1
        # 绩效—模仿 + 探索（按模式平均发展水平做 softmax）
        mu = np.array([u[mode == m].mean() if (mode == m).any() else 0.0 for m in range(5)])
        w = np.exp((mu - mu.max()) / p.temp); w /= w.sum()
        nm = rng.choice(5, N, p=w)
        ex = rng.random(N) < p.eps
        if ex.any():
            nm[ex] = rng.integers(0, 5, int(ex.sum()))
        # 更新角色
        nr = role.copy()
        for i in range(N):
            if nm[i] in (0, 3):
                nr[i] = 0
            elif nm[i] in (1, 2):
                nr[i] = 1
            else:
                Ea = E[role == 0].mean(); Eb = E[role == 1].mean()
                ua = dev(4, E[i], 1.0, Eb, 3.0, p)[0]
                ub = dev(4, E[i], 3.0, Ea, 1.0, p)[0]
                nr[i] = 0 if ua >= ub else 1
        role = nr.astype(int)
        mode = nm
        n = np.where(role == 0, 1.0, np.maximum(n, 2.0))
        # 遗传
        keep = rng.random(N) < p.inherit_p
        src = rng.integers(0, N, N)
        E = np.where(keep, E[src], rng.beta(2, 5, N) * 0.9 + 0.1)
        o = np.where(keep, o[src], rng.random(N))
        shares.append([float((mode == m).mean()) for m in range(5)])
        devs.append([float(u[mode == m].mean()) if (mode == m).any() else 0.0 for m in range(5)])
        mix.append(float(sum((mode == m).mean() for m in (2, 3, 4))))
        fail.append(f / N)
    return np.array(shares), np.array(devs), np.array(mix), np.array(fail)


def reps(p: P, k: int):
    S, D, M, F = [], [], [], []
    for r in range(k):
        s, d, m, f = run(replace(p, seed=p.seed + r))
        S.append(s); D.append(d); M.append(m); F.append(f)
    return np.array(S), np.array(D), np.array(M), np.array(F)


def tail(a, t=10):
    return float(a[:, -t:].mean())


def phase(p: P, k: int, g: int):
    """β × 寄养摩擦 c_foster（混合模式的主导通道是寄养，而非共居）"""
    betas = np.linspace(0.15, 1.20, g)
    cs = np.linspace(0.05, 0.50, g)
    Z = np.zeros((g, g))
    for i, b in enumerate(betas):
        for j, c in enumerate(cs):
            v = []
            for r in range(k):
                s, *_ = run(replace(p, beta=float(b), c_foster=float(c), seed=p.seed + 7919 * i + 37 * j + r))
                v.append(s[-10:, 2:5].sum(axis=1).mean())
            Z[j, i] = float(np.mean(v))
    return betas, cs, Z


def lam_scan(p: P, k: int, g: int):
    """λ（共居资源共享比例）× s_ext：共居在何处成为可行均衡（指标=共居占比）"""
    lams = np.linspace(0.0, 1.0, g)
    sxs = np.linspace(0.0, 3.0, g)
    Z = np.zeros((g, g))
    for i, sx in enumerate(sxs):
        for j, lm in enumerate(lams):
            v = []
            for r in range(k):
                s, *_ = run(replace(p, s_ext=float(sx), lam=float(lm), seed=p.seed + 613 * i + 29 * j + r))
                v.append(s[-10:, 4].mean())
            Z[j, i] = float(np.mean(v))
    return lams, sxs, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="baseline", choices=["baseline", "regime", "phase", "all"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--grid", type=int, default=12)
    a = ap.parse_args()
    p = P()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    out = []
    if a.mode in ("baseline", "all"):
        S, D, M, F = reps(p, a.reps)
        for r in range(a.reps):
            out.append("rep%d " % r + " ".join("%s=%.3f" % (NAMES[m], S[r, -10:, m].mean()) for m in range(5))
                       + " mixed=%.3f dev=%s fail=%.3f" % (M[r, -10:].mean(),
                       "/".join("%.3f" % D[r, -10:, m].mean() for m in range(5)), F[r, -10:].mean()))
        sm = S.mean(0); sd = S.std(0); dm = D.mean(0); mm = M.mean(0); fm = F.mean(0)
        with open(os.path.join(ROOT, "results", "baseline_evolution.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["gen"] + [f"share_{m}" for m in NAMES] + [f"dev_{m}" for m in NAMES] + ["mixed", "fail"])
            for g in range(p.generations):
                w.writerow([g] + ["%.5f" % v for v in list(sm[g]) + list(dm[g]) + [mm[g], fm[g]]])
        with open(os.path.join(ROOT, "results", "baseline_replicates.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rep"] + [f"share_{m}" for m in NAMES] + [f"dev_{m}" for m in NAMES] + ["mixed", "fail"])
            for r in range(a.reps):
                w.writerow([r] + ["%.5f" % v for v in list(S[r, -10:].mean(0)) + list(D[r, -10:].mean(0))
                                  + [M[r, -10:].mean(), F[r, -10:].mean()]])
        out.append("\n== 5 次重复（末 10 代均值 ± 标准差, n=%d 家庭, %d 代）==" % (p.n_agents, p.generations))
        for m in range(5):
            out.append("  %-10s share=%.3f±%.3f  dev=%.4f" % (LABELS[m], sm[-10:, m].mean(), sd[-10:, m].mean(), dm[-10:, m].mean()))
        out.append("  混合合计  share=%.3f±%.3f" % (mm[-10:].mean(), M[:, -10:].mean(0).std()))
        out.append("  平均发展水平 末代=%.4f 首代=%.4f" % (((dm[-1] * sm[-1]).sum()), ((dm[0] * sm[0]).sum())))
        out.append("  匹配失败率 末代=%.3f" % fm[-10:].mean())
        # 图 1
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=200)
        gens = np.arange(p.generations)
        ax[0].stackplot(gens, *[sm[:, m] * 100 for m in range(5)], labels=LABELS, colors=COLORS, alpha=.9)
        ax[0].set_xlabel("世代"); ax[0].set_ylabel("模式占比 (%)"); ax[0].legend(loc="upper right", fontsize=8)
        ax[0].set_xlim(0, p.generations - 1); ax[0].set_ylim(0, 100); ax[0].set_title("(a) 五种模式占比演化")
        for m in range(5):
            ax[1].plot(gens, dm[:, m], color=COLORS[m], label=LABELS[m])
        ax[1].set_xlabel("世代"); ax[1].set_ylabel("子女平均发展水平 U"); ax[1].legend(fontsize=8)
        ax[1].set_title("(b) 各模式发展水平")
        fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "abm_evolution.png"))
        plt.close(fig)
    if a.mode in ("regime", "all"):
        # 关键区制扫描：家庭外同伴 s_ext（学校/社区能否替代同胞）+ beta
        betas_r = [0.3, 0.6, 1.0]
        sxs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        tab = np.zeros((len(betas_r), len(sxs), 5))
        for ib, b in enumerate(betas_r):
            for ix, sx in enumerate(sxs):
                acc = []
                for r in range(3):
                    s, *_ = run(replace(p, beta=b, s_ext=sx, seed=p.seed + 131 * ix + r))
                    acc.append(s[-10:].mean(0))
                tab[ib, ix] = np.mean(acc, axis=0)
        with open(os.path.join(ROOT, "results", "regime_s_ext.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["beta", "s_ext"] + [f"share_{m}" for m in NAMES] + ["mixed"])
            for ib, b in enumerate(betas_r):
                for ix, sx in enumerate(sxs):
                    w.writerow([b, sx] + ["%.4f" % v for v in tab[ib, ix]] + ["%.4f" % tab[ib, ix, 2:5].sum()])
        out.append("\n== 区制扫描：家庭外同伴等效数 s_ext（每点 3 次重复，末 10 代均值）==")
        for ib, b in enumerate(betas_r):
            out.append("  beta=%.1f  纯A = %s" % (b, " ".join("%.2f" % tab[ib, ix, 0] for ix in range(len(sxs)))))
            out.append("          混合 = %s" % " ".join("%.2f" % tab[ib, ix, 2:5].sum() for ix in range(len(sxs))))
        out.append("  (s_ext 列 = %s)" % " ".join("%.1f" % s for s in sxs))
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=200)
        for ib, b in enumerate(betas_r):
            ax[0].plot(sxs, tab[ib, :, 0], "o-", label=r"$\beta$=%.1f" % b)
            ax[1].plot(sxs, tab[ib, :, 2:5].sum(axis=1), "o-", label=r"$\beta$=%.1f" % b)
        ax[0].set_xlabel(r"家庭外同伴等效数 $s_{ext}$"); ax[0].set_ylabel("纯A（独生）稳态占比")
        ax[0].set_title("(a) 独生模式的存续取决于家庭外同伴"); ax[0].legend(); ax[0].set_ylim(-0.03, 0.8)
        ax[1].set_xlabel(r"家庭外同伴等效数 $s_{ext}$"); ax[1].set_ylabel("混合模式稳态占比")
        ax[1].set_title("(b) 混合模式的相对优势随 $s_{ext}$ 衰减"); ax[1].legend(); ax[1].set_ylim(-0.03, 0.8)
        fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "abm_regime.png"))
        plt.close(fig)
    if a.mode in ("phase", "all"):
        betas, cs, Z = phase(p, 2, a.grid)
        with open(os.path.join(ROOT, "results", "phase_grid.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["c_f\\beta"] + ["%.3f" % b for b in betas])
            for j, c in enumerate(cs):
                w.writerow(["%.3f" % c] + ["%.4f" % v for v in Z[j]])
        out.append("\n== 相图（beta 0.15~1.20 × 寄养摩擦 c_f 0.05~0.50，%dx%d，每点 2 次重复）==" % (a.grid, a.grid))
        out.append("  混合占比 >0.60 的比例 = %.2f ; >0.80 的比例 = %.2f ; 最小 = %.3f ; 最大 = %.3f"
                   % ((Z > .6).mean(), (Z > .8).mean(), Z.min(), Z.max()))
        out.append("  高摩擦角落（c_f=%.2f, beta=%.2f）混合占比 = %.3f" % (cs[-1], betas[0], Z[-1, 0]))
        out.append("  低摩擦高 beta（c_f=%.2f, beta=%.2f）混合占比 = %.3f" % (cs[0], betas[-1], Z[0, -1]))
        fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
        im = ax.pcolormesh(betas, cs, Z, cmap="YlGnBu", shading="auto", vmin=0, vmax=0.6)
        cs_ = ax.contour(betas, cs, Z, levels=[0.3, 0.45], colors="k", linewidths=.8)
        ax.clabel(cs_, fmt="%.2f")
        ax.set_xlabel(r"社会化转化系数 $\beta$"); ax.set_ylabel(r"寄养摩擦成本 $c_f$")
        ax.set_title("混合模式稳态占比（%d×%d，每点 2 次重复均值）" % (a.grid, a.grid))
        fig.colorbar(im, ax=ax, label="混合模式（供子+献子+共居）占比")
        fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "abm_phase.png"))
        plt.close(fig)
        lams, sxs, Z2 = lam_scan(p, 2, 8)
        with open(os.path.join(ROOT, "results", "lam_scan.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["lam\\s_ext"] + ["%.2f" % s for s in sxs])
            for j, lm in enumerate(lams):
                w.writerow(["%.2f" % lm] + ["%.4f" % v for v in Z2[j]])
        out.append("\n== 共居可行性：λ（资源共享比例）× s_ext，指标 = 共居占比 ==")
        out.append("  共居占比 >0.10 的比例 = %.2f ; 最大 = %.3f @(lam=%.2f, s_ext=%.2f)"
                   % ((Z2 > .10).mean(), Z2.max(), lams[Z2.max(axis=0).argmax()], sxs[Z2.argmax(axis=0)[0]]))
        fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
        im = ax.pcolormesh(sxs, lams, Z2, cmap="YlOrRd", shading="auto", vmin=0, vmax=max(Z2.max(), 0.3))
        ax.set_xlabel(r"家庭外同伴等效数 $s_{ext}$"); ax.set_ylabel(r"共居资源共享比例 $\lambda$")
        ax.set_title("共居模式稳态占比")
        fig.colorbar(im, ax=ax, label="共居占比")
        fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "abm_lam.png"))
        plt.close(fig)
    txt = "\n".join(out)
    with open(os.path.join(ROOT, "results", "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

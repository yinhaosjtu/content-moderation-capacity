
import gc
import os
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

import math
import os
import queue
import threading
import time
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB


class ModelData(object):
    def __init__(self, inst):
        dim = inst["dimensions"]
        self.nL = int(dim["n_languages"])
        self.nC = int(dim["n_categories"])
        self.nK = int(dim["n_tiers"])
        self.nD = int(dim["n_days"])
        self.nP = int(dim["n_operating_points"])
        self.nW = int(dim["n_scenarios"])

        sets = inst["sets"]
        self.k_min = [int(v) for v in sets["k_min"]]
        self.C_k = {k: [int(c) for c in sets["C_k"][str(k)]] for k in range(1, self.nK + 1)}
        self.A_c = {c: [(str(a[0]), int(a[1])) for a in sets["A_c"][str(c)]]
                    for c in range(self.nC)}
        self.C_res = [int(c) for c in sets["C_res"]]
        self.C_fc = set(int(c) for c in sets["C_fc"])
        self.C_fo = set(int(c) for c in sets["C_fo"])
        self.C_free = [c for c in range(self.nC)
                       if c not in self.C_fc and c not in self.C_fo]

        det = inst["detection_coefficients"]
        self.phi_esc = np.asarray(det["phi_esc"], dtype=float)
        self.phi_mid = np.asarray(det["phi_mid"], dtype=float)
        self.phi_blk = np.asarray(det["phi_blk"], dtype=float)
        self.phi_pas = np.asarray(det["phi_pas"], dtype=float)
        self.e_fn = np.asarray(det["e_fn"], dtype=float)
        self.e_fp = np.asarray(det["e_fp"], dtype=float)
        self.pi_hat = np.asarray(det["pi_hat"], dtype=float)
        self.p_ref = int(inst["operating_points"]["p_ref"])

        vol = inst["volume"]
        self.a_bar = np.asarray(vol["a_bar"], dtype=float)
        self.psi = np.asarray(vol["psi"], dtype=float)

        wk = inst["workload"]
        self.H = float(wk["H"])
        self.theta_in = np.asarray(wk["theta_in"], dtype=float)
        self.theta_out = np.asarray(wk["theta_out"], dtype=float)
        self.theta_qc = np.asarray(wk["theta_qc"], dtype=float)
        self.o_bar = np.asarray(wk["o_bar"], dtype=float)
        self.acc_in = np.asarray(wk["accuracy_in"], dtype=float)
        self.acc_out = np.asarray(wk["accuracy_out"], dtype=float)

        insp = inst["inspection"]
        self.vs_blk = np.asarray(insp["varsigma_blk"], dtype=float)
        self.vs_pas = np.asarray(insp["varsigma_pas"], dtype=float)
        self.vs_adj_in = np.asarray(insp["varsigma_adj_in"], dtype=float)
        self.vs_adj_out = np.asarray(insp["varsigma_adj_out"], dtype=float)

        comp = inst["compliance"]
        self.epsilon_lng = float(comp["epsilon_lng"])
        self.eta_cell = np.asarray(comp["eta_cell"], dtype=float)
        self.varpi_lng = float(comp["varpi_lng"])
        self.varpi_cell = np.asarray(comp["varpi_cell"], dtype=float)
        self.varpi_qc = float(comp["varpi_qc"])

        cost = inst["cost"]
        self.c_hr = np.asarray(cost["c_hr"], dtype=float)
        self.c_ot = np.asarray(cost["c_ot"], dtype=float)
        self.c_out = np.asarray(cost["c_out"], dtype=float)
        self.c_ml = float(cost["c_ml"])
        self.c_fn = np.asarray(cost["c_fn"], dtype=float)
        self.c_fp = np.asarray(cost["c_fp"], dtype=float)
        self.n_bar = np.asarray(cost["n_bar"], dtype=float)
        self.B_bar = float(cost["B_bar"])

        unc = inst["uncertainty"]
        self.alpha = float(unc["alpha"])
        self.gamma = float(unc["gamma"])

        sc = inst["scenarios"]
        self.prob = np.asarray(sc["prob"], dtype=float)
        self.Z = np.asarray(sc["Z"], dtype=float)
        self.gidx = np.asarray(sc["delta_index"], dtype=int)
        self.v_bar = np.asarray(sc["v_bar"], dtype=float)
        self.U_bar = np.asarray(inst["derived"]["U_bar"], dtype=float)


DEFAULTS = dict(
    root_rounds=2000,
    root_ub_every=3,
    recourse_parts=0,
    adaptive_block_rounds=175,
    scen_frac=0.0,
    cut_purge=0,
    final_cut_keep=1200,
    lp_method=1,
    lp_presolve=0,
    count_frac=True,
    lam_repair=False,
    stall_rounds=10,
    workers=0,
    mip_start=True,
    incumbent_cuts=False,
    mip_outer=False,
    lazy_cut_limit=0,
    block_lazy_rounds=400,
    node_cut_rounds=12,
    tree_cut_rounds=0,
    tree_cut_max_node=32,
    mip_heur=None,
    mip_focus=None,
    gurobi_cuts=None,
    branch_y=0,
    branch_n=0,
    master_method=-1,
)


def _cvar(R, prob, alpha, want_xi=False):
    # CVaR calculation via sorting: CVaR_α(R) = min_ξ { ξ + E[(R-ξ)^+] / (1-α) }
    o = np.argsort(R, kind="stable")
    Rs = R[o]
    ps = prob[o]
    tp = np.cumsum(ps[::-1])[::-1]  # Tail probabilities
    tpr = np.cumsum((ps * Rs)[::-1])[::-1]  # Tail expectations
    val = Rs + (tpr - Rs * tp) / (1.0 - alpha)
    if want_xi:
        j = int(np.argmin(val))
        return float(val[j]), float(Rs[j])
    return float(val.min())


class _Pool(object):
    __slots__ = ("n", "qs", "done", "threads", "lock")

    def __init__(self, n, gthreads):
        self.n = n
        self.qs = [queue.Queue() for _ in range(n)]
        self.done = queue.Queue()
        self.lock = threading.Lock()
        self.threads = []
        ready = queue.Queue()
        err = None
        for i in range(n):
            t = threading.Thread(target=self._loop, args=(i, gthreads, ready),
                                 daemon=True)
            try:
                t.start()
            except BaseException as ex:
                err = ex
                break
            self.threads.append(t)
        for _ in range(len(self.threads)):
            exc = ready.get()
            if exc is not None and err is None:
                err = exc
        if err is not None:
            self.close()
            raise err

    def _loop(self, i, gthreads, ready):
        try:
            e = gp.Env(empty=True)
            e.setParam("OutputFlag", 0)
            if gthreads:
                e.setParam("Threads", int(gthreads))
            e.start()
        except BaseException as ex:
            ready.put(ex)
            return
        ready.put(None)
        q = self.qs[i]
        try:
            while True:
                fn = q.get()
                if fn is None:
                    break
                try:
                    fn(e)
                    self.done.put(None)
                except BaseException as ex:
                    self.done.put(ex)
        finally:
            try:
                e.dispose()
            except Exception:
                pass

    def run(self, fns):
        with self.lock:
            k = len(fns)
            for i in range(k):
                self.qs[i].put(fns[i])
            err = None
            for _ in range(k):
                ex = self.done.get()
                if ex is not None and err is None:
                    err = ex
            if err is not None:
                raise err

    def close(self):
        for q in self.qs:
            q.put(None)
        for t in self.threads:
            try:
                t.join(timeout=10.0)
            except RuntimeError:
                pass


class _Bundle(object):
    # Precomputed coefficient blocks for scenario group g (delta grid index)
    __slots__ = ("Bunit", "Aunit", "qunit", "objlam", "mmax", "objh", "objup",
                 "objum")

    def __init__(self, data, lay, g):
        psi = data.psi
        mid = data.phi_mid[:, :, :, g]
        esc = data.phi_esc[:, :, :, g]
        blk = data.phi_blk[:, :, :, g]
        pas = data.phi_pas[:, :, :, g]
        efn = data.e_fn[:, :, :, g]
        efp = data.e_fp[:, :, :, g]
        pih = data.pi_hat[:, :, :, g]
        cfn = data.c_fn[None, :, None]
        cfp = data.c_fp[None, :, None]
        ps3 = psi[:, :, None]
        self.Bunit = ps3 * mid
        self.Aunit = (data.theta_qc[:, :, None] * ps3
                      * (data.vs_blk[None, :, None] * blk
                         + data.vs_pas[None, :, None] * pas))
        self.qunit = psi * mid[:, :, data.p_ref]
        self.mmax = psi * mid.max(axis=2)
        self.objlam = ps3 * (data.c_ml * esc + cfn * efn + cfp * efp)
        HL, HC, HP, HR, HK = lay.HL, lay.HC, lay.HP, lay.HR, lay.HK
        ph = pih[HL, HC, HP]
        ac = np.where(HR == 0, data.acc_in[HL, HC, HK - 1], data.acc_out[HL, HC])
        oh = (1.0 - ac) * (ph * data.c_fn[HC] + (1.0 - ph) * data.c_fp[HC])
        self.objh = oh + np.where(HR == 1, data.c_out[HL] * data.theta_out[HL, HC], 0.0)
        self.objup = (data.c_fn[lay.UC] * pih[lay.UL, lay.UC, lay.UP]
                      if lay.n_up else np.zeros(0))
        self.objum = (data.c_fp[lay.MC] * (1.0 - pih[lay.ML, lay.MC, lay.MP])
                      if lay.n_um else np.zeros(0))


class _Layout(object):
    # Sparse matrix layout for scenario subproblems: variable blocks and constraint structure

    def __init__(self, data):
        nL, nC, nK, nP, nD = data.nL, data.nC, data.nK, data.nP, data.nD
        self.nL, self.nC, self.nK, self.nP, self.nD = nL, nC, nK, nP, nD

        ac_c, ac_r, ac_k = [], [], []
        for c in range(nC):
            for (r, k) in data.A_c[c]:
                ac_c.append(c)
                ac_r.append(0 if r == "in" else 1)
                ac_k.append(k)
        ac_c = np.asarray(ac_c, dtype=np.int64)
        ac_r = np.asarray(ac_r, dtype=np.int64)
        ac_k = np.asarray(ac_k, dtype=np.int64)
        nA = int(ac_c.size)

        cup = np.asarray([c for c in range(nC) if c not in data.C_fc], dtype=np.int64)
        cum = np.asarray([c for c in range(nC) if c not in data.C_fo], dtype=np.int64)
        cfr = np.asarray(sorted(data.C_free), dtype=np.int64)
        crs = np.asarray(sorted(data.C_res), dtype=np.int64)
        self.cup, self.cum, self.cfree, self.cres = cup, cum, cfr, crs
        nup, nem, nfr, nrs = int(cup.size), int(cum.size), int(cfr.size), int(crs.size)
        self.nfr, self.nrs = nfr, nrs
        pos_fr = -np.ones(nC, dtype=np.int64); pos_fr[cfr] = np.arange(nfr)
        pos_rs = -np.ones(nC, dtype=np.int64); pos_rs[crs] = np.arange(nrs)


        n_lam = nL * nC * nP
        n_h = nL * nA * nP
        n_up = nL * nup * nP
        n_um = nL * nem * nP
        n_ov = nL * nK
        o_lam = 0
        o_h = o_lam + n_lam
        o_up = o_h + n_h
        o_um = o_up + n_up
        o_ov = o_um + n_um
        o_ins = o_ov + n_ov
        o_sc = o_ins + nL
        o_sl = o_sc + nL * nrs
        o_st = o_sl + nL
        o_sq = o_st + nL
        o_T = o_sq + nL
        N = o_T + 1
        self.n_lam, self.n_h, self.n_up, self.n_um, self.N = n_lam, n_h, n_up, n_um, N
        self.o_lam, self.o_h, self.o_up, self.o_um = o_lam, o_h, o_up, o_um
        self.o_ov, self.o_ins, self.o_sc, self.o_sl, self.o_st, self.o_sq = (
            o_ov, o_ins, o_sc, o_sl, o_st, o_sq)
        self.n_ov = n_ov

        L3 = np.repeat(np.arange(nL), nC * nP)
        C3 = np.tile(np.repeat(np.arange(nC), nP), nL)
        P3 = np.tile(np.arange(nP), nL * nC)
        self.L3, self.C3, self.P3 = L3, C3, P3
        lam_col = o_lam + np.arange(n_lam)
        HL = np.repeat(np.arange(nL), nA * nP)
        HA = np.tile(np.repeat(np.arange(nA), nP), nL)
        HP = np.tile(np.arange(nP), nL * nA)
        HC, HR, HK = ac_c[HA], ac_r[HA], ac_k[HA]
        h_col = o_h + np.arange(n_h)
        self.HL, self.HC, self.HP, self.HR, self.HK = HL, HC, HP, HR, HK
        UL = np.repeat(np.arange(nL), nup * nP)
        UI = np.tile(np.repeat(np.arange(nup), nP), nL)
        UP = np.tile(np.arange(nP), nL * nup)
        UC = cup[UI] if nup else UI
        up_col = o_up + np.arange(n_up)
        self.UL, self.UC, self.UP = UL, UC, UP
        ML = np.repeat(np.arange(nL), nem * nP)
        MI = np.tile(np.repeat(np.arange(nem), nP), nL)
        MP = np.tile(np.arange(nP), nL * nem)
        MC = cum[MI] if nem else MI
        um_col = o_um + np.arange(n_um)
        self.ML, self.MC, self.MP = ML, MC, MP


        r_share = 0
        r_bal = r_share + nL * nC
        r_aud = r_bal + nL * nC * nP
        r_cell = r_aud + nL
        r_eq = r_cell + nL * nrs
        r_abs = r_eq + nL
        r_tot = r_abs + nL
        mF = r_tot + 1
        r_dp = mF
        r_dm = r_dp + nL * nfr
        r_cap = r_dm + nL * nfr
        r_ot = r_cap + nL * nK
        r_out = r_ot + nL * nK
        mAll = r_out + nL
        mS = mAll - mF
        self.mF, self.mS, self.r_cell = mF, mS, r_cell
        self.m_dp = nL * nfr
        self.m_cap = nL * nK

        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(np.asarray(r, dtype=np.int64).ravel())
            cols.append(np.asarray(c, dtype=np.int64).ravel())
            vals.append(np.asarray(v, dtype=float).ravel())

        one = np.ones
        add(r_share + L3 * nC + C3, lam_col, one(n_lam))
        slot_bal = len(vals)
        add(r_bal + (L3 * nC + C3) * nP + P3, lam_col, np.zeros(n_lam))
        add(r_bal + (HL * nC + HC) * nP + HP, h_col, one(n_h))
        if n_up:
            add(r_bal + (UL * nC + UC) * nP + UP, up_col, one(n_up))
        if n_um:
            add(r_bal + (ML * nC + MC) * nP + MP, um_col, one(n_um))
        add(r_aud + np.arange(nL), o_ins + np.arange(nL), one(nL))
        add(r_aud + np.arange(nL), o_sq + np.arange(nL), one(nL))
        slot_aud = len(vals)
        add(r_aud + L3, lam_col, np.zeros(n_lam))
        if n_up:
            add(r_aud + UL, up_col, -data.theta_qc[UL, UC] * data.vs_pas[UC])
        if n_um:
            add(r_aud + ML, um_col, -data.theta_qc[ML, MC] * data.vs_blk[MC])
        adj = np.where(HR == 0, data.vs_adj_in[HC], data.vs_adj_out[HC])
        add(r_aud + HL, h_col, -data.theta_qc[HL, HC] * adj)
        if nrs:
            mk = pos_rs[HC] >= 0
            add(r_cell + HL[mk] * nrs + pos_rs[HC[mk]], h_col[mk], one(int(mk.sum())))
            add(r_cell + np.arange(nL * nrs), o_sc + np.arange(nL * nrs), one(nL * nrs))
        add(r_eq + HL, h_col, one(n_h))
        add(r_eq + np.arange(nL), o_sl + np.arange(nL), one(nL))
        slot_eq = len(vals)
        add(r_eq + np.arange(nL), np.full(nL, o_T), np.zeros(nL))
        add(r_abs + np.arange(nL), o_st + np.arange(nL), one(nL))
        add(r_abs + np.arange(nL), o_sl + np.arange(nL), -one(nL))
        if nrs:
            add(r_abs + np.repeat(np.arange(nL), nrs), o_sc + np.arange(nL * nrs),
                one(nL * nrs))
        add(np.full(1, r_tot), np.full(1, o_T), one(1))
        add(np.full(n_h, r_tot), h_col, -one(n_h))
        if nfr:
            mk = pos_fr[UC] >= 0
            add(r_dp + UL[mk] * nfr + pos_fr[UC[mk]], up_col[mk], one(int(mk.sum())))
            mk = pos_fr[MC] >= 0
            add(r_dm + ML[mk] * nfr + pos_fr[MC[mk]], um_col[mk], one(int(mk.sum())))
        mk = HR == 0
        add(r_cap + HL[mk] * nK + (HK[mk] - 1), h_col[mk],
            data.theta_in[HL[mk], HC[mk], HK[mk] - 1])
        add(r_cap + np.arange(nL) * nK + (nK - 1), o_ins + np.arange(nL), one(nL))
        add(r_cap + np.arange(nL * nK), o_ov + np.arange(nL * nK), -one(nL * nK))
        add(r_ot + np.arange(nL * nK), o_ov + np.arange(nL * nK), one(nL * nK))
        mk = HR == 1
        add(r_out + HL[mk], h_col[mk], data.theta_out[HL[mk], HC[mk]])

        starts = np.cumsum([0] + [v.size for v in vals])
        self.sl_bal = slice(int(starts[slot_bal]), int(starts[slot_bal + 1]))
        self.sl_aud = slice(int(starts[slot_aud]), int(starts[slot_aud + 1]))
        self.sl_eq = slice(int(starts[slot_eq]), int(starts[slot_eq + 1]))
        drow = np.concatenate(rows)
        dcol = np.concatenate(cols)
        self.dval = np.concatenate(vals)
        self.nnz = int(self.dval.size)


        isF = drow < mF
        base = np.where(isF, drow, drow - mF + nD * mF)
        step = np.where(isF, mF, mS)
        dd = np.arange(nD)[:, None]
        self.srow = (base[None, :] + step[None, :] * dd).ravel()
        self.scol = (dcol[None, :] + N * dd).ravel()
        self.M = nD * mF + nD * mS
        self.NX = nD * N

        sf = np.empty(mF, dtype="<U1")
        sf[r_share:r_bal] = "="
        sf[r_bal:r_aud] = "="
        sf[r_aud:r_tot] = ">"
        sf[r_tot:mF] = "="
        self.sense = sf.tolist() * nD + ["<"] * (nD * mS)

        rf = np.zeros(mF)
        rf[r_share:r_bal] = 1.0
        self.rhsF0 = rf

        ub = np.full(N, GRB.INFINITY)
        ub[o_lam:o_lam + n_lam] = 1.0
        self.ubX = np.tile(ub, nD)
        self.lam_mask = np.tile(
            np.concatenate([np.ones(n_lam, dtype=bool),
                            np.zeros(N - n_lam, dtype=bool)]), nD)

        ob = np.zeros(N)
        ob[o_ov:o_ov + n_ov] = data.c_ot.reshape(-1)
        if nrs:
            ob[o_sc:o_sc + nL * nrs] = np.tile(data.varpi_cell[crs], nL)
        ob[o_st:o_st + nL] = data.varpi_lng
        ob[o_sq:o_sq + nL] = data.varpi_qc
        self.obj0 = ob


class _Scenario(object):
    __slots__ = ("model", "x", "qday", "cs_state", "buf", "Ubar", "vbar", "w",
                 "is_bin", "lam_idx")

    def __init__(self, lay, data, env, w, bundles, opts, threads, track_days):
        nL, nK, nD, N = lay.nL, lay.nK, lay.nD, lay.N
        nfr, nrs = lay.nfr, lay.nrs
        m_dp, m_cap = lay.m_dp, lay.m_cap
        vals = np.empty((nD, lay.nnz))
        obj = np.empty((nD, N))
        rhsF = np.empty((nD, lay.mF))
        Ub = np.empty((nD, m_dp))
        vb = np.empty((nD, nL))
        eps = 1.0 - data.epsilon_lng
        for d in range(nD):
            bd = bundles[int(data.gidx[d][w])]
            A = float(data.Z[d][w]) * data.a_bar[:, d]
            A3 = A[:, None, None]
            v = lay.dval.copy()
            v[lay.sl_bal] = -(bd.Bunit * A3).reshape(-1)
            v[lay.sl_aud] = -(bd.Aunit * A3).reshape(-1)
            qb = bd.qunit * A[:, None]
            pool = qb.sum(axis=1)
            gr = float(pool.sum())
            v[lay.sl_eq] = -eps * (pool / gr if gr > 0.0 else np.zeros(nL))
            vals[d] = v
            o = lay.obj0.copy()
            o[lay.o_lam:lay.o_lam + lay.n_lam] = (bd.objlam * A3).reshape(-1)
            o[lay.o_h:lay.o_h + lay.n_h] = bd.objh
            if lay.n_up:
                o[lay.o_up:lay.o_up + lay.n_up] = bd.objup
            if lay.n_um:
                o[lay.o_um:lay.o_um + lay.n_um] = bd.objum
            obj[d] = o
            r = lay.rhsF0.copy()
            if nrs:
                r[lay.r_cell:lay.r_cell + nL * nrs] = (
                    data.eta_cell[:, lay.cres] * qb[:, lay.cres]).reshape(-1)
            rhsF[d] = r
            if nfr:
                Ub[d] = (bd.mmax * A[:, None])[:, lay.cfree].reshape(-1)
            vb[d] = data.v_bar[:, d, w]

        Amat = sp.coo_matrix((vals.ravel(), (lay.srow, lay.scol)),
                             shape=(lay.M, lay.NX)).tocsr()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0
        if opts.get("lp_method") is not None:
            m.Params.Method = int(opts["lp_method"])
        if opts.get("lp_presolve") is not None:
            m.Params.Presolve = int(opts["lp_presolve"])
        m.Params.Threads = threads if threads else 1
        m.Params.MIPGap = 1e-9
        x = m.addMVar(lay.NX, lb=0.0, ub=lay.ubX, obj=obj.ravel())
        cons = m.addMConstr(Amat, x, lay.sense,
                            np.concatenate([rhsF.ravel(), np.zeros(nD * lay.mS)]))


        qday = None
        if track_days:
            qday = m.addMVar(nD, lb=0.0)
            qmat = sp.csr_matrix(
                (obj.ravel(),
                 (np.repeat(np.arange(nD, dtype=np.int64), N),
                  np.arange(lay.NX, dtype=np.int64))),
                shape=(nD, lay.NX))
            qmat.eliminate_zeros()
            m.addConstr(qday == qmat @ x)
        m.update()
        self.model = m
        self.x = x
        self.qday = qday
        self.cs_state = cons[nD * lay.mF:]
        self.buf = np.zeros((nD, lay.mS))
        self.Ubar = Ub
        self.vbar = vb
        self.w = w
        self.is_bin = False
        self.lam_idx = np.flatnonzero(lay.lam_mask)

    def flip_lambda(self, flag, lam_mask):
        if self.is_bin == flag:
            return
        self.x.VType = np.where(lam_mask, GRB.BINARY if flag else GRB.CONTINUOUS,
                                GRB.CONTINUOUS)
        self.is_bin = flag


def solve_instance(instance, params=None):
    p = dict(params or {})
    t_start = time.time()
    tl = float(p.get("time_limit", 3600.0))
    target = float(p.get("mip_gap", 1e-4))
    verbose = bool(p.get("output", False))
    deadline = t_start + tl
    thr_in = p.get("threads", None)
    threads = int(thr_in) if thr_in else 0
    max_root_rounds = int(p.get("root_rounds", 100000))

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    if threads:
        env.setParam("Threads", threads)
    env.start()

    scens = []
    pool = None
    nthr = 1
    mm = None
    UB = None
    bound = -1e30
    gap = None
    best_n = None
    best_y = None
    iters = 0
    data = None
    st = {}
    try:
        data = ModelData(instance)
        nL, nC, nK, nD, nP, nW = (data.nL, data.nC, data.nK, data.nD,
                                  data.nP, data.nW)
        nB = 1
        nLK = nL * nK
        gamma, alpha = data.gamma, data.alpha
        prob = data.prob
        chr_flat = data.c_hr.reshape(-1)
        obar_flat = data.o_bar.reshape(-1)
        nbar_flat = data.n_bar.reshape(-1)
        H = data.H

        lay = _Layout(data)
        nfr = lay.nfr
        nFY = nL * nfr
        m_dp, m_cap = lay.m_dp, lay.m_cap
        j_dm = m_dp
        j_cap = 2 * m_dp
        j_ot = j_cap + m_cap
        j_out = j_ot + m_cap

        bundles = {}
        for g in np.unique(data.gidx):
            bundles[int(g)] = _Bundle(data, lay, int(g))

        if threads:
            nthr = threads
        else:
            nthr = min(8, os.cpu_count() or 1)
        nthr = max(1, min(nthr, nW))
        pool = _Pool(nthr, 1)

        scens = [None] * nW

        def _build_shard(i):
            def f(wenv):
                for w in range(i, nW, nthr):
                    scens[w] = _Scenario(lay, data, wenv, w, bundles, p, 1,
                                         False)
            return f

        pool.run([_build_shard(i) for i in range(nthr)])

        eta_lb = np.zeros(nW)

        def _lb_shard(i):
            def f(wenv):
                for w in range(i, nW, nthr):
                    sc = scens[w]
                    buf = sc.buf
                    if nFY:
                        buf[:, :m_dp] = sc.Ubar
                        buf[:, j_dm:j_cap] = sc.Ubar
                    buf[:, j_cap:j_ot] = H * nbar_flat
                    buf[:, j_ot:j_out] = obar_flat * nbar_flat
                    buf[:, j_out:] = sc.vbar
                    sc.cs_state.RHS = buf.ravel()
                    sc.model.optimize()
                    if sc.model.Status == GRB.OPTIMAL:
                        eta_lb[w] = max(float(sc.model.ObjVal), 0.0)
            return f

        pool.run([_lb_shard(i) for i in range(nthr)])

        R = np.zeros(nW)
        Gn = np.zeros((nW, nLK))
        Gy = np.zeros((nW, nFY))
        bad = np.zeros(nthr, dtype=np.int64)

        def _price_shard(i, yfree, Hn, On):
            def f(wenv):
                for w in range(i, nW, nthr):
                    sc = scens[w]
                    buf = sc.buf
                    if nFY:
                        buf[:, :m_dp] = sc.Ubar * (1.0 - yfree)
                        buf[:, j_dm:j_cap] = sc.Ubar * yfree
                    buf[:, j_cap:j_ot] = Hn
                    buf[:, j_ot:j_out] = On
                    buf[:, j_out:] = sc.vbar
                    sc.cs_state.RHS = buf.ravel()
                    sc.model.optimize()
                    if sc.model.Status != GRB.OPTIMAL:
                        bad[i] = 1
                        return
                    R[w] = float(sc.model.ObjVal)
                    P = np.minimum(np.asarray(sc.cs_state.Pi), 0.0).reshape(
                        nD, lay.mS)
                    gd_n = (P[:, j_cap:j_ot] * H
                            + P[:, j_ot:j_out] * obar_flat).sum(axis=0)
                    Gn[w] = gd_n
                    if nFY:
                        gd_y = ((P[:, j_dm:j_cap] - P[:, :m_dp])
                                * sc.Ubar).sum(axis=0)
                        Gy[w] = gd_y
            return f

        def price(nflat, yfree):
            Hn = H * nflat
            On = obar_flat * nflat
            bad[:] = 0
            pool.run([_price_shard(i, yfree, Hn, On) for i in range(nthr)])
            iters_local = 1
            if bad.any():
                return None
            return iters_local

        def obj_of(nflat, R_):
            return (float(chr_flat @ nflat) + (1.0 - gamma) * float(prob @ R_)
                    + gamma * _cvar(R_, prob, alpha))

        mm = gp.Model(env=env)
        mm.Params.OutputFlag = 0
        if threads:
            mm.Params.Threads = threads
        nv = mm.addMVar(nLK, lb=0.0, ub=nbar_flat, vtype=GRB.INTEGER)
        yv = mm.addMVar(nFY, lb=0.0, ub=1.0, vtype=GRB.BINARY) if nFY else None
        xiv = mm.addVar(lb=0.0)
        wv = mm.addMVar(nW, lb=0.0)
        eta = mm.addMVar(nW, lb=eta_lb)
        nvl = nv.tolist()
        yvl = yv.tolist() if nFY else []
        etal = eta.tolist()
        wvl = wv.tolist()
        mvars = nvl + yvl
        xyv = gp.MVar.fromlist(mvars)
        com = gp.LinExpr(chr_flat.tolist(), nvl)
        mm.addConstr(com <= data.B_bar)
        for w in range(nW):
            mm.addConstr(wv[w] - eta[w] + xiv >= 0.0)
        mm.setObjective(com
                        + gp.LinExpr(((1.0 - gamma) * prob).tolist(), etal)
                        + gamma * xiv
                        + gp.LinExpr((gamma / (1.0 - alpha) * prob).tolist(),
                                     wvl),
                        GRB.MINIMIZE)
        mm.update()

        def add_cuts(widx, Rv, Gnv, Gyv, nflat, yfree):
            wi = np.asarray(list(widx), dtype=np.int64)
            if wi.size == 0:
                return
            if nFY:
                co = np.hstack((Gnv[wi], Gyv[wi]))
                cst = (Rv[wi] - Gnv[wi] @ nflat - Gyv[wi] @ yfree)
            else:
                co = Gnv[wi]
                cst = Rv[wi] - Gnv[wi] @ nflat
            mm.addConstr(eta[wi] - co @ xyv >= cst)

        nv.VType = GRB.CONTINUOUS
        if nFY:
            yv.VType = GRB.CONTINUOUS
        rootLB = -1e30
        rounds = 0
        for it in range(max_root_rounds):
            if time.time() > deadline:
                break
            mm.optimize()
            if mm.Status != GRB.OPTIMAL:
                break
            LB = float(mm.ObjVal)
            rootLB = max(rootLB, LB)
            nh = np.asarray(nv.X)
            yh = np.asarray(yv.X) if nFY else np.zeros(0)
            etax = np.asarray(eta.X)
            res = price(nh, yh)
            if res is None:
                break
            vr = R > etax + 1e-7 * np.maximum(1.0, np.abs(R))
            vio = np.flatnonzero(vr)
            rounds += 1
            if vio.size == 0:
                break
            add_cuts(vio, R, Gn, Gy, nh, yh)
            mm.update()

        nv.VType = GRB.INTEGER
        if nFY:
            yv.VType = GRB.BINARY
        mm.update()
        st["root_rounds"] = rounds

        cache = {}
        best = {"UB": UB, "n": None, "y": None, "R": None}
        lazy_state = {"calls": 0, "cuts": 0}

        def cb(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            lazy_state["calls"] += 1
            nh = np.round(np.asarray(model.cbGetSolution(nvl)))
            yh = (np.round(np.asarray(model.cbGetSolution(yvl)))
                  if nFY else np.zeros(0))
            kk = (nh.tobytes(), yh.tobytes())
            hit = cache.get(kk)
            if hit is None:
                if price(nh, yh) is None:
                    return
                hit = (R.copy(), Gn.copy(), Gy.copy())
                cache[kk] = hit
            Rv, Gnv, Gyv = hit
            etax = np.asarray(model.cbGetSolution(etal))
            vr = Rv > etax + 1e-7 * np.maximum(1.0, np.abs(Rv))
            wi = np.flatnonzero(vr)
            for w in wi:
                if nFY:
                    co = np.concatenate([Gnv[w], Gyv[w]])
                    cst = (Rv[w] - float(Gnv[w] @ nh) - float(Gyv[w] @ yh))
                else:
                    co = Gnv[w]
                    cst = Rv[w] - float(Gnv[w] @ nh)
                model.cbLazy(etal[int(w)] - gp.LinExpr(co.tolist(), mvars)
                             >= cst)
            lazy_state["cuts"] += int(wi.size)
            ub = obj_of(nh, Rv)
            if best["UB"] is None or ub < best["UB"] - 1e-9:
                best["UB"] = ub
                best["n"] = nh.copy()
                best["y"] = yh.copy()
                best["R"] = Rv.copy()

        t_mip0 = time.time()
        if time.time() < deadline:
            mm.Params.LazyConstraints = 1
            mm.Params.MIPGap = target
            mm.Params.TimeLimit = max(deadline - time.time(), 1.0)
            mm.optimize(cb)
            try:
                bound = max(rootLB, float(mm.ObjBound))
            except Exception:
                bound = rootLB
        else:
            bound = rootLB
        st["t_mip"] = time.time() - t_mip0
        st["lazy_calls"] = lazy_state["calls"]
        st["lazy_cuts"] = lazy_state["cuts"]

        UB = best["UB"]
        best_n = best["n"]
        best_y = best["y"]
        if best_n is not None:
            if price(best_n, best_y if nFY else np.zeros(0)) is not None:
                UB = obj_of(best_n, R)
        if UB is not None and bound > -1e30 and abs(UB) > 1e-12:
            gap = max(UB - bound, 0.0) / abs(UB) * 100.0
        iters = rounds + lazy_state["calls"]
        if verbose:
            print("[lshaped] L=%d C=%d K=%d D=%d W=%d | root_rounds=%d "
                  "lazy=%d/%d obj=%.10g bound=%.10g gap=%s time=%.1f"
                  % (nL, nC, nK, nD, nW, rounds, lazy_state["calls"],
                     lazy_state["cuts"], UB if UB else float("nan"), bound,
                     ("%.4g" % gap) if gap is not None else "-",
                     time.time() - t_start), flush=True)
    finally:
        if pool is not None:
            def _kill_shard(i):
                def f(wenv):
                    for w in range(i, len(scens), nthr):
                        sc = scens[w]
                        if sc is not None:
                            try:
                                sc.model.dispose()
                            except Exception:
                                pass
                return f
            try:
                pool.run([_kill_shard(i) for i in range(nthr)])
            except Exception:
                pass
            pool.close()
        try:
            if mm is not None:
                mm.dispose()
        except Exception:
            pass
        env.dispose()
        gc.collect()

    head = None
    dispo = None
    if best_n is not None:
        head = [[int(round(best_n[l * nK + k])) for k in range(nK)]
                for l in range(nL)]
        fr = sorted(data.C_free)
        dispo = []
        for l in range(nL):
            row = []
            for c in range(nC):
                if c in data.C_fc:
                    row.append(1)
                elif c in data.C_fo:
                    row.append(0)
                else:
                    row.append(int(round(best_y[l * nfr + fr.index(c)])))
            dispo.append(row)

    return {"objective": UB,
            "bound": None if bound <= -1e30 else bound,
            "gap": gap,
            "time": time.time() - t_start,
            "iterations": iters,
            "headcount": head,
            "disposition": dispo,
            "stats": st}

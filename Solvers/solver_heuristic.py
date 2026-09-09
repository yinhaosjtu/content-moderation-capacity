import gc
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

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
    o = np.argsort(R, kind="stable")
    Rs = R[o]
    ps = prob[o]
    tp = np.cumsum(ps[::-1])[::-1]
    tpr = np.cumsum((ps * Rs)[::-1])[::-1]
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


DEFAULTS = {
    "time_limit": 3600.0,
    "threads": 0,
    "output": False,
    "root_rounds": 10,
    "integer_rounds": 6,
    "local_rounds": 2,
    "max_neighbors": 12,
    "max_seed_candidates": 12,
    "full_candidate_count": 3,
    "full_scenario_blocks": 1000,
    "search_scenarios": 0,
    "cut_batch_size": 0,
    "exact_candidate_count": 0,
    "master_gap": 0.01,
    "quality_gap": 0.005,
    "workers": 0,
    "lp_backend": "auto",
    "lp_method": "highs-ds",
    "lp_presolve": True,
    "count_fractional": False,
    "adaptive": True,
    "enable_multistart": True,
    "enable_integer_stage": True,
    "enable_local_search": True,
}


class _ScenarioProgram:
    def __init__(self, layout, data, scenario, bundles, options):
        nL = layout.nL
        nD = layout.nD
        nfr = layout.nfr
        nrs = layout.nrs
        values = np.empty((nD, layout.nnz))
        objective = np.empty((nD, layout.N))
        fixed_rhs = np.empty((nD, layout.mF))
        disposition_bound = np.empty((nD, layout.m_dp))
        outsourcing_bound = np.empty((nD, nL))
        language_factor = 1.0 - data.epsilon_lng
        for d in range(nD):
            bundle = bundles[int(data.gidx[d][scenario])]
            arrivals = float(data.Z[d][scenario]) * data.a_bar[:, d]
            arrivals_3d = arrivals[:, None, None]
            daily_values = layout.dval.copy()
            daily_values[layout.sl_bal] = -(
                bundle.Bunit * arrivals_3d
            ).reshape(-1)
            daily_values[layout.sl_aud] = -(
                bundle.Aunit * arrivals_3d
            ).reshape(-1)
            reference_band = bundle.qunit * arrivals[:, None]
            pool_band = reference_band.sum(axis=1)
            total_band = float(pool_band.sum())
            daily_values[layout.sl_eq] = -language_factor * (
                pool_band / total_band
                if total_band > 0.0
                else np.zeros(nL)
            )
            values[d] = daily_values
            daily_objective = layout.obj0.copy()
            daily_objective[
                layout.o_lam : layout.o_lam + layout.n_lam
            ] = (bundle.objlam * arrivals_3d).reshape(-1)
            daily_objective[
                layout.o_h : layout.o_h + layout.n_h
            ] = bundle.objh
            if layout.n_up:
                daily_objective[
                    layout.o_up : layout.o_up + layout.n_up
                ] = bundle.objup
            if layout.n_um:
                daily_objective[
                    layout.o_um : layout.o_um + layout.n_um
                ] = bundle.objum
            objective[d] = daily_objective
            daily_rhs = layout.rhsF0.copy()
            if nrs:
                daily_rhs[
                    layout.r_cell : layout.r_cell + nL * nrs
                ] = (
                    data.eta_cell[:, layout.cres]
                    * reference_band[:, layout.cres]
                ).reshape(-1)
            fixed_rhs[d] = daily_rhs
            if nfr:
                disposition_bound[d] = (
                    bundle.mmax * arrivals[:, None]
                )[:, layout.cfree].reshape(-1)
            outsourcing_bound[d] = data.v_bar[:, d, scenario]

        matrix = sp.coo_matrix(
            (values.ravel(), (layout.srow, layout.scol)),
            shape=(layout.M, layout.NX),
        ).tocsr()
        sense = np.asarray(layout.sense)
        equality_rows = np.flatnonzero(sense == "=")
        inequality_rows = np.flatnonzero(sense != "=")
        inequality_sign = np.where(
            sense[inequality_rows] == "<", 1.0, -1.0
        )
        full_rhs = np.concatenate(
            [fixed_rhs.ravel(), np.zeros(nD * layout.mS)]
        )
        self.A_eq = matrix[equality_rows]
        self.b_eq = full_rhs[equality_rows]
        self.A_ub = sp.diags(inequality_sign).dot(
            matrix[inequality_rows]
        ).tocsr()
        self.inequality_rows = inequality_rows
        self.inequality_sign = inequality_sign
        self.full_rhs = full_rhs
        row_to_inequality = np.full(layout.M, -1, dtype=int)
        row_to_inequality[inequality_rows] = np.arange(
            len(inequality_rows)
        )
        dynamic_rows = np.arange(nD * layout.mF, layout.M)
        self.dynamic_positions = row_to_inequality[dynamic_rows]
        self.dynamic_sign = inequality_sign[self.dynamic_positions]
        self.objective = objective.ravel()
        self.bounds = np.column_stack(
            [np.zeros(layout.NX), layout.ubX]
        )
        self.buffer = np.zeros((nD, layout.mS))
        self.disposition_bound = disposition_bound
        self.outsourcing_bound = outsourcing_bound
        self.layout = layout
        self.method = str(options["lp_method"])
        self.presolve = bool(options["lp_presolve"])
        self.last_solution = None

    def solve(self, staffing, disposition, data):
        layout = self.layout
        nFY = layout.nL * layout.nfr
        m_dp = layout.m_dp
        m_cap = layout.m_cap
        j_dm = m_dp
        j_cap = 2 * m_dp
        j_ot = j_cap + m_cap
        j_out = j_ot + m_cap
        if nFY:
            self.buffer[:, :m_dp] = self.disposition_bound * (
                1.0 - disposition
            )
            self.buffer[:, j_dm:j_cap] = (
                self.disposition_bound * disposition
            )
        self.buffer[:, j_cap:j_ot] = data.H * staffing
        self.buffer[:, j_ot:j_out] = (
            data.o_bar.reshape(-1) * staffing
        )
        self.buffer[:, j_out:] = self.outsourcing_bound
        rhs = self.full_rhs.copy()
        rhs[layout.nD * layout.mF :] = self.buffer.ravel()
        result = linprog(
            self.objective,
            A_ub=self.A_ub,
            b_ub=rhs[self.inequality_rows]
            * self.inequality_sign,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
            method=self.method,
            options={"presolve": self.presolve},
        )
        if not result.success:
            raise RuntimeError(
                "Recourse evaluation failed with status {0}".format(
                    result.status
                )
            )
        dynamic_dual = (
            np.asarray(result.ineqlin.marginals)[
                self.dynamic_positions
            ]
            * self.dynamic_sign
        ).reshape(layout.nD, layout.mS)
        dynamic_dual = np.minimum(dynamic_dual, 0.0)
        gradient_staffing = (
            dynamic_dual[:, j_cap:j_ot] * data.H
            + dynamic_dual[:, j_ot:j_out]
            * data.o_bar.reshape(-1)
        ).sum(axis=0)
        if nFY:
            gradient_disposition = (
                (
                    dynamic_dual[:, j_dm:j_cap]
                    - dynamic_dual[:, :m_dp]
                )
                * self.disposition_bound
            ).sum(axis=0)
        else:
            gradient_disposition = np.zeros(0)
        self.last_solution = np.asarray(result.x)
        return (
            float(result.fun),
            gradient_staffing,
            gradient_disposition,
        )

    def count_fractional(self):
        if self.last_solution is None:
            return 0
        values = self.last_solution[self.layout.lam_mask].reshape(
            self.layout.nD, -1
        )
        return int(
            np.any(
                (values > 1e-6) & (values < 1.0 - 1e-6),
                axis=1,
            ).sum()
        )


def _scenario_order(data):
    grid_scale = max(float(np.max(data.gidx)), 1.0)
    score = (
        np.mean(np.log(np.maximum(data.Z, 1e-12)), axis=0)
        + 0.65 * np.mean(data.gidx / grid_scale, axis=0)
        + 0.20
        * np.max(np.log(np.maximum(data.Z, 1e-12)), axis=0)
        + 0.20 * np.max(data.gidx / grid_scale, axis=0)
    )
    return np.argsort(score, kind="stable")


def _select_scenarios(data, options):
    requested = int(options.get("search_scenarios", 0))
    if requested > 0:
        count = min(requested, data.nW)
    elif data.nD * data.nW <= int(options["full_scenario_blocks"]):
        count = data.nW
    else:
        count = min(
            data.nW, max(24, int(round(4.0 * math.sqrt(data.nW))))
        )
    if count >= data.nW:
        return np.arange(data.nW, dtype=int)
    order = _scenario_order(data)
    positions = np.rint(
        np.linspace(0, data.nW - 1, count)
    ).astype(int)
    selected = list(
        dict.fromkeys(int(order[position]) for position in positions)
    )
    if len(selected) < count:
        for index in order[::-1]:
            value = int(index)
            if value not in selected:
                selected.append(value)
            if len(selected) == count:
                break
    return np.asarray(sorted(selected), dtype=int)


def _normalized_probabilities(probability, indices):
    values = np.asarray(probability, dtype=float)[indices]
    total = float(values.sum())
    if total <= 0.0:
        return np.full(len(indices), 1.0 / len(indices))
    return values / total


def _initial_disposition(data, indices, probability):
    free = sorted(data.C_free)
    result = np.zeros(data.nL * len(free), dtype=float)
    if not free:
        return result
    for l in range(data.nL):
        for position, c in enumerate(free):
            release_loss = 0.0
            removal_loss = 0.0
            for local, w in enumerate(indices):
                scenario_weight = float(probability[local])
                for d in range(data.nD):
                    g = int(data.gidx[d, w])
                    volume = (
                        float(data.psi[l, c])
                        * float(data.Z[d, w])
                        * float(data.a_bar[l, d])
                        * float(
                            data.phi_mid[
                                l, c, data.p_ref, g
                            ]
                        )
                    )
                    prevalence = float(
                        data.pi_hat[l, c, data.p_ref, g]
                    )
                    release_loss += (
                        scenario_weight
                        * volume
                        * float(data.c_fn[c])
                        * prevalence
                    )
                    removal_loss += (
                        scenario_weight
                        * volume
                        * float(data.c_fp[c])
                        * (1.0 - prevalence)
                    )
            result[l * len(free) + position] = (
                1.0 if removal_loss <= release_loss else 0.0
            )
    return result


def _repair_staffing(values, reference, costs, upper, budget):
    staffing = np.clip(
        np.rint(np.asarray(values, dtype=float)), 0.0, upper
    )
    while float(costs @ staffing) > budget + 1e-7:
        available = np.flatnonzero(staffing > 0.0)
        if available.size == 0:
            break
        excess = staffing[available] - reference[available]
        score = excess + costs[available] / max(
            float(np.max(costs)), 1.0
        )
        selected = int(available[int(np.argmax(score))])
        staffing[selected] -= 1.0
    return staffing


def _staffing_variant(reference, mode, costs, upper, budget):
    if mode == "floor":
        raw = np.floor(reference + 1e-8)
    elif mode == "ceil":
        raw = np.ceil(reference - 1e-8)
    else:
        raw = np.rint(reference)
    return _repair_staffing(
        raw, reference, costs, upper, budget
    )


def _candidate_key(staffing, disposition):
    return (
        tuple(int(round(value)) for value in staffing),
        tuple(int(round(value)) for value in disposition),
    )


def _objective_value(
    staffing,
    recourse,
    costs,
    probability,
    gamma,
    alpha,
):
    commitment = float(costs @ staffing)
    expectation = float(probability @ recourse)
    return (
        commitment
        + (1.0 - gamma) * expectation
        + gamma * _cvar(recourse, probability, alpha)
    )


def _tail_weights(recourse, probability, gamma, alpha):
    weights = (1.0 - gamma) * probability.copy()
    if gamma <= 0.0:
        return weights
    _, threshold = _cvar(
        recourse, probability, alpha, True
    )
    tolerance = 1e-9 * max(1.0, abs(threshold))
    above = recourse > threshold + tolerance
    weights[above] += (
        gamma * probability[above] / (1.0 - alpha)
    )
    remaining = max(
        0.0, 1.0 - alpha - float(probability[above].sum())
    )
    tied = np.abs(recourse - threshold) <= tolerance
    tied_probability = float(probability[tied].sum())
    if remaining > 0.0 and tied_probability > 0.0:
        fraction = min(1.0, remaining / tied_probability)
        weights[tied] += (
            gamma
            * fraction
            * probability[tied]
            / (1.0 - alpha)
        )
    return weights


def _full_disposition(data, free_values):
    free = sorted(data.C_free)
    free_position = {
        c: position for position, c in enumerate(free)
    }
    rows = []
    for l in range(data.nL):
        row = []
        for c in range(data.nC):
            if c in data.C_fc:
                row.append(1)
            elif c in data.C_fo:
                row.append(0)
            else:
                row.append(
                    int(
                        round(
                            free_values[
                                l * len(free)
                                + free_position[c]
                            ]
                        )
                    )
                )
        rows.append(row)
    return rows


def _build_master_problem(
    data,
    costs,
    upper,
    probability,
    cuts,
    integer,
    time_limit,
    gap,
    output,
):
    nLK = len(costs)
    nFY = cuts["nFY"]
    nS = len(probability)
    offset_y = nLK
    offset_xi = offset_y + nFY
    offset_w = offset_xi + 1
    offset_eta = offset_w + nS
    variable_count = offset_eta + nS
    objective = np.zeros(variable_count)
    objective[:nLK] = costs
    objective[offset_xi] = data.gamma
    objective[offset_w:offset_eta] = (
        data.gamma / (1.0 - data.alpha) * probability
    )
    objective[offset_eta:] = (
        1.0 - data.gamma
    ) * probability
    lower = np.zeros(variable_count)
    upper_bound = np.full(variable_count, np.inf)
    upper_bound[:nLK] = upper
    if nFY:
        upper_bound[offset_y:offset_xi] = 1.0

    rows = []
    columns = []
    values = []
    constraint_lower = []
    constraint_upper = []

    def add_row(entries, row_lower, row_upper):
        row = len(constraint_lower)
        for column, value in entries:
            if abs(value) > 0.0:
                rows.append(row)
                columns.append(column)
                values.append(float(value))
        constraint_lower.append(float(row_lower))
        constraint_upper.append(float(row_upper))

    add_row(
        [(index, costs[index]) for index in range(nLK)],
        -np.inf,
        data.B_bar,
    )
    for local in range(nS):
        add_row(
            [
                (offset_xi, 1.0),
                (offset_w + local, 1.0),
                (offset_eta + local, -1.0),
            ],
            0.0,
            np.inf,
        )
    for local, staffing_gradient, disposition_gradient, constant in cuts[
        "rows"
    ]:
        entries = [
            (index, -staffing_gradient[index])
            for index in range(nLK)
        ]
        entries.extend(
            (
                offset_y + index,
                -disposition_gradient[index],
            )
            for index in range(nFY)
        )
        entries.append((offset_eta + local, 1.0))
        add_row(entries, constant, np.inf)
    matrix = sp.coo_matrix(
        (values, (rows, columns)),
        shape=(len(constraint_lower), variable_count),
    ).tocsr()
    integrality = np.zeros(variable_count, dtype=int)
    if integer:
        integrality[: offset_xi] = 1
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper_bound),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lower),
            np.asarray(constraint_upper),
        ),
        options={
            "disp": bool(output),
            "presolve": True,
            "time_limit": max(0.1, float(time_limit)),
            "mip_rel_gap": float(gap),
        },
    )
    if result.x is None:
        return None
    staffing = np.asarray(result.x[:nLK])
    disposition = np.asarray(
        result.x[offset_y:offset_xi]
    )
    recourse_estimate = np.asarray(result.x[offset_eta:])
    bound = None
    if integer and hasattr(result, "mip_dual_bound"):
        value = getattr(result, "mip_dual_bound")
        if value is not None and np.isfinite(value):
            bound = float(value)
    elif result.success and result.fun is not None:
        bound = float(result.fun)
    return {
        "staffing": staffing,
        "disposition": disposition,
        "recourse": recourse_estimate,
        "bound": bound,
        "success": bool(result.success),
    }


def solve_instance(instance, params=None):
    options = dict(DEFAULTS)
    options.update(params or {})
    started = time.time()
    deadline = started + float(options["time_limit"])
    round_cap_scale = max(1.0, float(options["time_limit"]) / 300.0)
    root_round_cap = 30.0 * round_cap_scale
    master_round_cap = 45.0 * round_cap_scale
    verbose = bool(options.get("output", False))
    statistics = {}
    evaluation_count = 0
    lp_count = 0
    lower_bound = None
    fractional_blocks = 0
    executor = None
    gurobi_pool = None
    programs = []
    worker_count = 1
    pricing_time = 0.0
    try:
        data = ModelData(instance)
        nL = data.nL
        nK = data.nK
        nW = data.nW
        nLK = nL * nK
        costs = data.c_hr.reshape(-1)
        upper = data.n_bar.reshape(-1)
        free = sorted(data.C_free)
        nFY = nL * len(free)
        if bool(options.get("adaptive", True)):
            scenario_blocks = data.nD * data.nW
            if scenario_blocks <= 500:
                options["root_rounds"] = max(
                    int(options["root_rounds"]), 20
                )
                options["integer_rounds"] = max(
                    int(options["integer_rounds"]), 12
                )
                options["local_rounds"] = min(
                    int(options["local_rounds"]), 1
                )
                options["max_neighbors"] = min(
                    int(options["max_neighbors"]), 12
                )
            elif scenario_blocks > 1500:
                if int(options.get("search_scenarios", 0)) <= 0:
                    options["search_scenarios"] = data.nW
                if int(options.get("cut_batch_size", 0)) <= 0:
                    options["cut_batch_size"] = min(20, data.nW)
                if int(options.get("exact_candidate_count", 0)) <= 0:
                    options["exact_candidate_count"] = 2
                options["root_rounds"] = max(
                    int(options["root_rounds"]), 20
                )
                options["integer_rounds"] = max(
                    int(options["integer_rounds"]), 10
                )
                options["local_rounds"] = min(
                    int(options["local_rounds"]), 1
                )
                options["full_candidate_count"] = min(
                    int(options["full_candidate_count"]),
                    1,
                )
                options["max_neighbors"] = min(
                    int(options["max_neighbors"]), 1
                )
                options["max_seed_candidates"] = min(
                    int(options["max_seed_candidates"]), 8
                )
            else:
                options["root_rounds"] = max(
                    int(options["root_rounds"]), 12
                )
                options["integer_rounds"] = max(
                    int(options["integer_rounds"]), 6
                )
                options["local_rounds"] = min(
                    int(options["local_rounds"]), 1
                )
                options["max_neighbors"] = min(
                    int(options["max_neighbors"]), 12
                )
        enable_multistart = bool(
            options.get("enable_multistart", True)
        )
        enable_integer_stage = bool(
            options.get("enable_integer_stage", True)
        )
        enable_local_search = bool(
            options.get("enable_local_search", True)
        )
        statistics["enable_multistart"] = enable_multistart
        statistics["enable_integer_stage"] = enable_integer_stage
        statistics["enable_local_search"] = enable_local_search
        search_indices = _select_scenarios(data, options)
        search_probability = _normalized_probabilities(
            data.prob, search_indices
        )
        all_indices = np.arange(nW, dtype=int)
        all_probability = np.asarray(
            data.prob, dtype=float
        )
        use_all = len(search_indices) == nW
        search_position = {
            int(scenario): local
            for local, scenario in enumerate(search_indices)
        }
        severity_order = [
            search_position[int(scenario)]
            for scenario in _scenario_order(data)
            if int(scenario) in search_position
        ]
        cut_batch_size = int(options.get("cut_batch_size", 0))
        if cut_batch_size <= 0 or cut_batch_size >= len(search_indices):
            cut_order = np.arange(len(search_indices), dtype=int)
            cut_batch_size = len(search_indices)
        else:
            stride = max(1, int(round(0.381966 * len(search_indices))))
            while math.gcd(stride, len(search_indices)) != 1:
                stride += 1
            cut_order = np.asarray(severity_order, dtype=int)[
                (np.arange(len(search_indices)) * stride)
                % len(search_indices)
            ]
        statistics["search_scenarios"] = int(
            len(search_indices)
        )
        statistics["total_scenarios"] = int(nW)
        statistics["cut_batch_size"] = int(cut_batch_size)

        setup_started = time.time()
        layout = _Layout(data)
        bundles = {
            int(g): _Bundle(data, layout, int(g))
            for g in np.unique(data.gidx)
        }
        requested_workers = int(
            options.get("workers", 0) or 0
        )
        thread_budget = int(options.get("threads", 0) or 0)
        if requested_workers:
            worker_count = requested_workers
        elif thread_budget:
            worker_count = thread_budget
        else:
            if (
                bool(options.get("adaptive", True))
                and data.nD * data.nW > 1500
            ):
                worker_count = min(
                    16, os.cpu_count() or 1
                )
            else:
                worker_count = min(
                    4, os.cpu_count() or 1
                )
        worker_count = max(1, min(worker_count, nW))
        backend = str(
            options.get("lp_backend", "auto")
        ).lower()
        if backend in ("auto", "gurobi"):
            candidate_pool = None
            try:
                candidate_pool = _Pool(worker_count, 1)
                programs = [None] * nW

                def build_shard(worker):
                    def build(environment):
                        for scenario in range(
                            worker, nW, worker_count
                        ):
                            programs[scenario] = _Scenario(
                                layout,
                                data,
                                environment,
                                scenario,
                                bundles,
                                {
                                    "lp_method": 1,
                                    "lp_presolve": 0,
                                },
                                1,
                                False,
                            )

                    return build

                candidate_pool.run(
                    [
                        build_shard(worker)
                        for worker in range(worker_count)
                    ]
                )
                gurobi_pool = candidate_pool
                backend = "gurobi"
            except Exception:
                if candidate_pool is not None:
                    candidate_pool.close()
                gurobi_pool = None
                programs = []
                backend = "scipy"
        else:
            backend = "scipy"
        if backend == "scipy":
            programs = [
                _ScenarioProgram(
                    layout, data, w, bundles, options
                )
                for w in range(nW)
            ]
            executor = ThreadPoolExecutor(
                max_workers=worker_count
            )
        statistics["setup_time"] = (
            time.time() - setup_started
        )
        statistics["workers"] = worker_count
        statistics["lp_backend"] = backend

        def price(
            staffing,
            disposition,
            indices,
            count_fractional=False,
        ):
            nonlocal evaluation_count, lp_count, pricing_time

            price_started = time.time()
            if backend == "gurobi":
                scenario_indices = np.asarray(
                    indices, dtype=int
                )
                recourse = np.empty(len(scenario_indices))
                gradient_staffing = np.empty(
                    (len(scenario_indices), nLK)
                )
                gradient_disposition = np.empty(
                    (len(scenario_indices), nFY)
                )
                fractional_by_scenario = np.zeros(
                    len(scenario_indices), dtype=int
                )
                m_dp = layout.m_dp
                m_cap = layout.m_cap
                j_dm = m_dp
                j_cap = 2 * m_dp
                j_ot = j_cap + m_cap
                j_out = j_ot + m_cap
                staffing_array = np.asarray(
                    staffing, dtype=float
                )
                disposition_array = np.asarray(
                    disposition, dtype=float
                )
                staffing_capacity = data.H * staffing_array
                overtime_capacity = (
                    data.o_bar.reshape(-1) * staffing_array
                )

                def solve_shard(worker):
                    def solve_all(environment):
                        for local, scenario in enumerate(
                            scenario_indices
                        ):
                            if (
                                int(scenario) % worker_count
                                != worker
                            ):
                                continue
                            program = programs[int(scenario)]
                            buffer = program.buf
                            if nFY:
                                buffer[:, :m_dp] = (
                                    program.Ubar
                                    * (
                                        1.0
                                        - disposition_array
                                    )
                                )
                                buffer[:, j_dm:j_cap] = (
                                    program.Ubar
                                    * disposition_array
                                )
                            buffer[:, j_cap:j_ot] = (
                                staffing_capacity
                            )
                            buffer[:, j_ot:j_out] = (
                                overtime_capacity
                            )
                            buffer[:, j_out:] = program.vbar
                            program.cs_state.RHS = (
                                buffer.ravel()
                            )
                            program.model.optimize()
                            if (
                                program.model.Status
                                != GRB.OPTIMAL
                            ):
                                raise RuntimeError(
                                    "Recourse evaluation failed "
                                    "with status {0}".format(
                                        program.model.Status
                                    )
                                )
                            recourse[local] = float(
                                program.model.ObjVal
                            )
                            dynamic_dual = np.minimum(
                                np.asarray(
                                    program.cs_state.Pi
                                ),
                                0.0,
                            ).reshape(
                                layout.nD, layout.mS
                            )
                            gradient_staffing[local] = (
                                dynamic_dual[
                                    :, j_cap:j_ot
                                ]
                                * data.H
                                + dynamic_dual[
                                    :, j_ot:j_out
                                ]
                                * data.o_bar.reshape(-1)
                            ).sum(axis=0)
                            if nFY:
                                gradient_disposition[
                                    local
                                ] = (
                                    (
                                        dynamic_dual[
                                            :, j_dm:j_cap
                                        ]
                                        - dynamic_dual[
                                            :, :m_dp
                                        ]
                                    )
                                    * program.Ubar
                                ).sum(axis=0)
                            if count_fractional:
                                values = np.asarray(
                                    program.x.X
                                )[
                                    program.lam_idx
                                ].reshape(
                                    layout.nD, -1
                                )
                                fractional_by_scenario[
                                    local
                                ] = int(
                                    np.any(
                                        (
                                            values
                                            > 1e-6
                                        )
                                        & (
                                            values
                                            < 1.0
                                            - 1e-6
                                        ),
                                        axis=1,
                                    ).sum()
                                )

                    return solve_all

                gurobi_pool.run(
                    [
                        solve_shard(worker)
                        for worker in range(worker_count)
                    ]
                )
                evaluation_count += 1
                lp_count += len(scenario_indices)
                pricing_time += (
                    time.time() - price_started
                )
                return (
                    recourse,
                    gradient_staffing,
                    gradient_disposition,
                    int(fractional_by_scenario.sum()),
                )

            def solve_one(w):
                value, staffing_gradient, disposition_gradient = (
                    programs[int(w)].solve(
                        staffing, disposition, data
                    )
                )
                fractional = (
                    programs[int(w)].count_fractional()
                    if count_fractional
                    else 0
                )
                return (
                    value,
                    staffing_gradient,
                    disposition_gradient,
                    fractional,
                )

            results = list(
                executor.map(solve_one, [int(w) for w in indices])
            )
            evaluation_count += 1
            lp_count += len(indices)
            recourse = np.asarray(
                [result[0] for result in results]
            )
            gradient_staffing = np.asarray(
                [result[1] for result in results]
            )
            gradient_disposition = np.asarray(
                [result[2] for result in results]
            )
            fractional = int(
                sum(result[3] for result in results)
            )
            pricing_time += time.time() - price_started
            return (
                recourse,
                gradient_staffing,
                gradient_disposition,
                fractional,
            )

        cuts = {"nFY": nFY, "rows": []}

        def add_cuts(
            staffing,
            disposition,
            recourse,
            gradient_staffing,
            gradient_disposition,
            estimated=None,
            local_indices=None,
        ):
            added = 0
            if local_indices is None:
                local_indices = np.arange(len(recourse), dtype=int)
            for priced_local, local in enumerate(local_indices):
                if estimated is not None:
                    tolerance = 1e-7 * max(
                        1.0, abs(float(recourse[priced_local]))
                    )
                    if (
                        recourse[priced_local]
                        <= estimated[priced_local] + tolerance
                    ):
                        continue
                constant = (
                    recourse[priced_local]
                    - float(
                        gradient_staffing[priced_local]
                        @ staffing
                    )
                    - (
                        float(
                            gradient_disposition[priced_local]
                            @ disposition
                        )
                        if nFY
                        else 0.0
                    )
                )
                cuts["rows"].append(
                    (
                        int(local),
                        gradient_staffing[priced_local].copy(),
                        gradient_disposition[priced_local].copy(),
                        float(constant),
                    )
                )
                added += 1
            return added

        candidate_cache = {}

        def pricing_batch(round_index):
            if cut_batch_size >= len(search_indices):
                return np.arange(len(search_indices), dtype=int)
            start = (
                int(round_index) * cut_batch_size
            ) % len(search_indices)
            positions = (
                start + np.arange(cut_batch_size)
            ) % len(search_indices)
            return cut_order[positions]

        def evaluate_search(staffing, disposition):
            staffing = _repair_staffing(
                staffing,
                staffing,
                costs,
                upper,
                data.B_bar,
            )
            disposition = np.rint(
                np.clip(disposition, 0.0, 1.0)
            ).astype(float)
            key = _candidate_key(staffing, disposition)
            cached = candidate_cache.get(key)
            if cached is not None:
                return cached
            (
                recourse,
                gradient_staffing,
                gradient_disposition,
                _,
            ) = price(
                staffing, disposition, search_indices
            )
            objective = _objective_value(
                staffing,
                recourse,
                costs,
                search_probability,
                data.gamma,
                data.alpha,
            )
            record = {
                "staffing": staffing.copy(),
                "disposition": disposition.copy(),
                "recourse": recourse,
                "gradient_staffing": gradient_staffing,
                "gradient_disposition": gradient_disposition,
                "objective": objective,
            }
            candidate_cache[key] = record
            return record

        def estimated_objective(staffing, disposition):
            recourse = np.zeros(len(search_indices))
            for (
                local,
                staffing_gradient,
                disposition_gradient,
                constant,
            ) in cuts["rows"]:
                value = (
                    float(constant)
                    + float(staffing_gradient @ staffing)
                    + (
                        float(disposition_gradient @ disposition)
                        if nFY
                        else 0.0
                    )
                )
                if value > recourse[local]:
                    recourse[local] = value
            return _objective_value(
                staffing,
                recourse,
                costs,
                search_probability,
                data.gamma,
                data.alpha,
            )

        root_points = []
        root_started = time.time()
        for root_round in range(int(options["root_rounds"])):
            if time.time() >= deadline:
                break
            master_result = _build_master_problem(
                data,
                costs,
                upper,
                search_probability,
                cuts,
                False,
                min(root_round_cap, deadline - time.time()),
                0.0,
                False,
            )
            if master_result is None:
                break
            if use_all and master_result["bound"] is not None:
                lower_bound = (
                    master_result["bound"]
                    if lower_bound is None
                    else max(
                        lower_bound,
                        master_result["bound"],
                    )
                )
            relaxed_staffing = master_result["staffing"]
            relaxed_disposition = master_result[
                "disposition"
            ]
            root_points.append(
                (
                    relaxed_staffing.copy(),
                    relaxed_disposition.copy(),
                )
            )
            batch_local = pricing_batch(root_round)
            batch_indices = search_indices[batch_local]
            (
                recourse,
                gradient_staffing,
                gradient_disposition,
                _,
            ) = price(
                relaxed_staffing,
                relaxed_disposition,
                batch_indices,
            )
            added = add_cuts(
                relaxed_staffing,
                relaxed_disposition,
                recourse,
                gradient_staffing,
                gradient_disposition,
                master_result["recourse"][batch_local],
                batch_local,
            )
            if (
                added == 0
                and cut_batch_size >= len(search_indices)
            ):
                break
        # Local search: Large Neighborhood Search (LNS) to improve integer solution
        statistics["root_time"] = (
            time.time() - root_started
        )
        statistics["root_rounds"] = len(root_points)

        seed_disposition = _initial_disposition(
            data, search_indices, search_probability
        )
        seeds = []
        seed_keys = set()

        def add_seed(staffing, disposition):
            key = _candidate_key(
                staffing, disposition
            )
            if key not in seed_keys:
                seed_keys.add(key)
                seeds.append((staffing, disposition))

        if enable_multistart:
            add_seed(np.zeros(nLK), seed_disposition)
            for (
                relaxed_staffing,
                relaxed_disposition,
            ) in root_points[::-1]:
                for mode in ("floor", "nearest", "ceil"):
                    staffing = _staffing_variant(
                        relaxed_staffing,
                        mode,
                        costs,
                        upper,
                        data.B_bar,
                    )
                    add_seed(
                        staffing,
                        np.rint(relaxed_disposition)
                        if nFY
                        else np.zeros(0),
                    )
                    add_seed(staffing, seed_disposition)
                    if len(seeds) >= int(
                        options["max_seed_candidates"]
                    ):
                        break
                if len(seeds) >= int(
                    options["max_seed_candidates"]
                ):
                    break
        elif root_points:
            relaxed_staffing, relaxed_disposition = root_points[-1]
            add_seed(
                _staffing_variant(
                    relaxed_staffing,
                    "nearest",
                    costs,
                    upper,
                    data.B_bar,
                ),
                np.rint(relaxed_disposition)
                if nFY
                else np.zeros(0),
            )
        else:
            add_seed(np.zeros(nLK), seed_disposition)
        exact_candidate_count = int(
            options.get("exact_candidate_count", 0)
        )
        if exact_candidate_count <= 0:
            for staffing, disposition in seeds[
                : int(options["max_seed_candidates"])
            ]:
                if time.time() >= deadline:
                    break
                evaluate_search(staffing, disposition)

        integer_started = time.time()
        integer_rounds = 0
        integer_round_limit = (
            int(options["integer_rounds"])
            if enable_integer_stage
            else 0
        )
        for integer_round in range(integer_round_limit):
            if time.time() >= deadline:
                break
            master_result = _build_master_problem(
                data,
                costs,
                upper,
                search_probability,
                cuts,
                True,
                min(master_round_cap, deadline - time.time()),
                float(options["master_gap"]),
                False,
            )
            if master_result is None:
                break
            integer_rounds += 1
            if use_all and master_result["bound"] is not None:
                lower_bound = (
                    master_result["bound"]
                    if lower_bound is None
                    else max(
                        lower_bound,
                        master_result["bound"],
                    )
                )
            staffing = np.rint(
                master_result["staffing"]
            )
            disposition = np.rint(
                master_result["disposition"]
            )
            if exact_candidate_count > 0:
                staffing = _repair_staffing(
                    staffing,
                    staffing,
                    costs,
                    upper,
                    data.B_bar,
                )
                disposition = np.rint(
                    np.clip(disposition, 0.0, 1.0)
                ).astype(float)
                add_seed(staffing, disposition)
                batch_local = pricing_batch(
                    int(options["root_rounds"])
                    + integer_round
                )
                batch_indices = search_indices[batch_local]
                (
                    recourse,
                    gradient_staffing,
                    gradient_disposition,
                    _,
                ) = price(
                    staffing,
                    disposition,
                    batch_indices,
                )
                added = add_cuts(
                    staffing,
                    disposition,
                    recourse,
                    gradient_staffing,
                    gradient_disposition,
                    master_result["recourse"][batch_local],
                    batch_local,
                )
            else:
                record = evaluate_search(
                    staffing, disposition
                )
                added = add_cuts(
                    record["staffing"],
                    record["disposition"],
                    record["recourse"],
                    record["gradient_staffing"],
                    record["gradient_disposition"],
                    master_result["recourse"],
                )
            if (
                added == 0
                and cut_batch_size >= len(search_indices)
            ):
                break
            if lower_bound is not None and candidate_cache:
                incumbent = min(
                    item["objective"]
                    for item in candidate_cache.values()
                )
                if (
                    incumbent - lower_bound
                    <= float(options["quality_gap"])
                    * max(abs(incumbent), 1.0)
                ):
                    break
        statistics["integer_time"] = (
            time.time() - integer_started
        )
        statistics["integer_rounds"] = integer_rounds

        if exact_candidate_count > 0:
            ranked_seeds = sorted(
                seeds,
                key=lambda candidate: estimated_objective(
                    candidate[0], candidate[1]
                ),
            )
            for staffing, disposition in ranked_seeds[
                :exact_candidate_count
            ]:
                if time.time() >= deadline and candidate_cache:
                    break
                evaluate_search(staffing, disposition)

        local_started = time.time()
        local_iterations = 0
        if enable_local_search and candidate_cache:
            current = min(
                candidate_cache.values(),
                key=lambda record: record["objective"],
            )
            for _ in range(int(options["local_rounds"])):
                if time.time() >= deadline:
                    break
                local_iterations += 1
                weights = _tail_weights(
                    current["recourse"],
                    search_probability,
                    data.gamma,
                    data.alpha,
                )
                staffing_gradient = costs + (
                    weights[:, None]
                    * current["gradient_staffing"]
                ).sum(axis=0)
                disposition_gradient = (
                    (
                        weights[:, None]
                        * current[
                            "gradient_disposition"
                        ]
                    ).sum(axis=0)
                    if nFY
                    else np.zeros(0)
                )
                moves = []
                staffing = current["staffing"]
                disposition = current["disposition"]
                commitment_value = float(
                    costs @ staffing
                )
                for index in range(nLK):
                    if (
                        staffing[index]
                        < upper[index] - 0.5
                        and commitment_value + costs[index]
                        <= data.B_bar + 1e-7
                    ):
                        candidate_staffing = (
                            staffing.copy()
                        )
                        candidate_staffing[index] += 1.0
                        moves.append(
                            (
                                float(
                                    staffing_gradient[index]
                                ),
                                candidate_staffing,
                                disposition.copy(),
                            )
                        )
                    if staffing[index] > 0.5:
                        candidate_staffing = (
                            staffing.copy()
                        )
                        candidate_staffing[index] -= 1.0
                        moves.append(
                            (
                                float(
                                    -staffing_gradient[index]
                                ),
                                candidate_staffing,
                                disposition.copy(),
                            )
                        )
                for index in range(nFY):
                    candidate_disposition = (
                        disposition.copy()
                    )
                    candidate_disposition[index] = (
                        1.0
                        - candidate_disposition[index]
                    )
                    prediction = (
                        disposition_gradient[index]
                        * (
                            1.0
                            - 2.0
                            * disposition[index]
                        )
                    )
                    moves.append(
                        (
                            float(prediction),
                            staffing.copy(),
                            candidate_disposition,
                        )
                    )
                negative_disposition = [
                    index
                    for index in np.argsort(
                        disposition_gradient
                        * (
                            1.0
                            - 2.0 * disposition
                        )
                    )
                    if disposition_gradient[index]
                    * (
                        1.0
                        - 2.0
                        * disposition[index]
                    )
                    < 0.0
                ]
                if len(negative_disposition) >= 2:
                    candidate_disposition = (
                        disposition.copy()
                    )
                    selected = negative_disposition[
                        : min(
                            5,
                            len(negative_disposition),
                        )
                    ]
                    candidate_disposition[selected] = (
                        1.0
                        - candidate_disposition[selected]
                    )
                    prediction = float(
                        np.sum(
                            disposition_gradient[selected]
                            * (
                                1.0
                                - 2.0
                                * disposition[selected]
                            )
                        )
                    )
                    moves.append(
                        (
                            prediction,
                            staffing.copy(),
                            candidate_disposition,
                        )
                    )
                moves.sort(key=lambda move: move[0])
                unique_moves = []
                seen = set()
                for (
                    prediction,
                    candidate_staffing,
                    candidate_disposition,
                ) in moves:
                    key = _candidate_key(
                        candidate_staffing,
                        candidate_disposition,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_moves.append(
                        (
                            prediction,
                            candidate_staffing,
                            candidate_disposition,
                        )
                    )
                    if len(unique_moves) >= int(
                        options["max_neighbors"]
                    ):
                        break
                improved = current
                for (
                    _,
                    candidate_staffing,
                    candidate_disposition,
                ) in unique_moves:
                    if time.time() >= deadline:
                        break
                    record = evaluate_search(
                        candidate_staffing,
                        candidate_disposition,
                    )
                    if (
                        record["objective"]
                        < improved["objective"] - 1e-7
                    ):
                        improved = record
                if improved is current:
                    break
                current = improved
        statistics["local_time"] = (
            time.time() - local_started
        )
        statistics["local_iterations"] = local_iterations

        ranked = sorted(
            candidate_cache.values(),
            key=lambda record: record["objective"],
        )
        best_record = ranked[0] if use_all and ranked else None
        if not use_all:
            full_records = []
            for record in ranked[
                : int(options["full_candidate_count"])
            ]:
                if (
                    time.time() >= deadline
                    and full_records
                ):
                    break
                (
                    recourse,
                    gradient_staffing,
                    gradient_disposition,
                    _,
                ) = price(
                    record["staffing"],
                    record["disposition"],
                    all_indices,
                )
                full_records.append(
                    {
                        "staffing": record[
                            "staffing"
                        ].copy(),
                        "disposition": record[
                            "disposition"
                        ].copy(),
                        "recourse": recourse,
                        "gradient_staffing": gradient_staffing,
                        "gradient_disposition": gradient_disposition,
                        "objective": _objective_value(
                            record["staffing"],
                            recourse,
                            costs,
                            all_probability,
                            data.gamma,
                            data.alpha,
                        ),
                    }
                )
            if full_records:
                best_record = min(
                    full_records,
                    key=lambda record: record[
                        "objective"
                    ],
                )

        if best_record is None:
            staffing = np.zeros(nLK)
            disposition = seed_disposition
            (
                recourse,
                gradient_staffing,
                gradient_disposition,
                _,
            ) = price(
                staffing, disposition, all_indices
            )
            best_record = {
                "staffing": staffing,
                "disposition": disposition,
                "recourse": recourse,
                "gradient_staffing": gradient_staffing,
                "gradient_disposition": gradient_disposition,
                "objective": _objective_value(
                    staffing,
                    recourse,
                    costs,
                    all_probability,
                    data.gamma,
                    data.alpha,
                ),
            }

        if bool(options.get("count_fractional", True)):
            (
                final_recourse,
                final_gradient_staffing,
                final_gradient_disposition,
                fractional_blocks,
            ) = price(
                best_record["staffing"],
                best_record["disposition"],
                all_indices,
                True,
            )
            best_record["recourse"] = final_recourse
            best_record[
                "gradient_staffing"
            ] = final_gradient_staffing
            best_record[
                "gradient_disposition"
            ] = final_gradient_disposition
            best_record["objective"] = _objective_value(
                best_record["staffing"],
                final_recourse,
                costs,
                all_probability,
                data.gamma,
                data.alpha,
            )

        objective = float(best_record["objective"])
        if not use_all:
            lower_bound = None
        if lower_bound is not None:
            lower_bound = min(
                float(lower_bound), objective
            )
            gap = (
                max(objective - lower_bound, 0.0)
                / max(abs(objective), 1e-12)
                * 100.0
            )
        else:
            gap = None
        staffing_matrix = [
            [
                int(
                    round(
                        best_record["staffing"][
                            l * nK + k
                        ]
                    )
                )
                for k in range(nK)
            ]
            for l in range(nL)
        ]
        disposition_matrix = _full_disposition(
            data, best_record["disposition"]
        )
        statistics["evaluations"] = evaluation_count
        statistics["lp_solves"] = lp_count
        statistics["pricing_time"] = pricing_time
        statistics["candidate_count"] = len(
            candidate_cache
        )
        statistics[
            "fractional_blocks"
        ] = fractional_blocks
        if verbose:
            print(
                "[heuristic] objective={0:.10g} bound={1} "
                "gap={2} evaluations={3} scenarios={4}/{5} "
                "time={6:.3f}".format(
                    objective,
                    "-"
                    if lower_bound is None
                    else "{0:.10g}".format(
                        lower_bound
                    ),
                    "-"
                    if gap is None
                    else "{0:.4f}".format(gap),
                    evaluation_count,
                    len(search_indices),
                    nW,
                    time.time() - started,
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if gurobi_pool is not None:

            def dispose_shard(worker):
                def dispose_all(environment):
                    for scenario in range(
                        worker, len(programs), worker_count
                    ):
                        program = programs[scenario]
                        if program is not None:
                            program.model.dispose()

                return dispose_all

            try:
                gurobi_pool.run(
                    [
                        dispose_shard(worker)
                        for worker in range(worker_count)
                    ]
                )
            finally:
                gurobi_pool.close()
        gc.collect()

    return {
        "objective": objective,
        "bound": lower_bound,
        "gap": gap,
        "time": time.time() - started,
        "iterations": evaluation_count,
        "fractional_blocks": fractional_blocks,
        "headcount": staffing_matrix,
        "disposition": disposition_matrix,
        "stats": statistics,
    }

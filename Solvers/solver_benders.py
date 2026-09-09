import gc, math, os, queue, threading, time
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB


_DIAG = bool(os.environ.get("BENDERS_DIAG"))


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
    incumbent_cuts=True,
    mip_outer=False,
    lazy_cut_limit=0,
    block_lazy_rounds=400,
    node_cut_rounds=12,
    node_cut_min_work=2048,
    tree_cut_rounds=1,
    tree_cut_max_node=32,
    tree_cut_min_work=2048,
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


class _AdaptiveRestart(Exception):
    pass


class _EvalDone(Exception):
    pass


def _channel_totals(scens, lay, data, prob, nthr, pool):
    nL, nK, nD, N = lay.nL, lay.nK, lay.nD, lay.N
    HL, HC, HR = lay.HL, lay.HC, lay.HR
    theta_out = data.theta_out
    c_out = data.c_out
    c_ot = data.c_ot.reshape(-1)
    varpi_cell_res = (data.varpi_cell[np.asarray(sorted(data.C_res), dtype=np.int64)]
                      if lay.nrs else np.zeros(0))
    out_mask = (HR == 1)
    out_theta = theta_out[HL, HC]

    tq = data.theta_qc
    qc_lam = tq[lay.L3, lay.C3]
    vsb_lam = data.vs_blk[lay.C3]
    vsp_lam = data.vs_pas[lay.C3]
    psi_lam = data.psi[lay.L3, lay.C3]
    Lidx_lam = lay.L3
    qc_up = (tq[lay.UL, lay.UC] * data.vs_pas[lay.UC]) if lay.n_up else np.zeros(0)
    qc_um = (tq[lay.ML, lay.MC] * data.vs_blk[lay.MC]) if lay.n_um else np.zeros(0)
    adj_rate = np.where(HR == 0, data.vs_adj_in[HC], data.vs_adj_out[HC])
    qc_h = tq[HL, HC] * adj_rate
    phi_blk = data.phi_blk
    phi_pas = data.phi_pas

    keys = ("exp_outsourced_hours", "exp_overtime_hours", "exp_inspection_hours",
            "exp_adjudicated_items", "exp_unreviewed_release_items",
            "exp_unreviewed_removal_items", "exp_shortfall_cell_items",
            "exp_shortfall_lng_net_items", "exp_shortfall_qc_hours",
            "exp_penalty_cost", "exp_flex_labour_cost")
    totals = {k: 0.0 for k in keys}
    R = np.zeros(len(scens))
    split_days = np.zeros(1, dtype=np.int64)
    per_lock = threading.Lock()

    def _shard(i):
        def f(wenv):
            local = {k: 0.0 for k in keys}
            loc_split = 0
            for w in range(i, len(scens), nthr):
                sc = scens[w]
                sc.model.optimize()
                if sc.model.Status != GRB.OPTIMAL:
                    continue
                R[w] = float(sc.model.ObjVal)
                x = np.asarray(sc.x.X).reshape(nD, N)
                pw = float(prob[w])
                h = x[:, lay.o_h:lay.o_h + lay.n_h]
                out_hours = (h[:, out_mask] * out_theta[None, out_mask]).sum()
                over = x[:, lay.o_ov:lay.o_ov + lay.n_ov]
                over_hours = over.sum()
                ot_cost = float((over * c_ot[None, :]).sum())
                out_cost = 0.0
                if out_mask.any():
                    hl_out = HL[out_mask]
                    per_col = out_theta[out_mask] * c_out[hl_out]
                    out_cost = float((h[:, out_mask] * per_col[None, :]).sum())
                adj = h.sum()
                lam = x[:, lay.o_lam:lay.o_lam + lay.n_lam]
                lam3 = lam.reshape(nD, nL * data.nC, data.nP)
                active = (lam3 > 1e-4).sum(axis=2)
                day_split = (active >= 2).any(axis=1)
                n_split = int(day_split.sum())
                up = x[:, lay.o_up:lay.o_up + lay.n_up] if lay.n_up else None
                um = x[:, lay.o_um:lay.o_um + lay.n_um] if lay.n_um else None
                insp = 0.0
                for d in range(nD):
                    g = int(data.gidx[d][w])
                    A = float(data.Z[d][w]) * data.a_bar[:, d]
                    fblk = phi_blk[lay.L3, lay.C3, lay.P3, g]
                    fpas = phi_pas[lay.L3, lay.C3, lay.P3, g]
                    lam_coef = (qc_lam * psi_lam * A[Lidx_lam]
                                * (vsb_lam * fblk + vsp_lam * fpas))
                    insp += float(lam[d] @ lam_coef)
                    insp += float(h[d] @ qc_h)
                    if up is not None:
                        insp += float(up[d] @ qc_up)
                    if um is not None:
                        insp += float(um[d] @ qc_um)
                up_tot = up.sum() if up is not None else 0.0
                um_tot = um.sum() if um is not None else 0.0
                scell = x[:, lay.o_sc:lay.o_sc + nL * lay.nrs]
                cell_items = scell.sum()
                pen_cell = 0.0
                if lay.nrs:
                    tiled = np.tile(varpi_cell_res, nL)
                    pen_cell = float((scell * tiled[None, :]).sum())
                stilde = x[:, lay.o_st:lay.o_st + nL].sum()
                sqc = x[:, lay.o_sq:lay.o_sq + nL].sum()
                pen = (pen_cell + data.varpi_lng * stilde
                       + data.varpi_qc * sqc)
                local["exp_outsourced_hours"] += pw * out_hours
                local["exp_overtime_hours"] += pw * over_hours
                local["exp_inspection_hours"] += pw * insp
                local["exp_adjudicated_items"] += pw * adj
                local["exp_unreviewed_release_items"] += pw * up_tot
                local["exp_unreviewed_removal_items"] += pw * um_tot
                local["exp_shortfall_cell_items"] += pw * cell_items
                local["exp_shortfall_lng_net_items"] += pw * stilde
                local["exp_shortfall_qc_hours"] += pw * sqc
                local["exp_penalty_cost"] += pw * pen
                local["exp_flex_labour_cost"] += pw * (out_cost + ot_cost)
                loc_split += n_split
            with per_lock:
                for k in keys:
                    totals[k] += local[k]
                split_days[0] += loc_split
        return f

    pool.run([_shard(i) for i in range(nthr)])
    result = {k: float(v) for k, v in totals.items()}
    result["E_recourse"] = float(prob @ R)
    result["cvar95"] = _cvar(R, prob, 0.95)
    result["mix_fraction"] = (float(split_days[0]) / (len(scens) * nD)
                              if len(scens) * nD else 0.0)
    result["q_scenarios"] = R.tolist()
    return result


def solve_instance(instance, params=None):
    p = dict(DEFAULTS)
    p.update(params or {})
    t_start = time.time()
    tl = float(p.get("time_limit", 3600.0))
    target = float(p.get("mip_gap", 1e-4))
    verbose = bool(p.get("output", False)) or _DIAG
    deadline = t_start + tl
    thr_in = p.get("threads", None)
    threads = int(thr_in) if thr_in else 0

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    if threads:
        env.setParam("Threads", threads)
    env.start()

    st = {}
    scens = []
    pool = None
    nthr = 1
    mm = None
    UB = None
    bound = -1e30
    gap = None
    best_n = None
    best_y = None
    nfrac = 0
    iters = 0
    channels = None
    nfr = 0
    nL = nC = nK = 0
    data = None
    restart_params = None
    restart_meta = None
    try:
        data = ModelData(instance)
        nL, nC, nK, nD, nP, nW = (data.nL, data.nC, data.nK, data.nD, data.nP, data.nW)
        node_cut_rounds = max(0, int(p["node_cut_rounds"]))
        tree_cut_rounds = max(0, int(p["tree_cut_rounds"]))
        tree_cut_work = nW * nD
        node_cut_min_work = max(0, int(p["node_cut_min_work"]))
        tree_cut_min_work = max(0, int(p["tree_cut_min_work"]))
        if node_cut_min_work and tree_cut_work < node_cut_min_work:
            node_cut_rounds = 0
        if tree_cut_min_work and tree_cut_work < tree_cut_min_work:
            tree_cut_rounds = 0
        st["node_cut_rounds_effective"] = node_cut_rounds
        st["node_cut_min_work"] = node_cut_min_work
        st["tree_cut_rounds_effective"] = tree_cut_rounds
        st["tree_cut_work"] = tree_cut_work
        st["tree_cut_min_work"] = tree_cut_min_work
        parts = int(p["recourse_parts"])
        if parts <= 0:


            parts = 2 if nW * nD <= 200 else 1
        nB = max(1, min(nD, parts))
        day_block = np.minimum(np.arange(nD) * nB // nD, nB - 1)
        block_days = [np.flatnonzero(day_block == b) for b in range(nB)]
        nLK = nL * nK
        gamma, alpha = data.gamma, data.alpha
        prob = data.prob
        chr_flat = data.c_hr.reshape(-1)
        obar_flat = data.o_bar.reshape(-1)
        nbar_flat = data.n_bar.reshape(-1)
        H = data.H

        t0 = time.time()
        lay = _Layout(data)
        if bool(p.get("lock_pref")):
            ub_day = lay.ubX[:lay.N].copy()
            p_ref = data.p_ref
            for l in range(nL):
                for c in range(nC):
                    for pp in range(nP):
                        if pp != p_ref:
                            ub_day[lay.o_lam + (l * nC + c) * nP + pp] = 0.0
            lay.ubX = np.tile(ub_day, nD)
        nfr = lay.nfr
        nFY = nL * nfr
        m_dp, m_cap = lay.m_dp, lay.m_cap
        j_dm = m_dp
        j_cap = 2 * m_dp
        j_ot = j_cap + m_cap
        j_out = j_ot + m_cap
        st["t_layout"] = time.time() - t0

        t0 = time.time()
        bundles = {}
        for g in np.unique(data.gidx):
            bundles[int(g)] = _Bundle(data, lay, int(g))
        st["t_bundle"] = time.time() - t0
        st["n_bundle"] = len(bundles)


        t0 = time.time()
        wk_in = p.get("workers", 0)
        if wk_in:
            nthr = int(wk_in)
        elif threads:
            nthr = threads
        else:
            nthr = min(8, os.cpu_count() or 1)
        nthr = max(1, min(nthr, nW))
        pool = _Pool(nthr, 1)
        st["t_pool"] = time.time() - t0
        st["workers"] = nthr

        t0 = time.time()
        scens = [None] * nW

        def _build_shard(i):
            def f(wenv):
                for w in range(i, nW, nthr):
                    scens[w] = _Scenario(lay, data, wenv, w, bundles, p, 1,
                                         nB > 1)
            return f

        pool.run([_build_shard(i) for i in range(nthr)])
        st["t_build"] = time.time() - t0

        ev = p.get("eval_only")
        if ev is not None:
            n_fix = np.asarray(ev["n"], dtype=float).reshape(-1)
            if nFY:
                fr = sorted(data.C_free)
                y_fix = np.array([float(ev["y"][l][c])
                                  for l in range(nL) for c in fr], dtype=float)
            else:
                y_fix = np.zeros(0)
            Hn = H * n_fix
            On = obar_flat * n_fix
            for w in range(nW):
                sc = scens[w]
                buf = sc.buf
                if nFY:
                    buf[:, :m_dp] = sc.Ubar * (1.0 - y_fix)
                    buf[:, j_dm:j_cap] = sc.Ubar * y_fix
                buf[:, j_cap:j_ot] = Hn
                buf[:, j_ot:j_out] = On
                buf[:, j_out:] = sc.vbar
                sc.cs_state.RHS = buf.ravel()
            channels = _channel_totals(scens, lay, data, prob, nthr, pool)
            channels["F_commit"] = float(chr_flat @ n_fix)
            best_n = n_fix
            best_y = y_fix
            UB = (channels["F_commit"]
                  + (1.0 - gamma) * channels["E_recourse"]
                  + gamma * _cvar(np.asarray(channels["q_scenarios"]),
                                  prob, alpha))
            raise _EvalDone()


        t0 = time.time()
        eta_day_lb = np.zeros((nW, nD))

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
                        if sc.qday is None:
                            eta_day_lb[w, 0] = max(float(sc.model.ObjVal), 0.0)
                        else:
                            eta_day_lb[w] = np.maximum(np.asarray(sc.qday.X), 0.0)
            return f

        pool.run([_lb_shard(i) for i in range(nthr)])
        eta_block_lb = np.column_stack(
            [eta_day_lb[:, ds].sum(axis=1) for ds in block_days])
        st["t_bounds"] = time.time() - t0

        if verbose:
            print("[benders] L=%d C=%d K=%d D=%d P=%d W=%d grid=%d | scen LP "
                  "cols=%d rows=%d nnz=%d | bundles=%d workers=%d layout=%.3f "
                  "bundle=%.3f pool=%.3f build=%.3f bounds=%.3f"
                  % (nL, nC, nK, nD, nP, nW, int(data.phi_mid.shape[3]), lay.NX,
                     lay.M, lay.nnz * nD, len(bundles), nthr, st["t_layout"],
                     st["t_bundle"], st["t_pool"], st["t_build"], st["t_bounds"]),
                  flush=True)


        state = {"ev": 0, "nlp": 0, "nmip": 0, "t_lp": 0.0}
        repair = bool(p["lam_repair"])


        cnt_bad = np.zeros(nthr, dtype=np.int64)
        cnt_lp = np.zeros(nthr, dtype=np.int64)
        cnt_mip = np.zeros(nthr, dtype=np.int64)
        cnt_bad_status = np.zeros(nthr, dtype=np.int64)

        def _price_shard(i, yfree, Hn, On, sel, count, R, Rx, Gn, Gy,
                         Rd, Gnd, Gyd):
            def f(wenv):
                nb = nl = nm = 0
                for w in range(i, nW, nthr):
                    if sel is not None and not sel[w]:
                        continue
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
                    nl += 1
                    if sc.model.Status != GRB.OPTIMAL:
                        cnt_bad_status[i] = 1
                        break
                    v = float(sc.model.ObjVal)
                    R[w] = v
                    P = np.minimum(np.asarray(sc.cs_state.Pi), 0.0).reshape(nD, lay.mS)
                    gd_y = ((P[:, j_dm:j_cap] - P[:, :m_dp]) * sc.Ubar
                            if nFY else None)
                    gd_n = (P[:, j_cap:j_ot] * H
                            + P[:, j_ot:j_out] * obar_flat)
                    if nB == 1:
                        Rd[w, 0] = v
                        Gnd[w, 0] = gd_n.sum(axis=0)
                        if nFY:
                            Gyd[w, 0] = gd_y.sum(axis=0)
                    else:
                        qd = np.asarray(sc.qday.X)
                        for b, ds in enumerate(block_days):
                            Rd[w, b] = qd[ds].sum()
                            Gnd[w, b] = gd_n[ds].sum(axis=0)
                            if nFY:
                                Gyd[w, b] = gd_y[ds].sum(axis=0)
                    Gn[w] = Gnd[w].sum(axis=0)
                    if nFY:
                        Gy[w] = Gyd[w].sum(axis=0)
                    if count:
                        lv = np.asarray(sc.x.X)[sc.lam_idx].reshape(nD, -1)
                        fr = np.any((lv > 1e-6) & (lv < 1.0 - 1e-6), axis=1)
                        nb += int(fr.sum())
                        if repair and fr.any():
                            nm += 1
                            sc.flip_lambda(True, lay.lam_mask)
                            sc.model.optimize()
                            if sc.model.SolCount > 0:
                                v = float(sc.model.ObjVal)
                            sc.flip_lambda(False, lay.lam_mask)
                    Rx[w] = v
                cnt_bad[i] = nb
                cnt_lp[i] = nl
                cnt_mip[i] = nm
            return f


        price_lock = threading.Lock()

        def price(nflat, yfree, sel=None, count=False):
            with price_lock:
                R = np.zeros(nW)
                Rx = np.zeros(nW)
                Gn = np.zeros((nW, nLK))
                Gy = np.zeros((nW, nFY))
                Rd = np.zeros((nW, nB))
                Gnd = np.zeros((nW, nB, nLK))
                Gyd = np.zeros((nW, nB, nFY))
                Hn = H * nflat
                On = obar_flat * nflat
                cnt_bad_status[:] = 0
                t1 = time.time()
                pool.run([_price_shard(i, yfree, Hn, On, sel, count, R, Rx, Gn, Gy,
                                       Rd, Gnd, Gyd)
                          for i in range(nthr)])
                state["t_lp"] += time.time() - t1
                state["nlp"] += int(cnt_lp.sum())
                state["nmip"] += int(cnt_mip.sum())
                state["ev"] += 1
                if cnt_bad_status.any():
                    return None
                return R, Gn, Gy, Rx, int(cnt_bad.sum()), Rd, Gnd, Gyd

        def obj_of(nflat, R):
            return (float(chr_flat @ nflat) + (1.0 - gamma) * float(prob @ R)
                    + gamma * _cvar(R, prob, alpha))


        mm = gp.Model(env=env)
        mm.Params.OutputFlag = 0
        if threads:
            mm.Params.Threads = threads
        nv = mm.addMVar(nLK, lb=0.0, ub=nbar_flat, vtype=GRB.INTEGER)
        yv = mm.addMVar(nFY, lb=0.0, ub=1.0, vtype=GRB.BINARY) if nFY else None
        xiv = mm.addVar(lb=0.0)
        wv = mm.addMVar(nW, lb=0.0)
        ed = mm.addMVar(nW * nB, lb=eta_block_lb.ravel())
        nvl = nv.tolist()
        yvl = yv.tolist() if nFY else []
        edl = ed.tolist()
        wvl = wv.tolist()
        mvars = nvl + yvl
        xyv = gp.MVar.fromlist(mvars)
        com = gp.LinExpr(chr_flat.tolist(), nvl)
        mm.addConstr(com <= data.B_bar)
        day_sum = sp.kron(sp.eye(nW, format="csr"),
                          np.ones((1, nB)), format="csr")
        mm.addConstr(wv - day_sum @ ed + xiv >= 0.0)
        mm.setObjective(com
                        + gp.LinExpr(
                            np.repeat((1.0 - gamma) * prob, nB).tolist(), edl)
                        + gamma * xiv
                        + gp.LinExpr((gamma / (1.0 - alpha) * prob).tolist(), wvl),
                        GRB.MINIMIZE)
        mm.update()

        def cut_of(w, R, Gn, Gy, nflat, yfree):
            if nFY:
                co = np.concatenate([Gn[w], Gy[w]])
                cst = R[w] - float(Gn[w] @ nflat) - float(Gy[w] @ yfree)
            else:
                co = Gn[w]
                cst = R[w] - float(Gn[w] @ nflat)
            return gp.LinExpr(co.tolist(), mvars), float(cst)

        def add_cuts(widx, R, Gn, Gy, nflat, yfree):
            wi = np.asarray(list(widx), dtype=np.int64)
            if wi.size == 0:
                return []
            if nFY:
                co = np.hstack((Gn[wi], Gy[wi]))
                cst = (R[wi] - Gn[wi] @ nflat - Gy[wi] @ yfree)
            else:
                co = Gn[wi]
                cst = R[wi] - Gn[wi] @ nflat
            mc = mm.addConstr(day_sum[wi] @ ed - co @ xyv >= cst)
            return np.asarray(mc.tolist(), dtype=object).ravel().tolist()

        def day_cut_of(w, d, Rd, Gnd, Gyd, nflat, yfree):
            if nFY:
                co = np.concatenate([Gnd[w, d], Gyd[w, d]])
                cst = (Rd[w, d] - float(Gnd[w, d] @ nflat)
                       - float(Gyd[w, d] @ yfree))
            else:
                co = Gnd[w, d]
                cst = Rd[w, d] - float(Gnd[w, d] @ nflat)
            return gp.LinExpr(co.tolist(), mvars), float(cst)

        def add_day_cuts(widx, didx, Rd, Gnd, Gyd, nflat, yfree):
            wi = np.asarray(widx, dtype=np.int64)
            di = np.asarray(didx, dtype=np.int64)
            if wi.size == 0:
                return []
            if nFY:
                co = np.hstack((Gnd[wi, di], Gyd[wi, di]))
                cst = (Rd[wi, di] - Gnd[wi, di] @ nflat
                        - Gyd[wi, di] @ yfree)
            else:
                co = Gnd[wi, di]
                cst = Rd[wi, di] - Gnd[wi, di] @ nflat
            ei = wi * nB + di
            mc = mm.addConstr(ed[ei] - co @ xyv >= cst)
            return np.asarray(mc.tolist(), dtype=object).ravel().tolist()


        t0 = time.time()
        t_master = 0.0
        nv.VType = GRB.CONTINUOUS
        if nFY:
            yv.VType = GRB.CONTINUOUS
        if int(p["master_method"]) >= 0:
            mm.Params.Method = int(p["master_method"])
        cutrows = []
        rootLB = -1e30
        rootstop = False
        prevLB = -1e30
        stall = 0
        rounds = 0
        best_R = None
        best_Rd = None
        best_Gnd = None
        best_Gyd = None
        sf = float(p["scen_frac"])
        if sf <= 0.0:
            sf = 1.0 if nW * nD <= 200 else 0.60
        frac_n = nW if sf >= 1.0 else max(1, min(nW, int(math.ceil(sf * nW))))
        ub_every = int(p["root_ub_every"])
        for it in range(int(p["root_rounds"])):
            if time.time() > deadline:
                break
            t2 = time.time()
            mm.optimize()
            t_master += time.time() - t2
            if mm.Status != GRB.OPTIMAL:
                break
            LB = float(mm.ObjVal)
            rootLB = max(rootLB, LB)
            nh = np.asarray(nv.X)
            yh = np.asarray(yv.X) if nFY else np.zeros(0)
            edx = np.asarray(ed.X).reshape(nW, nB)
            lim = int(p["cut_purge"])
            if lim and len(cutrows) > lim:
                bas = mm.getAttr("CBasis", cutrows)
                drop = [c for c, b in zip(cutrows, bas) if b == 0]
                if drop:
                    cutrows = [c for c, b in zip(cutrows, bas) if b != 0]
                    mm.remove(drop)
                    mm.update()
            if frac_n >= nW:
                sel = None
                wlist = range(nW)
            else:
                sel = np.zeros(nW, dtype=bool)
                s0 = (it * frac_n) % nW
                wlist = [(s0 + j) % nW for j in range(frac_n)]
                sel[wlist] = True
            res = price(nh, yh, sel=sel)
            if res is None:
                break
            R, Gn, Gy, Rd, Gnd, Gyd = (res[0], res[1], res[2],
                                       res[5], res[6], res[7])
            wl = np.asarray(list(wlist), dtype=np.int64)
            vr = (Rd[wl] > edx[wl]
                  + 1e-7 * np.maximum(1.0, np.abs(Rd[wl])))
            iw, iday = np.nonzero(vr)
            vio_w = wl[iw]
            added = int(vio_w.size)
            if added:
                cutrows.extend(add_day_cuts(vio_w, iday, Rd, Gnd, Gyd, nh, yh))
            rounds += 1
            if added:
                mm.update()
            if ub_every and (it % ub_every == 0):
                ni = np.minimum(np.ceil(nh - 1e-7), nbar_flat)
                if float(chr_flat @ ni) > data.B_bar + 1e-6:
                    ni = np.floor(nh + 1e-7)
                yi = np.round(yh) if nFY else np.zeros(0)
                if float(chr_flat @ ni) <= data.B_bar + 1e-6:
                    r2 = price(ni, yi)
                    if r2 is not None:
                        ub = obj_of(ni, r2[3])
                        if UB is None or ub < UB - 1e-9:
                            UB = ub
                            best_n = ni.copy()
                            best_y = yi.copy()
                            best_R = r2[0].copy()
                            best_Rd = r2[5].copy()
                            best_Gnd = r2[6].copy()
                            best_Gyd = r2[7].copy()


                        if nFY:
                            at_root = (r2[5]
                                       + np.einsum("wdk,k->wd", r2[6], nh - ni)
                                       + np.einsum("wdk,k->wd", r2[7], yh - yi))
                        else:
                            at_root = (r2[5]
                                       + np.einsum("wdk,k->wd", r2[6], nh - ni))
                        v2 = (at_root > edx
                              + 1e-7 * np.maximum(1.0, np.abs(at_root)))
                        vw, vd = np.nonzero(v2)
                        if vw.size:
                            cutrows.extend(add_day_cuts(vw, vd, r2[5], r2[6],
                                                        r2[7], ni, yi))
                            added += int(vw.size)
                            mm.update()
                if UB is not None and UB - rootLB <= target * abs(UB):
                    rootstop = True
                    break
            if added == 0:
                break
            if LB <= prevLB + 1e-9 * max(1.0, abs(LB)):
                stall += 1
                if stall >= int(p["stall_rounds"]):
                    break
            else:
                stall = 0
            prevLB = LB


        adapt_at = int(p["adaptive_block_rounds"])
        if (int(p["recourse_parts"]) <= 0 and nB == 1 and adapt_at > 0
                and rounds >= adapt_at and deadline - time.time() > 5.0):
            restart_params = dict(p)
            restart_params["recourse_parts"] = 2
            restart_params["adaptive_block_rounds"] = 0
            restart_meta = {"rounds": rounds, "evals": state["ev"]}
            if verbose:
                print("[benders] adaptive epigraph refinement after %d root rounds"
                      % rounds, flush=True)
            raise _AdaptiveRestart()


        final_keep = int(p["final_cut_keep"]) * nB
        final_purged = 0
        if final_keep >= 0 and len(cutrows) > final_keep and time.time() < deadline:
            t2 = time.time()
            mm.optimize()
            t_master += time.time() - t2
            if mm.Status == GRB.OPTIMAL:
                rootLB = max(rootLB, float(mm.ObjVal))
                bas = mm.getAttr("CBasis", cutrows)
                old = len(cutrows) - final_keep
                drop = [c for j, (c, b) in enumerate(zip(cutrows, bas))
                        if j < old and b == 0]
                if drop:
                    final_purged = len(drop)
                    keep_ids = set(id(c) for c in drop)
                    cutrows = [c for c in cutrows if id(c) not in keep_ids]
                    mm.remove(drop)
                    mm.update()


        incumbent_cut_count = 0
        if (bool(p["incumbent_cuts"]) and best_Gnd is not None
                and best_n is not None):
            iw = np.repeat(np.arange(nW, dtype=np.int64), nB)
            ib = np.tile(np.arange(nB, dtype=np.int64), nW)
            rows = add_day_cuts(iw, ib, best_Rd, best_Gnd, best_Gyd,
                                best_n, best_y)
            cutrows.extend(rows)
            incumbent_cut_count = len(rows)
            mm.update()
        nv.VType = GRB.INTEGER
        if nFY:
            yv.VType = GRB.BINARY
            yv.BranchPriority = np.full(nFY, int(p["branch_y"]), dtype=int)
        nv.BranchPriority = np.full(nLK, int(p["branch_n"]), dtype=int)
        if int(p["master_method"]) >= 0:
            mm.Params.Method = -1
        mm.update()
        st["t_root"] = time.time() - t0
        st["t_master"] = t_master
        st["root_rounds"] = rounds
        st["final_purged"] = final_purged
        st["incumbent_cuts"] = incumbent_cut_count


        cache = {}
        best = {"UB": UB, "n": best_n, "y": best_y, "R": best_R,
                "Rd": best_Rd}


        if bool(p["mip_start"]) and best_R is not None and best_n is not None:
            _, xi0 = _cvar(best_R, prob, alpha, True)
            nv.Start = best_n
            if nFY:
                yv.Start = best_y
            if best_Rd is not None:
                ed.Start = best_Rd.ravel()
            xiv.Start = xi0
            wv.Start = np.maximum(best_R - xi0, 0.0)

        def set_start(n0, y0, R0, Rd0):
            _, xi0 = _cvar(R0, prob, alpha, True)
            nv.Start = n0
            if nFY:
                yv.Start = y0
            ed.Start = Rd0.ravel()
            xiv.Start = xi0
            wv.Start = np.maximum(R0 - xi0, 0.0)

        lazy_state = {"calls": 0, "cuts": 0, "hits": 0,
                      "node_rounds": 0, "node_cuts": 0,
                      "root_done": False,
                      "tree_rounds": 0, "tree_cuts": 0}

        def cb(model, where):
            if where == GRB.Callback.MIPNODE:
                if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                    return
                nodcnt = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                at_root = nodcnt <= 0.5
                if at_root:
                    lim = node_cut_rounds
                    if (lim <= 0 or lazy_state["root_done"]
                            or lazy_state["node_rounds"] >= lim):
                        return
                else:
                    lim = tree_cut_rounds
                    if (lim <= 0 or lazy_state["tree_rounds"] >= lim
                            or nodcnt > float(p["tree_cut_max_node"])):
                        return
                nh = np.asarray(model.cbGetNodeRel(nvl))
                yh = (np.asarray(model.cbGetNodeRel(yvl))
                      if nFY else np.zeros(0))
                edx = np.asarray(model.cbGetNodeRel(edl)).reshape(nW, nB)
                hit = price(nh, yh)
                if hit is None:
                    if at_root:
                        lazy_state["root_done"] = True
                    return
                Rd, Gnd, Gyd = hit[5], hit[6], hit[7]
                vr = (Rd > edx + 1e-7 * np.maximum(1.0, np.abs(Rd)))
                wi, di = np.nonzero(vr)
                if at_root:
                    lazy_state["node_rounds"] += 1
                else:
                    lazy_state["tree_rounds"] += 1
                if wi.size == 0:
                    if at_root:
                        lazy_state["root_done"] = True
                    return
                for w, d in zip(wi, di):
                    ex, cst = day_cut_of(int(w), int(d), Rd, Gnd, Gyd, nh, yh)
                    model.cbCut(edl[int(w) * nB + int(d)] - ex >= cst)
                if at_root:
                    lazy_state["node_cuts"] += int(wi.size)
                else:
                    lazy_state["tree_cuts"] += int(wi.size)
                return
            if where != GRB.Callback.MIPSOL:
                return
            lazy_state["calls"] += 1
            nh = np.round(np.asarray(model.cbGetSolution(nvl)))
            yh = np.round(np.asarray(model.cbGetSolution(yvl))) if nFY else np.zeros(0)
            kk = (nh.tobytes(), yh.tobytes())
            hit = cache.get(kk)
            if hit is None:
                hit = price(nh, yh)
                if hit is None:
                    return
                cache[kk] = hit
            else:
                lazy_state["hits"] += 1
            R, Gn, Gy, Rx = hit[0], hit[1], hit[2], hit[3]
            cutlim = int(p["lazy_cut_limit"])


            block_rounds = int(p["block_lazy_rounds"])
            if (block_rounds <= 0
                    or lazy_state["calls"] <= block_rounds):
                Rd, Gnd, Gyd = hit[5], hit[6], hit[7]
                edx = np.asarray(
                    model.cbGetSolution(edl)).reshape(nW, nB)
                viol = Rd - edx
                wi, di = np.nonzero(
                    viol > 1e-7 * np.maximum(1.0, np.abs(Rd)))
                if cutlim > 0 and wi.size > cutlim:
                    score = (viol[wi, di]
                             / np.maximum(1.0, np.abs(Rd[wi, di])))
                    keep = np.argpartition(score, -cutlim)[-cutlim:]
                    wi, di = wi[keep], di[keep]
                for w, d in zip(wi, di):
                    ex, cst = day_cut_of(
                        int(w), int(d), Rd, Gnd, Gyd, nh, yh)
                    model.cbLazy(
                        edl[int(w) * nB + int(d)] - ex >= cst)
                n_added = int(wi.size)
            else:
                edx = np.asarray(
                    model.cbGetSolution(edl)).reshape(nW, nB)
                viol = R - edx.sum(axis=1)
                wi = np.flatnonzero(
                    viol > 1e-7 * np.maximum(1.0, np.abs(R)))
                if cutlim > 0 and wi.size > cutlim:
                    score = viol[wi] / np.maximum(1.0, np.abs(R[wi]))
                    wi = wi[np.argpartition(score, -cutlim)[-cutlim:]]
                for w in wi:
                    ex, cst = cut_of(int(w), R, Gn, Gy, nh, yh)
                    model.cbLazy(
                        gp.quicksum(edl[int(w) * nB:(int(w) + 1) * nB])
                        - ex >= cst)
                n_added = int(wi.size)
            lazy_state["cuts"] += n_added
            ub = obj_of(nh, Rx)
            if best["UB"] is None or ub < best["UB"] - 1e-9:
                best["UB"] = ub
                best["n"] = nh.copy()
                best["y"] = yh.copy()
                best["R"] = R.copy()
                best["Rd"] = hit[5].copy()
            if best["UB"] is not None:
                lb = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                if lb > -1e30 and (best["UB"] - lb) <= target * abs(best["UB"]):
                    model.terminate()

        t0 = time.time()
        mip_rounds = 0
        mip_cuts = 0
        if p["mip_focus"] is not None:
            mm.Params.MIPFocus = int(p["mip_focus"])
        if p["gurobi_cuts"] is not None:
            mm.Params.Cuts = int(p["gurobi_cuts"])
        if not rootstop and bool(p["mip_outer"]):


            mm.Params.MIPGap = target
            if p["mip_heur"] is not None:
                mm.Params.Heuristics = float(p["mip_heur"])
            while time.time() < deadline:
                mm.Params.TimeLimit = max(deadline - time.time(), 1.0)
                mm.optimize()
                mip_rounds += 1
                try:
                    bound = max(rootLB, float(mm.ObjBound))
                except Exception:
                    bound = rootLB
                if mm.SolCount <= 0:
                    break
                nh = np.round(np.asarray(nv.X))
                yh = np.round(np.asarray(yv.X)) if nFY else np.zeros(0)
                kk = (nh.tobytes(), yh.tobytes())
                hit = cache.get(kk)
                if hit is None:
                    hit = price(nh, yh)
                    if hit is None:
                        break
                    cache[kk] = hit
                R, Gn, Gy, Rx = hit[0], hit[1], hit[2], hit[3]
                ub = obj_of(nh, Rx)
                if best["UB"] is None or ub < best["UB"] - 1e-9:
                    best["UB"] = ub
                    best["n"] = nh.copy()
                    best["y"] = yh.copy()
                    best["R"] = R.copy()
                    best["Rd"] = hit[5].copy()
                evx = np.asarray(ed.X).reshape(nW, nB).sum(axis=1)
                vio = np.flatnonzero(
                    R > evx + 1e-7 * np.maximum(1.0, np.abs(R)))
                if vio.size == 0:
                    break
                rows = add_cuts(vio, R, Gn, Gy, nh, yh)
                cutrows.extend(rows)
                mip_cuts += len(rows)
                mm.update()
                set_start(nh, yh, R, hit[5])
                if (best["UB"] is not None and bound > -1e30
                        and best["UB"] - bound <= target * abs(best["UB"])):
                    break
        elif not rootstop:
            mm.Params.LazyConstraints = 1
            if node_cut_rounds > 0 or tree_cut_rounds > 0:
                mm.Params.PreCrush = 1
            mm.Params.MIPGap = target
            mm.Params.TimeLimit = max(deadline - time.time(), 1.0)
            if p["mip_heur"] is not None:
                mm.Params.Heuristics = float(p["mip_heur"])
            mm.optimize(cb)
            mip_rounds = 1
            try:
                bound = max(rootLB, float(mm.ObjBound))
            except Exception:
                bound = rootLB
        else:
            bound = rootLB
        st["t_mip"] = time.time() - t0
        st["mip_rounds"] = mip_rounds
        st["mip_cuts"] = mip_cuts
        st["lazy_calls"] = lazy_state["calls"]
        st["lazy_cuts"] = lazy_state["cuts"]
        st["cache_hits"] = lazy_state["hits"]
        st["node_rounds"] = lazy_state["node_rounds"]
        st["node_cuts"] = lazy_state["node_cuts"]
        st["tree_rounds"] = lazy_state["tree_rounds"]
        st["tree_cuts"] = lazy_state["tree_cuts"]

        UB = best["UB"]
        best_n = best["n"]
        best_y = best["y"]


        if best_n is not None and bool(p["count_frac"]):
            chk = price(best_n, best_y, count=True)
            if chk is not None:
                nfrac = chk[4]
                UB = obj_of(best_n, chk[3])
        if bool(p.get("extract_channels")) and best_n is not None:
            yfree = best_y if nFY else np.zeros(0)
            Hn = H * best_n
            On = obar_flat * best_n
            for w in range(nW):
                sc = scens[w]
                buf = sc.buf
                if nFY:
                    buf[:, :m_dp] = sc.Ubar * (1.0 - yfree)
                    buf[:, j_dm:j_cap] = sc.Ubar * yfree
                buf[:, j_cap:j_ot] = Hn
                buf[:, j_ot:j_out] = On
                buf[:, j_out:] = sc.vbar
                sc.cs_state.RHS = buf.ravel()
            channels = _channel_totals(scens, lay, data, prob, nthr, pool)
            channels["F_commit"] = float(chr_flat @ best_n)
        if UB is not None and bound > -1e30 and abs(UB) > 1e-12:
            gap = max(UB - bound, 0.0) / abs(UB) * 100.0
        iters = state["ev"]
        st.update(nlp=state["nlp"], nmip=state["nmip"], cuts=len(cutrows),
                  t_price=state["t_lp"])
        if verbose:
            print("[benders] root=%.3f(%d) t_master=%.3f mip=%.3f t_price=%.3f | "
                  "lp=%d mip_sub=%d eval=%d frac=%d cuts=%d scen_frac=%.2f "
                  "obj=%.10g bound=%.10g gap=%s target=%g threads=%s | "
                  "lazy=%d/%d hit=%d node=%d/%d purged=%d"
                  % (st["t_root"], st["root_rounds"], st.get("t_master", 0.0),
                     st["t_mip"], state["t_lp"], state["nlp"], state["nmip"],
                     state["ev"], nfrac, len(cutrows), sf,
                     UB if UB else float("nan"), bound,
                     ("%.4g" % gap) if gap is not None else "-", target, thr_in,
                     lazy_state["calls"], lazy_state["cuts"], lazy_state["hits"],
                     lazy_state["node_rounds"], lazy_state["node_cuts"],
                     final_purged),
                  flush=True)
    except _AdaptiveRestart:
        pass
    except _EvalDone:
        gap = None
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

    if restart_params is not None:
        probe_time = time.time() - t_start
        restart_params["time_limit"] = max(deadline - time.time(), 1.0)
        result = solve_instance(instance, restart_params)
        result["time"] += probe_time
        result["iterations"] += int(restart_meta["evals"])
        result.setdefault("stats", {})["adaptive_restarted"] = 1
        result["stats"]["adaptive_probe_time"] = probe_time
        result["stats"]["adaptive_probe_rounds"] = int(restart_meta["rounds"])
        result["stats"]["adaptive_probe_evals"] = int(restart_meta["evals"])
        return result

    head = None
    dispo = None
    if best_n is not None:
        head = [[int(round(best_n[l * nK + k])) for k in range(nK)] for l in range(nL)]
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
            "fractional_blocks": nfrac,
            "headcount": head,
            "disposition": dispo,
            "channels": channels,
            "stats": st}

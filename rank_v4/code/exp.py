"""Experiment core for protocol v4.2: setup (splits, source products, refitted ridge), source pools,
attacks, per-scenario scores and within-target removal (scenario S1), plus the false-flag (FF) control."""
import math
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
import common as C

SCORE_METHODS = ("gap_rm", "gap_ens", "saad_core", "gap_cal", "R4")     # ties: shared pi_tie
RAND_METHODS = ("B2p", "R2", "R3", "LAF")                                # randomness: method-specific pi_rand
ALL_METHODS = ("B0",) + SCORE_METHODS + RAND_METHODS


class Infeasible(Exception):
    pass


class Setup:
    def __init__(self, work, splits=("val", "test")):
        tr = pd.read_parquet(f"{work}/tr_corpus.parquet")
        self.E = np.load(f"{work}/emb_tr.npy"); self.S = np.load(f"{work}/sent_tr.npy")
        assert len(self.E) == len(tr) == len(self.S)
        self.tr = tr
        self.rid = tr.rid.values; self.rating = tr.rating.values.astype(float); self.pid = tr.pid.values
        self.cat = tr.category.values; self.norm = np.array([C.norm_text(t) for t in tr.text.values], dtype=object)
        self.rid_num = np.array([int(r.split("-")[1]) for r in self.rid])
        ev = tr[tr.part == "evaluation"]; cnt = ev.pid.value_counts()
        pool = np.array(sorted(cnt[cnt >= 40].index))
        self.pool = pool
        # 192-product clean ranking -> halves -> stratified 1/3 validation split
        mu = ev[ev.pid.isin(pool)].rating.mean()
        sc = {p: C.bayes(self.rating[self.pid == p], mu) for p in pool}
        order = sorted(pool, key=lambda p: (-sc[p], p)); h = len(order) // 2
        g = C.rng(0); val = []
        for half in (sorted(order[:h]), sorted(order[h:])):
            perm = g.permutation(np.array(half)); val += list(perm[: len(half) // 3])
        self.split_pids = {"val": np.array(sorted(val)), "test": np.array(sorted(set(pool) - set(val)))}
        # source products and refitted ridge
        ref_p = np.array(sorted(tr[tr.part == "reference"].pid.unique()))
        perm = C.rng(1).permutation(ref_p); ns = int(round(C.SOURCE_SHARE * len(ref_p)))
        self.src_pids = np.array(sorted(perm[:ns]))
        is_ref = (tr.part == "reference").values; is_src = np.isin(self.pid, self.src_pids)
        fit = is_ref & ~is_src
        self.ridge = Ridge(alpha=10.0).fit(self.E[fit], self.rating[fit])
        self.ridge_meta = C.model_meta("ridge_srcout", self.ridge, fit.sum())
        self.pred = self.ridge.predict(self.E)
        self.src_idx = np.where(is_src)[0]
        self.universes = {s: Universe(self, s) for s in splits}   # selection run builds only "val"

    def source_pool(self, fam, p, target_norms):
        """Eligible source indices for target p (protocol 2): category, star class, no text match with the
        target, de-duplicated by normalised text (smallest corpus id kept); returned in corpus-id order."""
        cat = self.cat[self.pid == p][0]
        i = self.src_idx
        i = i[(self.cat[i] == cat) & np.isin(self.rating[i], C.FAM_SRC_STARS[fam])]
        i = i[np.argsort(self.rid_num[i], kind="stable")]
        seen, out = set(target_norms), []
        for j in i:
            if self.norm[j] not in seen:
                seen.add(self.norm[j]); out.append(j)
        return np.array(out, int)


class Universe:
    """One split: clean scores, per-review raw signals and the fixed clean ranking universe."""

    def __init__(self, st, split):
        self.st, self.split = st, split
        pids = st.split_pids[split]; self.pids = pids
        idx = {p: np.where(st.pid == p)[0] for p in pids}
        self.idx = idx
        allidx = np.concatenate([idx[p] for p in pids])
        self.mu = st.rating[allidx].mean()
        self.clean = {p: C.bayes(st.rating[idx[p]], self.mu) for p in pids}
        order = sorted(pids, key=lambda p: (-self.clean[p], p)); h = len(order) // 2
        self.half = {p: ("upper" if i < h else "lower") for i, p in enumerate(order)}
        self.n = {p: len(idx[p]) for p in pids}
        self.L = {p: int(np.sum(st.rating[idx[p]] <= 2)) for p in pids}
        # per-review raw signals of the clean universe (product order = pids order)
        self.pre = {p: np.abs((st.rating[idx[p]] - 1) / 4 - st.S[idx[p]]) for p in pids}
        self.rm = {p: np.abs(st.rating[idx[p]] - st.pred[idx[p]]) for p in pids}
        self.ctx = {p: C.ctx_distance(st.E[idx[p]]) for p in pids}
        self.rat = {p: st.rating[idx[p]] for p in pids}

    def others_clean(self, p):
        return np.array([self.clean[q] for q in self.pids if q != p])

    def universe_vectors(self, p, pre_t, rm_t, ctx_t, rat_t):
        """Raw signal vectors of the whole universe with product p replaced by its (attacked) reviews."""
        o = [q for q in self.pids if q != p]
        cat = lambda d, t: np.concatenate([d[q] for q in o] + [t])
        return cat(self.pre, pre_t), cat(self.rm, rm_t), cat(self.ctx, ctx_t), cat(self.rat, rat_t)


def attack(st, U, fam, r, p, alpha):
    """Inject k_p reviews into target p (replicate r). Returns dict with the target's review table."""
    i = U.idx[p]
    pool = st.source_pool(fam, p, set(st.norm[i]))
    k = math.ceil(alpha * U.n[p])
    if len(pool) < k:
        raise Infeasible(f"{fam} p={p}: pool {len(pool)} < k {k}")
    draw = pool[C.rng(2, C.FAM_NO[fam], C.SPLIT_NO[U.split], r, p).choice(len(pool), k, replace=False)]
    inj_rids = [C.inj_rid(fam, U.split, r, p, j, st.rid[s]) for j, s in enumerate(draw)]
    return dict(fam=fam, r=r, p=p, k=k, src=draw,
                rid=np.concatenate([st.rid[i], np.array(inj_rids, dtype=object)]),
                rating=np.concatenate([st.rating[i], np.full(k, C.FAM_RATING[fam])]),
                sent=np.concatenate([st.S[i], st.S[draw]]), pred=np.concatenate([st.pred[i], st.pred[draw]]),
                emb=np.vstack([st.E[i], st.E[draw]]), norm=np.concatenate([st.norm[i], st.norm[draw]]),
                inj=np.concatenate([np.zeros(len(i), bool), np.ones(k, bool)]))


def clean_target(st, U, p):
    i = U.idx[p]
    return dict(fam=None, r=None, p=p, k=0, src=np.array([], int), rid=st.rid[i], rating=st.rating[i],
                sent=st.S[i], pred=st.pred[i], emb=st.E[i], norm=st.norm[i], inj=np.zeros(len(i), bool))


def target_scores(U, T):
    """All method scores for the target's reviews, with robust-z statistics over the whole universe."""
    r = T["rating"]
    pre_t = np.abs((r - 1) / 4 - T["sent"]); rm_t = np.abs(r - T["pred"]); ctx_t = C.ctx_distance(T["emb"])
    pre_u, rm_u, ctx_u, rat_u = U.universe_vectors(T["p"], pre_t, rm_t, ctx_t, r)
    z = lambda x, u: (x - C.rz_params(u)[0]) / C.rz_params(u)[1]
    ens = (z(pre_t, pre_u) + z(rm_t, rm_u)) / 2
    cal = np.empty(len(r))
    for c in np.unique(r):
        m, s = C.rz_params(rm_u[rat_u == c]); cal[r == c] = (rm_t[r == c] - m) / s
    n = len(r); loo = (r.sum() - r) / (n - 1)
    return {"gap_rm": rm_t, "gap_ens": ens, "saad_core": np.maximum(ens, z(ctx_t, ctx_u)), "gap_cal": cal,
            "R4": np.abs(r - loo),
            "B2p": np.zeros(n), "R2": (r == 1).astype(float), "R3": -r, "LAF": T["inj"].astype(float)}


def run_methods(U, T, b, stage, fam, r_key, unit_kind, methods=ALL_METHODS):
    """Remove b of the target's reviews with each method; returns {method: metrics} and the removal sets."""
    p = T["p"]; rids = list(T["rid"]); sc = target_scores(U, T)
    tie = [C.priority("tie", stage, fam, U.split, None if stage == "ff" else r_key, unit_kind, p, "*", x) for x in rids]
    others = U.others_clean(p); rank_clean = C.rank_of(U.clean[p], others)
    orig = ~T["inj"]; low = orig & (T["rating"] <= 2)
    out, removed = {}, {}
    for m in methods:
        if m == "B0":
            rem = np.zeros(len(rids), bool); near = "none"
        else:
            if m in RAND_METHODS:
                prio = [C.priority("rand", stage, fam, U.split, r_key, unit_kind, p, m, x) for x in rids]
            else:
                prio = tie
            order = C.removal_order(sc[m], prio, rids)
            bm = min(b, T["k"]) if m == "LAF" else b          # LAF removes min(b, k) injected reviews, no fill
            rem = np.zeros(len(rids), bool); rem[order[:bm]] = True
            near = C.ulp_near_boundary(sc[m], order, bm)
        s_f = C.bayes(T["rating"][~rem], U.mu)
        star = {f"del_{c}": (float(rem[orig & (T['rating'] == c)].mean()) if np.any(orig & (T["rating"] == c)) else np.nan)
                for c in (1, 2, 3, 4, 5)}
        L = int(low.sum())
        out[m] = dict(n_removed=int(rem.sum()), D=abs(C.rank_of(s_f, others) - rank_clean), score_dev=abs(s_f - U.clean[p]),
                      capture=(float(rem[T["inj"]].mean()) if T["k"] else np.nan), L=L,
                      retained_low=L - int(rem[low].sum()), TRR=((L - int(rem[low].sum())) / L if L else np.nan),
                      ulp_near=near, **star)
        removed[m] = rem
    return out, removed, sc


def eligibility(st, U, fam, alpha):
    """Eligible targets of a family in one split (protocol section 14): pool >= k_p (A3 also needs the paired A2 pool)
    and L_p >= 1. Returns (eligible pids, pool-excluded pids)."""
    T = sorted(p for p in U.pids if U.half[p] == C.FAM_HALF[fam])
    pools = [len(st.source_pool(fam, p, set(st.norm[U.idx[p]]))) for p in T]
    if fam == "A3":
        a2 = [len(st.source_pool("A2", p, set(st.norm[U.idx[p]]))) for p in T]
        pools = [min(x, y) for x, y in zip(pools, a2)]
    ok = C.eligible_targets([U.n[p] for p in T], [U.L[p] for p in T], pools, alpha)
    excluded = [p for p, q in zip(T, pools) if q < math.ceil(alpha * U.n[p])]
    return [p for p, e in zip(T, ok) if e], excluded


def polarity_pool_rids(st, fam):
    """Source-review ids of a family's polarity check: all source-product reviews in the family's source star classes."""
    i = st.src_idx[np.isin(st.rating[st.src_idx], C.FAM_SRC_STARS[fam])]
    return list(st.rid[i])

"""Cluster all training samples by RNA sequence (MMseqs2) and split them 90/5/5 by cluster.

Samples = what the h5 will contain:
  * small entries the builder reads from pdb_files/ (from the v8 h5, those passing the training filters)
  * files written by split_entries_to_pdb.py: whole RNA chains with context, and crops of large rRNA chains

Each sample gets the sequence cluster(s) of its RNA:
  * whole-chain file: its RNA chain (chain A)          * crop: the rRNA chain it was cut from
  * small entry: all its RNA chains (>= --min-len nt); an entry with several RNA chains joins their clusters
Sequences of at least 11 nt are clustered with MMseqs2 (--min-seq-id, -c coverage of both sequences);
shorter ones are only grouped when identical. Groups of joined clusters never cross splits.

Groups are shuffled (--seed). Groups containing the --balance modification (PSU) go first to validation or test,
until each holds --frac of its distinct residues; other groups then fill each split to --frac of the samples;
groups larger than --max-group samples always go to train, so that one RNA family cannot fill a test set.

Outputs (in --out-dir):
  splits_mods_{train,validation,test}.txt   sample ids as stored in the h5 (e.g. 1EHZ, 7F5S_L5-CROP3)
  splits_mods_samples.csv                   sample, kind, cluster group, split, sequence length, PSU/5MC/OMG

Usage:
    python make_splits.py
"""
import argparse
import os
import subprocess
from collections import Counter, defaultdict
from multiprocessing import Pool

import gemmi
import h5py
import numpy as np
import pandas as pd

ROOT = "/scratch/izar/maazoun"
MMSEQS = f"{ROOT}/tools/mmseqs/bin/mmseqs"
MODS = ["PSU", "5MC", "OMG"]
TRAIN_MAX_SIZE, TRAIN_MIN_RES, TRAIN_MAX_BA = 1024 * 8, 10, 1   # config.py
MIN_KMER_LEN = 11                                                 # mmseqs -k 11


def to_nt(seq):
    """RNA sequence -> MMseqs2 nucleotide alphabet (U -> T, anything unusual -> N)."""
    seq = seq.upper().replace("U", "T")
    return "".join(c if c in "ACGT" else "N" for c in seq)


def small_entry_chains(args):
    """RNA chain sequences and modification counts of one small entry (first model)."""
    pdb_id, path, min_len = args
    try:
        st = gemmi.read_structure(path)
        st.setup_entities()
        seqs, mods = [], Counter()
        for ch in st[0]:
            poly = ch.get_polymer()
            if poly.check_polymer_type() not in (gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid):
                continue
            names = [r.name for r in poly]
            mods.update(n for n in names if n in MODS)
            if len(names) >= min_len:
                seqs.append(to_nt(gemmi.one_letter_code(names)))
        return pdb_id, seqs, mods, ""
    except Exception as e:
        return pdb_id, [], Counter(), repr(e)


def run_mmseqs(seqs, work, min_seq_id, cov, threads):
    """Cluster unique sequences; returns {sequence: representative sequence}."""
    os.makedirs(work, exist_ok=True)
    fa = os.path.join(work, "seqs.fa")
    with open(fa, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">s{i}\n{s}\n")
    subprocess.run([MMSEQS, "easy-cluster", fa, os.path.join(work, "res"), os.path.join(work, "tmp"),
                    "--min-seq-id", str(min_seq_id), "-c", str(cov), "--cov-mode", "0", "--dbtype", "2",
                    "-k", "11", "--split-memory-limit", "4G", "--threads", str(threads), "-v", "1"], check=True)
    tsv = pd.read_csv(os.path.join(work, "res_cluster.tsv"), sep="\t", header=None, names=["rep", "member"])
    idx = {f"s{i}": s for i, s in enumerate(seqs)}
    return {idx[m]: idx[r] for r, m in zip(tsv.rep, tsv.member)}


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=f"{ROOT}/pdb_download/pdb_split/manifest.csv")
    ap.add_argument("--h5", default=f"{ROOT}/risotto/datasets/pdb_bprna_structures_nuc_v8_mods.h5",
                    help="current h5, used for the small entries that pass the training filters")
    ap.add_argument("--out-dir", default=f"{ROOT}/risotto/datasets")
    ap.add_argument("--min-seq-id", type=float, default=0.8)
    ap.add_argument("--cov", type=float, default=0.8, help="minimum alignment coverage of both sequences")
    ap.add_argument("--min-len", type=int, default=10, help="RNA chains shorter than this are ignored for clustering")
    ap.add_argument("--frac", type=float, default=0.05, help="fraction of samples in validation and in test")
    ap.add_argument("--max-group", type=float, default=0.01,
                    help="groups with more than this fraction of all samples always go to train")
    ap.add_argument("--balance", default="PSU", choices=MODS,
                    help="validation and test each get ~frac of the distinct residues of this modification")
    ap.add_argument("--balance-tol", type=float, default=1.25,
                    help="a split may exceed its share of the balanced modification by this factor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    a = ap.parse_args()

    # --- samples from the split files
    man = pd.read_csv(a.manifest, keep_default_na=False, low_memory=False)
    for m in MODS:
        man[f"n_{m}"] = pd.to_numeric(man[f"n_{m}"], errors="coerce").fillna(0).astype(int)
    split_ids = set(man.pdb_id[man.status != "not split: builder reads original"])
    chain_seq = {(r.pdb_id, r.chain): r.sequence for r in man.itertuples() if r.status in ("written", "cropped")}

    # distinct modified residues of each cropped chain (crops overlap, so their own counts double-count)
    crop_chain_mods = {f"{r.pdb_id}/{r.chain}": {m: getattr(r, f"n_{m}") for m in MODS}
                       for r in man[man.status == "cropped"].itertuples()}

    samples = []   # dicts: sample, kind, pdb_id, seqs, mods
    for r in man[man.status.isin(["written", "written crop"])].itertuples():
        crop = r.status == "written crop"
        samples.append({"sample": r.file.split(".")[0].upper(), "kind": "crop" if crop else "chain",
                        "pdb_id": r.pdb_id, "seqs": [to_nt(chain_seq[(r.pdb_id, r.chain)])],
                        "crop_chain": f"{r.pdb_id}/{r.chain}" if crop else "",
                        **{f"n_{m}": getattr(r, f"n_{m}") for m in MODS}})
    print(f"split files: {len(samples)}")

    # --- small entries: in the current h5, passing the training filters, not replaced by split files
    with h5py.File(a.h5, "r") as hf:
        keys = np.array(hf["metadata/keys"]).astype("U")
        rsizes = np.array(hf["metadata/rsizes"])
    ok = [k.split("/")[1] for k, (n_atoms, n_res) in zip(keys, rsizes)
          if int(k.split("/")[2]) <= TRAIN_MAX_BA and n_atoms <= TRAIN_MAX_SIZE and n_res >= TRAIN_MIN_RES]
    small_ids = sorted(set(ok) - split_ids)
    jobs = [(i, f"{ROOT}/pdb_download/pdb_files/{i}.pdb.gz", a.min_len) for i in small_ids]
    with Pool(a.workers) as pool:
        out = pool.map(small_entry_chains, jobs, chunksize=16)
    errors = [(i, e) for i, _, _, e in out if e]
    for pdb_id, seqs, mods, _ in out:
        samples.append({"sample": pdb_id, "kind": "entry", "pdb_id": pdb_id, "seqs": seqs, "crop_chain": "",
                        **{f"n_{m}": mods[m] for m in MODS}})
    print(f"small entries passing the training filters: {len(small_ids)} (read errors: {len(errors)})")

    # --- cluster unique sequences
    uniq = sorted({s for smp in samples for s in smp["seqs"]})
    long_seqs = [s for s in uniq if len(s) >= MIN_KMER_LEN]
    rep = run_mmseqs(long_seqs, os.path.join(a.out_dir, "mmseqs_splits"), a.min_seq_id, a.cov, a.workers)
    for s in uniq:
        rep.setdefault(s, s)          # short or unclustered sequences: their own cluster
    print(f"unique sequences: {len(uniq)} -> clusters: {len(set(rep.values()))}")

    # --- join clusters that share a sample, one group per connected set
    uf = UnionFind()
    for smp in samples:
        reps = [rep[s] for s in smp["seqs"]] or [f"nochain:{smp['sample']}"]
        for r in reps:
            uf.find(r)
        for r in reps[1:]:
            uf.union(reps[0], r)
        smp["group_key"] = reps[0]
    groups = {}
    for smp in samples:
        smp["group"] = groups.setdefault(uf.find(smp["group_key"]), len(groups))
    df = pd.DataFrame(samples)
    sizes = df.group.value_counts()
    print(f"groups: {len(sizes)} | largest: {sizes.iloc[:5].tolist()} samples")

    # distinct modified residues per group: whole chains and entries as they are, cropped chains once
    def distinct_mods(d):
        out = {m: int(d[f"n_{m}"][d.kind != "crop"].sum()) for m in MODS}
        for cc in d.crop_chain[d.kind == "crop"].unique():
            for m in MODS:
                out[m] += crop_chain_mods[cc][m]
        return out
    gmods = {g: distinct_mods(d) for g, d in df.groupby("group")}
    total_bal = sum(v[a.balance] for v in gmods.values())

    # --- 90/5/5 by group, validation and test each with ~frac of the samples and of the distinct --balance residues
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(sizes.index.values)
    n_target, b_target, cap = a.frac * len(df), a.frac * total_bal, a.max_group * len(df)
    split_of = {g: "train" for g in order}
    filled = {s: {"n": 0, "b": 0} for s in ("validation", "test")}

    # pass 1: groups with the balanced modification, into the split that has the least of it so far
    for g in order:
        n, b = sizes[g], gmods[g][a.balance]
        if b == 0 or n > cap:
            continue
        for s in sorted(filled, key=lambda s: filled[s]["b"]):
            f = filled[s]
            if f["b"] < b_target and f["b"] + b <= a.balance_tol * b_target and f["n"] + n <= n_target:
                split_of[g] = s
                f["n"] += n
                f["b"] += b
                break
    # pass 2: groups without it, until each split holds frac of the samples
    for g in order:
        n = sizes[g]
        if gmods[g][a.balance] > 0 or n > cap:
            continue
        for s in ("validation", "test"):
            if filled[s]["n"] + n <= n_target:
                split_of[g] = s
                filled[s]["n"] += n
                break
    df["split"] = df.group.map(split_of)

    os.makedirs(a.out_dir, exist_ok=True)
    for s in ("train", "validation", "test"):
        ids = sorted(df["sample"][df.split == s])
        with open(os.path.join(a.out_dir, f"splits_mods_{s}.txt"), "w") as f:
            f.write("\n".join(ids) + "\n")
    df["seq_lengths"] = df.seqs.map(lambda x: ";".join(str(len(s)) for s in x))
    df.drop(columns=["seqs", "group_key", "crop_chain"]).to_csv(os.path.join(a.out_dir, "splits_mods_samples.csv"), index=False)

    # --- summary
    summ = df.groupby("split").agg(samples=("sample", "size"), groups=("group", "nunique"),
                                   entries=("kind", lambda k: (k == "entry").sum()),
                                   chain_files=("kind", lambda k: (k == "chain").sum()),
                                   crops=("kind", lambda k: (k == "crop").sum()),
                                   **{f"{m} occurrences": (f"n_{m}", "sum") for m in MODS})
    summ["fraction"] = (summ.samples / len(df)).round(3)
    for m in MODS:
        summ[f"{m} distinct"] = [sum(gmods[g][m] for g, s in split_of.items() if s == sp) for sp in summ.index]
    summ[f"{a.balance} fraction"] = (summ[f"{a.balance} distinct"] / summ[f"{a.balance} distinct"].sum()).round(3)
    with pd.option_context("display.width", 250):
        print("\n" + summ.loc[["train", "validation", "test"]].to_string())
    print(f"\nwritten: {a.out_dir}/splits_mods_{{train,validation,test}}.txt and splits_mods_samples.csv")


if __name__ == "__main__":
    main()

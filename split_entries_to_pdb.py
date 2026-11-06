"""Split large PDB entries into one .pdb.gz file per RNA chain (with its surroundings), cropping chains that are too big.

Entries the builder cannot read (.cif only, gzipped file >= 1 MB, or >= 16384 atoms in the first model)
are processed here; small entries are left alone and the builder keeps reading their original file.
The builder keeps every chain of an entry (proteins, ions, ligands), so the files written here keep the
surroundings of the RNA too, within the training size limit (--budget, 8192 atoms).

First model, hydrogens removed, first altloc, water removed. For every RNA (or DNA/RNA hybrid) polymer chain:
  * RNA chain <= budget atoms : the whole chain + context      -> <PDBID>_<CHAIN>.pdb.gz
      context = residues of all other chains (proteins, other RNA, ions, ligands) within --context-radius
      of the chain, nearest first, while the file stays within the budget
  * RNA chain >  budget, with PSU/5MC/OMG : spherical crops of the entry -> <PDBID>_<CHAIN>-crop<k>.pdb.gz
      each crop = all residues (any chain) closest to one modified residue, until the budget is used;
      the distance to the last residue taken is the crop radius R; crops are made until every
      modified residue of these chains lies within --cover-frac * R of some crop centre
  * RNA chain >  budget, without modification : skipped
  * shorter than --min-res, or same sequence as a chain already handled in the entry : skipped
In every file the RNA chain of interest (for a crop: the chain of the centre residue) is chain A,
the other chains are B, C, ... Distances are measured between residue centres (C4', CA, or centroid).
--context-radius 0 writes the RNA alone (crops then only contain the centre's chain).

<out>/manifest.csv has one row per RNA chain and one row per crop.

Usage:
    python split_entries_to_pdb.py                      # all downloaded entries
    python split_entries_to_pdb.py --ids 4V5B 1VY4 --out /tmp/test
"""
import argparse
import csv
import glob
import gzip
import os
from collections import Counter
from multiprocessing import Pool

import gemmi
import numpy as np
from scipy.spatial import cKDTree

DL_DIR = "/scratch/izar/maazoun/pdb_download"
MODS = ["PSU", "5MC", "OMG"]
STD_RNA = ["A", "G", "C", "U"]

# filters of build_structures_dataset.py: entries failing one of them are split
BUILDER_MAX_FILE_SIZE = 1e6
BUILDER_MAX_NUM_ATOMS = 1024 * 8 * 2
# training filter of config.py ('max_size'): atoms per file
TRAIN_MAX_SIZE = 1024 * 8

# chain ids for the PDB format: A = RNA chain of interest, then context chains
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

FIELDS = ["pdb_id", "source", "chain", "polymer_type", "crop", "file", "status",
          "n_res", "n_atoms", "n_file_atoms", "n_context_chains", "radius",
          "n_PSU", "n_5MC", "n_OMG", "n_crops", "other_mods", "renumbered", "sequence"]


def entry_paths():
    """One file per PDB id, .pdb.gz preferred (same choice as modification_survey.ipynb)."""
    pdb = {os.path.basename(p).split(".")[0].upper(): p for p in glob.glob(f"{DL_DIR}/pdb_files/*.pdb.gz")}
    cif = {os.path.basename(p).split(".")[0].upper(): p for p in glob.glob(f"{DL_DIR}/cif_files/*.cif.gz")}
    return {**cif, **pdb}


def residue_centers(residues):
    """C4' (nucleotides), CA (amino acids), else atom centroid of each residue."""
    xyz = []
    for r in residues:
        a = r.find_atom("C4'", "*")
        if a is None:
            a = r.find_atom("CA", "*")
        if a is not None:
            xyz.append([a.pos.x, a.pos.y, a.pos.z])
        else:
            xyz.append(np.mean([[at.pos.x, at.pos.y, at.pos.z] for at in r], axis=0))
    return np.array(xyz).reshape(-1, 3)


def res_key(chain_name, r):
    return (chain_name, r.subchain, r.seqid.num, r.seqid.icode, r.name)


class EntryPool:
    """All non-water residues of a model, flattened, with chain, centre, atom count and RNA-polymer membership."""

    def __init__(self, model, rna_keys):
        self.res, self.chain, self.pos = [], [], []
        self.n_dropped = 0
        for ch in model:
            for i, r in enumerate(ch):
                if r.is_water():
                    continue
                if len(r.name) > 3:      # does not fit the PDB format (long CCD codes)
                    self.n_dropped += 1
                    continue
                self.res.append(r)
                self.chain.append(ch.name)
                self.pos.append(i)
        self.chain = np.array(self.chain)
        self.pos = np.array(self.pos)
        self.xyz = residue_centers(self.res)
        self.size = np.array([len(r) for r in self.res])
        self.names = np.array([r.name for r in self.res])
        keys = [res_key(c, r) for c, r in zip(self.chain, self.res)]
        self.is_rna = np.array([k in rna_keys for k in keys])
        self.rna_chain = np.array([rna_keys.get(k, "") for k in keys])   # RNA chain the residue belongs to


def nearest_first(pool, cand, d, budget):
    """Candidates sorted by distance d, kept while the cumulated atom count stays within budget."""
    order = np.argsort(d, kind="stable")
    n = int(np.searchsorted(np.cumsum(pool.size[cand[order]]), budget, side="right"))
    return cand[order[:n]], d[order[:n]]


def spatial_crops(pool, targets, budget, cover_frac=0.75, candidates=None):
    """Spherical crops around the target residues (pool indices).

    candidates(t) -> pool indices a crop centred on t may take (default: the whole entry).
    Returns a list of (centre, selected pool indices, radius).
    """
    crops, covered = [], set()
    all_idx = np.arange(len(pool.res))
    for t in targets:
        if t in covered:
            continue
        cand = all_idx if candidates is None else candidates(t)
        d = np.linalg.norm(pool.xyz[cand] - pool.xyz[t], axis=1)
        sel, dsel = nearest_first(pool, cand, d, budget)
        radius = float(dsel[-1])
        crops.append((t, sel, radius))
        in_sel = set(sel.tolist())
        covered.add(t)
        covered.update(i for i in targets if i in in_sel
                       and np.linalg.norm(pool.xyz[i] - pool.xyz[t]) <= cover_frac * radius)
    return crops


def build_structure(src, pool, sel, main_chain, d=None):
    """Structure with the selected pool residues: main_chain as chain A, other chains B, C, ... (nearest first).

    Residues keep their original order inside each chain. Chains beyond the 62 available ids are dropped.
    Returns (structure, renumbered, n_context_chains, kept pool indices).
    """
    sel = np.asarray(sel)
    order = np.argsort(d, kind="stable") if d is not None else np.arange(len(sel))
    # chain A = the RNA polymer only; ions / ligands of the same author chain get their own chain
    group = np.where((pool.chain[sel] == main_chain) & (pool.rna_chain[sel] != main_chain),
                     main_chain + ":het", pool.chain[sel])
    chains = [main_chain] + [c for c in dict.fromkeys(group[order]) if c != main_chain]
    chains = chains[:len(CHAIN_IDS)]

    model = gemmi.Model(1)
    renumber, kept = False, []
    for cid, cname in zip(CHAIN_IDS, chains):
        idx = sel[group == cname]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(pool.pos[idx])]
        ch = gemmi.Chain(cid)
        for i in idx:
            ch.add_residue(pool.res[i])
        # PDB residue numbers have 4 columns: renumber 1..n if they would not fit
        if any(not (-999 <= r.seqid.num <= 9999) for r in ch):
            renumber = True
            for k, r in enumerate(ch):
                r.seqid = gemmi.SeqId(k + 1, " ")
        model.add_chain(ch)
        kept.extend(idx.tolist())

    st = gemmi.Structure()
    st.name = src.name
    st.cell = src.cell
    st.spacegroup_hm = src.spacegroup_hm
    st.add_model(model)
    return st, renumber, len(model) - 1, np.array(kept)


def write_pdb(st, fn):
    with gzip.open(fn, "wt") as f:
        f.write(st.make_pdb_string(gemmi.PdbWriteOptions(minimal=True)))


def file_stats(pool, idx):
    """RNA residues / RNA atoms / all atoms / modified residue counts of the residues in idx."""
    idx = np.asarray(idx)
    rna = idx[pool.is_rna[idx]]
    c = Counter(pool.names[rna])
    return {"n_res": len(rna), "n_atoms": int(pool.size[rna].sum()), "n_file_atoms": int(pool.size[idx].sum()),
            **{f"n_{m}": c[m] for m in MODS}}


def process_entry(args):
    pdb_id, path, out_dir, min_res, budget, cover_frac, context_radius, dedupe = args
    source = "cif" if path.endswith(".cif.gz") else "pdb"
    rows = []
    try:
        st = gemmi.read_structure(path)
        st.setup_entities()

        # does the builder already read this entry as it is?
        too_big = (source == "cif") or (os.path.getsize(path) >= BUILDER_MAX_FILE_SIZE) \
                  or (st[0].count_atom_sites() >= BUILDER_MAX_NUM_ATOMS)
        if not too_big:
            return [{"pdb_id": pdb_id, "source": source, "status": "not split: builder reads original"}]

        # keep first model only, no hydrogens, first altloc
        while len(st) > 1:
            del st[1]
        st.remove_hydrogens()
        st.remove_alternative_conformations()
        model = st[0]

        # RNA polymer chains
        rna_chains, rna_keys = [], {}
        for ch in model:
            poly = ch.get_polymer()
            ptype = poly.check_polymer_type()
            # RNA chains, and DNA/RNA hybrid chains (their DNA residues are kept as they are)
            if ptype not in (gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid):
                continue
            rna_chains.append((ch.name, ptype, list(poly)))
            for r in poly:
                rna_keys[res_key(ch.name, r)] = ch.name

        if not rna_chains:
            return [{"pdb_id": pdb_id, "source": source, "status": "no RNA chain"}]

        pool = EntryPool(model, rna_keys)
        tree_all = cKDTree(pool.xyz)

        seen_seq, used_tags = set(), set()
        crop_chains = {}          # chain name -> (manifest row, tag)
        for cname, ptype, residues in rna_chains:
            names = [r.name for r in residues]
            n_atoms = sum(len(r) for r in residues)
            base = {"pdb_id": pdb_id, "source": source, "chain": cname,
                    "polymer_type": "hybrid" if ptype == gemmi.PolymerType.DnaRnaHybrid else "rna"}
            row = {**base, "n_res": len(residues), "n_atoms": n_atoms,
                   **{f"n_{m}": names.count(m) for m in MODS},
                   "other_mods": ";".join(f"{n}:{c}" for n, c in Counter(names).items() if n not in STD_RNA + MODS),
                   "sequence": gemmi.one_letter_code(names)}
            rows.append(row)

            if len(residues) < min_res:
                row["status"] = "skipped: too short"
                continue
            if n_atoms > budget and not any(n in MODS for n in names):
                row["status"] = "skipped: too many atoms, no modification"
                continue
            if dedupe and tuple(names) in seen_seq:
                row["status"] = "skipped: duplicate sequence in entry"
                continue
            seen_seq.add(tuple(names))

            # the builder upper-cases ids: chains "a" and "A" would collide
            tag = cname
            if tag.upper() in used_tags:
                tag = f"{cname}-{len(used_tags)}"
            used_tags.add(tag.upper())

            if n_atoms > budget:
                crop_chains[cname] = (row, tag)
                continue

            # whole chain + context: other residues within context_radius, nearest first, within budget
            main = np.where((pool.chain == cname) & pool.is_rna)[0]
            sel, d = main, np.zeros(len(main))
            if context_radius > 0:
                near = np.unique(np.concatenate(
                    tree_all.query_ball_point(pool.xyz[main], r=context_radius))).astype(int)
                ctx = np.setdiff1d(near, main)
                if len(ctx):
                    d_ctx, _ = cKDTree(pool.xyz[main]).query(pool.xyz[ctx])
                    ctx, d_ctx = nearest_first(pool, ctx, d_ctx, budget - int(pool.size[main].sum()))
                    sel, d = np.concatenate([main, ctx]), np.concatenate([d, d_ctx])
            out, renumber, n_ctx, kept = build_structure(st, pool, sel, cname, d)
            fn = f"{pdb_id}_{tag}.pdb.gz"
            write_pdb(out, os.path.join(out_dir, fn))
            row.update(status="written", file=fn, renumbered=int(renumber), n_context_chains=n_ctx,
                       n_file_atoms=int(pool.size[kept].sum()))

        # crops of the chains that are too big, around their modified residues
        if crop_chains:
            targets = [i for i in range(len(pool.res))
                       if pool.rna_chain[i] in crop_chains and pool.names[i] in MODS]
            if context_radius > 0:
                candidates = None                       # whole entry
            else:
                candidates = lambda t: np.where((pool.chain == pool.rna_chain[t]) & pool.is_rna)[0]
            crops = spatial_crops(pool, targets, budget, cover_frac, candidates)

            k_of = Counter()
            for centre, sel, radius in crops:
                cname = pool.rna_chain[centre]
                row, tag = crop_chains[cname]
                k_of[cname] += 1
                d = np.linalg.norm(pool.xyz[sel] - pool.xyz[centre], axis=1)
                out, renumber, n_ctx, kept = build_structure(st, pool, sel, cname, d)
                fn = f"{pdb_id}_{tag}-crop{k_of[cname]}.pdb.gz"
                write_pdb(out, os.path.join(out_dir, fn))
                rows.append({"pdb_id": pdb_id, "source": source, "chain": cname,
                             "polymer_type": row["polymer_type"], "crop": k_of[cname], "file": fn,
                             "status": "written crop", **file_stats(pool, kept), "n_context_chains": n_ctx,
                             "radius": round(radius, 1), "renumbered": int(renumber)})
            for cname, (row, tag) in crop_chains.items():
                row.update(status="cropped", n_crops=k_of[cname])

    except Exception as e:
        rows.append({"pdb_id": pdb_id, "source": source, "status": f"error: {e!r}"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{DL_DIR}/pdb_split")
    ap.add_argument("--ids", nargs="*", help="only these PDB ids (for testing)")
    ap.add_argument("--min-res", type=int, default=10)
    ap.add_argument("--budget", type=int, default=TRAIN_MAX_SIZE,
                    help="max heavy atoms per file (default: training max_size); larger RNA chains are cropped")
    ap.add_argument("--cover-frac", type=float, default=0.75,
                    help="a modification is covered when within this fraction of a crop's radius (lower = more crops)")
    ap.add_argument("--context-radius", type=float, default=15.0,
                    help="Å around a whole RNA chain from which other chains are added as context (0 = RNA only)")
    ap.add_argument("--no-dedupe", action="store_true", help="also write chains identical to one already written")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    paths = entry_paths()
    if a.ids:
        paths = {k: v for k, v in paths.items() if k in {i.upper() for i in a.ids}}

    # big files first so the pool does not finish on a ribosome
    items = sorted(paths.items(), key=lambda kv: -os.path.getsize(kv[1]))
    jobs = [(k, p, a.out, a.min_res, a.budget, a.cover_frac, a.context_radius, not a.no_dedupe) for k, p in items]

    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_entry, jobs), 1):
            rows.extend(r)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} entries", flush=True)

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["pdb_id"], r.get("chain", ""), r.get("crop", 0))))

    # summary
    status = Counter(r["status"] if not r["status"].startswith("error") else "error" for r in rows)
    whole = [r for r in rows if r["status"] == "written"]
    cropped = [r for r in rows if r["status"] == "cropped"]
    crops = [r for r in rows if r["status"] == "written crop"]
    print(f"\nentries: {len(jobs)}")
    for s, n in status.most_common():
        print(f"  {s:<45} {n}")
    print(f"files written: {len(whole)} whole chains + {len(crops)} crops of {len(cropped)} chains, in {a.out}")
    for m in MODS:
        print(f"  {m}: {sum(r[f'n_{m}'] for r in whole)} in whole chains, "
              f"{sum(r[f'n_{m}'] for r in cropped)} in cropped chains "
              f"({sum(r[f'n_{m}'] for r in crops)} counting overlaps between crops)")
    if crops:
        print(f"crops: {np.mean([r['n_res'] for r in crops]):.0f} RNA residues and "
              f"{np.mean([r['n_file_atoms'] for r in crops]):.0f} atoms on average, "
              f"{np.mean([r['n_context_chains'] for r in crops]):.1f} other chains")
    if whole:
        print(f"whole chains: {np.mean([r['n_context_chains'] for r in whole]):.1f} context chains, "
              f"{np.mean([r['n_file_atoms'] - r['n_atoms'] for r in whole]):.0f} context atoms on average")


if __name__ == "__main__":
    main()

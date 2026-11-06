"""Split large PDB entries into one .pdb.gz file per RNA chain, cropping chains that are still too big.

Entries the builder cannot read (.cif only, gzipped file >= 1 MB, or >= 16384 atoms in the first model)
are processed here; small entries are left alone and the builder keeps reading their original file.

For every RNA (or DNA/RNA hybrid) polymer chain of the first model, hydrogens removed, first altloc:
  * <= --crop-atoms heavy atoms : written whole        -> <PDBID>_<CHAIN>.pdb.gz
  * >  --crop-atoms, with PSU/5MC/OMG : spatial crops  -> <PDBID>_<CHAIN>-crop<k>.pdb.gz
      each crop = residues closest (C4') to one modified residue, added until the atom budget is used;
      crops are made until every modified residue lies within --cover-frac (0.75) of some crop's radius
  * >  --crop-atoms, without modification : skipped
  * shorter than --min-res, or same sequence as a chain already written for the entry : skipped
Chains are written as chain "A"; ions, ligands, water and proteins are not written.

<out>/manifest.csv has one row per RNA chain, and one row per crop.

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

DL_DIR = "/scratch/izar/maazoun/pdb_download"
MODS = ["PSU", "5MC", "OMG"]
STD_RNA = ["A", "G", "C", "U"]

# filters of build_structures_dataset.py: entries failing one of them are split
BUILDER_MAX_FILE_SIZE = 1e6
BUILDER_MAX_NUM_ATOMS = 1024 * 8 * 2
# training filter of config.py ('max_size'): larger chains are cropped
TRAIN_MAX_SIZE = 1024 * 8

FIELDS = ["pdb_id", "source", "chain", "polymer_type", "crop", "file", "status", "n_res", "n_atoms",
          "n_PSU", "n_5MC", "n_OMG", "n_crops", "other_mods", "renumbered", "sequence"]


def entry_paths():
    """One file per PDB id, .pdb.gz preferred (same choice as modification_survey.ipynb)."""
    pdb = {os.path.basename(p).split(".")[0].upper(): p for p in glob.glob(f"{DL_DIR}/pdb_files/*.pdb.gz")}
    cif = {os.path.basename(p).split(".")[0].upper(): p for p in glob.glob(f"{DL_DIR}/cif_files/*.cif.gz")}
    return {**cif, **pdb}


def chain_structure(src, residues):
    """New single-model structure holding `residues` as chain A."""
    ch = gemmi.Chain("A")
    for r in residues:
        ch.add_residue(r)

    # PDB residue numbers have 4 columns: renumber 1..n if they would not fit
    renumber = any(not (-999 <= r.seqid.num <= 9999) for r in ch)
    if renumber:
        for i, r in enumerate(ch):
            r.seqid = gemmi.SeqId(i + 1, " ")

    model = gemmi.Model(1)
    model.add_chain(ch)
    st = gemmi.Structure()
    st.name = src.name
    st.cell = src.cell
    st.spacegroup_hm = src.spacegroup_hm
    st.add_model(model)
    return st, renumber


def write_pdb(st, fn):
    with gzip.open(fn, "wt") as f:
        f.write(st.make_pdb_string(gemmi.PdbWriteOptions(minimal=True)))


def residue_centers(residues):
    """C4' position of each residue, atom centroid if C4' is missing."""
    xyz = []
    for r in residues:
        a = r.find_atom("C4'", "*")
        if a is not None:
            xyz.append([a.pos.x, a.pos.y, a.pos.z])
        else:
            xyz.append(np.mean([[at.pos.x, at.pos.y, at.pos.z] for at in r], axis=0))
    return np.array(xyz)


def spatial_crops(residues, budget, cover_frac=0.75, return_centers=False):
    """Index sets (sorted, chain order) of crops of at most `budget` atoms around modified residues.

    A crop around residue t takes residues by increasing distance from t while the atom count stays
    within budget; the distance to the last residue taken is the crop radius. A modified residue counts
    as covered once it lies within cover_frac * radius of a crop's centre, so every modification gets a
    crop where it is not at the edge. Uncovered modifications become the centre of a new crop.
    """
    xyz = residue_centers(residues)
    sizes = np.array([len(r) for r in residues])
    targets = [i for i, r in enumerate(residues) if r.name in MODS]

    crops, centers, covered = [], [], set()
    for t in targets:
        if t in covered:
            continue
        centers.append(t)
        d = np.linalg.norm(xyz - xyz[t], axis=1)
        order = np.argsort(d, kind="stable")
        n = int(np.searchsorted(np.cumsum(sizes[order]), budget, side="right"))
        sel = order[:n]
        r_cover = d[sel[-1]] * cover_frac
        crops.append(np.sort(sel))
        covered.add(t)
        covered.update(i for i in targets if d[i] <= r_cover)
    return (crops, centers) if return_centers else crops


def mod_counts(names):
    c = Counter(names)
    return {f"n_{m}": c[m] for m in MODS}


def process_entry(args):
    pdb_id, path, out_dir, min_res, crop_atoms, cover_frac, dedupe = args
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

        seen_seq = set()
        used_tags = set()
        for ch in st[0]:
            poly = ch.get_polymer()
            ptype = poly.check_polymer_type()
            # RNA chains, and DNA/RNA hybrid chains (their DNA residues are kept as they are)
            if ptype not in (gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid):
                continue
            residues = list(poly)
            names = [r.name for r in residues]
            n_atoms = sum(len(r) for r in residues)
            base = {"pdb_id": pdb_id, "source": source, "chain": ch.name,
                    "polymer_type": "hybrid" if ptype == gemmi.PolymerType.DnaRnaHybrid else "rna"}
            row = {**base, "n_res": len(residues), "n_atoms": n_atoms, **mod_counts(names),
                   "other_mods": ";".join(f"{n}:{c}" for n, c in Counter(names).items() if n not in STD_RNA + MODS),
                   "sequence": gemmi.one_letter_code(names)}
            rows.append(row)
            has_mod = any(n in MODS for n in names)

            if len(residues) < min_res:
                row["status"] = "skipped: too short"
                continue
            if n_atoms > crop_atoms and not has_mod:
                row["status"] = "skipped: too many atoms, no modification"
                continue
            if dedupe and tuple(names) in seen_seq:
                row["status"] = "skipped: duplicate sequence in entry"
                continue
            seen_seq.add(tuple(names))

            # the builder upper-cases ids: chains "a" and "A" would collide
            tag = ch.name
            if tag.upper() in used_tags:
                tag = f"{ch.name}-{len(used_tags)}"
            used_tags.add(tag.upper())

            if n_atoms <= crop_atoms:
                out, renumber = chain_structure(st, residues)
                fn = f"{pdb_id}_{tag}.pdb.gz"
                write_pdb(out, os.path.join(out_dir, fn))
                row.update(status="written", file=fn, renumbered=int(renumber))
                continue

            # chain too large: crops around the modified residues
            crops = spatial_crops(residues, crop_atoms, cover_frac)
            row.update(status="cropped", n_crops=len(crops))
            for k, sel in enumerate(crops, 1):
                res_k = [residues[i] for i in sel]
                out, renumber = chain_structure(st, res_k)
                fn = f"{pdb_id}_{tag}-crop{k}.pdb.gz"
                write_pdb(out, os.path.join(out_dir, fn))
                rows.append({**base, "crop": k, "file": fn, "status": "written crop",
                             "n_res": len(res_k), "n_atoms": sum(len(r) for r in res_k),
                             **mod_counts([r.name for r in res_k]), "renumbered": int(renumber)})

        if not rows:
            rows.append({"pdb_id": pdb_id, "source": source, "status": "no RNA chain"})
    except Exception as e:
        rows.append({"pdb_id": pdb_id, "source": source, "status": f"error: {e!r}"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{DL_DIR}/pdb_split")
    ap.add_argument("--ids", nargs="*", help="only these PDB ids (for testing)")
    ap.add_argument("--min-res", type=int, default=10)
    ap.add_argument("--crop-atoms", type=int, default=TRAIN_MAX_SIZE,
                    help="chains with more heavy atoms are cropped around modifications (default: training max_size)")
    ap.add_argument("--cover-frac", type=float, default=0.75,
                    help="a modification is covered when within this fraction of a crop's radius (lower = more crops)")
    ap.add_argument("--no-dedupe", action="store_true", help="also write chains identical to one already written")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    paths = entry_paths()
    if a.ids:
        paths = {k: v for k, v in paths.items() if k in {i.upper() for i in a.ids}}

    # big files first so the pool does not finish on a ribosome
    items = sorted(paths.items(), key=lambda kv: -os.path.getsize(kv[1]))
    jobs = [(k, p, a.out, a.min_res, a.crop_atoms, a.cover_frac, not a.no_dedupe) for k, p in items]

    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_entry, jobs), 1):
            rows.extend(r)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} entries")

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


if __name__ == "__main__":
    main()

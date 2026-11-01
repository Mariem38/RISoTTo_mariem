"""Count PSU, 5MC and OMG residues in each split of the RISoTTo h5, using main.py's selection rules."""
import sys
import h5py
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."
h5 = f"{root}/datasets/pdb_bprna_structures_nuc_v8_mods.h5"
MODS = ['PSU', '5MC', 'OMG']
RNA = ['G', 'A', 'U', 'C'] + MODS

with h5py.File(h5, 'r') as hf:
    keys = np.array(hf['metadata/keys']).astype('U')
    sizes = np.array(hf['metadata/rsizes'])
    resnames = np.array(hf['metadata/std_resnames']).astype('U')
    rna_ids = np.where(np.isin(resnames, RNA))[0]
    mod_ids = {m: int(np.where(resnames == m)[0][0]) for m in MODS}

    # per-entry counts of RNA residues and modified residues
    n_rna = np.zeros(len(keys), dtype=int)
    n_mod = {m: np.zeros(len(keys), dtype=int) for m in MODS}
    for i, k in enumerate(keys):
        g = hf['data/structures/' + k]
        qr = np.array(g['qr'])  # (atom, resname_idx)
        mr = np.array(g['Mr'])  # (atom, residue_idx)
        atom_res = np.empty(g.attrs['Mr_shape'][0], dtype=int)
        atom_res[mr[:, 0]] = mr[:, 1]
        atom_name = np.full(g.attrs['qr_shape'][0], -1)
        atom_name[qr[:, 0]] = qr[:, 1]
        # one resname per residue (first atom of each residue)
        _, first = np.unique(atom_res, return_index=True)
        res_name = atom_name[first]
        n_rna[i] = np.isin(res_name, rna_ids).sum()
        for m in MODS:
            n_mod[m][i] = (res_name == mod_ids[m]).sum()

pdb = np.array([k.split('/')[1] for k in keys])
# same filters as config.py / main.py
filt = (np.array([int(k.split('/')[2]) for k in keys]) <= 1) \
     & (sizes[:, 0] <= 1024 * 8) & (sizes[:, 1] >= 10)


def report(name, m):
    e, r = m.sum(), n_rna[m].sum()
    print(f"{name:<22} entries={e:5d}  RNA_res={r:8d}")
    for mod in MODS:
        p = n_mod[mod][m].sum()
        ep = (m & (n_mod[mod] > 0)).sum()
        print(f"  {mod:<4} entries={ep:4d} ({ep / max(e, 1):6.2%})  res={p:6d} ({p / max(r, 1):.4%})")


report("whole h5", np.ones(len(keys), bool))
splits = {}
for s in ['train', 'validation', 'test']:
    ids = np.genfromtxt(f"{root}/datasets/grnade_das_{s}_set.txt", dtype='U')
    ids = np.unique([x.split('_')[0] for x in ids])
    splits[s] = ids
    m = np.isin(pdb, ids)
    print(f"\n[{s}] {len(ids)} PDB ids in split file, {m.sum()} found in h5")
    report(f"{s} (in h5)", m)
    report(f"{s} (after filters)", m & filt)
    for mod in MODS:
        mod_pdbs = sorted(set(pdb[m & filt & (n_mod[mod] > 0)]))
        print(f"  PDBs with {mod}: {mod_pdbs if len(mod_pdbs) <= 30 else str(len(mod_pdbs)) + ' entries'}")

print("\nPDB-id overlap: train&val", len(np.intersect1d(splits['train'], splits['validation'])),
      " train&test", len(np.intersect1d(splits['train'], splits['test'])),
      " val&test", len(np.intersect1d(splits['validation'], splits['test'])))

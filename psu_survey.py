

import os
import json
import time
import gzip
import argparse
import requests
import pandas as pd
import gemmi

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


PSU_IDS = ["PSU"]


# ---------------------------------------------------------------- 1. search

def find_entities(comp_ids):
    """RNA polymer entities containing any of the given chemical components."""
    nodes = [{
        "type": "terminal", "service": "text",
        "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers.chem_comp_monomers",
            "operator": "in", "value": comp_ids},
    }, {
        "type": "terminal", "service": "text",
        "parameters": {
            "attribute": "entity_poly.rcsb_entity_polymer_type",
            "operator": "exact_match", "value": "RNA"},
    }]
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "polymer_entity",
        "request_options": {"paginate": {"start": 0, "rows": 10000}},
    }
    r = requests.post(SEARCH_URL, json=query, timeout=60)
    r.raise_for_status()
    data = r.json()
    return [hit["identifier"] for hit in data.get("result_set", [])]


# -------------------------------------------------------------- 2. metadata

ENTITY_QUERY = """
query($ids: [String!]!) {
  polymer_entities(entity_ids: $ids) {
    rcsb_id
    entity_poly { rcsb_entity_polymer_type rcsb_sample_sequence_length }
    rcsb_polymer_entity { pdbx_description }
    rcsb_polymer_entity_container_identifiers { auth_asym_ids }
    entry {
      rcsb_id
      rcsb_entry_info {
        resolution_combined
        experimental_method
        polymer_entity_count_protein
        polymer_entity_count_DNA
        deposited_atom_count
      }
    }
  }
}
"""


def fetch_metadata(entity_ids, batch=50):
    rows = []
    for i in range(0, len(entity_ids), batch):
        chunk = entity_ids[i:i + batch]
        r = requests.post(GRAPHQL_URL,
                          json={"query": ENTITY_QUERY, "variables": {"ids": chunk}},
                          timeout=120)
        r.raise_for_status()
        for e in r.json()["data"]["polymer_entities"]:
            if e is None:
                continue
            info = e["entry"]["rcsb_entry_info"]
            res = info.get("resolution_combined") or [None]
            for chain in e["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]:
                rows.append({
                    "pdb_id": e["entry"]["rcsb_id"],
                    "entity_id": e["rcsb_id"],
                    "chain": chain,
                    "description": e["rcsb_polymer_entity"]["pdbx_description"],
                    "n_res_entity": e["entity_poly"]["rcsb_sample_sequence_length"],
                    "method": info.get("experimental_method"),
                    "resolution": res[0],
                    "n_protein_entities": info.get("polymer_entity_count_protein"),
                    "n_dna_entities": info.get("polymer_entity_count_DNA"),
                    "deposited_atoms": info.get("deposited_atom_count"),
                })
        time.sleep(0.2)
    return pd.DataFrame(rows)


# ------------------------------------------------- 3. per-chain PSU counting

def download_cif(pdb_id, cache="cif_cache"):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{pdb_id}.cif.gz")
    if not os.path.exists(path):
        r = requests.get(CIF_URL.format(pdb_id=pdb_id), timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        time.sleep(0.1)
    return path


def count_psu(pdb_id, comp_ids, cache="cif_cache"):
    """Per auth chain: residue count, PSU count, PSU sequence positions."""
    path = download_cif(pdb_id, cache)
    with gzip.open(path, "rt") as f:
        st = gemmi.read_structure_from_string(f.read())
    st.setup_entities()
    out = {}
    model = st[0]
    for chain in model:
        psu_pos = [str(r.seqid.num) for r in chain if r.name in comp_ids]
        n_res = sum(1 for _ in chain)
        out[chain.name] = {
            "n_res_modelled": n_res,
            "n_psu": len(psu_pos),
            "psu_positions": ";".join(psu_pos),
        }
    return out


# --------------------------------------------- 4. joins with local resources

def load_id_list(path):
    """RNAsolo TXT list or a RISoTTo split file: PDBID_model_chain per line."""
    if not path or not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_")
            if len(parts) >= 3:
                keys.add((parts[0].upper(), parts[2]))
            elif len(parts) == 2:
                keys.add((parts[0].upper(), parts[1]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="psu_chains.csv")
    ap.add_argument("--rnasolo_list", default=None,
                    help="TXT list downloaded from the RNAsolo Download page")
    ap.add_argument("--train_list", default="datasets/grnade_das_train_set.txt")
    ap.add_argument("--val_list", default="datasets/grnade_das_validation_set.txt")
    ap.add_argument("--skip_counting", action="store_true",
                    help="Stop after metadata; no mmCIF downloads")
    args = ap.parse_args()

    print("Searching RCSB ...")
    entity_ids = find_entities(PSU_IDS)
    print(f"  {len(entity_ids)} RNA polymer entities contain {PSU_IDS}")

    df = fetch_metadata(entity_ids)
    print(f"  -> {len(df)} chains across {df.pdb_id.nunique()} entries")

    if not args.skip_counting:
        print("Counting PSU per chain (downloading mmCIF) ...")
        records = []
        for pdb_id, sub in df.groupby("pdb_id"):
            try:
                counts = count_psu(pdb_id, set(PSU_IDS))
            except Exception as exc:
                print(f"  ! {pdb_id}: {exc}")
                counts = {}
            for _, row in sub.iterrows():
                c = counts.get(row["chain"], {})
                records.append({**row.to_dict(), **c})
        df = pd.DataFrame(records)

    rnasolo = load_id_list(args.rnasolo_list)
    train = load_id_list(args.train_list)
    val = load_id_list(args.val_list)
    key = list(zip(df.pdb_id.str.upper(), df.chain))
    df["in_rnasolo"] = [k in rnasolo for k in key] if rnasolo else None
    df["in_train"] = [k in train for k in key] if train else None
    df["in_val"] = [k in val for k in key] if val else None

    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    # headline numbers
    if "n_psu" in df:
        d = df[df.n_psu > 0]
        print(f"chains with >=1 PSU : {len(d)}")
        print(f"total PSU residues  : {int(d.n_psu.sum())}")
        print(f"modelled nucleotides: {int(d.n_res_modelled.sum())}")
        print(f"PSU fraction        : {d.n_psu.sum()/max(d.n_res_modelled.sum(),1):.4%}")
        small = d[d.deposited_atoms < 12288]
        print(f"entries under max_size (12288 atoms): {small.pdb_id.nunique()}")
        hires = d[(d.resolution.notna()) & (d.resolution <= 3.0)]
        print(f"chains at <= 3.0 A  : {len(hires)}")
        if train:
            print(f"already in RISoTTo train split: {int(d.in_train.sum())}")


if __name__ == "__main__":
    main()
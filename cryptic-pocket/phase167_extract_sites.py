#!/usr/bin/env python3

import os
import csv
import math
from pathlib import Path
from collections import defaultdict

from Bio.PDB import MMCIFParser, is_aa
import numpy as np


# ============================================================
# Phase 167: Experimental ligand-binding-site extraction
# ============================================================

ROOT = Path.home() / "bioproj" / "cryptic-pocket"

MAPPING_FILE = ROOT / "cryptobench_binding_mapping.csv"
HOLO_DIR = ROOT / "phase167_holo"

OUTPUT_FILE = ROOT / "phase167_experimental_sites.csv"
FAILURE_FILE = ROOT / "phase167_structure_failures.csv"

CUTOFF = 4.5


# Known mmCIF identifier mismatches diagnosed during Phase 167.
#
# Format:
#   CryptoBench holo PDB -> (actual author chain, actual ligand,
#                            actual ligand residue number)
#
SPECIAL_CASES = {
    "1KDP": ("B", "DCM", 6407),
    "5V0A": ("B", "D5M", 102),
    "6M3S": ("A", "ICT", 404),
}


def normalize_pdb_id(value):
    """Convert values such as 3rj9C into PDB ID 3RJ9."""
    value = str(value).strip()
    return value[:4].upper()


def normalize_chain(value):
    return str(value).strip()


def normalize_resseq(value):
    """
    Convert residue number/index to an integer where possible.
    """
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def distance(a, b):
    return float(np.linalg.norm(a - b))


def get_structure(path, pdb_id):
    parser = MMCIFParser(
        QUIET=True,
        auth_chains=True,
        auth_residues=True
    )
    return parser.get_structure(pdb_id, str(path))


def find_ligand_residue(structure, chain_id, ligand_name, ligand_index):
    """
    Locate the ligand residue using author chain + residue number.

    First tries the exact CryptoBench identifiers.
    Then applies the known Phase 167 special cases.
    """

    ligand_name = str(ligand_name).strip().upper()
    chain_id = normalize_chain(chain_id)
    ligand_index = normalize_resseq(ligand_index)

    # --------------------------------------------------------
    # Special cases
    # --------------------------------------------------------
    pdb_id = structure.id.upper()

    if pdb_id in SPECIAL_CASES:
        actual_chain, actual_ligand, actual_index = SPECIAL_CASES[pdb_id]

        chain = structure[0][actual_chain]

        matches = []

        for residue in chain:
            hetflag, resseq, icode = residue.id

            if (
                str(actual_ligand).upper() == residue.resname.upper()
                and int(resseq) == int(actual_index)
            ):
                matches.append(residue)

        if len(matches) == 1:
            return matches[0], actual_chain, actual_ligand, actual_index

    # --------------------------------------------------------
    # Exact author chain + residue number
    # --------------------------------------------------------
    if chain_id in structure[0]:
        chain = structure[0][chain_id]

        matches = []

        for residue in chain:
            hetflag, resseq, icode = residue.id

            if ligand_index is not None and int(resseq) != ligand_index:
                continue

            if residue.resname.upper() != ligand_name:
                continue

            # Hetero residues have H_... in BioPython
            if str(hetflag).strip() == "":
                continue

            matches.append(residue)

        if len(matches) == 1:
            return matches[0], chain_id, ligand_name, ligand_index

    # --------------------------------------------------------
    # Broader search: ligand name + residue number
    # --------------------------------------------------------
    matches = []

    for chain in structure[0]:
        for residue in chain:
            hetflag, resseq, icode = residue.id

            if ligand_index is not None and int(resseq) != ligand_index:
                continue

            if residue.resname.upper() != ligand_name:
                continue

            if str(hetflag).strip() == "":
                continue

            matches.append(
                (residue, chain.id, residue.resname, int(resseq))
            )

    if len(matches) == 1:
        return matches[0]

    # --------------------------------------------------------
    # If exact ligand name differs, search by residue number.
    # This helps with CryptoBench component-name mismatches.
    # --------------------------------------------------------
    if ligand_index is not None:
        matches = []

        for chain in structure[0]:
            for residue in chain:
                hetflag, resseq, icode = residue.id

                if str(hetflag).strip() == "":
                    continue

                if int(resseq) == ligand_index:
                    matches.append(
                        (residue, chain.id, residue.resname, int(resseq))
                    )

        if len(matches) == 1:
            return matches[0]

    raise ValueError(
        f"could not uniquely identify ligand "
        f"{ligand_name} chain={chain_id} index={ligand_index}"
    )


def get_ligand_atoms(ligand_residue):
    """
    Return heavy-atom coordinates for the ligand.
    """
    atoms = []

    for atom in ligand_residue.get_atoms():
        element = getattr(atom, "element", "").strip().upper()

        # Exclude hydrogen/deuterium.
        if element in {"H", "D"}:
            continue

        # Some mmCIF files don't populate element cleanly.
        atom_name = atom.get_name().strip().upper()

        if atom_name.startswith("H"):
            continue

        atoms.append(atom)

    if not atoms:
        raise ValueError("ligand contains no heavy atoms")

    return atoms


def get_binding_residues(structure, ligand_atoms, ligand_chain):
    """
    A residue is considered experimentally ligand-contacting if
    any protein heavy atom is within CUTOFF Å of any ligand heavy atom.

    Water and hetero residues are excluded.
    """

    binding = []

    ligand_coords = np.array(
        [atom.coord for atom in ligand_atoms],
        dtype=float
    )

    model = structure[0]

    for chain in model:

        # We evaluate protein residues in the whole structure,
        # because the experimental pocket can include residues
        # neighboring the ligand across chain boundaries.
        for residue in chain:

            if not is_aa(residue, standard=False):
                continue

            protein_atoms = []

            for atom in residue.get_atoms():

                element = getattr(atom, "element", "").strip().upper()

                if element in {"H", "D"}:
                    continue

                atom_name = atom.get_name().strip().upper()

                if atom_name.startswith("H"):
                    continue

                protein_atoms.append(atom)

            if not protein_atoms:
                continue

            protein_coords = np.array(
                [atom.coord for atom in protein_atoms],
                dtype=float
            )

            # Pairwise distance calculation.
            delta = (
                protein_coords[:, None, :]
                - ligand_coords[None, :, :]
            )

            dists = np.sqrt(np.sum(delta * delta, axis=2))

            if np.any(dists <= CUTOFF):
                binding.append(residue)

    return binding


def residue_identifier(residue):
    """
    Return author-style chain/residue information.
    """
    chain = residue.get_parent()

    hetflag, resseq, icode = residue.id

    return (
        chain.id,
        int(resseq),
        str(icode).strip(),
        residue.resname.upper()
    )


def calculate_ligand_center(ligand_atoms):
    coords = np.array(
        [atom.coord for atom in ligand_atoms],
        dtype=float
    )

    return coords.mean(axis=0)


def calculate_site_center(binding_residues):
    coords = []

    for residue in binding_residues:
        for atom in residue.get_atoms():

            element = getattr(atom, "element", "").strip().upper()

            if element in {"H", "D"}:
                continue

            atom_name = atom.get_name().strip().upper()

            if atom_name.startswith("H"):
                continue

            coords.append(atom.coord)

    if not coords:
        return np.array([np.nan, np.nan, np.nan])

    return np.mean(np.array(coords, dtype=float), axis=0)


def main():

    print("=" * 70)
    print("Phase 167 experimental binding-site extraction")
    print("=" * 70)

    print(f"Mapping:       {MAPPING_FILE}")
    print(f"Holo directory:{HOLO_DIR}")
    print(f"Cutoff:        {CUTOFF} Å")
    print()

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {MAPPING_FILE}"
        )

    if not HOLO_DIR.exists():
        raise FileNotFoundError(
            f"Holo directory not found: {HOLO_DIR}"
        )

    # --------------------------------------------------------
    # Read mapping
    # --------------------------------------------------------
    with open(MAPPING_FILE, newline="") as f:
        reader = csv.DictReader(f)
        mappings = list(reader)

    print(f"Mappings found: {len(mappings)}")
    print()

    results = []
    failures = []

    # --------------------------------------------------------
    # Process every graph
    # --------------------------------------------------------
    for i, row in enumerate(mappings, start=1):

        apo_pdb = str(
            row.get("apo_pdb") or ""
        ).strip().upper()

        apo_chain = str(
            row.get("apo_chain")
            or row.get("chain")
            or ""
        ).strip()

        holo_pdb = str(
            row.get("holo_pdb") or ""
        ).strip().upper()

        holo_chain = str(
            row.get("holo_chain")
            or ""
        ).strip()

        ligand = str(
            row.get("ligand")
            or ""
        ).strip().upper()

        ligand_index = row.get(
            "ligand_index",
            ""
        )

        ligand_chain = str(
            row.get("ligand_chain")
            or ""
        ).strip()

        uniprot_id = str(
            row.get("uniprot_id")
            or ""
        ).strip()

        split = str(
            row.get("split")
            or ""
        ).strip()

        # ----------------------------------------------------
        # Locate CIF
        # ----------------------------------------------------
        candidates = [
            HOLO_DIR / f"{holo_pdb}.cif",
            HOLO_DIR / f"{holo_pdb.lower()}.cif",
            HOLO_DIR / f"{holo_pdb}_holo.cif",
            HOLO_DIR / f"{holo_pdb.lower()}_holo.cif",
        ]

        cif_path = next(
            (p for p in candidates if p.exists()),
            None
        )

        if cif_path is None:
            failures.append({
                "apo_pdb_id": apo_pdb,
                "apo_chain": apo_chain,
                "holo_pdb_id": holo_pdb,
                "holo_chain": holo_chain,
                "ligand": ligand,
                "ligand_index": ligand_index,
                "ligand_chain": ligand_chain,
                "error": "holo CIF not found",
            })
            continue

        try:

            structure = get_structure(
                cif_path,
                holo_pdb
            )

            # ------------------------------------------------
            # Locate ligand
            # ------------------------------------------------
            ligand_result = find_ligand_residue(
                structure,
                ligand_chain,
                ligand,
                ligand_index
            )

            if isinstance(ligand_result, tuple):
                ligand_residue = ligand_result[0]
                actual_ligand_chain = ligand_result[1]
                actual_ligand_name = ligand_result[2]
                actual_ligand_index = ligand_result[3]
            else:
                ligand_residue = ligand_result
                actual_ligand_chain = ligand_chain
                actual_ligand_name = ligand
                actual_ligand_index = ligand_index

            ligand_atoms = get_ligand_atoms(
                ligand_residue
            )

            ligand_center = calculate_ligand_center(
                ligand_atoms
            )

            # ------------------------------------------------
            # Experimental binding residues
            # ------------------------------------------------
            binding_residues = get_binding_residues(
                structure,
                ligand_atoms,
                actual_ligand_chain
            )

            if not binding_residues:
                raise ValueError(
                    "no protein residues found within cutoff"
                )

            site_center = calculate_site_center(
                binding_residues
            )

            residue_ids = []

            for residue in binding_residues:

                chain_id, resseq, icode, resname = (
                    residue_identifier(residue)
                )

                if icode:
                    residue_ids.append(
                        f"{chain_id}:{resseq}{icode}"
                    )
                else:
                    residue_ids.append(
                        f"{chain_id}:{resseq}"
                    )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------
            results.append({
                "apo_pdb_id": apo_pdb,
                "apo_chain": apo_chain,
                "holo_pdb_id": holo_pdb,
                "holo_chain": holo_chain,
                "ligand": ligand,
                "ligand_index": ligand_index,
                "ligand_chain": ligand_chain,
                "actual_ligand": actual_ligand_name,
                "actual_ligand_chain": actual_ligand_chain,
                "actual_ligand_index": actual_ligand_index,
                "uniprot_id": uniprot_id,
                "split": split,
                "cutoff_angstrom": CUTOFF,
                "n_ligand_heavy_atoms": len(ligand_atoms),
                "n_binding_residues": len(binding_residues),
                "ligand_center_x": ligand_center[0],
                "ligand_center_y": ligand_center[1],
                "ligand_center_z": ligand_center[2],
                "site_center_x": site_center[0],
                "site_center_y": site_center[1],
                "site_center_z": site_center[2],
                "binding_residues": ";".join(residue_ids),
            })

            if i % 25 == 0 or i == len(mappings):
                print(
                    f"[{i:3d}/{len(mappings)}] "
                    f"success={len(results)} "
                    f"fail={len(failures)} "
                    f"{apo_pdb}{apo_chain} -> "
                    f"{holo_pdb}{actual_ligand_chain} "
                    f"{actual_ligand_name}"
                )

        except Exception as e:

            failures.append({
                "apo_pdb_id": apo_pdb,
                "apo_chain": apo_chain,
                "holo_pdb_id": holo_pdb,
                "holo_chain": holo_chain,
                "ligand": ligand,
                "ligand_index": ligand_index,
                "ligand_chain": ligand_chain,
                "error": str(e),
            })

            print(
                f"[FAIL] {apo_pdb}{apo_chain} -> "
                f"{holo_pdb}: {e}"
            )

    # --------------------------------------------------------
    # Write successful results
    # --------------------------------------------------------
    result_fields = [
        "apo_pdb_id",
        "apo_chain",
        "holo_pdb_id",
        "holo_chain",
        "ligand",
        "ligand_index",
        "ligand_chain",
        "actual_ligand",
        "actual_ligand_chain",
        "actual_ligand_index",
        "uniprot_id",
        "split",
        "cutoff_angstrom",
        "n_ligand_heavy_atoms",
        "n_binding_residues",
        "ligand_center_x",
        "ligand_center_y",
        "ligand_center_z",
        "site_center_x",
        "site_center_y",
        "site_center_z",
        "binding_residues",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result_fields
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # Write failures
    # --------------------------------------------------------
    failure_fields = [
        "apo_pdb_id",
        "apo_chain",
        "holo_pdb_id",
        "holo_chain",
        "ligand",
        "ligand_index",
        "ligand_chain",
        "error",
    ]

    with open(
        FAILURE_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=failure_fields
        )

        writer.writeheader()
        writer.writerows(failures)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Mappings:  {len(mappings)}")
    print(f"Successes: {len(results)}")
    print(f"Failures:  {len(failures)}")

    print()
    print(f"Output:")
    print(f"  {OUTPUT_FILE}")

    print()
    print(f"Failures:")
    print(f"  {FAILURE_FILE}")

    if failures:
        print()
        print("Failed structures:")

        for failure in failures:
            print(
                f"  {failure['apo_pdb_id']}"
                f"{failure['apo_chain']} -> "
                f"{failure['holo_pdb_id']}: "
                f"{failure['error']}"
            )


if __name__ == "__main__":
    main()

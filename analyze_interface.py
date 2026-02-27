#!/usr/bin/env python3
"""Analyze the 15-PGDH dimer interface from PDB 2GDZ to determine optimal binder scaffold."""

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Superimposer, NeighborSearch
from Bio.PDB.vectors import Vector, rotmat
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


def apply_biomt(structure, chain_id="A"):
    """Apply BIOMT2 transformation from 2GDZ to generate chain B of the dimer.
    
    BIOMT2 from PDB:
      0  1  0  0
      1  0  0  0
      0  0 -1  195.771
    """
    model = structure[0]
    chain_a = model[chain_id]

    from Bio.PDB.Chain import Chain
    chain_b = Chain("B")

    rot = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0]
    ])
    trans = np.array([0.0, 0.0, 195.771])

    for residue in chain_a.get_residues():
        from Bio.PDB.Residue import Residue
        new_res = Residue(residue.id, residue.resname, residue.segid)
        for atom in residue.get_atoms():
            new_atom = atom.copy()
            coord = atom.get_vector().get_array()
            new_coord = rot @ coord + trans
            new_atom.set_coord(new_coord)
            new_res.add(new_atom)
        chain_b.add(new_res)

    model.add(chain_b)
    return structure


def get_interface_residues(structure, chain1_id="A", chain2_id="B", cutoff=8.0):
    """Find interface residues between two chains using CA-CA distance cutoff."""
    model = structure[0]
    chain1 = model[chain1_id]
    chain2 = model[chain2_id]

    interface_res_1 = {}
    interface_res_2 = {}

    for res1 in chain1.get_residues():
        if res1.id[0] != " ":
            continue
        for res2 in chain2.get_residues():
            if res2.id[0] != " ":
                continue
            if "CA" in res1 and "CA" in res2:
                dist = res1["CA"] - res2["CA"]
                if dist < cutoff:
                    resnum1 = res1.id[1]
                    resnum2 = res2.id[1]
                    if resnum1 not in interface_res_1:
                        interface_res_1[resnum1] = {"resname": res1.resname, "min_dist": dist, "contacts": []}
                    interface_res_1[resnum1]["min_dist"] = min(interface_res_1[resnum1]["min_dist"], dist)
                    interface_res_1[resnum1]["contacts"].append((resnum2, dist))

                    if resnum2 not in interface_res_2:
                        interface_res_2[resnum2] = {"resname": res2.resname, "min_dist": dist, "contacts": []}
                    interface_res_2[resnum2]["min_dist"] = min(interface_res_2[resnum2]["min_dist"], dist)
                    interface_res_2[resnum2]["contacts"].append((resnum1, dist))

    return interface_res_1, interface_res_2


def get_heavy_atom_contacts(structure, chain1_id="A", chain2_id="B", cutoff=4.5):
    """Find inter-chain heavy atom contacts."""
    model = structure[0]
    chain1 = model[chain1_id]
    chain2 = model[chain2_id]

    contacts = []
    contact_residues_1 = set()
    contact_residues_2 = set()

    for res1 in chain1.get_residues():
        if res1.id[0] != " ":
            continue
        for res2 in chain2.get_residues():
            if res2.id[0] != " ":
                continue
            for atom1 in res1.get_atoms():
                if atom1.element == "H":
                    continue
                for atom2 in res2.get_atoms():
                    if atom2.element == "H":
                        continue
                    dist = atom1 - atom2
                    if dist < cutoff:
                        contacts.append({
                            "res1": f"{res1.resname}{res1.id[1]}",
                            "atom1": atom1.name,
                            "res2": f"{res2.resname}{res2.id[1]}",
                            "atom2": atom2.name,
                            "dist": dist
                        })
                        contact_residues_1.add(res1.id[1])
                        contact_residues_2.add(res2.id[1])

    return contacts, contact_residues_1, contact_residues_2


def compute_interface_geometry(structure, interface_res, chain_id="A"):
    """Compute geometric properties of the interface patch."""
    model = structure[0]
    chain = model[chain_id]

    ca_coords = []
    for resnum in interface_res:
        res = chain[(" ", resnum, " ")]
        if "CA" in res:
            ca_coords.append(res["CA"].get_vector().get_array())

    ca_coords = np.array(ca_coords)
    centroid = ca_coords.mean(axis=0)

    centered = ca_coords - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    spread = np.sqrt(eigenvalues)
    return {
        "centroid": centroid,
        "n_residues": len(ca_coords),
        "spread_axes": spread,
        "principal_axes": eigenvectors,
        "extent_1": spread[0] * 2,
        "extent_2": spread[1] * 2,
        "extent_3": spread[2] * 2,
        "flatness": spread[2] / spread[0] if spread[0] > 0 else 0,
        "aspect_ratio": spread[1] / spread[0] if spread[0] > 0 else 0,
    }


def compute_sasa_interface(structure, chain1_id="A", chain2_id="B"):
    """Estimate buried surface area at the interface."""
    from Bio.PDB.DSSP import dssp_dict_from_pdb_file
    # Simple estimation: count interface atoms and approximate
    model = structure[0]
    chain1 = model[chain1_id]
    chain2 = model[chain2_id]

    all_atoms_1 = [a for r in chain1.get_residues() if r.id[0] == " " for a in r.get_atoms() if a.element != "H"]
    all_atoms_2 = [a for r in chain2.get_residues() if r.id[0] == " " for a in r.get_atoms() if a.element != "H"]

    ns2 = NeighborSearch(all_atoms_2)
    buried_atoms_1 = set()
    for atom in all_atoms_1:
        nearby = ns2.search(atom.get_vector().get_array(), 4.5)
        if nearby:
            buried_atoms_1.add(atom.get_full_id())

    ns1 = NeighborSearch(all_atoms_1)
    buried_atoms_2 = set()
    for atom in all_atoms_2:
        nearby = ns1.search(atom.get_vector().get_array(), 4.5)
        if nearby:
            buried_atoms_2.add(atom.get_full_id())

    return len(buried_atoms_1), len(buried_atoms_2)


def classify_residue(resname):
    """Classify residue by physicochemical property."""
    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET"}
    polar = {"SER", "THR", "ASN", "GLN", "CYS", "TYR"}
    positive = {"LYS", "ARG", "HIS"}
    negative = {"ASP", "GLU"}
    if resname in hydrophobic:
        return "hydrophobic"
    elif resname in polar:
        return "polar"
    elif resname in positive:
        return "positive"
    elif resname in negative:
        return "negative"
    return "other"


def analyze_interface_composition(structure, interface_residues, chain_id="A"):
    """Analyze the chemical composition of the interface."""
    model = structure[0]
    chain = model[chain_id]

    composition = defaultdict(int)
    residue_details = []

    for resnum in sorted(interface_residues):
        res = chain[(" ", resnum, " ")]
        rtype = classify_residue(res.resname)
        composition[rtype] += 1
        residue_details.append({
            "resnum": resnum,
            "resname": res.resname,
            "type": rtype
        })

    return composition, residue_details


def estimate_cavity_depth(structure, interface_res_1, interface_res_2, chain1_id="A", chain2_id="B"):
    """Estimate the concavity/convexity of the interface."""
    model = structure[0]
    chain1 = model[chain1_id]
    chain2 = model[chain2_id]

    coords_1 = []
    coords_2 = []
    for resnum in interface_res_1:
        res = chain1[(" ", resnum, " ")]
        if "CA" in res:
            coords_1.append(res["CA"].get_vector().get_array())
    for resnum in interface_res_2:
        res = chain2[(" ", resnum, " ")]
        if "CA" in res:
            coords_2.append(res["CA"].get_vector().get_array())

    coords_1 = np.array(coords_1)
    coords_2 = np.array(coords_2)

    centroid_1 = coords_1.mean(axis=0)
    centroid_2 = coords_2.mean(axis=0)
    interface_axis = centroid_2 - centroid_1
    interface_axis /= np.linalg.norm(interface_axis)

    proj_1 = (coords_1 - centroid_1) @ interface_axis
    proj_2 = (coords_2 - centroid_2) @ (-interface_axis)

    return {
        "interface_distance": np.linalg.norm(centroid_2 - centroid_1),
        "depth_variation_chain1": proj_1.std(),
        "depth_variation_chain2": proj_2.std(),
        "depth_range_chain1": proj_1.max() - proj_1.min(),
        "depth_range_chain2": proj_2.max() - proj_2.min(),
    }


def main():
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("15PGDH", "2GDZ.pdb")

    print("=" * 70)
    print("15-PGDH DIMER INTERFACE ANALYSIS (PDB: 2GDZ)")
    print("=" * 70)

    # Chain A info
    model = structure[0]
    chain_a = model["A"]
    protein_residues = [r for r in chain_a.get_residues() if r.id[0] == " "]
    print(f"\nChain A: {len(protein_residues)} protein residues")
    print(f"  First: {protein_residues[0].resname}{protein_residues[0].id[1]}")
    print(f"  Last:  {protein_residues[-1].resname}{protein_residues[-1].id[1]}")

    # Build dimer
    print("\n--- Building biological dimer from BIOMT ---")
    structure = apply_biomt(structure)

    # Save dimer PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save("2GDZ_dimer.pdb")
    print("Saved: 2GDZ_dimer.pdb")

    # Interface residues (CA-CA < 8 A)
    print("\n" + "=" * 70)
    print("INTERFACE RESIDUES (CA-CA < 8.0 A)")
    print("=" * 70)
    iface_res_1, iface_res_2 = get_interface_residues(structure, cutoff=8.0)
    print(f"\nChain A interface residues: {len(iface_res_1)}")
    for resnum in sorted(iface_res_1):
        info = iface_res_1[resnum]
        print(f"  {info['resname']:>3s} {resnum:>4d}  min_dist={info['min_dist']:.1f}A  n_contacts={len(info['contacts'])}")

    # Close contacts (heavy atoms < 4.5 A)
    print("\n" + "=" * 70)
    print("CLOSE INTER-CHAIN CONTACTS (heavy atom < 4.5 A)")
    print("=" * 70)
    contacts, contact_res_1, contact_res_2 = get_heavy_atom_contacts(structure, cutoff=4.5)
    print(f"\nTotal atomic contacts: {len(contacts)}")
    print(f"Chain A contact residues: {sorted(contact_res_1)}")
    print(f"Chain B contact residues: {sorted(contact_res_2)}")

    print("\nTop contacts by distance:")
    sorted_contacts = sorted(contacts, key=lambda x: x["dist"])
    for c in sorted_contacts[:30]:
        print(f"  {c['res1']:>8s}.{c['atom1']:<4s} -- {c['res2']:>8s}.{c['atom2']:<4s}  {c['dist']:.2f} A")

    # Interface geometry
    print("\n" + "=" * 70)
    print("INTERFACE GEOMETRY")
    print("=" * 70)
    geom = compute_interface_geometry(structure, iface_res_1, "A")
    print(f"\nInterface patch on Chain A:")
    print(f"  Number of residues: {geom['n_residues']}")
    print(f"  Extent axis 1 (longest):  {geom['extent_1']:.1f} A")
    print(f"  Extent axis 2 (medium):   {geom['extent_2']:.1f} A")
    print(f"  Extent axis 3 (shortest): {geom['extent_3']:.1f} A")
    print(f"  Flatness (axis3/axis1):   {geom['flatness']:.3f}")
    print(f"  Aspect ratio (axis2/axis1): {geom['aspect_ratio']:.3f}")

    # Buried atoms estimate
    buried_1, buried_2 = compute_sasa_interface(structure)
    print(f"\n  Buried atoms chain A: {buried_1}")
    print(f"  Buried atoms chain B: {buried_2}")
    avg_sasa_per_atom = 10.0  # rough A^2 per heavy atom
    print(f"  Estimated BSA (rough): ~{(buried_1 + buried_2) * avg_sasa_per_atom:.0f} A^2")
    print(f"  PDB REMARK 350 BSA: 7160 A^2")

    # Cavity depth
    cavity = estimate_cavity_depth(structure, iface_res_1, iface_res_2)
    print(f"\n  Inter-centroid distance: {cavity['interface_distance']:.1f} A")
    print(f"  Depth variation chain A: {cavity['depth_variation_chain1']:.1f} A")
    print(f"  Depth range chain A: {cavity['depth_range_chain1']:.1f} A")

    # Chemical composition
    print("\n" + "=" * 70)
    print("INTERFACE CHEMICAL COMPOSITION")
    print("=" * 70)
    comp, details = analyze_interface_composition(structure, contact_res_1, "A")
    total = sum(comp.values())
    print(f"\nContact residue composition (chain A, {total} residues):")
    for rtype in ["hydrophobic", "polar", "positive", "negative", "other"]:
        count = comp.get(rtype, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"  {rtype:>12s}: {count:3d} ({pct:5.1f}%)")

    print("\nDetailed interface residues:")
    for d in details:
        print(f"  {d['resname']:>3s} {d['resnum']:>4d}  [{d['type']}]")

    # Secondary structure assignment (manual from known literature)
    print("\n" + "=" * 70)
    print("INTERFACE MAPPING TO STRUCTURAL ELEMENTS")
    print("=" * 70)
    # From the literature: alpha9 helix is the main dimer interface
    # alpha9 is approximately residues 143-172
    # Lid domain: alpha10-12, residues ~185-217
    alpha9_range = set(range(143, 173))
    lid_range = set(range(185, 218))
    other_interface = set()

    print("\nInterface residues by structural element:")
    a9_iface = sorted(contact_res_1 & alpha9_range)
    lid_iface = sorted(contact_res_1 & lid_range)
    other_iface = sorted(contact_res_1 - alpha9_range - lid_range)

    print(f"\n  alpha9 helix (dimer interface helix, ~143-172):")
    for r in a9_iface:
        res = model["A"][(" ", r, " ")]
        print(f"    {res.resname}{r}")

    print(f"\n  Lid domain (alpha10-12, ~185-217):")
    for r in lid_iface:
        res = model["A"][(" ", r, " ")]
        print(f"    {res.resname}{r}")

    print(f"\n  Other regions:")
    for r in other_iface:
        res = model["A"][(" ", r, " ")]
        print(f"    {res.resname}{r}")

    # Scaffold suitability analysis
    print("\n" + "=" * 70)
    print("SCAFFOLD SUITABILITY ANALYSIS")
    print("=" * 70)

    print(f"""
Interface Summary:
  - Total BSA: ~7160 A^2 (from PDB REMARK)
  - Per-chain BSA: ~3580 A^2
  - Interface extent: {geom['extent_1']:.0f} x {geom['extent_2']:.0f} A (2D footprint)
  - Flatness: {geom['flatness']:.3f} (low = flat interface)
  - Chemistry: {100*comp.get('hydrophobic',0)/total:.0f}% hydrophobic, {100*comp.get('polar',0)/total:.0f}% polar
  - Inter-centroid distance: {cavity['interface_distance']:.1f} A
  - Contact residues: {len(contact_res_1)} on each chain

Scaffold Comparison for Dimer Interface Disruption:
""")

    scaffolds = [
        {
            "name": "DARPin (Designed Ankyrin Repeat Protein)",
            "typical_size": "~130-170 aa (4-5 repeats)",
            "binding_surface": "~800-1200 A^2",
            "shape": "Flat, slightly concave",
            "pros": [
                "Flat binding surface matches flat dimer interfaces well",
                "Modular repeat architecture = tunable size",
                "Very high stability (Tm > 90C typically)",
                "No disulfides = robust intracellular folding",
                "Good expression in E. coli",
                "Well within 250 aa limit",
            ],
            "cons": [
                "Binding surface may not cover full interface (~3580 A^2)",
                "Less established in de novo computational design pipelines",
                "Standard DARPin libraries not publicly available for RFdiffusion/BindCraft",
            ],
        },
        {
            "name": "De novo miniprotein (helix bundle)",
            "typical_size": "~50-90 aa",
            "binding_surface": "~500-900 A^2",
            "shape": "Compact globular, mixed",
            "pros": [
                "Best supported by BindCraft and RFdiffusion",
                "EGFR competition winner was BindCraft miniprotein",
                "High designability, well-characterized pipelines",
                "Small = efficient to synthesize",
                "Can be designed to wedge into interface groove",
            ],
            "cons": [
                "Small binding surface vs large interface",
                "May need to target a subregion (hotspot) of interface",
                "Single miniprotein unlikely to cover full dimer seam",
            ],
        },
        {
            "name": "Nanobody / VHH",
            "typical_size": "~120-130 aa",
            "binding_surface": "~600-1000 A^2",
            "shape": "Ig-fold with protruding CDR loops",
            "pros": [
                "CDR loops can reach into grooves and crevices",
                "Well-characterized, clinically validated format",
                "Good for targeting concave epitopes",
            ],
            "cons": [
                "Disulfide bond needed (intracellular challenge)",
                "CDR loop design is harder for de novo",
                "Flat interface may not have deep pockets for loop insertion",
            ],
        },
        {
            "name": "De novo beta-sheet binder",
            "typical_size": "~60-120 aa",
            "binding_surface": "~600-1000 A^2",
            "shape": "Flat beta-sheet face",
            "pros": [
                "Beta-sheet face can make flat extended contacts",
                "RFdiffusion beta-pairing mode available (2025 paper)",
                "Could mimic the alpha9 helix interaction pattern",
                "Better surface complementarity for flat interfaces",
            ],
            "cons": [
                "Beta-sheet designs historically harder to express",
                "Edge effects / aggregation risk",
                "Newer approach, less validated experimentally",
            ],
        },
        {
            "name": "Bivalent / biparatopic miniprotein",
            "typical_size": "~120-200 aa (2 domains + linker)",
            "binding_surface": "~1000-1800 A^2",
            "shape": "Two binding heads connected by linker",
            "pros": [
                "Can span the dimer seam = contacts both monomers",
                "Dimer-selective via avidity (only binds assembled dimer)",
                "Larger total BSA coverage",
                "Within 250 aa limit",
                "Differentiated IP story (bioPROTAC-ready)",
            ],
            "cons": [
                "More complex design (two interfaces + linker optimization)",
                "Linker flexibility may reduce binding",
                "Harder to validate computationally",
            ],
        },
        {
            "name": "Cyclic peptide",
            "typical_size": "~8-30 aa",
            "binding_surface": "~300-500 A^2",
            "shape": "Constrained macrocycle",
            "pros": [
                "Cell-permeable (solves intracellular delivery)",
                "Can target hotspot residues at interface",
                "BoltzGen can design these",
                "Oral bioavailability potential",
            ],
            "cons": [
                "Very small binding surface",
                "Unlikely to disrupt strong dimer interface alone",
                "Limited computational design tools",
                "Low affinity expected for flat interface",
            ],
        },
    ]

    for s in scaffolds:
        print(f"\n{'─' * 60}")
        print(f"  {s['name']}")
        print(f"  Size: {s['typical_size']}  |  BSA: {s['binding_surface']}  |  Shape: {s['shape']}")
        print(f"  PROS:")
        for p in s["pros"]:
            print(f"    + {p}")
        print(f"  CONS:")
        for c in s["cons"]:
            print(f"    - {c}")

    # Recommendation
    print(f"""
{'=' * 70}
RECOMMENDATION
{'=' * 70}

Given the 15-PGDH dimer interface characteristics:
  - Large flat interface (~3580 A^2 per side)
  - Predominantly hydrophobic (alpha9 antiparallel helices)
  - The interface is relatively FLAT (flatness = {geom['flatness']:.3f})
  - Key hotspot residues: F161, L150, A153, A146, L167, A168, Y206, L171, M172

PRIMARY RECOMMENDATION: De novo helical miniprotein via BindCraft
  - Target a hotspot subregion of the alpha9 interface
  - BindCraft has the best experimental validation track record
  - Aim for ~60-90 aa binder targeting F161/A146/L150/A153 cluster
  - Use alpha-helical design to pack against the alpha9 helix

SECONDARY RECOMMENDATION: Beta-sheet binder via RFdiffusion (beta-pairing mode)
  - Flat beta-sheet face complements the flat helix-helix interface  
  - New beta-pairing RFdiffusion can make extended contacts

ALTERNATIVE: Bivalent design spanning the dimer seam
  - Two miniprotein heads + linker
  - Contacts both monomers = dimer-selective
  - Best differentiation story for pitch

DARPin ASSESSMENT: Suitable but harder to design computationally
  - Natural DARPin binding surface is well-matched to flat interfaces
  - However, no established BindCraft/RFdiffusion workflow for DARPin scaffolds
  - Would require: (1) use DARPin backbone as a fixed scaffold and redesign
    binding surface with ProteinMPNN, or (2) use RFdiffusion partial diffusion
    from a DARPin template
  - Risk: less validated pipeline = lower confidence for hackathon

NOTE ON INTRACELLULAR DELIVERY:
  15-PGDH is an intracellular target. For the competition, binding affinity 
  (SPR/KD) is what matters. For the pitch, mention mRNA delivery as the
  clinical path (like approved mRNA vaccines/therapeutics).
""")


if __name__ == "__main__":
    main()

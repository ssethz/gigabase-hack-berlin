# 15-PGDH Interface Binder Design Pipeline

**Target:** 15-Hydroxyprostaglandin Dehydrogenase (15-PGDH / HPGD, UniProt P15428)
**Goal:** Design a DARPin-scaffold protein binder targeting the 15-PGDH homodimer interface surface
**Mechanism:** Occupying the functionally critical alpha-9 helix region required for dimerization and catalytic residue positioning

## Scientific Rationale

15-PGDH is a "gerozyme" — an enzyme whose activity increases with age and drives aging phenotypes. It degrades prostaglandin E2 (PGE2), a key tissue regeneration signal. Inhibiting 15-PGDH rejuvenates aged muscle, nerve, and cartilage tissue ([Blau et al., Science Transl Med 2023](https://www.science.org/doi/10.1126/scitranslmed.adg1485)).

**Why target the dimer interface surface?** 15-PGDH functions as a homodimer. The alpha-9 helix (residues 143-172) forms the core of the dimer interface and is essential for proper positioning of catalytic residues S138, Y151, and K155. A binder that occupies this surface with high affinity and specificity targets a functionally critical region that is completely distinct from the active-site pocket targeted by existing small-molecule inhibitors.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  1. TARGET PREPARATION                                          │
│     PDB 2GDZ → build dimer → interface mapping → hotspot select │
│     Position DARPin scaffold at interface                        │
├─────────────────────────────────────────────────────────────────┤
│  2. BACKBONE GENERATION                                         │
│     RFdiffusion partial diffusion from DARPin scaffold (~156 aa)│
│     Multiple noise levels (partial_T = 15, 25, 35)              │
├─────────────────────────────────────────────────────────────────┤
│  3. SEQUENCE DESIGN                                             │
│     ProteinMPNN (soluble model, 8 seqs/backbone, T=0.1)         │
├─────────────────────────────────────────────────────────────────┤
│  4. STRUCTURE PREDICTION                                        │
│     Boltz-2 complex prediction (5 diffusion samples per design) │
├─────────────────────────────────────────────────────────────────┤
│  5. SCORING                                                     │
│     ipSAE (primary) + Boltz confidence + ipTM + ipLDDT          │
├─────────────────────────────────────────────────────────────────┤
│  6. SELECTION                                                   │
│     Quality filters → multi-criteria ranking → diversity cluster │
│     → Top 20 designs for experimental validation                │
└─────────────────────────────────────────────────────────────────┘
```

## Why a DARPin Scaffold

DARPins (Designed Ankyrin Repeat Proteins) are well-suited for this target:

- **Flat, concave binding surface** matches the flat alpha-helix dimer interface geometry
- **~156 aa** — well within the 250 aa limit, leaving room for functional tags
- **No disulfide bonds** — robust folding for an intracellular target
- **High thermostability** (Tm > 90°C typical) — tolerant of partial diffusion redesign
- **Modular repeat architecture** — tunable binding surface via repeat number

RFdiffusion partial diffusion adapts the DARPin binding surface to the 15-PGDH interface while preserving the overall fold stability.

## Dimer Interface Analysis

From structural analysis of PDB 2GDZ (1.65 Å resolution):

| Property | Value |
|----------|-------|
| Total buried surface area | ~7,160 Å² |
| Per-chain interface residues | 60 (heavy atom cutoff 5.0 Å) |
| Interface composition | 45% hydrophobic, 34% polar |
| Core structural element | Alpha-9 helix (residues 143-172) |
| Key contact residues | F161, A146, L150, A153, L167, A168, Y206, M172 |
| Lid cross-contacts | Y203/Y206 ↔ Y116/A168/L171/M172 (opposite protomer) |

## Quick Start

```bash
# 1. Prepare target (builds dimer, identifies hotspots, positions DARPin)
bash run_pipeline.sh prepare

# 2. Submit RFdiffusion partial diffusion jobs (SLURM)
bash run_pipeline.sh rfdiff

# 3. After rfdiff completes: sequence design
bash run_pipeline.sh mpnn

# 4. After mpnn completes: structure prediction
bash run_pipeline.sh boltz

# 5. After boltz completes: score with ipSAE
bash run_pipeline.sh score

# 6. Final selection
bash run_pipeline.sh select
```

## Repository Structure

```
├── run_pipeline.sh              # Master orchestrator
├── pipeline/
│   ├── prepare_target.py        # Target preparation + interface analysis
│   ├── prepare_darpin.py        # DARPin scaffold positioning
│   ├── prepare_mpnn_inputs.py   # ProteinMPNN input generation
│   ├── prepare_boltz_inputs.py  # Boltz-2 YAML generation
│   ├── score_designs.py         # ipSAE + Boltz confidence scoring
│   └── filter_and_select.py     # Multi-criteria filtering + diversity selection
├── scripts/
│   ├── setup_env.sh             # Environment configuration
│   ├── run_rfdiffusion.sh       # SLURM: RFdiffusion partial diffusion
│   ├── run_proteinmpnn.sh       # SLURM: ProteinMPNN sequence design
│   └── run_boltz.sh             # SLURM: Boltz-2 complex prediction
├── configs/
│   └── pipeline_config.json     # Target info, hotspots, design parameters
├── targets/
│   ├── 15pgdh_chainA.pdb        # Clean monomer target
│   ├── 15pgdh_dimer.pdb         # Biological dimer
│   ├── darpin_template.pdb      # DARPin scaffold (PDB 4DX5 chain D)
│   └── darpin_on_15pgdh.pdb     # DARPin positioned at interface
├── analyze_interface.py         # Detailed interface geometry analysis
└── results/                     # Final scored and selected designs
```

## Tools and Dependencies

| Tool | Version | Purpose |
|------|---------|---------|
| RFdiffusion | 1.1.0 | Backbone generation (partial diffusion from DARPin) |
| ProteinMPNN | - | Sequence design (soluble model) |
| Boltz-2 | - | Complex structure prediction |
| ipSAE | v4 | Interface quality scoring (Dunbrack Lab) |
| BioPython | 1.84 | Structure analysis |

## Compute Environment

ETH Euler cluster, SLURM scheduler, NVIDIA A100 80GB GPUs.

## Positioning

- **First biologic targeting the 15-PGDH dimer interface** — all current programs are active-site small molecules
- **Distinct from Epirium Bio's MF-300** — different binding site, different modality
- **DARPin scaffold** — proven clinical-stage format (e.g., abicipar pegol)
- **Extensible** — interface binder can be fused to E3 ligase recruiter for targeted degradation (bioPROTAC)
- **Intracellular delivery** via mRNA expression (clinical precedent in mRNA therapeutics field)

## References

- Blau et al. (2023) "15-PGDH is a gerozyme" — *Science Transl Med*
- Chougule et al. (2023) "15-PGDH lid dynamics" — *Nature Communications*
- Watson et al. (2023) RFdiffusion — *Nature*
- Dauparas et al. (2022) ProteinMPNN — *Science*
- Dunbrack (2025) ipSAE scoring — *bioRxiv*

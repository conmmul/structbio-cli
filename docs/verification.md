# Verification before sharing with the lab

Most of this repository has been tested against stand-in programs and against
the upstream projects' published interfaces. One path is now proven on real
hardware: on 2026-08-29, `env repair` produced a working CUDA PyTorch for
RFdiffusion on a workstation with two RTX 4090s, confirmed by `env verify`
computing on the card. **Everything else — the wrapped commands themselves, and
every other tool — is still unproven against a real installation.** This
checklist closes that gap.

Work through it in order: each stage costs more time than the one before, and a
failure early makes the later stages pointless. Record the result of each stage
somewhere you can refer back to.

---

## Stage 0 — the toolkit itself (any computer, no GPU, ~10 minutes)

Proves the package installs cleanly from a fresh clone.

```bash
cd /tmp
git clone https://github.com/conmmul/structbio-cli.git verify-structbio
cd verify-structbio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

- [ ] The clone, the install, and the test suite all succeed.
- [ ] `structbio --version` prints a version.
- [ ] `structbio doctor` runs and its `package` line points at this clone.
- [ ] `structbio detect` completes in a second or two.

Then check every command generates something sensible without any tool
installed:

```bash
structbio rfdiffusion monomer 150 t1 --dry-run
structbio rfdiffusion symmetry c4 400 t2 --dry-run
structbio proteinmpnn design tests/fixtures/tiny.pdb 2 t3 --chains A --dry-run
structbio cryozeta predict-json examples/cryozeta/native_input.example.json t4 --dry-run
```

- [ ] Each prints a command and says nothing was created.
- [ ] No folders `t1`–`t4` exist afterwards (`ls`).

Finally remove the clone: `cd /tmp && rm -rf verify-structbio`.

---

## Stage 0b — the environments (workstation, ~15 minutes)

Before any tool test, settle the environment question. It is where the time
goes, and `env verify` answers it in seconds.

```bash
structbio gpu
structbio env verify rfdiffusion
structbio env verify proteinmpnn
```

- [ ] `structbio gpu` names every card and its compute capability.
- [ ] Each `env verify` prints the card's name and no `FAILED` lines.

`CUDA none` or `no GPU` means the installed PyTorch has no CUDA support, which
is the usual outcome of RFdiffusion's own install because `env/SE3nv.yml` lists
the general conda channels ahead of the `pytorch` one. Repair rather than
rebuild:

```bash
structbio env repair rfdiffusion
```

- [ ] After repairing, `env verify` passes.
- [ ] `structbio doctor` reads `FOUND` rather than `FOUND, WITH WARNINGS`.

Confirmed working on 2026-08-29: two RTX 4090s (compute capability 8.9, driver
580.173.02), Python 3.9 in a miniconda `SE3nv`, repaired with
`pytorch=2.3.1=py3.9_cuda11.8_cudnn8.7.0_0` and `dgl=2.4.0.th23.cu118=py39_0`
from the conda channels. That machine cannot reach `download.pytorch.org`, so
the conda route was the only one that worked.

## Stage 1 — ProteinMPNN (workstation, ~30 minutes)

The cheapest real test: its model weights ship inside the repository, and it
runs on the CPU if no GPU is free.

```bash
structbio install proteinmpnn --into ~/software
# run the printed conda steps, then:
structbio setup
structbio doctor
```

- [ ] `doctor` reports ProteinMPNN as `FOUND`.

**The critical check.** Compare the command structbio generates with the
options your installed copy actually accepts:

```bash
structbio proteinmpnn design tests/fixtures/tiny.pdb 2 t --chains A --dry-run
conda activate mlfold
python ~/software/ProteinMPNN/protein_mpnn_run.py --help
```

- [ ] Every flag in the generated command appears in that help output.
      In particular `--pdb_path`, `--pdb_path_chains`, `--fixed_positions_jsonl`,
      `--out_folder`, `--num_seq_per_target`, `--sampling_temp`, `--batch_size`,
      `--seed`, `--model_name`, `--omit_AAs`.

If any flag is missing or renamed, stop: that version differs from the one the
wrapper was written against. Report it before going further.

Then a real run on a small structure of your own:

```bash
proteinmpnn design my_backbone.pdb 4 check_mpnn --chains A --designable A:10-30
```

- [ ] The printed designable/fixed residue list matches what you intended.
      Check two or three residue numbers against the PDB by eye.
- [ ] `check_mpnn/seqs/` contains a FASTA with 4 designed sequences.
- [ ] Running the identical command again is refused, and the first results are
      untouched.
- [ ] `check_mpnn/.structbio/command.txt` matches what was printed.

---

## Stage 2 — RFdiffusion (workstation, ~1 hour plus install)

```bash
structbio install rfdiffusion --into ~/software
# run the printed steps: conda env, SE3Transformer, pip install -e ., 7 weight files
structbio setup
structbio doctor
```

- [ ] `doctor` reports RFdiffusion as `FOUND`.

```bash
rfdiffusion monomer 50 check_rf -n 1 --dry-run
```

- [ ] The Hydra arguments match the RFdiffusion documentation for your version:
      `contigmap.contigs`, `inference.output_prefix`, `inference.num_designs`.

Then the smallest real run:

```bash
rfdiffusion monomer 50 check_rf -n 1
```

- [ ] It completes and `check_rf/check_rf_0.pdb` exists and opens in PyMOL.
- [ ] Progress appeared live in the terminal rather than only at the end.

Then the binder path, which is where structbio derives a contig for you:

```bash
rfdiffusion binder your_target.pdb 80 check_binder --chain B --hotspots B30 --dry-run
```

- [ ] The generated `contigmap.contigs` matches the residue range actually
      present in chain B of that file. This is the highest-value single check
      in the whole list: a wrong contig produces plausible-looking rubbish.

---

## Stage 3 — ColabFold (workstation, ~30 minutes)

Point it at the ProteinMPNN output from Stage 1:

```bash
colabfold predict check_mpnn check_folds --msa-mode single_sequence --dry-run
```

- [ ] The report says it found the sequences inside the ProteinMPNN output.
- [ ] No warning about sequences leaving the machine appears with
      `--msa-mode single_sequence`.
- [ ] The warning **does** appear when that option is removed. Confirm this: it
      is the safeguard for unpublished sequences.

Then run it for real on a handful of short sequences.

- [ ] Ranked PDB files appear in `check_folds/`.
- [ ] Folding a ProteinMPNN batch (a folder of several structures) works, and
      `check_folds/.structbio/inputs/sequences.fa` holds every record from every
      subfolder.

---

## Stage 4 — CryoZeta (workstation, half a day)

The most important tool for this lab, and the one whose input file structbio
now generates. Test upstream first, then compare.

```bash
structbio install cryozeta --into ~/software
# read WEIGHT_LICENSE.md, then run: pixi run setup
cd ~/software/CryoZeta
bash inference_demo.sh          # the bundled example, no structbio involved
```

- [ ] The bundled example completes. If it does not, the problem is the
      CryoZeta installation, not structbio.

Now the same target through structbio, using CryoZeta's own example JSON:

```bash
cryozeta predict-json ~/software/CryoZeta/assets/examples/example.json check_cz_json
```

- [ ] It completes, and the output is equivalent to the bundled example's.

Now the generated-JSON path, which is the new code:

```bash
cryozeta predict <the example map> <a FASTA of its chains> check_cz \
  --resolution <as in the example JSON> --contour <as in the example JSON> --dry-run
diff <(python -m json.tool ~/software/CryoZeta/assets/examples/example.json) \
     <(python -m json.tool check_cz/.structbio/inputs/targets.json)
```

- [ ] The generated JSON differs from CryoZeta's own only in ways you expect
      (target name, absolute paths). **The chain entries, counts, and sequence
      strings must match.** This is the check that the JSON generation is
      correct; do not skip it.
- [ ] The real run then produces a model comparable to the bundled example's.
- [ ] A map file that is not MRC/CCP4 is rejected in about a second.

---

## Stage 5 — a colleague, unaided (~1 hour of their time)

The point of the whole exercise. Pick the least command-line-confident person
who is willing.

- [ ] Give them only [PROTOCOL.md](../PROTOCOL.md) and a login. Answer nothing.
- [ ] Watch, and write down every place they hesitate, re-read, or guess.
- [ ] Note anything they type that the protocol did not tell them to.

Every hesitation is a documentation bug. Fix those before a wider rollout: the
protocol is cheaper to change than everyone's habits.

---

## Before the announcement

- [ ] One person other than the author has completed Stage 5 successfully.
- [ ] Every tool the lab actually uses reads `FOUND` in `structbio doctor` on
      the workstation.
- [ ] There is exactly one clone of this repository on the workstation, so
      nobody runs an old copy. `structbio doctor` names the one in use.
- [ ] Someone other than the author knows how to run
      `structbio install-wrappers` after the environment is rebuilt.
- [ ] The lab knows where to send problems, and that
      "the command, the whole output, and `structbio doctor`" is what to send.

## Known gaps to state plainly when you announce it

- Wrapper flags were verified against each project's published interface, not
  against your installed versions, until the stages above are done.
- RFdiffusion on an Ada, Hopper or Blackwell card needs a PyTorch newer than
  the one it pins. That combination works but is not one the RFdiffusion
  authors publish, so check early designs against a known result.
- SLURM support exists but is untested on a real cluster; it is not needed on a
  workstation.
- There is no filtering or scoring step yet: ColabFold will fold designs, but
  deciding which ones are good is still manual.
- Ligands, ions, and chain modifications in CryoZeta need a hand-written JSON
  and `cryozeta predict-json`.

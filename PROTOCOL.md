# structbio: a step-by-step protocol

This guide assumes you have never used a command line. It explains every step,
including the ones experienced users skip. Nothing here can damage the
workstation or anyone else's data if you follow it in order.

Read [Part 1](#part-1--the-terminal-in-ten-minutes) once. After that you will
mostly use [Part 4](#part-4--running-your-first-design) and
[Part 7](#part-7--when-something-goes-wrong).

**Contents**

1. [The terminal in ten minutes](#part-1--the-terminal-in-ten-minutes)
2. [One-time setup](#part-2--one-time-setup)
3. [Checking that it works](#part-3--checking-that-it-works)
4. [Running your first design](#part-4--running-your-first-design)
5. [Where your results are](#part-5--where-your-results-are)
6. [The four tools](#part-6--the-four-tools)
7. [When something goes wrong](#part-7--when-something-goes-wrong)
8. [Long jobs, and sharing the workstation](#part-8--long-jobs-and-sharing-the-workstation)
9. [Glossary](#part-9--glossary)

---

## Part 1 — the terminal in ten minutes

### What the terminal is

The terminal is a window where you type a command, press Enter, and the
computer does it. That is all. It looks unfriendly because it gives no hints,
not because it is dangerous.

### Opening it

- **On a Mac:** press `Cmd` + `Space`, type `Terminal`, press Enter.
- **On Linux:** press `Ctrl` + `Alt` + `T`, or find "Terminal" in the
  applications menu.
- **On the lab workstation from your own computer:** ask whoever runs the
  workstation for the `ssh` command. It will look like
  `ssh yourname@workstation.example.edu`. Type it, press Enter, and give your
  password when asked. **The password will not appear as you type it** — not
  even dots. That is normal. Type it and press Enter.

### The prompt

You will see something like this, with a blinking cursor:

```text
connor@workstation:~$
```

That is the **prompt**. It is the computer saying "ready". You type after it.

In this guide, and in most documentation, a command is shown in a box:

```bash
structbio doctor
```

Type `structbio doctor` and press Enter. If you ever see a line that starts
with `$` or `%`, **do not type that symbol** — it just represents the prompt.

### Copying and pasting

You will copy commands from this document a lot. Do that rather than retyping:
a single wrong character is the most common cause of an error.

- **Mac Terminal:** `Cmd` + `C` to copy, `Cmd` + `V` to paste.
- **Linux terminal:** `Ctrl` + `Shift` + `C` to copy, `Ctrl` + `Shift` + `V` to
  paste. (Plain `Ctrl` + `C` means something else — see below.)

Paste one command at a time and press Enter after each.

### Six things that will save you

| What | How | Why |
| --- | --- | --- |
| Where am I? | `pwd` | Prints the folder you are currently in. |
| What is here? | `ls` | Lists the files and folders here. |
| Go into a folder | `cd foldername` | Moves you into it. |
| Go back up one | `cd ..` | Moves to the folder above. |
| Finish a name for me | Start typing, press `Tab` | Completes file names. Use it constantly; it prevents typos. |
| Stop this! | `Ctrl` + `C` | Cancels whatever is running. |

Press the **up arrow** to bring back the last command you typed. Press it again
for the one before. This is much faster than retyping.

### Folders and paths

A path is the address of a file. There are two kinds:

- **Absolute** — starts from the top: `/home/connor/data/target.pdb`. It means
  the same thing from anywhere.
- **Relative** — starts from where you are: `data/target.pdb`. It means "the
  `data` folder inside the folder I am in right now".

`~` is shorthand for your home folder, so `~/data/target.pdb` is usually the
same as `/home/yourname/data/target.pdb`.

Two rules that prevent most confusion:

1. **Avoid spaces in file and folder names.** Use `my_target.pdb`, not
   `my target.pdb`. If a name does contain a space, wrap the whole path in
   quotes: `"~/my data/target.pdb"`.
2. **To get a file's exact path without typing it,** drag the file from your
   file browser onto the terminal window. The full path appears at the cursor.

### Reading and editing a text file

To look at a file without changing it:

```bash
cat ~/.config/structbio/config.yaml
```

To edit it, use `nano`, which is the friendliest editor:

```bash
nano ~/.config/structbio/config.yaml
```

Inside `nano`:

- Move with the arrow keys. There is no mouse.
- Type normally to make changes.
- `Ctrl` + `O`, then Enter, **saves** the file.
- `Ctrl` + `X` **exits**.

The shortcuts are listed at the bottom of the screen. `^O` there means
`Ctrl` + `O`.

---

## Part 2 — one-time setup

Do this once per computer. It is one command, plus however long the scientific
software takes to install.

### Step 1 — check you have Python

```bash
python3 --version
```

You need `3.10` or higher. If the number is lower, or the command is not found,
stop and ask whoever runs the workstation. Do not try to install Python
yourself.

### Step 2 — get structbio and install it

```bash
cd ~
git clone https://github.com/conmmul/structbio-cli.git
cd structbio-cli
./install.sh
```

Line by line: the first two download structbio into a folder called
`structbio-cli` in your home folder and move you into it. The third does
everything else, and prints what it is doing as it goes.

`./install.sh` is safe to run again at any time. It never deletes anything.

### Step 3 — read what it printed

The output is a short report. Read it once, from the top:

```text
Scanning for installed software...
  rfdiffusion    found      /home/connor/software/RFdiffusion
  proteinmpnn    not found
  colabfold      not found
  cryozeta       found      /home/connor/software/CryoZeta

Configuration     /home/connor/.config/structbio/config.yaml (2 tool(s))
Commands          /home/connor/.local/bin  (colabfold, cryozeta, proteinmpnn, rfdiffusion, structbio)
PATH              added to /home/connor/.zshrc

Checking that each tool can run (this uses the GPU briefly)...
  rfdiffusion    ready            PyTorch 2.4.0 (CUDA 12.1), device: NVIDIA RTX 4090
  cryozeta       not checked      manages its own environment

Open a new terminal, or run:  source /home/connor/.zshrc

Ready to run: rfdiffusion

Try it:
  rfdiffusion monomer 100 my_first_designs -n 2
```

Line by line:

| Line | What it means |
| --- | --- |
| `Scanning...` | Which of the four programs are already on this machine. |
| `Configuration` | The file recording where they are. You may edit it. |
| `Commands` | The short commands it just created for you. |
| `PATH` | The file it edited so your shell can find those commands. |
| `Checking...` | Whether each one can actually run, proved by running code. |
| `Ready to run` | Which ones you can use right now. |

If a line says **`fix: something`**, that is the exact command to type. There
is never more than one.

### Step 4 — open a new terminal

Close this terminal window and open a new one, so it picks up the change to
your PATH. Then check:

```bash
structbio --version
```

A version number means setup worked. If instead you see
`command not found: structbio`, go to
[`command not found`](#command-not-found-rfdiffusion) in Part 7.

If setup said it **could not** set your PATH, it printed a line beginning
`export PATH=`. Send that message to whoever set up the machine: it means your
shell configuration file belongs to someone else, which is not something you
can fix from here.

### Step 5 — only if a tool said `needs attention` or `no environment`

RFdiffusion and ProteinMPNN run from a Conda environment that has to contain a
version of PyTorch built for the graphics card in this machine. That single
requirement causes almost all setup trouble. Setup already told you which of
these three applies; you do not have to work it out.

**`ready`.** Nothing to do. If an environment already worked, setup adopted it
rather than building a second one beside it.

If you know of an environment that works and setup did not look for it, name it
yourself:

```bash
structbio env adopt rfdiffusion --environment the_env_that_works
```

It runs a short check inside that environment, and records it only if that
passes. Nothing is installed, changed or removed. This is always the better
option: a working environment took somebody hours, and rebuilding it risks
losing it.

**`needs attention`.** The environment exists but its PyTorch cannot use the
card. Replace only PyTorch, leaving the rest of the environment alone:

```bash
structbio env repair rfdiffusion
```

**`no environment`.** There is nothing to repair, so build one. This chooses
versions to match the card in this machine:

```bash
structbio env create rfdiffusion
```

Add `--dry-run` to either one first, to see exactly what it will install
without installing it. `env create` asks before starting, and it downloads
several gigabytes, so it takes a while.

If an environment of that name already exists, `--force` does **not** delete
it. It is renamed to `SE3nv-before-1` and left alone, and the command tells you
how to put it back:

```bash
conda rename -n SE3nv-before-1 SE3nv
```

Neither command simply claims success: each finishes by running code on the
graphics card to prove the environment works, and tells you if it does not. You
can repeat that check at any time:

```bash
structbio env verify rfdiffusion
```

If it says the environment cannot be built for this machine, read what it says
carefully and take it to whoever runs the workstation. It means no combination
of versions exists, not that you did something wrong.

ColabFold and CryoZeta manage their own environments, so they are reported as
`not checked`; follow their own setup steps instead.

### Step 6 — install any missing tools

For anything reported as `not found`:

```bash
structbio install proteinmpnn --into ~/software
```

Add `--dry-run` first if you want to see exactly what it will do without doing
it. The command downloads the project, records where it put it, and then prints
that project's own remaining steps — usually creating a software environment
and downloading model weights.

**structbio stops before those steps on purpose.** They differ between
machines, they change with each new release, and some model weights are
licensed for academic use only, which is not a decision a tool should make for
you. Run the printed steps yourself, then:

```bash
structbio setup
structbio doctor
```

If those steps look intimidating, this is the right moment to ask a colleague.
It is a one-time job per tool, and it is the only genuinely fiddly part.

---

## Part 3 — checking that it works

```bash
structbio doctor
```

Read the output from the top:

```text
structbio              0.2.0
  package              /home/connor/structbio-cli/src/structbio
  interpreter          /home/connor/structbio-cli/.venv/bin/python
Python                 OK
Git                    OK
GPU                    OK
                       NVIDIA RTX A6000

RFdiffusion            FOUND
ProteinMPNN            NOT CONFIGURED
```

What the tool statuses mean:

| Status | Meaning | What to do |
| --- | --- | --- |
| `FOUND` | Installed and reachable. | Nothing. You can use it. |
| `CONFIGURED, UNAVAILABLE` | structbio has been told where it is, but it is not really there. | Read the lines underneath: they say why, and the `fix:` lines say what to run. |
| `NOT CONFIGURED` | structbio does not know about it. | `structbio detect`, then `structbio setup`. If still missing, `structbio install TOOL`. |

`CONFIGURED, UNAVAILABLE` is the one people meet most often, usually straight
after a first install, and it always explains itself:

```text
RFdiffusion            CONFIGURED, UNAVAILABLE
                       environment=SE3nv
                       the configured path does not exist: /home/connor/software/RFdiffusion
                       the conda environment 'SE3nv' does not exist
  fix:                 structbio install rfdiffusion --into /home/connor/software
  fix:                 or, if it is installed elsewhere, correct the path and re-run 'structbio detect'
  fix:                 create it with the steps from 'structbio install rfdiffusion --dry-run', or correct 'environment' in the configuration
```

Read it as a sentence: **the folder is not there, and neither is the software
environment.** Nothing is broken — the tool simply has not been installed yet.
The configuration written by `structbio setup` lists all four tools as a
starting point, so an entry existing does not mean the software does.

The three things it distinguishes, and what each means:

| The message says | What actually happened |
| --- | --- |
| `the configured path does not exist` | The tool was never installed, or it is somewhere else. |
| `... exists but does not contain scripts/run_inference.py` | The folder is there but the download was incomplete, or it is a different project. |
| `the conda environment 'X' does not exist` | The code is there but its software environment was never created — usually the install steps were stopped partway. |
| `PyTorch is not installed in the conda environment 'X'` | The environment exists but is empty of the maths library the tool needs. See below. |
| `PyTorch ... is a CPU-only build` | A warning, not an error. It will run, but on the processor rather than the graphics card, so far more slowly. |
| `PyTorch ... was built for CUDA X, which this driver cannot run` | The wrong version was installed for this machine's graphics driver. |

A tool can also show `FOUND, WITH WARNINGS`. **The tool works.** The warning is
about something that may make it slower — usually a PyTorch older than the
graphics card, which still runs but spends time compiling for the card on its
first call. Nothing is stopped, and if your runs are fine you can ignore it.

To settle whether a warning matters, ask the environment itself rather than
believing a version comparison:

```bash
structbio env verify rfdiffusion
```

That runs code on the graphics card. Its answer is the one to trust; anything
structbio says from reading files is only a hint.

### PyTorch problems

RFdiffusion and ProteinMPNN both need PyTorch built for the graphics card in
this machine. Getting that wrong is the single most common installation
problem. First ask the environment itself:

```bash
structbio env verify rfdiffusion
```

If it reports `CUDA none` or `no GPU`, the installed PyTorch has no
graphics-card support at all. Repair it without rebuilding anything:

```bash
structbio env repair rfdiffusion
```

It shows what it will change and asks before doing it, replacing only PyTorch
and the graph library that goes with it. Add `--dry-run` to look first. When it
finishes it checks the graphics card again, so you will know straight away.

#### RFdiffusion is a special case

RFdiffusion's environment is built from a file, `env/SE3nv.yml`, that fixes
which version of everything gets installed — including an old PyTorch that the
rest of the checkout is built against. **Installing a newer PyTorch there will
break it**, so structbio refuses to, and tells you the right repair instead:

```text
RFdiffusion    SE3nv: PyTorch 1.9.1.post3, CPU-only build
               this environment is defined by RFdiffusion's env/SE3nv.yml, so
               structbio will not install PyTorch into it
               repair it with: conda install -n SE3nv -c pytorch -c nvidia pytorch=1.9 cudatoolkit=11.1
               or rebuild it: conda env remove -n SE3nv, then re-create it from RFdiffusion's env/SE3nv.yml
```

This happens because `env/SE3nv.yml` lists the general-purpose conda channels
before the PyTorch one, so conda often picks a version of PyTorch built without
graphics-card support. It is a quirk of the upstream file, not a mistake you
made.

#### If PyTorch cannot see the graphics card

`structbio env verify rfdiffusion` saying `CUDA none` and `no GPU` means the
PyTorch that was installed has no graphics-card support built in. It is a very
common outcome of the standard install, and it is why runs are slow or fail.

Fix it without rebuilding anything:

```bash
structbio env repair rfdiffusion
```

This replaces only PyTorch, and for RFdiffusion the graph library that goes
with it, leaving everything else in the environment alone. Then check it:

```bash
structbio env verify rfdiffusion
```

You want to see your card's name and no `FAILED` lines.

#### When no fix will work

On a recent graphics card you may instead see:

```text
this machine's GPU is Blackwell, such as the RTX 50 series, which needs CUDA
12.8 or newer, but RFdiffusion's env/SE3nv.yml pins CUDA 11.1. No PyTorch
install can bridge that: the pinned version has no kernels for this card
```

This is not something to keep trying to fix. RFdiffusion pins software from
2021, which contains no instructions for graphics cards designed after it. Stop
and speak to whoever runs the workstation: the options are an environment built
with newer versions, which somebody has to test, or running on an older card.
Reinstalling will not help, and neither will any command in this guide.

Run the `fix:` line that matches. If you have the tool installed somewhere
else, `structbio detect` will find it and `structbio setup` will record
it, which is quicker than editing the configuration by hand.

`GPU NOT FOUND` on a laptop is normal. On the workstation it means something is
wrong, and this command says what:

```bash
structbio gpu
```

It reports the graphics card, its driver, and — when it cannot see one — the
reason:

| What it says | What it means |
| --- | --- |
| `nvidia-smi was not found` | The tool that reports on the card is not on this terminal's PATH, or the driver is not installed. |
| `exited with code N` | The driver is installed but unhappy; the message after the code comes from the driver itself. |
| `did not answer within 20 seconds` | The driver is busy or stuck. Try again; if it persists, report it. |
| `ran but listed no GPUs` | The driver works but sees no card, which usually means a hardware or configuration fault. |

First check whether the card is visible outside structbio at all:

```bash
nvidia-smi
```

If that works but `structbio gpu` does not, the two are running with different
PATHs. Point structbio at it directly:

```bash
STRUCTBIO_NVIDIA_SMI=/usr/bin/nvidia-smi structbio gpu
```

If that fixes it, add that line to the shell start-up file `structbio setup`
named when it set your PATH. If `nvidia-smi` does not work either, the
problem is the driver, not structbio — that is one for whoever runs the
workstation.

To see where structbio thinks everything is:

```bash
structbio config
```

---

## Part 4 — running your first design

### The shape of every command

```text
toolname  whattodo  [inputs]  outputfolder  [options]
```

For example:

```bash
rfdiffusion monomer 150 my_first_designs -n 5
```

Read that as: *RFdiffusion, design monomers, 150 residues each, put them in a
folder called `my_first_designs`, and make 5 of them.*

**The last thing before the options is always the output folder**, and its name
becomes the prefix of the files inside it. `my_first_designs` will contain
`my_first_designs_0.pdb`, `my_first_designs_1.pdb`, and so on.

### Always dry-run first

Add `--dry-run` and nothing will be created or executed. You will see exactly
what would happen:

```bash
rfdiffusion monomer 150 my_first_designs -n 5 --dry-run
```

Get into this habit. It costs one second and catches mistakes before they cost
GPU hours.

### Then run it for real

First, move to a sensible place to work:

```bash
cd ~
mkdir -p projects/my_first_project
cd projects/my_first_project
```

Then run the same command without `--dry-run`:

```bash
rfdiffusion monomer 150 my_first_designs -n 5
```

The tool's own output appears as it runs, so you can watch progress. When it
finishes you will see:

```text
Done. Results are in /home/connor/projects/my_first_project/my_first_designs
```

Two things structbio guarantees:

- **It will never write over an earlier result.** If the output folder already
  exists and has anything in it, the run stops and asks you to pick another
  name. Your previous work is safe.
- **Your input files are never modified.** They are only read.

---

## Part 5 — where your results are

```bash
ls my_first_designs
```

```text
my_first_designs_0.pdb
my_first_designs_1.pdb
my_first_designs_2.pdb
...
```

The results sit at the top level of the folder, ready to open in PyMOL or
ChimeraX.

There is also a hidden folder called `.structbio` recording what produced them.
`ls` does not show hidden folders unless you ask:

```bash
ls -a my_first_designs
ls my_first_designs/.structbio
```

| File | What it holds |
| --- | --- |
| `command.txt` | The exact command that ran. |
| `config.yaml` | Every setting used, including defaults. |
| `metadata.json` | Tool, computer, versions, software commit, inputs, outputs, timings. |
| `environment.txt` | Python, CUDA and GPU details. |
| `stdout.log` | Everything the tool printed. |
| `stderr.log` | Warnings and errors. |

This is what makes a run reproducible six months later, and it is what to look
at first when something went wrong. For a summary:

```bash
structbio status my_first_designs
```

Copy the whole folder, hidden part included, when you archive or share results:

```bash
cp -r my_first_designs ~/backups/
```

---

## Part 6 — the four tools

Run any command with `--help` to see all of its options, for example
`rfdiffusion binder --help`.

### RFdiffusion — designing backbones

Makes new protein shapes. It produces structures, not sequences.

```bash
# A new protein of 150 residues, 10 of them
rfdiffusion monomer 150 my_monomers -n 10

# A symmetric assembly: 400 residues total across 4 identical subunits
rfdiffusion symmetry c4 400 my_tetramers -n 10

# A binder against chain B of a target, touching specific residues
rfdiffusion binder target.pdb 100 my_binders --chain B --hotspots B30,B33,B34

# Variations on an existing structure
rfdiffusion partial start.pdb 10 my_variants -n 20
```

For `symmetry`, the number is the **total** across all subunits and must divide
by the subunit count: `c4` needs a multiple of 4. structbio will tell you the
nearest workable numbers if you get it wrong.

For `binder`, structbio reads the target's residue numbering out of the PDB
file itself, so you only name the chain. If the file has more than one chain
and you do not say which, it will ask rather than guess.

### ProteinMPNN — designing sequences

Takes a backbone and works out amino-acid sequences that would fold into it.

```bash
# 8 sequences for one structure
proteinmpnn design my_backbone.pdb 8 my_sequences

# Only let part of the protein change
proteinmpnn design 7kdp.pdb 8 my_sequences --chains A --designable A:697-749

# Every backbone in a folder at once
proteinmpnn design my_monomers 4 my_sequences
```

Positions use the numbering in your PDB file: `A:697-749` means chain A,
residues 697 to 749, exactly as written in the file. Before it runs, it prints
which residues may change and which are fixed. **Read that list.** It is the
single easiest thing to get wrong, and the check is there so you catch it.

Without `--designable`, every residue of the selected chains may change.

### ColabFold — predicting structures

Folds sequences so you can check whether a design came out as intended.

```bash
colabfold predict my_sequences my_folds --msa-mode single_sequence
```

**Important for unpublished work:** by default ColabFold sends your sequences
to a public server to build alignments. structbio warns you every time a run
would do that. `--msa-mode single_sequence` keeps them on this machine, and for
designed sequences it is usually the better choice anyway, because a brand-new
design has no natural relatives for an alignment to find.

### CryoZeta — building models into cryo-EM maps

```bash
cryozeta predict emd_44046.map.gz chains.fasta my_model --resolution 2.99 --contour 0.3
```

You give it the map, a FASTA file containing every chain in the complex, the
output folder, and the map's resolution and recommended contour level.
structbio writes CryoZeta's own input file for you.

Two things to know:

- If a sequence is written with only the letters A, C, G and T, structbio
  cannot tell whether it is DNA or a short peptide — both are valid — so it
  stops and asks. Answer with `--dna chain_C`, `--rna chain_C`, or
  `--protein chain_C`.
- For complexes above roughly 2800 residues, add `--large` to use CryoZeta's
  large-complex pipeline. structbio warns you when it thinks you need it.

### Chaining them together

The output of one becomes the input of the next:

```bash
rfdiffusion monomer 150 my_backbones -n 20
proteinmpnn design my_backbones 4 my_sequences
colabfold predict my_sequences my_folds --msa-mode single_sequence
```

Design 20 backbones, give each 4 candidate sequences, then fold all 80 to see
which ones came out looking like what you asked for.

---

## Part 7 — when something goes wrong

Nothing here breaks anything. Work through it calmly.

### `command not found: rfdiffusion`

Your terminal does not know where the command is. Almost always this means the
terminal was open before setup ran: close it, open a new one, and try again.

If that does not help, run the installer again — it is safe to repeat, and it
puts the commands back on your PATH:

```bash
cd ~/structbio-cli && ./install.sh
```

To fix only the terminal you are in right now, without changing anything:

```bash
eval "$(~/structbio-cli/.venv/bin/structbio shell-init)"
```

### `No such command '...'`

You are running an older copy of structbio. This happens when more than one
copy exists on the machine. Check which one is running:

```bash
structbio doctor
```

The `package` line near the top tells you. If it is not the copy you expect,
run `structbio install-wrappers` from the copy you do want.

### `Refusing to write into the existing non-empty folder`

You have used that output name before. This is structbio protecting your
earlier results. Pick a new name:

```bash
rfdiffusion monomer 150 my_designs_v2 -n 5
```

If you are sure you no longer want the old results, look inside first, then
remove it deliberately:

```bash
ls my_designs
rm -r my_designs
```

`rm -r` cannot be undone. There is no recycle bin. Check the folder name twice.

### `... is not available on this machine`

The same situation as `CONFIGURED, UNAVAILABLE`, met while trying to run
something. The error lists the reasons and then a `To fix it:` section:

```text
Error: RFdiffusion is not available on this machine.
  the configured path does not exist: /home/connor/software/RFdiffusion
  the conda environment 'SE3nv' does not exist

To fix it:
  structbio install rfdiffusion --into /home/connor/software
  or, if it is installed elsewhere, correct the path and re-run 'structbio detect'
  create it with the steps from 'structbio install rfdiffusion --dry-run', or correct 'environment' in the configuration
```

Nothing was created and nothing was harmed; the run stopped before it started.
See the table in [Part 3](#part-3--checking-that-it-works) for what each reason
means.

If you do not intend to use that tool at all, remove its entry from
`~/.config/structbio/config.yaml` — see
[reading and editing a text file](#reading-and-editing-a-text-file) — and it
will stop being reported.

### `Chain 'X' is absent` or `Selection includes absent residue numbers`

Your selection does not match the structure file. structbio uses the numbering
in the file itself and will not renumber anything. Look at what is actually
there:

```bash
grep "^ATOM" target.pdb | cut -c22-27 | sort -u | head -30
```

That prints the chain letters and residue numbers present. Adjust your
selection to match. Do not edit the PDB to fit the command.

### `Cannot tell whether ... is protein, DNA, or RNA`

A sequence uses only A, C, G and T, which are valid as both amino acids and
nucleotides. Say which it is: `--dna chain_C`, `--rna chain_C`, or
`--protein chain_C`.

### The run started, then failed partway

Look at the log:

```bash
cat my_designs/.structbio/stderr.log
```

The last ten lines are usually the informative part. `CUDA out of memory` means
the design was too big for the GPU, or somebody else is using it — see
[Part 8](#part-8--long-jobs-and-sharing-the-workstation).

### It seems stuck

Some stages are genuinely quiet for minutes: downloading model weights the
first time, building alignments, or compiling GPU code. Give it ten minutes
before worrying. To check it is really working, open a second terminal:

```bash
nvidia-smi
```

If a process is using the GPU, it is running. `Ctrl` + `C` stops it if you need
to.

### Asking for help

Send these three things. They almost always contain the answer:

1. The exact command you typed.
2. The complete output, not a summary.
3. The result of `structbio doctor`.

To capture all of it to a file you can attach:

```bash
structbio doctor > ~/help.txt 2>&1
cat my_designs/.structbio/stderr.log >> ~/help.txt
```

---

## Part 8 — long jobs, and sharing the workstation

### Check the GPU before you start

```bash
nvidia-smi
```

The table shows each GPU, how much memory is in use, and whose processes are
running. If a colleague is using GPU 0, use another one:

```bash
rfdiffusion monomer 150 my_designs -n 50 --gpu 1
```

Or let structbio choose the least busy one:

```bash
rfdiffusion monomer 150 my_designs -n 50 --gpu auto
```

### Jobs that outlive your terminal

A big run can take hours. If you close the terminal or your `ssh` connection
drops, the job dies with it. Use `tmux`, which keeps a session alive on the
workstation:

```bash
tmux new -s mydesigns
```

Run your command inside it as normal. Then press `Ctrl` + `B`, release both,
and press `D`. This **detaches**: the job keeps running and you get your prompt
back. You can now close the terminal or disconnect entirely.

To come back later, log in again and:

```bash
tmux attach -t mydesigns
```

Everything will be exactly as you left it, including output printed while you
were away. To list your sessions: `tmux ls`. To finish one for good, attach to
it and type `exit`.

### Being a good neighbour

- Check `nvidia-smi` before starting something large.
- Start with a small run — `-n 1` — to confirm the settings, then scale up.
- Big design runs are best started at the end of the day.
- Results belong in your own project folder, not in shared space.

---

## Part 9 — glossary

| Term | Meaning |
| --- | --- |
| **Terminal / shell** | The window where you type commands. |
| **Prompt** | The text before your cursor, showing the computer is ready. |
| **Command** | An instruction you type, then Enter. |
| **Option / flag** | An extra setting starting with `-` or `--`, such as `--gpu 1`. |
| **Argument** | A value you give a command, such as a file name. |
| **Path** | The address of a file or folder. |
| **Absolute path** | A path from the top of the disk, starting with `/`. |
| **Relative path** | A path starting from where you currently are. |
| **`~`** | Your home folder. |
| **Directory** | Another word for folder. |
| **`ssh`** | Logging in to another computer over the network. |
| **`tmux`** | Keeps a session running after you disconnect. |
| **Virtual environment (venv)** | A private Python area, so one program's packages cannot disturb another's. |
| **PATH** | The list of folders your terminal searches for commands. |
| **Dry run** | A preview that changes nothing. |
| **GPU** | The graphics card that does the heavy computation. |
| **Backbone** | A protein structure without sequence assigned. |
| **Contig** | RFdiffusion's notation for which parts to keep and which to design. |
| **MSA** | Multiple sequence alignment: related natural sequences, used to improve structure prediction. |
| **Hotspot** | A target residue you want a designed binder to engage. |
| **YAML** | The plain-text format of structbio's configuration files. |

---

## Where to go next

- [README](README.md) — the condensed version of everything here.
- [docs/installation.md](docs/installation.md) — installation in more detail.
- [docs/rfdiffusion.md](docs/rfdiffusion.md),
  [docs/proteinmpnn.md](docs/proteinmpnn.md),
  [docs/colabfold.md](docs/colabfold.md),
  [docs/cryozeta.md](docs/cryozeta.md) — every option each tool accepts,
  including the ones the short commands do not expose.
- [docs/troubleshooting.md](docs/troubleshooting.md) — a longer error list.
- [examples/](examples/README.md) — configuration files for runs too detailed
  for a short command.

# Installation and environments

Install `structbio` in its own lightweight Python environment. Wrapped tools
remain in their vendor-recommended environments; `structbio` uses `conda run -n`
instead of relying on an activated interactive shell.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Put installation paths in `~/.config/structbio/config.yaml`. A lab manager can
instead set `STRUCTBIO_LAB_CONFIG` to a shared read-only YAML file. Run
`structbio doctor` after any environment or driver change.

Do not install RFdiffusion, ProteinMPNN, or CryoZeta merely by installing this
package. Follow each upstream project's instructions and license terms.

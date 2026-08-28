"""Built-in tool backend registry."""

from __future__ import annotations

from structbio.tools.base import ToolBackend


def get_backends() -> dict[str, ToolBackend]:
    from structbio.tools.colabfold import ColabFoldBackend
    from structbio.tools.cryozeta import CryoZetaBackend
    from structbio.tools.proteinmpnn import ProteinMPNNBackend
    from structbio.tools.rfdiffusion import RFDiffusionBackend

    backends: list[ToolBackend] = [
        RFDiffusionBackend(),
        ProteinMPNNBackend(),
        ColabFoldBackend(),
        CryoZetaBackend(),
    ]
    return {backend.name: backend for backend in backends}


def get_backend(name: str) -> ToolBackend:
    try:
        return get_backends()[name]
    except KeyError as exc:
        raise ValueError(f"No backend is implemented for tool {name!r}") from exc

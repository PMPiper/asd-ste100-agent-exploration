"""Text embeddings with an on-disk cache."""

import hashlib
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Turns texts into L2-normalized vectors."""

    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns one unit-length row per text."""
        ...


class SentenceTransformerEmbedder:
    """Embeds texts as documents with a sentence-transformers model.

    Uses CUDA in bfloat16 when available, and the CPU in float32 otherwise.
    No query instruction is added. Inputs longer than `max_tokens` are
    truncated, and each truncation is logged.
    """

    def __init__(self, model_name: str, max_tokens: int, batch_size: int = 8) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.truncated = 0
        self._batch_size = batch_size
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
            logger.info("Embedding on %s", torch.cuda.get_device_name(0))
        else:
            device, dtype = "cpu", torch.float32
            logger.warning("CUDA is not available; embedding on the CPU.")
        self.device = device
        self._model = SentenceTransformer(
            model_name, device=device, model_kwargs={"torch_dtype": dtype}
        )
        self._model.max_seq_length = max_tokens

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns one unit-length float32 row per text."""
        limit = self._model.max_seq_length
        tokenizer = self._model.tokenizer
        for text in texts:
            n_tokens = len(tokenizer(text, truncation=False)["input_ids"])
            if n_tokens > limit:
                self.truncated += 1
                logger.warning(
                    "Truncated a text of %d tokens to %d: %.60s...",
                    n_tokens,
                    limit,
                    text,
                )
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return vectors.float().cpu().numpy()


class CachedEmbedder:
    """Wraps an embedder with one .npy file per text.

    The file name is a hash of the model name and the text, so a re-run only
    embeds texts that are new.
    """

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.name = inner.name
        self._inner = inner
        self._cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def truncated(self) -> int | None:
        """Texts truncated while embedding in this process, if known."""
        return getattr(self._inner, "truncated", None)

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.name}\0{text}".encode()).hexdigest()
        return self._cache_dir / f"{digest}.npy"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns one unit-length row per text, from the cache if present."""
        missing = sorted({t for t in texts if not self._path(t).exists()})
        if missing:
            for text, vector in zip(missing, self._inner.embed(missing)):
                np.save(self._path(text), vector)
        return np.stack([np.load(self._path(text)) for text in texts])

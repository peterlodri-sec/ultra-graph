"""Composed ternary transformer: pre-norm blocks and a ``GPT`` with generation.

``GPT`` wires an embedding, a shared :class:`~ultragraph.nn.RoPE`, N pre-norm
:class:`TransformerBlock` s, a final norm, and a ternary output head. Every
weight matrix is a dense ternary :class:`~ultragraph.core.Tree`; the whole model
is one byte-graph. ``generate`` decodes autoregressively with a per-layer
KV-cache — and because activation quantization is per token, a cached step is
byte-for-byte the same as the full forward pass at that position.
"""

import numpy as np

from .autograd import Tensor
from .core import Embedding, Tree
from .nn import MultiHeadAttention, RMSNorm, RoPE


class TransformerBlock:
    """Pre-norm block: ``x = x + attn(norm1(x)); x = x + ff(norm2(x))``."""

    def __init__(self, d_model, n_heads, mlp_ratio=4, rope=None, name="block"):
        self.name = name
        self.norm1 = RMSNorm(d_model, name=f"{name}.n1")
        self.attn = MultiHeadAttention(d_model, n_heads, causal=True, rope=rope, name=f"{name}.attn")
        self.norm2 = RMSNorm(d_model, name=f"{name}.n2")
        h = d_model * int(mlp_ratio)
        self.ff1 = Tree.dense(d_model, h, f"{name}.ff1", act="relu")
        self.ff2 = Tree.dense(h, d_model, f"{name}.ff2", act="none")

    def __call__(self, x, cache=None):
        if cache is not None:
            a, new_cache = self.attn(self.norm1(x), cache=cache)
            x = x + a
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.ff2.forward(self.ff1.forward(self.norm2(x)))
        return (x, new_cache) if cache is not None else x

    def parameters(self):
        ps = list(self.norm1.parameters()) + list(self.attn.parameters())
        ps += list(self.norm2.parameters()) + list(self.ff1.parameters()) + list(self.ff2.parameters())
        return ps

    def requantize(self):
        self.attn.requantize()
        self.ff1.requantize()
        self.ff2.requantize()


class GPT:
    """A ternary GPT: token embedding + RoPE + N pre-norm blocks + ternary head."""

    def __init__(self, vocab, d_model, n_layers, n_heads, max_len=512, mlp_ratio=4, name="gpt",
                 threaded=False):
        assert (d_model // n_heads) % 2 == 0, "d_head (d_model // n_heads) must be even for RoPE"
        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_len = int(max_len)
        self.mlp_ratio = int(mlp_ratio)
        self.name = name
        self.threaded = bool(threaded)
        self.embed = Embedding(vocab, d_model, name=f"{name}.embed")
        self.rope = RoPE(d_model // n_heads, max_len=max_len, name=f"{name}.rope")
        self.blocks = [
            TransformerBlock(d_model, n_heads, mlp_ratio, rope=self.rope, name=f"{name}.b{i}")
            for i in range(self.n_layers)
        ]
        self.norm = RMSNorm(d_model, name=f"{name}.nf")
        self.head = Tree.dense(d_model, vocab, f"{name}.head", act="none")

    def __call__(self, ids):
        """ids: int array ``[T]`` or ``[B, T]`` -> logits ``[..., T, vocab]``.

        When ``self.threaded=True`` and input is batched (B > 1), forward
        passes run in parallel across threads for free-threading Python builds.
        """
        ids_arr = np.asarray(ids)
        if self.threaded and ids_arr.ndim == 2 and ids_arr.shape[0] > 1:
            return self._forward_threaded(ids_arr)
        return self._forward_sequential(ids_arr)

    def _forward_sequential(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head.forward(self.norm(x))

    def _forward_threaded(self, ids):
        """Run batched forward pass with one thread per sequence (inference only)."""
        import concurrent.futures

        bsz = ids.shape[0]

        def _run_one(row_idx):
            x = self.embed(ids[row_idx:row_idx + 1])
            for blk in self.blocks:
                x = blk(x)
            return self.head.forward(self.norm(x))

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(bsz, 8)) as ex:
            results = list(ex.map(_run_one, range(bsz)))

        from .autograd import cat as cat_tensors
        return cat_tensors(results, 0)

    def parameters(self):
        ps = list(self.embed.parameters())
        for b in self.blocks:
            ps += b.parameters()
        ps += list(self.norm.parameters()) + list(self.head.parameters())
        return ps

    def requantize(self):
        for b in self.blocks:
            b.requantize()
        self.head.requantize()

    # -- persistence ----------------------------------------------------------
    def save(self, path):
        """Save the fp32 parameters, tagged with this model's hyper-parameters so
        :meth:`load_model` can rebuild the architecture. Restore with :meth:`load` onto
        a ``GPT`` built with the same hyper-parameters (byte-exact for inference after
        re-quantize), or with :meth:`load_model` (reads the embedded hp)."""
        from .io import save_params
        hp = {"vocab": self.vocab, "d_model": self.d_model, "n_layers": self.n_layers,
              "n_heads": self.n_heads, "max_len": self.max_len, "mlp_ratio": self.mlp_ratio}
        save_params([self], path, meta={"format": "params", "hp": hp})

    def load(self, path):
        """Load parameters saved by :meth:`save` into this model in place, then
        re-quantize the ternary weights. Returns ``self``."""
        from .io import load_params
        load_params([self], path)
        return self

    def n_params(self):
        """Total trainable scalar count (fp32 masters + biases + norm gains + embed)."""
        return int(sum(p.data.size for p in self.parameters()))

    def _dense_trees(self):
        """Every ternary Tree in the model, in a stable order (save/load rely on it)."""
        ts = []
        for b in self.blocks:
            ts += [b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.ff1, b.ff2]
        ts.append(self.head)
        return ts

    def _fp32_gains(self):
        """The tiny full-precision tensors (embedding + norm gains), named + ordered."""
        g = [("embed", self.embed.table)]
        for i, b in enumerate(self.blocks):
            g += [(f"n1_{i}", b.norm1.gain), (f"n2_{i}", b.norm2.gain)]
        g.append(("nf", self.norm.gain))
        return g

    def save_deployed(self, path, packed=True):
        """Save the *deployed* model: ternary weight bytes (bit-packed at ~1.6
        bits/weight when ``packed``) plus the small fp32 pieces (embedding, norm
        gains, biases). No fp32 masters — smaller, and inference-only. Restore with
        :meth:`load_deployed`; forward then runs straight from the ternary bytes."""
        import json

        from .pack import pack_ternary
        hp = {"vocab": self.vocab, "d_model": self.d_model, "n_layers": self.n_layers,
              "n_heads": self.n_heads, "max_len": self.max_len, "mlp_ratio": self.mlp_ratio}
        meta = {"format": "deployed", "hp": hp, "packed": bool(packed), "trees": []}
        arrays = {}
        for i, t in enumerate(self._dense_trees()):
            entry = {"w_scale": float(t.w_scale), "shape": list(t.wq.shape), "packed": bool(packed)}
            arrays[f"t{i}_wq"] = pack_ternary(t.wq) if packed else t.wq
            arrays[f"t{i}_bias"] = t.adhoc["bias"].data
            meta["trees"].append(entry)
        for name, ten in self._fp32_gains():
            arrays[f"f_{name}"] = ten.data
        with open(path, "wb") as fh:
            np.savez(fh, __meta__=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load_deployed(cls, path):
        """Rebuild an inference-only GPT from a :meth:`save_deployed` checkpoint (no
        fp32 masters). Its forward runs from the ternary bytes — byte-exact versus
        the trained model. Returns the model."""
        import json

        from .pack import unpack_ternary
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            hp = meta["hp"]
            m = cls(hp["vocab"], hp["d_model"], hp["n_layers"], hp["n_heads"],
                    max_len=hp["max_len"], mlp_ratio=hp["mlp_ratio"])
            for i, (t, entry) in enumerate(zip(m._dense_trees(), meta["trees"])):
                shape = tuple(entry["shape"])
                if entry.get("packed"):
                    t.wq = unpack_ternary(data[f"t{i}_wq"], int(np.prod(shape))).reshape(shape)
                else:
                    t.wq = np.array(data[f"t{i}_wq"])
                t.w_scale = float(entry["w_scale"])
                t.adhoc["bias"] = Tensor(np.array(data[f"t{i}_bias"]))
                t.adhoc["w_master"] = None       # -> deployed path in Tree.forward
            for name, ten in m._fp32_gains():
                ten.data[...] = data[f"f_{name}"]
            return m

    # -- generation -----------------------------------------------------------
    def generate(self, prompt, n_new, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, stop=None, seed=None, stream=False):
        """Autoregressive decode with a per-layer KV-cache. ``temperature <= 0`` is
        greedy; ``top_k`` / ``top_p`` (nucleus) truncate the sampling distribution;
        ``repetition_penalty`` (> 1) discourages already-seen tokens; ``stop`` is a
        token id (or iterable of ids) that halts generation once produced.

        Default returns ``prompt + generated`` token ids. With ``stream=True`` returns
        a generator that yields each *new* token id as it is produced."""
        ids = [int(t) for t in np.asarray(prompt, dtype=np.int64).reshape(-1)]
        if not ids:
            raise ValueError("prompt must be non-empty")
        stops = set() if stop is None else ({int(stop)} if isinstance(stop, int) else {int(s) for s in stop})
        it = self._stream(ids, int(n_new), temperature, top_k, top_p, float(repetition_penalty), stops, seed)
        if stream:
            return it
        ids.extend(it)
        return ids

    def generate_batch(self, prompts, n_new, temperature=1.0, top_k=None, top_p=None,
                       repetition_penalty=1.0, stop=None, seed=None):
        """Batched decode over ``prompts`` ``[B, T]`` (equal-length) with one shared
        per-layer KV-cache. Each sequence samples independently and stops on its own
        ``stop`` token. Returns a list of ``B`` id lists (prompt + generated, truncated
        at the first stop token)."""
        prompts = np.asarray(prompts, dtype=np.int64)
        if prompts.ndim == 1:
            prompts = prompts[None, :]
        if prompts.ndim != 2:
            raise ValueError("generate_batch expects equal-length prompts shaped [B, T]")
        b = prompts.shape[0]
        rng = np.random.default_rng(seed)
        stops = set() if stop is None else ({int(stop)} if isinstance(stop, int) else {int(s) for s in stop})
        caches = [{"k": None, "v": None} for _ in self.blocks]
        logits = self._decode(prompts, caches)          # [B, T, vocab]
        outs = [list(prompts[i]) for i in range(b)]
        hist = [list(prompts[i]) for i in range(b)]
        done = [False] * b
        for _ in range(int(n_new)):
            nxt = np.empty((b, 1), dtype=np.int64)
            for i in range(b):
                nxt[i, 0] = self._sample(logits[i, -1], temperature, top_k, top_p,
                                         repetition_penalty, hist[i], rng)
            for i in range(b):
                tok = int(nxt[i, 0])
                hist[i].append(tok)
                if not done[i]:
                    outs[i].append(tok)
                    if tok in stops:
                        done[i] = True
            if all(done):
                break
            logits = self._decode(nxt, caches)           # feed one token per row -> [B, 1, vocab]
        return outs

    def _stream(self, prompt_ids, n_new, temperature, top_k, top_p, rep_pen, stops, seed):
        rng = np.random.default_rng(seed)
        caches = [{"k": None, "v": None} for _ in self.blocks]
        history = list(prompt_ids)
        logits = self._decode(np.array(prompt_ids, dtype=np.int64)[None, :], caches)  # prime
        for _ in range(n_new):
            nxt = self._sample(logits[0, -1], temperature, top_k, top_p, rep_pen, history, rng)
            history.append(nxt)
            yield nxt
            if nxt in stops:
                return
            logits = self._decode(np.array([[nxt]], dtype=np.int64), caches)

    def _decode(self, ids, caches):
        x = self.embed(ids)
        for b, c in zip(self.blocks, caches):
            x, nc = b(x, cache=c)
            c["k"], c["v"] = nc["k"], nc["v"]
        return self.head.forward(self.norm(x)).data  # [B, T, vocab] fp32

    @staticmethod
    def _sample(logits, temperature, top_k, top_p, rep_pen, history, rng):
        logits = logits.astype(np.float64)  # fresh copy; safe to mutate in place
        if rep_pen > 0.0 and rep_pen != 1.0 and history:
            seen = np.unique(np.asarray(history, dtype=np.int64))
            seen = seen[(seen >= 0) & (seen < logits.shape[-1])]
            pos = logits[seen] > 0
            logits[seen[pos]] /= rep_pen          # CTRL-style: shrink positive, grow negative
            logits[seen[~pos]] *= rep_pen
        if temperature <= 0:
            return int(np.argmax(logits))
        logits = logits / float(temperature)
        if top_k:
            k = min(int(top_k), logits.shape[-1])
            logits = np.where(logits < np.sort(logits)[-k], -np.inf, logits)
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        if top_p and 0.0 < top_p < 1.0:
            order = np.argsort(p)[::-1]
            csum = np.cumsum(p[order])
            cut = int(np.searchsorted(csum, top_p)) + 1  # smallest nucleus with cumprob >= top_p
            keep = order[:cut]
            masked = np.zeros_like(p)
            masked[keep] = p[keep]
            p = masked / masked.sum()
        return int(rng.choice(len(p), p=p))

    def summary(self):
        """Print a formatted table of model layers with param counts and byte sizes.

        Example:
            model.summary()
            # Layer           Kind          Shape        Params    Bytes (ternary)
            # embed           Embedding     256x128      32768     0
            # b0.attn.q       dense         32x128       4096      5
            # ...
        """
        rows = []
        rows.append(("embed", "Embedding", f"{self.vocab}x{self.d_model}", self.vocab * self.d_model, 0))
        for i, b in enumerate(self.blocks):
            for name in ("attn.q", "attn.k", "attn.v", "attn.o"):
                t = getattr(b.attn, f"w{name[-1]}")
                w = t.adhoc["w_master"]
                rows.append((f"b{i}.{name}", "dense", f"{t.out_dim}x{t.in_dim}",
                             w.data.size, int(w.data.size * 1.6 / 8)))
            for j, ff in enumerate((b.ff1, b.ff2)):
                w = ff.adhoc["w_master"]
                rows.append((f"b{i}.ff{j + 1}", "dense", f"{ff.out_dim}x{ff.in_dim}",
                             w.data.size, int(w.data.size * 1.6 / 8)))
        rows.append(("head", "dense", f"vocab x {self.d_model}",
                     self.head.adhoc["w_master"].data.size,
                     int(self.head.adhoc["w_master"].data.size * 1.6 / 8)))

        header = f"{'Layer':<16} {'Kind':<12} {'Shape':<16} {'Params':>10} {'Bytes':>10}"
        sep = "-" * len(header)
        lines = [header, sep]
        for name, kind, shape, params, byte_size in rows:
            lines.append(f"{name:<16} {kind:<12} {shape:<16} {params:>10} {byte_size:>10}")
        lines.append(sep)
        total_params = sum(r[3] for r in rows)
        total_bytes = sum(r[4] for r in rows)
        lines.append(f"{'TOTAL':<16} {'':12} {'':16} {total_params:>10} {total_bytes:>10}")
        print("\n".join(lines))

    # -- one-line training ----------------------------------------------------
    def fit(self, x, y, epochs=1, lr=0.1, batch_size=None, seed=0, quiet=False):
        """Train the model on ``x`` → ``y`` pair with SGD.

        ``x`` and ``y`` are int token-id arrays (``[seq]`` or ``[batch, seq]``).
        Returns a list of per-epoch losses.

        Example:
            loss = model.fit(x_ids, y_ids, epochs=10, lr=0.01)
        """
        from .optim import SGD

        np.random.seed(seed)
        x_arr = np.asarray(x, dtype=np.int64)
        y_arr = np.asarray(y, dtype=np.int64)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
            y_arr = y_arr[None, :]

        opt = SGD(self, lr=lr, accum_steps=1)
        losses = []

        use_rich = False
        if not quiet:
            try:
                from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
                use_rich = True
            except ImportError:
                use_rich = False

        if use_rich and not quiet:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                          TextColumn("loss={task.fields[loss]:.4f}")) as progress:
                task = progress.add_task("training", total=epochs, loss=float("inf"))
                for epoch in range(epochs):
                    opt.zero_grad()
                    logits = self(x_arr)
                    loss = logits.cross_entropy(y_arr)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.data))
                    progress.update(task, advance=1, loss=float(loss.data))
        else:
            for epoch in range(epochs):
                opt.zero_grad()
                logits = self(x_arr)
                loss = logits.cross_entropy(y_arr)
                loss.backward()
                opt.step()
                losses.append(float(loss.data))
                if not quiet:
                    print(f"epoch {epoch + 1}/{epochs}  loss={loss.data:.4f}")

        return losses

    # -- unified save ---------------------------------------------------------
    def save_model(self, path, deployed=False):
        """Save the model.

        - ``deployed=False`` (default): save fp32 masters for training resume.
        - ``deployed=True``: save bit-packed ternary bytes for inference-only.

        Use ``GPT.load_model(path)`` to restore.

        Example:
            model.save_model("model.ugm")              # training checkpoint
            model.save_model("model.ugm", deployed=True)  # inference-only
        """
        if deployed:
            self.save_deployed(path)
        else:
            self.save(path)

    @classmethod
    def load_model(cls, path):
        """Load a model saved by ``save_model()``. Auto-detects format.

        Example:
            model = GPT.load_model("model.ugm")
        """
        import json

        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"])) if "__meta__" in data else None
        if meta is not None:
            if meta.get("format") == "deployed" or "trees" in meta:
                return cls.load_deployed(path)
            if "hp" in meta:
                hp = meta["hp"]
                instance = cls(hp["vocab"], hp["d_model"], hp["n_layers"],
                               hp["n_heads"], max_len=hp["max_len"],
                               mlp_ratio=hp["mlp_ratio"])
                instance.load(path)
                return instance
        raise ValueError(
            f"{path}: checkpoint has no architecture metadata, so the model shape "
            "cannot be inferred (n_heads and max_len are not recoverable from weight "
            "shapes). Rebuild the model with its hyper-parameters and call "
            "GPT(...).load(path), or re-save it with GPT.save to embed them."
        )


class Mesh:
    """A graph of minds — a learned soft (or top-k) mixture of *full* models.

    Each expert maps token ids ``[B, T]`` to logits ``[B, T, vocab]`` (a `GPT`, or
    anything with that signature + ``parameters`` / ``requantize``). A small ternary
    router mixes them *per sequence* from a pooled token embedding::

        logits = sum_e  gate(ids)_e * expert_e(ids)

    This is `nn.MoE`'s routing lifted to whole networks — the mesh of the Low-Bit
    stories, made real. Fully differentiable; the router and every expert train
    together. (Mixture is over logits, so generation is per-expert, not joint.)
    """

    def __init__(self, experts, vocab, d_router=32, top_k=None, name="mesh"):
        assert experts, "a mesh needs at least one expert"
        self.experts = list(experts)
        self.n_experts = len(self.experts)
        if top_k is None:
            self.top_k = self.n_experts
        else:
            self.top_k = int(top_k)
            if not (1 <= self.top_k <= self.n_experts):
                raise ValueError(f"top_k must be in [1, {self.n_experts}], got {top_k}")
        self.vocab = int(vocab)
        self.name = name
        self.r_embed = Embedding(vocab, d_router, name=f"{name}.rembed")
        self.router = Tree.dense(d_router, self.n_experts, f"{name}.router", act="none")

    def _gate(self, ids):
        pooled = self.r_embed(ids).mean(axis=-2)                 # [B, d_router]
        gate = self.router.forward(pooled).softmax(axis=-1)      # [B, n_experts]
        if self.top_k < self.n_experts:
            g = gate.data
            idx = np.argpartition(-g, self.top_k - 1, axis=-1)[..., : self.top_k]
            mask = np.zeros_like(g)
            np.put_along_axis(mask, idx, 1.0, axis=-1)
            gate = gate * Tensor(mask)
            gate = gate / (gate.sum(axis=-1, keepdims=True) + 1e-9)
        return gate

    def __call__(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        two_d = ids.ndim == 1
        if two_d:
            ids = ids[None, :]
        b = ids.shape[0]
        gate = self._gate(ids)                                   # [B, n_experts]
        out = None
        for e in range(self.n_experts):
            logits = self.experts[e](ids)                        # [B, T, vocab]
            term = logits * gate[:, e : e + 1].reshape(b, 1, 1)  # broadcast over [T, vocab]
            out = term if out is None else out + term
        if two_d:
            out = out.reshape(out.shape[1], out.shape[2])
        return out

    def parameters(self):
        ps = list(self.r_embed.parameters()) + list(self.router.parameters())
        for e in self.experts:
            if hasattr(e, "parameters"):
                ps.extend(e.parameters())
        return ps

    def requantize(self):
        self.router.requantize()
        for e in self.experts:
            r = getattr(e, "requantize", None)
            if callable(r):
                r()

    def n_params(self):
        return int(sum(p.data.size for p in self.parameters()))

    def generate(self, prompt, n_new, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, stop=None, seed=None):
        """Joint autoregressive decoding across the mixture. At each step every
        expert's cached next-token logits are mixed by the (re-evaluated) router gate,
        then sampled once. Requires GPT-like experts (with ``.blocks`` / ``._decode``).
        Returns ``prompt + generated`` ids."""
        if not all(hasattr(e, "blocks") and hasattr(e, "_decode") for e in self.experts):
            raise TypeError("Mesh.generate requires GPT-like experts (with .blocks and ._decode)")
        ids = [int(t) for t in np.asarray(prompt, dtype=np.int64).reshape(-1)]
        if not ids:
            raise ValueError("prompt must be non-empty")
        stops = set() if stop is None else ({int(stop)} if isinstance(stop, int) else {int(s) for s in stop})
        rng = np.random.default_rng(seed)
        caches = [[{"k": None, "v": None} for _ in e.blocks] for e in self.experts]
        per = [self.experts[e]._decode(np.array(ids, dtype=np.int64)[None, :], caches[e])
               for e in range(self.n_experts)]                     # each [1, T, vocab]
        for _ in range(int(n_new)):
            gate = self._gate(np.array(ids, dtype=np.int64)[None, :]).data[0]  # [n_experts]
            mixed = sum(gate[e] * per[e][0, -1] for e in range(self.n_experts))  # [vocab]
            nxt = GPT._sample(mixed, temperature, top_k, top_p, repetition_penalty, ids, rng)
            ids.append(nxt)
            if nxt in stops:
                break
            per = [self.experts[e]._decode(np.array([[nxt]], dtype=np.int64), caches[e])
                   for e in range(self.n_experts)]
        return ids

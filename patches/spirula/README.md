# Strict prior sentinel patch proposal

Target: `harry7557558/spirula-studio` commit
`bddda193ee09f03ea1b0aad44b5c6b96f81e249a`.

This patch is a proposal, **not installed or trainer-certified**. It keeps CPU
RGB resizing unchanged, uses conservative validity for CPU geometry resizing,
and rejects GPU geometry samples with any positive-weight invalid tap. The
shared weight check also gates its scatter helper. It can reduce prior coverage;
that tradeoff needs real trainer tests, particularly at multiscale boundaries.

Reproduce from the SphereAlign root:

```sh
python scripts/geometry_fetch_spike.py
python scripts/geometry_spirula_sentinel.py
```

The second command uses `g++ -std=c++17` on exact pinned function bodies and
generates the same patch, without changing the downloaded source snapshot.
It verifies CPU math and GPU weight logic on the host. It does not establish
CUDA/Slang backward behavior, normal sign, range semantics, all warp paths or
equivalence to another Spirula binary. Do not enable profiles on its basis.

Upstream source context and extracted function bodies are covered by Spirula's
GNU GPL version 3 license; see [the pinned LICENSE](https://github.com/harry7557558/spirula-studio/blob/bddda193ee09f03ea1b0aad44b5c6b96f81e249a/LICENSE).
The proposed additions are offered under GPL-3.0 for integration with that
source. Full upstream sources remain only in the ignored `.work` snapshot.

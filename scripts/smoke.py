"""Import/compile smoke test.

There is no unit-test suite here, and app.py sys.exit()s at import when the
generated assets/ payload is absent (it is not committed), so this deliberately
stops short of importing app. What it does cover is enough to catch the thing
dependency bumps actually break: whether the package still imports against the
installed torch / Pillow / open_clip, and whether every module still parses.
"""

import importlib
import pathlib
import py_compile
import sys

# Python puts THIS file's directory (scripts/) on sys.path, not the repo
# root, so orchid_clip is not importable unless we add the root ourselves.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

mods = [
    "orchid_clip",
    "orchid_clip.abstain",
    "orchid_clip.embedder",
    "orchid_clip.genus",
    "orchid_clip.references",
]
for m in mods:
    importlib.import_module(m)
    print(f"import ok   {m}")

for f in sorted(ROOT.rglob("*.py")):
    if ".venv" in f.parts:
        continue
    py_compile.compile(str(f), doraise=True)
    print(f"compiles    {f}")

import PIL
import torch

print(f"torch       {torch.__version__}")
print(f"pillow      {PIL.__version__}")


def check_references() -> None:
    """Round trip of the demo's extra references on made-up embeddings.

    Species 0 has two looks, so its centroid sits between them, and species 1
    resembles the first look. A new photo of that first look goes to species 1
    on centroids alone, and to species 0 with photo-level references or with a
    form centroid built from the first look.
    """
    import tempfile

    import numpy as np

    from orchid_clip import references as R

    rng = np.random.default_rng(0)
    n_sp, per, dims = 6, 40, [16, 16]
    d = sum(dims)
    axes = np.linalg.qr(rng.normal(size=(d, d)))[0]
    look_a, look_b = axes[n_sp], axes[n_sp + 1]
    centers = [look_a] * (per // 2) + [look_b] * (per // 2)  # species 0
    centers += [look_a + 0.8 * axes[1]] * per  # species 1
    centers += [axes[k] for k in range(2, n_sp) for _ in range(per)]
    rows = np.stack([c + 0.02 * rng.normal(size=d) for c in centers]).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    sp = np.repeat(np.arange(n_sp), per)
    names = [f"Genus species{k}" for k in range(n_sp)]
    gallery = np.stack([R.tower_centroid(rows[sp == k], dims) for k in range(n_sp)])
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    query = (look_a + 0.02 * rng.normal(size=d)).astype(np.float32)
    query /= np.linalg.norm(query)
    first_look = list(range(per // 2))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        shuffle = rng.permutation(len(rows))  # the builder must regroup by species
        built = R.build_photo_rows(rows[shuffle], sp[shuffle], n_sp, dims=16)
        built.pop("order")
        np.savez(tmp / "photo_rows.npz", ckpt_sha256_prefix="abc", **built)
        forms = R.build_form_refs(
            rows, {names[0]: first_look, names[3]: [3 * per]}, dims
        )
        assert list(forms["binomials"]) == [names[0]], "one-photo species not skipped"
        np.savez(tmp / "form_refs.npz", ckpt_sha256_prefix="abc", **forms)

        assert R.load_photo_rows(tmp / "absent.npz", names, "abc") is None
        photo = R.load_photo_rows(tmp / "photo_rows.npz", names, "abc")
        form = R.load_form_refs(tmp / "form_refs.npz", names, "abc")

        plain = R.species_sims(gallery, query)
        assert int(plain.argmax()) == 1, "setup: centroids alone should pick species 1"
        with_rows = R.species_sims(gallery, query, rows=photo)
        assert int(with_rows.argmax()) == 0, "photo-level references did not engage"
        with_form = R.species_sims(gallery, query, forms=form)
        assert int(with_form.argmax()) == 0, "form centroid did not engage"
        assert np.allclose(np.delete(with_form, 0), np.delete(plain, 0))

        # a species with no photos in the file keeps its centroid score
        keep = built["species_index"] != 4
        part = {
            k: (v[keep] if k in ("rows", "species_index") else v)
            for k, v in built.items()
        }
        np.savez(tmp / "part.npz", ckpt_sha256_prefix="abc", **part)
        some = R.species_sims(
            gallery, query, rows=R.load_photo_rows(tmp / "part.npz", names, "abc")
        )
        assert np.isclose(some[4], plain[4]) and np.isclose(some[0], with_rows[0])

        # stale files must fail loudly, and the good file above must not
        bad_order = dict(built, species_index=built["species_index"][::-1].copy())
        np.savez(tmp / "bad_order.npz", ckpt_sha256_prefix="abc", **bad_order)
        bad_range = dict(built, species_index=built["species_index"].copy())
        bad_range["species_index"][-1] = n_sp
        np.savez(tmp / "bad_range.npz", ckpt_sha256_prefix="abc", **bad_range)
        for label, fn in [
            (
                "another checkpoint",
                lambda: R.load_photo_rows(tmp / "photo_rows.npz", names, "xyz"),
            ),
            (
                "a shorter species list",
                lambda: R.load_photo_rows(tmp / "photo_rows.npz", names[:-1], "abc"),
            ),
            (
                "a longer species list",
                lambda: R.load_photo_rows(
                    tmp / "photo_rows.npz", names + ["Genus extra"], "abc"
                ),
            ),
            (
                "a species outside the list",
                lambda: R.load_photo_rows(tmp / "bad_range.npz", names, "abc"),
            ),
            (
                "ungrouped rows",
                lambda: R.load_photo_rows(tmp / "bad_order.npz", names, "abc"),
            ),
            (
                "form: another checkpoint",
                lambda: R.load_form_refs(tmp / "form_refs.npz", names, "xyz"),
            ),
            (
                "form: unknown species",
                lambda: R.load_form_refs(tmp / "form_refs.npz", names[1:], "abc"),
            ),
        ]:
            try:
                fn()
            except ValueError:
                continue
            raise AssertionError(f"loader accepted a file built for {label}")
    print("references  round trip ok")


check_references()
print("smoke OK")

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

        # photos added later: species 2 gets a second look that leans toward
        # species 3. The old rows keep their bytes and the new look is found.
        lean = 0.65 * axes[2] + 0.76 * axes[3]
        new_rows = np.stack([lean + 0.02 * rng.normal(size=d) for _ in range(per)])
        new_rows = (new_rows / np.linalg.norm(new_rows, axis=1, keepdims=True)).astype(
            np.float32
        )
        query2 = (lean / np.linalg.norm(lean)).astype(np.float32)
        before = R.species_sims(gallery, query2, rows=photo)
        assert int(before.argmax()) == 3, "setup: species 3 should win before adding"
        more = R.append_photo_rows(built, new_rows, np.full(per, 2))
        is_new = more.pop("is_new")
        assert float(more.pop("kept_energy")) > 0.9, "projection should fit these rows"
        assert is_new.sum() == per and (more["species_index"][is_new] == 2).all()
        assert np.array_equal(more["rows"][~is_new], built["rows"]), "old rows changed"
        assert np.array_equal(more["species_index"][~is_new], built["species_index"])
        np.savez(tmp / "more.npz", ckpt_sha256_prefix="abc", **more)
        after = R.species_sims(
            gallery, query2, rows=R.load_photo_rows(tmp / "more.npz", names, "abc")
        )
        assert int(after.argmax()) == 2, "added reference photos did not engage"
        assert np.allclose(np.delete(after, 2), np.delete(before, 2), atol=1e-6)
        wide = R.append_photo_rows(built, new_rows.astype(np.float64), np.full(per, 2))
        step = np.abs(wide["rows"].astype(int) - more["rows"].astype(int))
        assert step.max() <= 1 and (step > 0).mean() < 0.01, "float64 rows differ"
        # a look outside what the projection was fit on is flagged, not hidden
        odd = np.tile(axes[n_sp + 2], (per, 1)).astype(np.float32)
        assert (
            float(R.append_photo_rows(built, odd, np.full(per, 2))["kept_energy"]) < 0.5
        )
        for label, said, args in [
            ("rows of another size", "dimensions", (new_rows[:, :-1], np.full(per, 2))),
            ("a species outside the list", "outside", (new_rows, np.full(per, n_sp))),
            ("no rows", "no rows", (new_rows[:0], np.zeros(0, np.int64))),
        ]:
            try:
                R.append_photo_rows(built, *args)
            except ValueError as err:
                assert said in str(err), f"{label}: unclear error {err!r}"
                continue
            raise AssertionError(f"append_photo_rows accepted {label}")

        # a species can hold two extra centroids; the closer one counts
        second = list(range(per // 2, per))
        both = R.concat_form_refs(
            forms, R.build_form_refs(rows, {names[0]: second}, dims)
        )
        np.savez(tmp / "both.npz", ckpt_sha256_prefix="abc", **both)
        query_b = (look_b / np.linalg.norm(look_b)).astype(np.float32)
        one = R.species_sims(gallery, query_b, forms=form)
        two = R.species_sims(
            gallery, query_b, forms=R.load_form_refs(tmp / "both.npz", names, "abc")
        )
        assert two[0] > one[0] + 0.1, "second extra centroid did not engage"
        assert np.allclose(np.delete(two, 0), np.delete(one, 0))
        assert np.isclose(
            R.species_sims(gallery, query, forms=form)[0],
            R.species_sims(
                gallery, query, forms=R.load_form_refs(tmp / "both.npz", names, "abc")
            )[0],
        ), "first extra centroid lost when a second was added"

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


def check_abstain() -> None:
    """The abstain's two conditions: top1-top2 margin, and lead over other genera."""
    import json
    import math
    import tempfile

    import numpy as np

    from infer import score_and_format
    from orchid_clip import references as R
    from orchid_clip.abstain import AbstainConfig, decide, load_config

    ids, sc = ["Aa bb", "Aa cc", "Bb dd"], [0.80, 0.70, 0.60]
    off = AbstainConfig("margin", 0.05, 0.9)
    on = AbstainConfig("margin", 0.05, 0.9, True, 0.25)
    d = decide(ids, sc, off)
    assert (d.tier, d.species, d.reason) == ("species", "Aa bb", ""), d
    d = decide(ids, sc, on, 0.30)
    assert (d.species, d.other_genus_lead, d.reason) == ("Aa bb", 0.30, ""), d
    d = decide(ids, sc, on, 0.25)  # the threshold itself passes
    assert d.species == "Aa bb", d
    d = decide(ids, sc, on, 0.20)
    assert (d.tier, d.species, d.reason) == ("genus_only", None, "other_genus_lead"), d
    d = decide(ids, [0.80, 0.78, 0.10], on, 0.70)  # margin fails first
    assert (d.species, d.reason) == (None, "signal"), d
    assert decide(ids, sc, on, math.inf).species == "Aa bb"
    for bad, text in [(None, "passed no measured other_genus_lead"), (math.nan, "NaN")]:
        try:
            decide(ids, sc, on, bad)
        except ValueError as e:
            assert text in str(e), e
        else:
            raise AssertionError(f"decide accepted other_genus_lead={bad}")

    g = R.genus_ids(["Aa bb", "Aa cc", "Bb dd", "Cc ee"])
    assert g.tolist() == [0, 0, 1, 2], g
    assert abs(R.other_genus_lead(np.array([0.8, 0.79, 0.6, 0.65]), g) - 0.15) < 1e-12
    assert R.other_genus_lead(np.array([0.8, 0.7]), np.array([0, 0])) == math.inf
    try:
        R.other_genus_lead(np.array([0.8, 0.7]), g)
    except ValueError as e:
        assert "differ" in str(e), e
    else:
        raise AssertionError("other_genus_lead accepted mismatched shapes")

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/genus_abstain.json"
        blob = {"signal": "margin", "tau": 0.05, "target_precision": 0.9}
        json.dump(blob, open(path, "w"))
        assert load_config(path).other_genus_lead is None
        json.dump(dict(blob, other_genus_lead=0.25), open(path, "w"))
        assert load_config(path).other_genus_lead == 0.25

    # End to end through the result card: same query, the second condition alone
    # turns a named species into a genus-only answer and says why.
    bank = np.array([[1.0, 0.0], [0.8, 0.6], [0.95, 0.3122499]], np.float32)
    names = ["Aa bb", "Aa cc", "Bb dd"]
    q = np.array(
        [1.0, 0.0], np.float32
    )  # scores 1.0, 0.8, 0.95: margin 0.05, lead 0.05
    v_off, _, _ = score_and_format(q, 3, bank, names, off, {})
    v_on, _, _ = score_and_format(q, 3, bank, names, on, {})
    assert v_off.startswith("### ✅ Aa bb"), v_off
    assert v_on.startswith("### 🌿 Genus") and "another genus" in v_on, v_on
    print("abstain     two conditions ok")


def check_remote_assets() -> None:
    """Large asset files named in model.json: pinned revision, sha256-checked."""
    import hashlib
    import tempfile

    from orchid_clip import references as R

    sha = lambda b: hashlib.sha256(b).hexdigest()
    rev = "a" * 40
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        assets, hub = tmp / "assets", tmp / "hub"
        assets.mkdir()
        hub.mkdir()
        (assets / "model.json").write_text("{}")
        (assets / "small.json").write_text("small")
        (hub / "big.npz").write_bytes(b"big file")
        (hub / "other.npz").write_bytes(b"other file")
        calls = []

        def fake(repo_id, filename, repo_type, revision, subfolder):
            calls.append((repo_id, filename, repo_type, revision, subfolder))
            return str(hub / filename)

        def meta(**over):
            remote = {
                "repo_id": "someone/refs",
                "repo_type": "dataset",
                "revision": rev,
                "subfolder": "v1",
                "files": {"big.npz": sha(b"big file"), "other.npz": sha(b"other file")},
            }
            remote.update(over)
            return {"remote_assets": remote}

        def refused(m, text, **kw):
            try:
                R.resolve_assets(assets, m, **kw)
            except ValueError as e:
                assert text in str(e), e
            else:
                raise AssertionError(f"resolve_assets accepted a config with: {text}")

        # No remote_assets: the directory itself, and nothing is downloaded.
        assert R.resolve_assets(assets, {}, download=fake) == assets and not calls

        # Both files downloaded at the pinned revision; small files linked through.
        out = R.resolve_assets(assets, meta(), download=fake)
        assert out != assets
        assert sorted(f.name for f in out.iterdir()) == [
            "big.npz",
            "model.json",
            "other.npz",
            "small.json",
        ]
        assert (out / "big.npz").read_bytes() == b"big file"
        assert (out / "small.json").read_text() == "small"
        assert calls == [
            ("someone/refs", "big.npz", "dataset", rev, "v1"),
            ("someone/refs", "other.npz", "dataset", rev, "v1"),
        ], calls

        # A downloaded file whose content is not the recorded one is refused.
        bad = meta()
        bad["remote_assets"]["files"]["other.npz"] = sha(b"something else")
        refused(bad, "other.npz from someone/refs@aaaaaaaa has sha256", download=fake)
        # So are a branch name, a short sha, a malformed hash and missing keys.
        refused(meta(revision="main"), "full commit sha", download=fake)
        refused(meta(revision="a" * 39), "full commit sha", download=fake)
        refused(meta(files={"big.npz": "abc"}), "is not a sha256", download=fake)
        refused(meta(files={}), "files is empty", download=fake)
        no_rev = meta()
        del no_rev["remote_assets"]["revision"]
        refused(no_rev, "lacks ['revision']", download=fake)

        # A local copy with the recorded hash is used and never downloaded ...
        (assets / "big.npz").write_bytes(b"big file")
        calls.clear()
        out = R.resolve_assets(assets, meta(), download=fake)
        assert [c[1] for c in calls] == ["other.npz"], calls
        assert (out / "big.npz").resolve() == (assets / "big.npz").resolve()
        # ... and a local copy with other content is refused, not replaced.
        (assets / "big.npz").write_bytes(b"stale file")
        refused(meta(), "(local copy) has sha256", download=fake)
    print("assets      remote files ok")


check_references()
check_abstain()
check_remote_assets()
print("smoke OK")

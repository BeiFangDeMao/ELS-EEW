# Third-party source notices

Original upstream source, license files and attribution are retained. Copying a research experiment does not transfer authorship of its baseline architectures.

| Source | Local directory | Retained license |
| --- | --- | --- |
| [SeisT, Sen Li et al.](https://github.com/senli1073/SeisT) | `discussion/model_comparison/upstream/SeisT-main/` | MIT, `LICENSE` |
| [E3WS, Pablo E. Lara et al.](https://github.com/PabloELara/E3WS) | `discussion/model_comparison/upstream/E3WS/` | CC BY 4.0, `LICENSE` |

The SeisT repository supplies the MagNet, BAZ-Network and distPT implementations used or inspected by the comparison code. Inclusion of source does not imply every baseline was trained or included in the final paper. In particular, keeping distPT source is not evidence of a completed distPT experiment.

Provenance snapshots: `discussion/model_comparison/source_manifest.json` and `E3WS_source_manifest.json`. These upstream manifests describe the original downloaded snapshots; this code-only archive intentionally omits their non-code assets. They are not manifests of the current archive. Use root `SOURCE_MANIFEST.json` for this archive's file checksums.

Third-party Python packages, Android SDK binaries, external dataset files and installed tooling were not bundled. Local table-export scripts referring to Codex-installed JavaScript tooling retain those references for provenance; that tooling is not distributed here.

No blanket license has been added to the original project code. Select the author's intended license before public release; retain upstream license requirements separately.

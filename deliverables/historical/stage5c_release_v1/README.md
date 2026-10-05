# Stage 5C acceptance package (release v1) - historical

**Status: historical, superseded. This is not the current state of the project.**

It is the manual acceptance package recorded in June 2026 for the first production posture,
`foodflow-10doc-release-v1`, kept unchanged as the record of that acceptance. The current evidence
is in [`deliverables/evidence/`](../../evidence/README.md).

| Item | In this package | Current |
|------|-----------------|---------|
| Release | `foodflow-10doc-release-v1` | `foodflow-10doc-release-v2` |
| Corpus / index | 215 chunks, fingerprint `bf3df0d4...`, `data/04_index_backup_10docs_215chunks` (a manually provisioned archive, no longer part of the release path) | 216 chunks, fingerprint `aa1005e1...`, `data/04_index_production`, built from the repository |
| CLI output | pre-stage-2C fields (`risk_level`, citations that were really retrieved fragments) | `assessment_status`, `risk_floor`, `answer_provenance`, `citations` vs `retrieved_materials` |
| Customer text | model drafts shown after a text check that has since been replaced | customer-output policy with deterministic templates for critical categories |
| Execution | real OpenAI calls (M01-M07), Docker Streamlit, local Windows machine | deterministic stubs on the real pipeline, no credential (see the current evidence) |

Known properties of these files, left as they are on purpose:

- the `cli/*.json` captures were written through the wrong console code page, so their Cyrillic text is garbled (mojibake);
- the screenshots show the UI as it was before the customer-output policy and the current risk presentation;
- `evidence/evidence_manifest.json` records commit `b5739c9...` and the report names `32ec3f7...`; both are SHAs from before the history cleanup and do not exist on `main` any more. The manifest's paths were relocated to this directory; its digests are unchanged;
- `M00`-`M09` are the scenario numbers of that package and are unrelated to the `S01`-`S14` scenarios of the current evidence.

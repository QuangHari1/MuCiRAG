# Dataset layout

`dataset/3gpp/` is the immutable original GSMA/paper-compatible corpus used by
the default configuration. Do not place enhanced files, overlays, or alternate
embedding profiles in that tree.

```text
dataset/
├── teleqna/
│   └── TeleQnA.json
├── 3gpp/                              # Original baseline only
│   ├── marked/Rel-18/
│   ├── Metadata/Rel-18/
│   ├── Chunk/Rel-18/
│   ├── Reference/Rel-18/
│   ├── Embeddings/Rel-18/
│   └── embedding_selections/
└── 3gpp-enhanced/                     # Separate experimental namespace
    ├── raw/Rel-18/
    ├── selections/
    ├── profiles/
    │   └── <selection-id>/
    │       ├── Metadata/Rel-18/
    │       ├── Chunk/Rel-18/
    │       ├── Reference/Rel-18/
    │       └── Embeddings/Rel-18/<selection-id>/
    └── overlays/image-descriptions/Rel-18/
```

Each enhanced profile owns its processed artifacts. Its embedding manifest's
`source_chunk_directory` must point to the matching profile's `Chunk/Rel-18`
directory. Enhanced selections live only under `3gpp-enhanced/selections/` and
the image-description overlay stays separate because it enriches baseline
context after retrieval rather than supplying vectors.

The default `MuCiRAG/config.toml` points only to `dataset/3gpp/`. An
enhanced experiment must use an explicit, dedicated configuration profile; it
must never overwrite the baseline paths or manifest.

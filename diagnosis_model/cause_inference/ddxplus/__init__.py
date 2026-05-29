"""DDXPlus cross-dataset adaptation of FaCE-R.

Text-only smoke / generalization probe: maps patient summary -> global_emb and
evidence tokens -> lesion_embs so the existing Phase 1 retrieval + Phase 2 CEAH
machinery applies unchanged. See README.md for the end-to-end command flow.
"""

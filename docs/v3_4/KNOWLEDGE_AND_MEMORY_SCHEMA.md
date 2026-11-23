# V3.4 Schema

`v34_entities`, `v34_evidence`, `v34_assertions`, `v34_conflicts`, and `v34_experiences` are additive SQLite tables. Assertion rows retain recorded time, effective interval, evidence IDs, verification status, and revocation timestamp. Relations must point to valid entities and assertions; no evidence means `UNVERIFIED`.

# Language V2 data provenance

The registry is fail-closed. FineWeb-2, CulturaX and Wikimedia Wikipedia are candidates only and remain `REVIEW_REQUIRED` until a current dataset revision, subset, license, attribution, privacy and permitted-use review is recorded locally. This round performs zero network reads and downloads zero bytes.

The only immediately usable data are locally generated toy-world observations and manually supplied fixtures. They are tagged as program-generated rather than public corpus data. The existing 30-page Python knowledge base remains retrieval-only and is not silently promoted to neural training data.

`DataSourceRegistry` records revocation. Revocation prevents future use and supports rebuild manifests; it cannot erase information from an already trained checkpoint, so such a checkpoint must be quarantined and dispositioned manually.

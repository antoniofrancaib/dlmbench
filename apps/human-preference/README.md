# OWT human preference collection

Public study: https://owt-human-preference.antoniofrancaib.workers.dev

The hosted study continues the original 13-source OWT experiment: 12 trained generators and the OWT train reference, 1,024 samples per source, and unchanged 35-word prefixes. It retains anonymous A/B preferences, skips, audited Undo, balanced source-pair exposure and sample selection, and browser-side pending-response retries.

## Hosting and persistence

Cloudflare Workers serves the application and a single SQLite-backed Durable Object owns the journal. Every event is appended to the primary journal and mirrored into the separate D1 database before success is acknowledged. An interrupted mirror leaves the original event pending; the next request retries the mirror and the same vote request ID cannot be counted twice. Historical event bytes, their hashes, participant IDs, import exclusions and sample provenance are preserved.

Private backups outside Cloudflare run every six hours in `antoniofrancaib/owt-human-preference-backups`. The export-only token cannot import or mutate data. Each backup validates the hash chain and rejects any rewrite of an earlier snapshot. Git history retains older snapshots; the private repository also contains the original complete recovery ZIP. GitHub schedules can be delayed, so immediate durability comes from the two Cloudflare stores.

The initial production import contained 561 events, 323 response rows and 301 scored judgments: 18 skips and four flagged imported exclusions. The preserved journal endpoint was `54a3c62ba71d95fe5113e6a9581db5e01ac157fb72a2482c1bb89cb4c86622d0`. Original events were verified byte for byte, and every original Elo score matched the retained Python reference to floating-point precision. Synthetic test responses belong only to staging.

## Participant experience

No name, email or account is requested. A signed, secure cookie identifies the browser. Rankings and source names are withheld until the participant selects **Finish and see rankings**, which permanently finishes that browser's voting session. The owner dashboard stays live throughout. Reopening the same browser resumes its pending comparison; clearing cookies or switching browsers creates a new participant. IDs do not prove distinct people, and cookies from the retired temporary hostname do not transfer to this site.

The hosted cohort is tagged `hosted-2026-11`; original records retain their earlier presentation and name-access flags. The new sampler uses recorded 128-bit seeds and xoshiro128** with rejection sampling, preserving the original exposure-balancing rules. All old assignments remain unchanged. Counts measure saved judgments, not a guarantee of statistical precision. Existing LLM correlations remain exploratory because those judges evaluated longer samples.

## Build and operate

Requirements: Node.js with npm, Python 3 for recovery and off-provider backup verification, and access to the existing Cloudflare account. From this folder:

```sh
npm ci
npm run build
npm test
npx wrangler deploy --env ''
```

`npm run build:data` verifies the frozen study and sample hashes before generating the server-only data module. Neither bank files nor raw journals are served as public static assets. Do not modify `study/study.json` or sample banks after collection starts.

Production uses the top-level Wrangler configuration. Staging is a separate Worker and Durable Object namespace and uses a separate key in the backup database. Use `--env staging` for test deployments. Never run synthetic voting tests against production. `scripts/verify-host.mjs` enforces the staging hostname. The storage test uses temporary local databases and exercises a backup outage plus full runtime restart.

Secrets `ADMIN_TOKEN`, `SESSION_SECRET`, and `BACKUP_TOKEN` are separate and are not committed. Set them using `wrangler secret bulk` from a private file. Preserve the session secret across deployments. The authenticated `/api/admin/export` endpoint returns JSON including original `event_raw`, or CSV with `?format=csv`; `/api/admin/stats` supports the owner dashboard. `/api/admin/import` only initializes an empty collection or retries the identical original import; conflicting data are rejected.

For recovery, decompress the latest private snapshot, validate its chain with `backup.py`, and submit its `event_raw` array to a new empty collection. Verify response counts and every original raw event before routing users to it. The original recovery ZIP includes frozen banks and the reference app. Keep one active production writer and never merge independent collections by overwriting sequence numbers.

The source code belongs to `antoniofrancaib/dLMbench`; raw participant responses and secrets are excluded from this public repository. The copied historical Python implementation in `reference/server.py` is a validation reference, not the running service.

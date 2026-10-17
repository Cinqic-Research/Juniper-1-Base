# Drive-only recovery drill — 2026-10-10

Sources used: Google Drive `juniper1/drills/drive-only-2026-10-10/` only (manifest v2, `remote_listing_v2.json`).
GitHub, Hugging Face, RunPod volumes and the HDD copy were not used for the restore.

1. `restore_verify_flowbox.txt`: Drive → new FLOWBOX directory. Every byte verified against the Drive manifest; both
   Git bundles clone and `git fsck` at their recorded HEADs; 4/4 representative corpus shards match the inventory;
   tokenizer round trips; checkpoint verified by the restored code. (The first attempt failed: the root-level exclude
   of the completion marker also excluded the checkpoint's own `COMPLETE`; fixed and republished as v2.)
2. The resumed computation was moved to an A100 (FLOWBOX has 16 GB RAM, too little for the CPU resume). The restored
   checkpoint, data prefix, tokenizer and restored `juniper-1-train` checkout were copied from the FLOWBOX restore to
   the pod by rsync. The pod copy of the code is that restored checkout (HEAD e942abec, verified by `git fsck` in step 1;
   `git rev-parse` on the pod returned empty, so the pod-side HEAD is not separately recorded). `uploaded_subset.sha256`
   holds the 15 manifest entries that were uploaded; the operator's `sha256sum -c` over that subset on the pod printed
   no mismatch (`SUBSET_OK`, recorded in the session log, not as a file here). The bitwise-equal resume is consistent with
   intact bytes but is indirect evidence for it.
3. `resume_a100_summary.txt`: resume from `ckpt-000000025-b2150637`, run updates 26–27 with the reference contract.
   Pre-registered criterion: batch fingerprints exact and losses within 1 %. Result: **PASS**. Fingerprints, learning
   rates and losses are bitwise equal to the reference run.

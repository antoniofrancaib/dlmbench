from dlmbench.generators.registry import catalog_sha256, generators


def test_catalog_has_exact_paper_generator_closure() -> None:
    rows = {row["id"]: row for row in generators()}
    expected_naive = {
        "lm1b_topk_iid_k32", "lm1b_mirror_5000", "lm1b_periodic_k_64",
        "lm1b_phrase_bank_1000", "owt_topk_iid_k64", "owt_mirror_5000",
        "owt_periodic_k_400", "owt_phrase_bank_5000",
    }
    expected_trained = {
        "lm1b_flm_1024_nfe", "lm1b_fmlm_1_nfe", "lm1b_fmlm_4_nfe",
        "lm1b_fmlm_32_nfe", "lm1b_mdlm_1024_nfe", "lm1b_duo_1024_nfe",
        "lm1b_langflow_128_nfe", "lm1b_langflow_1024_nfe", "owt_flm_1024_nfe",
        "owt_fmlm_32_nfe", "owt_ar_base", "owt_mdlm_1024_nfe",
        "owt_sedd_1024_nfe", "owt_duo_base_1024_nfe", "owt_langflow_1024_nfe",
        "owt_sdtt_mdlm_teacher_1024_nfe", "lm1b_rdlm_1000_nfe",
        "owt_v2_cobit_m_256_nfe", "owt_v2_elf_m_64_nfe",
        "owt_v2_plaid_1024_nfe", "owt_v2_replaid_nosc_ddpm_1024_nfe",
    }
    assert set(rows) == expected_naive | expected_trained
    assert len(rows) == 29
    assert catalog_sha256()
    assert rows["owt_v2_plaid_1024_nfe"]["support"] == "unsupported"
    assert rows["owt_v2_replaid_nosc_ddpm_1024_nfe"]["support"] == "samples-only"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]\n", argv[0]);
    LOG("\n  MTP (multi-token prediction) example:\n");
    LOG("\n    %s -m model.gguf --spec-type draft-mtp --spec-draft-n-max 2 -ngl 99 -c 2048 -b 2048 -ub 512 -npp 128,256 -ntg 128,256 -npl 1\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    // Use LLAMA_EXAMPLE_SPECULATIVE so that --spec-type / --spec-draft-n-max /
    // --spec-draft-ngl and related flags are parsed and visible in --help.
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE, print_usage)) {
        return 1;
    }

    int is_pp_shared   = params.is_pp_shared;
    int is_tg_separate = params.is_tg_separate;

    std::vector<int> n_pp = params.n_pp;
    std::vector<int> n_tg = params.n_tg;
    std::vector<int> n_pl = params.n_pl;

    // MTP is single-sequence only
    const bool use_mtp = !params.speculative.types.empty() &&
                         params.speculative.types[0] == COMMON_SPECULATIVE_TYPE_DRAFT_MTP;

    if (use_mtp && n_pl.size() == 1 && n_pl[0] > 1) {
        fprintf(stderr, "%s: warn: MTP does not support parallel sequences (--parallel > 1); forcing -npl 1\n", __func__);
        n_pl = { 1 };
    }

    const int32_t mtp_n_max = use_mtp ? params.speculative.draft.n_max : 0;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    auto llama_init_tgt = common_init_from_params(params);

    llama_model   * model = llama_init_tgt->model();
    llama_context * ctx   = llama_init_tgt->context();

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize draft context for MTP
    llama_model_ptr   model_dft;
    llama_context_ptr ctx_dft;
    common_speculative_ptr spec;

    if (use_mtp) {
        const auto & params_spec = params.speculative.draft;

        // For local paths, common_params_handle_models only sets mparams.path when
        // loading via -hf. Fall back to the main model (same-file MTP) if no
        // --model-draft was provided explicitly.
        if (params.speculative.draft.mparams.path.empty()) {
            params.speculative.draft.mparams.path = params.model.path;
        }

        if (params_spec.mparams.path.empty()) {
            fprintf(stderr, "%s: error: --spec-type draft-mtp requires --model-draft or a model path via -m\n", __func__);
            return 1;
        }

        // Reuse the already-loaded model so the draft context shares GPU weights
        // with the target — no second VRAM allocation for weights.
        llama_context_params cparams_dft = llama_context_default_params();
        cparams_dft.n_ctx       = (uint32_t)(params.n_ctx + mtp_n_max + 1);
        cparams_dft.n_batch     = params.n_batch;
        cparams_dft.n_ubatch    = mtp_n_max + 1;
        cparams_dft.type_k      = params_spec.cache_type_k;
        cparams_dft.type_v      = params_spec.cache_type_v;
        cparams_dft.offload_kqv = true;
        ctx_dft.reset(llama_init_from_model(model, cparams_dft));
        if (!ctx_dft) {
            fprintf(stderr, "%s: error: failed to create MTP draft context\n", __func__);
            return 1;
        }

        params.speculative.draft.ctx_tgt = ctx;
        params.speculative.draft.ctx_dft = ctx_dft.get();

        spec.reset(common_speculative_init(params.speculative, 1));
        if (!spec) {
            fprintf(stderr, "%s: error: failed to init speculative engine\n", __func__);
            return 1;
        }
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    const auto get_token_rand = [n_vocab]() -> llama_token {
        return std::rand() % n_vocab;
    };

    auto * mem = llama_get_memory(ctx);

    const int32_t n_kv_max = llama_n_ctx(ctx);

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [](llama_context * lctx, llama_batch & lbatch, int32_t nb, bool synchronize) {
        for (int32_t i = 0; i < lbatch.n_tokens; i += nb) {
            const int32_t n_tokens = std::min(nb, lbatch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                lbatch.token    + i,
                nullptr,
                lbatch.pos      + i,
                lbatch.n_seq_id + i,
                lbatch.seq_id   + i,
                lbatch.logits   + i,
            };

            const int ret = llama_decode(lctx, batch_view);
            if (ret != 0) {
                LOG_ERR("failed to decode the batch, n_batch = %d, ret = %d\n", nb, ret);
                return false;
            }

            if (synchronize) {
                llama_synchronize(lctx);
            }
        }

        return true;
    };

    const int32_t n_batch_main = params.n_batch;

    // warm up
    {
        for (int i = 0; i < 16; ++i) {
            common_batch_add(batch, get_token_rand(), i, { 0 }, false);
        }

        if (!decode_helper(ctx, batch, n_batch_main, true)) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    const bool has_mtp = use_mtp && spec;

    if (!params.batched_bench_output_jsonl) {
        LOG("\n");
        LOG("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, is_pp_shared = %d, "
            "is_tg_separate = %d, n_gpu_layers = %d, n_threads = %u, mtp = %s, mtp_n_max = %d\n",
            __func__, n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type),
            is_pp_shared, is_tg_separate, params.n_gpu_layers, (uint32_t)params.cpuparams.n_threads,
            has_mtp ? "yes" : "no", mtp_n_max);
        LOG("\n");
        if (has_mtp) {
            LOG("|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s |\n",
                "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s", "D_ACC %", "SPEC_UP");
            LOG("|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n",
                "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------", "--------", "--------");
        } else {
            LOG("|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s |\n",
                "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s");
            LOG("|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n",
                "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------");
        }
    }

    common_sampler_ptr smpl;
    if (has_mtp) {
        smpl.reset(common_sampler_init(model, params.sampling));
    }

    for (        int i_pp = 0; i_pp < (int) n_pp.size(); ++i_pp) {
        for (    int i_tg = 0; i_tg < (int) n_tg.size(); ++i_tg) {
            for (int i_pl = 0; i_pl < (int) n_pl.size(); ++i_pl) {
                const int pp = n_pp[i_pp];
                const int tg = n_tg[i_tg];
                const int pl = n_pl[i_pl];

                const int n_ctx_req = is_pp_shared ? (params.kv_unified ? pp : pl*pp) + pl*tg : pl*(pp + tg);

                if (n_ctx_req > n_kv_max) {
                    continue;
                }

                // ── prompt phase ─────────────────────────────────────────────

                common_batch_clear(batch);

                for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                    for (int i = 0; i < pp; ++i) {
                        common_batch_add(batch, get_token_rand(), i, { j }, i == pp - 1);
                    }
                }

                llama_memory_clear(mem, false);

                const auto t_pp_start = ggml_time_us();

                if (!decode_helper(ctx, batch, n_batch_main, false)) {
                    LOG_ERR("%s: llama_decode() failed\n", __func__);
                    return 1;
                }

                llama_synchronize(ctx);

                const auto t_pp_end = ggml_time_us();

                if (is_pp_shared) {
                    for (int32_t i = 1; i < pl; ++i) {
                        llama_memory_seq_cp(mem, 0, i, -1, -1);
                    }

                    if (!params.kv_unified) {
                        common_batch_clear(batch);
                        common_batch_add(batch, get_token_rand(), pp + 0, { 0 }, true);
                        if (!decode_helper(ctx, batch, n_batch_main, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            return 1;
                        }
                        llama_memory_seq_rm(mem, 0, pp, -1);
                    }
                }

                // ── generation phase ─────────────────────────────────────────

                const auto t_tg_start = ggml_time_us();

                int64_t n_drafted  = 0;
                int64_t n_accepted = 0;

                if (has_mtp && pl == 1) {
                    // MTP speculative generation (single sequence only)
                    const llama_seq_id seq_id = 0;

                    llama_tokens prompt_tgt;
                    prompt_tgt.reserve(pp);
                    for (int i = 0; i < pp; ++i) {
                        prompt_tgt.push_back(get_token_rand());
                    }

                    common_speculative_begin(spec.get(), seq_id, prompt_tgt);

                    llama_token   id_last  = get_token_rand();
                    int32_t       n_past   = pp;
                    int32_t       tg_done  = 0;

                    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx), 0, 1);
                    llama_tokens draft;

                    while (tg_done < tg) {
                        draft.clear();

                        common_speculative_get_draft_params(spec.get(), seq_id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ mtp_n_max,
                            /* .n_past   = */ n_past,
                            /* .id_last  = */ id_last,
                            /* .prompt   = */ &prompt_tgt,
                            /* .result   = */ &draft,
                        };
                        common_speculative_draft(spec.get());

                        n_drafted += (int64_t) draft.size();

                        common_batch_clear(batch_tgt);
                        common_batch_add(batch_tgt, id_last, n_past, { seq_id }, true);
                        for (int i = 0; i < (int) draft.size(); ++i) {
                            common_batch_add(batch_tgt, draft[i], n_past + 1 + i, { seq_id }, true);
                        }

                        llama_decode(ctx, batch_tgt);
                        llama_synchronize(ctx);

                        // accept/reject draft tokens via greedy sampler
                        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx, draft);

                        const int32_t n_acc = (int32_t) ids.size() - 1;
                        n_accepted += n_acc;

                        common_speculative_accept(spec.get(), seq_id, n_acc);

                        n_past  += (int32_t) ids.size();
                        tg_done += (int32_t) ids.size();

                        for (auto tok : ids) {
                            prompt_tgt.push_back(tok);
                        }
                        id_last = ids.back();
                    }

                    llama_batch_free(batch_tgt);

                } else if (is_tg_separate) {
                    // decode pattern:
                    // 0 0 0 ... 1 1 1 ... 2 2 2 ... 3 3 3 ...
                    for (int j = 0; j < pl; ++j) {
                        for (int i = 0; i < tg; ++i) {
                            common_batch_clear(batch);

                            common_batch_add(batch, get_token_rand(), pp + i, { j }, true);

                            if (!decode_helper(ctx, batch, n_batch_main, true)) {
                                LOG_ERR("%s: llama_decode() failed\n", __func__);
                                return 1;
                            }
                        }
                    }
                } else {
                    // decode pattern:
                    // 0123 0123 0123 ...
                    for (int i = 0; i < tg; ++i) {
                        common_batch_clear(batch);

                        for (int j = 0; j < pl; ++j) {
                            common_batch_add(batch, get_token_rand(), pp + i, { j }, true);
                        }

                        if (!decode_helper(ctx, batch, n_batch_main, true)) {
                            LOG_ERR("%s: llama_decode() failed\n", __func__);
                            return 1;
                        }
                    }
                }

                const auto t_tg_end = ggml_time_us();

                const int32_t n_kv = n_ctx_req;

                const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
                const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;
                const float t    = t_pp + t_tg;

                const float speed_pp = is_pp_shared ? pp / t_pp : pl*pp / t_pp;
                const float speed_tg = pl*tg / t_tg;
                const float speed    = ((is_pp_shared ? pp : pl*pp) + pl*tg) / t;

                const float draft_acc_pct  = (n_drafted > 0) ? 100.0f * (float)n_accepted / (float)n_drafted : -1.0f;
                const float spec_speedup   = (n_drafted > 0 && tg > 0) ? (float)(tg) / (float)(tg - n_accepted) : -1.0f;

                if (params.batched_bench_output_jsonl) {
                    if (has_mtp && pl == 1) {
                        LOG(
                            "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, "
                            "\"is_pp_shared\": %d, \"n_gpu_layers\": %d, \"mtp_n_max\": %d, "
                            "\"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, "
                            "\"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, "
                            "\"t\": %f, \"speed\": %f, \"draft_acc_pct\": %f, \"spec_speedup\": %f}\n",
                            n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type),
                            params.is_pp_shared, params.n_gpu_layers, mtp_n_max,
                            pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed,
                            draft_acc_pct, spec_speedup);
                    } else {
                        LOG(
                            "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, "
                            "\"is_pp_shared\": %d, \"n_gpu_layers\": %d, "
                            "\"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, "
                            "\"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, "
                            "\"t\": %f, \"speed\": %f}\n",
                            n_kv_max, params.n_batch, params.n_ubatch, int(params.flash_attn_type),
                            params.is_pp_shared, params.n_gpu_layers,
                            pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed);
                    }
                } else {
                    if (has_mtp && pl == 1) {
                        LOG("|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f | %7.1f%% | %8.2f |\n",
                            pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed,
                            draft_acc_pct, spec_speedup);
                    } else {
                        LOG("|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n",
                            pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed);
                    }
                }
            }
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    if (has_mtp) {
        common_speculative_print_stats(spec.get());
    }

    llama_batch_free(batch);

    llama_backend_free();

    return 0;
}

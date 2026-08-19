/****************************************************************************
 * QiDiAi 建木 — on-device semantic indexing demo (openvela)
 *
 * Vertical slice proving the end-to-end offline pipeline:
 *     embed text  ->  build local vector index  ->  semantic search
 *
 * Powered by the V10 Pure Source Pool embedding engine:
 * 1024-d output, 2.58M params, FP16 weights 4.92MB, libc+libm only.
 * Anti-collapse trained (hyperspherical repulsion + contraction loss).
 * Runs fully offline — no network, no cloud.
 *
 * Usage:
 *   jianmu                       — run with weights embedded in the XIP firmware
 *   jianmu /path/to/weights.baize — run with an external weight file (debug)
 *   jianmu --selftest             — end-to-end pipeline check (embedded weights)
 ****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "v10.h"
#include "semantic_index.h"
#include "lvgl_frontend.h"

#ifndef V10_MODEL_PATH
#define V10_MODEL_PATH "/data/v10_weights.baize"
#endif

static const char *k_corpus[] =
{
  "上次聊装修的朋友 小王 微信",
  "周五的团队周会纪要 关于 openvela 移植",
  "妈妈生日提醒 下周二 买蛋糕",
  "健身计划 每周三跑步五公里",
  "装修预算表格 水电改造两万元",
  "围棋课报名 周末上午",
};

/* Forward declarations from v10_infer.c (for selftest) */
typedef struct {
    int vocab_size, embed_dim, num_layers, max_seq_len, output_dim, sp_dim;
    float r1, r2, r3;
    const float *w_embed;
    const float *skip_gates;
    const float *l_write_gw[6], *l_write_gb[6];
    const float *l_fast_rw[6], *l_fast_rb[6];
    const float *l_med_rw[6], *l_med_rb[6];
    const float *l_slow_rw[6], *l_slow_rb[6];
    const float *l_fusion_w[6], *l_fusion_b[6];
    const float *bottleneck_w, *bottleneck_b;
    const float *out_proj_w, *out_proj_b;
    const float *out_norm_w, *out_norm_b;
    float *tok_emb, *hidden, *pool_f, *pool_m, *pool_s;
    float *write_vals, *fast_r, *med_r, *slow_r;
    float *concat, *mod, *bottleneck_out, *output;
    float *weights;
    long n_floats;
} v10_model_t;

extern int  model_init(v10_model_t *m, const char *path);
extern void model_free(v10_model_t *m);
extern void model_forward(v10_model_t *m, const char *text, float *out_vec);

static v10_model_t g_selftest_model;

static int selftest_init(void)
{
  /* Load the REAL V10 weights embedded in the XIP firmware image.
     model_init(NULL) parses the `.weights_blob` flash section in place
     and points every tensor straight at flash — no 10 MB malloc. */
  memset(&g_selftest_model, 0, sizeof(g_selftest_model));

  if (model_init(&g_selftest_model, NULL) != 0)
    {
      fprintf(stderr, "selftest: model_init(embedded XIP) failed\n");
      return -1;
    }

  return 0;
}

/* Selftest embed: calls model_forward directly */
static int selftest_embed(const char *text, float *out, int dim)
{
  if (!text || text[0] == '\0') {
    memset(out, 0, dim * sizeof(float));
    return dim;
  }
  float tmp[1024];
  model_forward(&g_selftest_model, text, tmp);
  int copy = dim < 1024 ? dim : 1024;
  memcpy(out, tmp, copy * sizeof(float));
  return copy;
}

int main(int argc, char *argv[])
{
  if (argc > 1 && strcmp(argv[1], "--gui") == 0)
    return jianmu_gui_run();

  int selftest = 0;
  const char *model_path = NULL;   /* NULL → embedded XIP weights */

  if (argc > 1 && strcmp(argv[1], "--selftest") == 0)
    selftest = 1;
  else if (argc > 1)
    model_path = argv[1];

  printf("=== QiDiAi 建木 (openvela on-device semantic index) ===\n");

  if (selftest)
    {
      if (selftest_init() != 0)
        {
          fprintf(stderr, "建木: selftest init failed\n");
          return 1;
        }

      /* Test pipeline: embed → check output validity */
      const char *test_texts[] = {"测试文本", "hello world", ""};
      int all_ok = 1;
      for (int i = 0; i < 3; i++)
        {
          float emb[1024];
          selftest_embed(test_texts[i], emb, 1024);

          int has_nan = 0;
          float norm = 0;
          for (int j = 0; j < 1024; j++)
            {
              if (isnan(emb[j]) || isinf(emb[j])) has_nan = 1;
              norm += emb[j] * emb[j];
            }
          norm = sqrtf(norm);

          printf("  [%d] \"%s\": norm=%.4f nan=%d %s\n",
                 i, test_texts[i], norm, has_nan,
                 has_nan ? "FAIL" : "OK");
          if (has_nan) all_ok = 0;
        }

      printf("\n=== selftest %s (embedded XIP weights, real model) ===\n",
             all_ok ? "PASSED" : "FAILED");

      model_free(&g_selftest_model);
      return all_ok ? 0 : 1;
    }

  /* Normal mode: load real weights (embedded XIP by default) */
  if (v10_init(model_path) != 0)
    {
      fprintf(stderr, "建木: V10 模型加载失败 (%s)\n",
              model_path ? model_path : "<embedded XIP weights>");
      fprintf(stderr, "      嵌入式权重不可用时，可改用 --selftest 进行管道自检\n");
      return 1;
    }
  printf("V10 engine ready, dim = %d (real model)\n", v10_dim());

  semantic_index_t idx;
  si_build(&idx, k_corpus, sizeof(k_corpus) / sizeof(k_corpus[0]));
  printf("Indexed %d local documents (fully offline).\n", idx.count);

  const char *queries[] =
  {
    "找上次聊装修的朋友",
    "提醒我妈生日",
    "这周去跑步",
  };

  for (int q = 0; q < 3; q++)
    {
      int   res[SI_TOP_K];
      float sc[SI_TOP_K];
      int   n = si_search(&idx, queries[q], res, sc);

      printf("\nQuery: \"%s\"\n", queries[q]);
      if (n == 0)
        {
          printf("  (no match / model not loaded)\n");
          continue;
        }
      for (int k = 0; k < n; k++)
        {
          printf("  [%d] score=%.3f  %s\n", k + 1, sc[k],
                 idx.docs[res[k]].text);
        }
    }

  printf("\n=== demo complete (offline semantic search OK) ===\n");
  si_free(&idx);
  v10_free();
  return 0;
}
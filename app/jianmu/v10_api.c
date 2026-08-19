/**
 * v10_api.c — External API for V10 embedding engine
 *
 * Wraps v10_infer.c behind a clean init/embed/free interface.
 * Matches the API used by semantic_index.h / jianmu_main.c.
 *
 * The Model struct and forward pass live in v10_infer.c.
 * This file provides the singleton + guard + empty-string safety
 * that the index/demo layer expects.
 *
 * Compile:
 *   gcc -O2 -shared -fPIC -lm -o libv10.so v10_infer.c v10_api.c
 */

#include <string.h>

/* Re-declare Model from v10_infer.c — must match exactly */
#define VOCAB   4958
#define E       384
#define S       48
#define L       6
#define N       96
#define O       1024
#define S3      (S * 3)
#define E3      (E * 3)
#define EPS     1e-5f

typedef struct {
    int vocab_size, embed_dim, num_layers, max_seq_len, output_dim, sp_dim;
    float r1, r2, r3;
    const float *w_embed;
    const float *skip_gates;
    const float *l_write_gw[L];
    const float *l_write_gb[L];
    const float *l_fast_rw[L];
    const float *l_fast_rb[L];
    const float *l_med_rw[L];
    const float *l_med_rb[L];
    const float *l_slow_rw[L];
    const float *l_slow_rb[L];
    const float *l_fusion_w[L];
    const float *l_fusion_b[L];
    const float *bottleneck_w;
    const float *bottleneck_b;
    const float *out_proj_w;
    const float *out_proj_b;
    const float *out_norm_w;
    const float *out_norm_b;
    float *tok_emb;
    float *hidden;
    float *pool_f;
    float *pool_m;
    float *pool_s;
    float *write_vals;
    float *fast_r;
    float *med_r;
    float *slow_r;
    float *concat;
    float *mod;
    float *bottleneck_out;
    float *output;
    float *weights;
    long n_floats;
} Model;

/* Forward declarations from v10_infer.c */
extern float *load_baize(const char *path, long *out_n_floats, int *out_version);
extern int  model_init(Model *m, const char *path);
extern void model_free(Model *m);
extern void model_forward(Model *m, const char *text, float *out_vec);

/* Global singleton */
static Model g_model;
static int g_initialized = 0;

/* ============ Public C API ============ */

int v10_init(const char *model_path) {
    if (g_initialized) {
        model_free(&g_model);
        g_initialized = 0;
    }
    int ret = model_init(&g_model, model_path);
    if (ret == 0) g_initialized = 1;
    return ret;
}

int v10_embed(const char *text, float *out, int dim) {
    if (!g_initialized) return -1;
    int copy = dim < O ? dim : O;

    /* Guard: empty string returns zero vector (safe for index layer) */
    if (!text || text[0] == '\0') {
        memset(out, 0, copy * sizeof(float));
        return copy;
    }

    float tmp_out[O];
    memset(tmp_out, 0, sizeof(tmp_out));
    model_forward(&g_model, text, tmp_out);
    memcpy(out, tmp_out, copy * sizeof(float));
    return copy;
}

void v10_free(void) {
    if (g_initialized) {
        model_free(&g_model);
        g_initialized = 0;
    }
}

int v10_dim(void) {
    return g_initialized ? g_model.output_dim : O;
}

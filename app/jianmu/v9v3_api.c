/**
 * v9v3_api.c — External API for V9v3 embedding engine
 *
 * Compiled together with v9v3_infer.c to produce libv9v3.so/.dll.
 * Wraps the inference engine behind a clean init/embed/get_pools/free interface.
 *
 * Compile:
 *   gcc -O2 -shared -fPIC -lm -o libv9v3.so v9v3_infer.c v9v3_api.c
 *   cl /O2 /shared /fp:fast v9v3_infer.c v9v3_api.c /Fe:v9v3.dll
 */

#include <string.h>

/* ---- Re-declare Model (must match v9v3_infer.c exactly) ---- */

#define VOCAB   4096
#define E       384
#define S       24
#define L       6
#define N       24
#define O       1024
#define EPS     1e-5f

typedef struct {
    int vocab_size, embed_dim, num_layers, max_seq_len, output_dim, sp_dim;
    float r1, r2, r3;
    const float *w_embed;
    const float *w_pos;
    const float *w_out;
    const float *b_out;
    const float *w_out_norm;
    const float *b_out_norm;
    const float *skip_gates;
    const float *l_linear_w[L];
    const float *l_linear_b[L];
    const float *l_norm_w[L];
    const float *l_norm_b[L];
    const float *l_write_gw[L];
    const float *l_write_gb[L];
    const float *l_fast_rw[L];
    const float *l_fast_rb[L];
    const float *l_slow_rw[L];
    const float *l_slow_rb[L];
    const float *l_med_rw[L];
    const float *l_med_rb[L];
    const float *l_fusion_w[L];
    const float *l_fusion_b[L];
    float *tok_emb;
    float *hidden;
    float *pool_f;
    float *pool_m;
    float *pool_s;
    float *tmp;
    float *output;
    float *weights;
    long   n_floats;
} Model;

/* Forward declarations from v9v3_infer.c */
extern float *load_baize(const char *path, long *out_n_floats, int *out_params);
extern int model_init(Model *m, const char *path);
extern void model_free(Model *m);
extern int tokenize(const char *text, int *tokens);
extern void model_forward(Model *m, const char *text, float *out_vec, int verbose);

/* Global singleton */
static Model g_model;
static int g_initialized = 0;

/* ============ Public C API ============ */

int v9v3_init(const char *model_path) {
    if (g_initialized) {
        model_free(&g_model);
        g_initialized = 0;
    }
    int ret = model_init(&g_model, model_path);
    if (ret == 0) g_initialized = 1;
    return ret;
}

int v9v3_embed(const char *text, float *out, int dim) {
    if (!g_initialized) return -1;
    int copy = dim < O ? dim : O;
    /* 守卫：空串/ NULL 直接返回零向量。
       实测 v9v3_embed("") 在 model_forward 早返路径下会 segfault
       （二进制自检只测了 model_forward 直连，未覆盖此 API 路径）。
       产品层索引空备忘录时必须避免崩溃。 */
    if (!text || text[0] == '\0') {
        memset(out, 0, copy * sizeof(float));
        return copy;
    }
    float tmp_out[O];
    memset(tmp_out, 0, sizeof(tmp_out));
    model_forward(&g_model, text, tmp_out, 0);
    memcpy(out, tmp_out, copy * sizeof(float));
    return copy;
}

int v9v3_get_pools(float *fast_pool, float *med_pool, float *slow_pool, int dim) {
    if (!g_initialized) return -1;
    int copy = dim < S ? dim : S;
    memcpy(fast_pool, g_model.pool_f, copy * sizeof(float));
    memcpy(med_pool,  g_model.pool_m, copy * sizeof(float));
    memcpy(slow_pool, g_model.pool_s, copy * sizeof(float));
    return copy;
}

void v9v3_free(void) {
    if (g_initialized) {
        model_free(&g_model);
        g_initialized = 0;
    }
}

int v9v3_output_dim(void) {
    return g_initialized ? g_model.output_dim : O;
}

int v9v3_pool_dim(void) {
    return g_initialized ? g_model.sp_dim : S;
}

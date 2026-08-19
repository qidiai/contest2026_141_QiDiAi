/**
 * v9v3_infer.c — V9v3 embedding model C inference engine (single file)
 *
 * Compile:
 *   gcc -O2 -lm -o v9v3_infer v9v3_infer.c
 *   cl /O2 /fp:fast v9v3_infer.c
 *
 * Usage:
 *   ./v9v3_infer "your text"
 *   ./v9v3_infer /path/to/model.baize "your text"
 *
 * Weight file (.baize):
 *   [Header 68B] [Float32 weights (3,146,664 floats = 12.58 MB)]
 */

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ============ Model constants ============ */
#define VOCAB   4096
#define E       384
#define S       24
#define L       6
#define N       96
#define O       1024
#define EPS     1e-5f

/* ============ Model struct ============ */
typedef struct {
    int vocab_size, embed_dim, num_layers, max_seq_len, output_dim, sp_dim;
    float r1, r2, r3;

    /* Weight pointers into weights[] */
    const float *w_embed;
    const float *w_pos;
    const float *w_out;
    const float *b_out;
    const float *w_out_norm;
    const float *b_out_norm;
    const float *skip_gates;

    /* Per-layer weights (14 tensors per layer) */
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

    /* Pre-allocated runtime buffers */
    float *tok_emb;
    float *hidden;
    float *pool_f;
    float *pool_m;
    float *pool_s;
    float *tmp;
    float *output;

    /* Owned memory block */
    float *weights;
    long   n_floats;
} Model;

/* ============ .baize loader ============ */
float *load_baize(const char *path, long *out_n_floats, int *out_params) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);
    uint8_t *buf = (uint8_t*)malloc(fsize);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, fsize, fp) != (size_t)fsize) {
        fprintf(stderr, "Read failed\n");
        free(buf); fclose(fp); return NULL;
    }
    fclose(fp);
    if (buf[0] != 'B' || buf[1] != 'A' || buf[2] != 'I' || buf[3] != 'Z') {
        fprintf(stderr, "Not a valid .baize file\n");
        free(buf); return NULL;
    }
    uint32_t *hdr = (uint32_t*)buf;
    *out_params = (int)hdr[2];
    *out_n_floats = (fsize - 68) / 4;
    float *data = (float*)(buf + 68);
    if (*out_n_floats != *out_params) {
        fprintf(stderr, "Param count mismatch: expected %d, got %ld\n", *out_params, *out_n_floats);
        free(buf); return NULL;
    }
    return data;
}

/* ============ Model init / free ============ */
int model_init(Model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    long n_floats; int total_params;
    float *data = load_baize(path, &n_floats, &total_params);
    if (!data) return -1;
    m->weights = data;
    m->n_floats = n_floats;
    m->vocab_size = VOCAB; m->embed_dim = E; m->num_layers = L;
    m->max_seq_len = N; m->output_dim = O; m->sp_dim = S;
    m->r1 = 0.85f; m->r2 = 0.995f; m->r3 = 0.95f;

    int off = 0;
    m->w_embed    = data + off; off += VOCAB * E;
    m->w_pos      = data + off; off += N * E;
    m->w_out      = data + off; off += O * E;
    m->b_out      = data + off; off += O;
    m->w_out_norm = data + off; off += O;
    m->b_out_norm = data + off; off += O;
    m->skip_gates = data + off; off += L;

    for (int i = 0; i < L; i++) {
        m->l_linear_w[i] = data + off; off += E * E;
        m->l_linear_b[i] = data + off; off += E;
        m->l_norm_w[i]   = data + off; off += E;
        m->l_norm_b[i]   = data + off; off += E;
        m->l_write_gw[i] = data + off; off += S * E;
        m->l_write_gb[i] = data + off; off += S;
        m->l_fast_rw[i]  = data + off; off += E * S;
        m->l_fast_rb[i]  = data + off; off += E;
        m->l_slow_rw[i]  = data + off; off += E * S;
        m->l_slow_rb[i]  = data + off; off += E;
        m->l_med_rw[i]   = data + off; off += E * S;
        m->l_med_rb[i]   = data + off; off += E;
        m->l_fusion_w[i] = data + off; off += 3 * (3 * E);
        m->l_fusion_b[i] = data + off; off += 3;
    }

    m->tok_emb = (float*)calloc(N * E, sizeof(float));
    m->hidden  = (float*)calloc(N * E, sizeof(float));
    m->pool_f  = (float*)calloc(S, sizeof(float));
    m->pool_m  = (float*)calloc(S, sizeof(float));
    m->pool_s  = (float*)calloc(S, sizeof(float));
    m->tmp     = (float*)calloc((E > 3*E ? E : 3*E) > O ? (E > 3*E ? E : 3*E) : O, sizeof(float));
    m->output  = (float*)calloc(O, sizeof(float));
    if (!m->tok_emb || !m->hidden || !m->pool_f || !m->pool_m ||
        !m->pool_s || !m->tmp || !m->output) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    printf("[Model] %d params, weights=%.1f MB, runtime=%.1f KB\n",
           total_params,
           n_floats * 4.0 / 1024 / 1024,
           (N*E*2 + S*3 + O + (E > 3*E ? E : 3*E) + O) * 4.0 / 1024);
    return 0;
}

void model_free(Model *m) {
    if (m->weights) free((void*)(m->weights - 17));
    free(m->tok_emb); free(m->hidden);
    free(m->pool_f); free(m->pool_m); free(m->pool_s);
    free(m->tmp); free(m->output);
    memset(m, 0, sizeof(*m));
}

/* ============ Tokenizer (UTF-8 to Unicode) ============ */
static int decode_utf8(const unsigned char *s, int *consumed) {
    if (s[0] < 0x80) { *consumed = 1; return s[0]; }
    else if ((s[0] & 0xE0) == 0xC0) { *consumed = 2; return ((s[0] & 0x1F) << 6) | (s[1] & 0x3F); }
    else if ((s[0] & 0xF0) == 0xE0) { *consumed = 3; return ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    else if ((s[0] & 0xF8) == 0xF0) { *consumed = 4; return ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); }
    *consumed = 1; return s[0];
}

int tokenize(const char *text, int *tokens) {
    int len = 0;
    const unsigned char *p = (const unsigned char*)text;
    while (*p && len < N) {
        int consumed, cp = decode_utf8(p, &consumed);
        tokens[len++] = cp % VOCAB;
        p += consumed;
    }
    return len;
}

/* ============ Core math ============ */
static void matvec(const float *W, const float *x, const float *b,
                   float *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = b ? b[i] : 0.0f;
        const float *restrict w = W + i * cols;
        int j = 0;
        for (; j + 3 < cols; j += 4)
            sum += w[j]*x[j] + w[j+1]*x[j+1] + w[j+2]*x[j+2] + w[j+3]*x[j+3];
        for (; j < cols; j++)
            sum += w[j] * x[j];
        y[i] = sum;
    }
}

static void batch_linear(const float *restrict W, const float *restrict X,
                          const float *restrict b, float *restrict Y,
                          int n_pos, int dim) {
    for (int n = 0; n < n_pos; n++) {
        float *restrict y = Y + n * dim;
        for (int i = 0; i < dim; i++)
            y[i] = b ? b[i] : 0.0f;
    }
    for (int i = 0; i < dim; i++) {
        const float *restrict w_i = W + i * dim;
        for (int n = 0; n < n_pos; n++) {
            const float *restrict x_n = X + n * dim;
            float sum = 0.0f;
            int j = 0;
            for (; j + 3 < dim; j += 4)
                sum += w_i[j]*x_n[j] + w_i[j+1]*x_n[j+1]
                     + w_i[j+2]*x_n[j+2] + w_i[j+3]*x_n[j+3];
            for (; j < dim; j++)
                sum += w_i[j] * x_n[j];
            Y[n * dim + i] += sum;
        }
    }
}

static void layernorm(float *restrict x, const float *restrict w,
                       const float *restrict b, int n) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + EPS);
    for (int i = 0; i < n; i++)
        x[i] = (x[i] - mean) * inv * w[i] + b[i];
}

static void softmax(float *restrict x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void silu(float *restrict x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void l2norm(float *restrict x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float inv = 1.0f / sqrtf(sum + 1e-12f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ============ Forward pass ============ */
void model_forward(Model *m, const char *text, float *out_vec, int verbose) {
    int tokens[N];
    int seq_len = tokenize(text, tokens);

    /* Empty input → return zero vector */
    if (seq_len < 1) {
        memset(out_vec, 0, O * sizeof(float));
        return;
    }

    memset(m->pool_f, 0, S * sizeof(float));
    memset(m->pool_m, 0, S * sizeof(float));
    memset(m->pool_s, 0, S * sizeof(float));

    /* 1. Token embedding + position encoding */
    for (int i = 0; i < seq_len; i++) {
        int base = i * E;
        int tok = tokens[i];
        const float *emb = m->w_embed + tok * E;
        const float *pos = m->w_pos + i * E;
        for (int j = 0; j < E; j++)
            m->tok_emb[base + j] = emb[j] + pos[j];
    }

    float *buf_input  = m->tok_emb;
    float *buf_output = m->hidden;

    /* 2. L layers of SourcePool */
    for (int layer = 0; layer < L; layer++) {
        batch_linear(m->l_linear_w[layer], buf_input, m->l_linear_b[layer],
                     buf_output, seq_len, E);

        for (int i = 0; i < seq_len; i++)
            layernorm(buf_output + i * E, m->l_norm_w[layer], m->l_norm_b[layer], E);

        /* SourcePool: write gate + three-speed EMA decay + fusion */
        memset(m->pool_f, 0, S * sizeof(float));
        memset(m->pool_m, 0, S * sizeof(float));
        memset(m->pool_s, 0, S * sizeof(float));

        const float *x_last = buf_output + (seq_len - 1) * E;
        float gw[S];
        matvec(m->l_write_gw[layer], x_last, m->l_write_gb[layer], gw, S, E);
        for (int i = 0; i < S; i++) gw[i] = tanhf(gw[i]);

        float r1 = m->r1, r2 = m->r2, r3 = m->r3;
        for (int i = 0; i < S; i++) {
            m->pool_f[i] = r1 * m->pool_f[i] + (1.0f - r1) * gw[i];
            m->pool_s[i] = r2 * m->pool_s[i] + (1.0f - r2) * gw[i];
            m->pool_m[i] = r3 * m->pool_m[i] + (1.0f - r3) * gw[i];
        }

        float fast_r[E], slow_r[E], med_r[E];
        matvec(m->l_fast_rw[layer], m->pool_f, m->l_fast_rb[layer], fast_r, E, S);
        matvec(m->l_slow_rw[layer], m->pool_s, m->l_slow_rb[layer], slow_r, E, S);
        matvec(m->l_med_rw[layer],  m->pool_m,  m->l_med_rb[layer],  med_r,  E, S);

        float concat[3 * E];
        memcpy(concat, fast_r, E * sizeof(float));
        memcpy(concat + E, med_r, E * sizeof(float));
        memcpy(concat + 2 * E, slow_r, E * sizeof(float));

        float logits[3];
        matvec(m->l_fusion_w[layer], concat, m->l_fusion_b[layer], logits, 3, 3 * E);
        softmax(logits, 3);

        float mod[E];
        for (int i = 0; i < E; i++)
            mod[i] = logits[0] * fast_r[i] + logits[1] * med_r[i] + logits[2] * slow_r[i];

        for (int i = 0; i < seq_len; i++) {
            float *h = buf_output + i * E;
            for (int j = 0; j < E; j++) h[j] += mod[j];
            silu(h, E);
            float sg = 1.0f / (1.0f + expf(-m->skip_gates[layer]));
            for (int j = 0; j < E; j++)
                h[j] += sg * buf_input[i * E + j];
        }

        float *swap = buf_input;
        buf_input = buf_output;
        buf_output = swap;
    }

    float *final_hidden = buf_input;

    /* 3. Mean pooling */
    float pooled[E];
    memset(pooled, 0, sizeof(pooled));
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < E; j++)
            pooled[j] += final_hidden[i * E + j];
    float inv_len = 1.0f / seq_len;
    for (int j = 0; j < E; j++) pooled[j] *= inv_len;

    /* 4. Output projection */
    matvec(m->w_out, pooled, m->b_out, m->output, O, E);

    /* 5. Output LayerNorm */
    layernorm(m->output, m->w_out_norm, m->b_out_norm, O);

    /* 6. L2 normalization */
    l2norm(m->output, O);

    memcpy(out_vec, m->output, O * sizeof(float));

    if (verbose) {
        printf("Output [0..5]: %.6f %.6f %.6f %.6f %.6f\n",
               out_vec[0], out_vec[1], out_vec[2], out_vec[3], out_vec[4]);
    }
}

/* ============ Utilities ============ */
static void print_vec(const float *v, int n, const char *label) {
    printf("%s [", label);
    for (int i = 0; i < (n < 6 ? n : 6); i++)
        printf("%.6f%c", v[i], i < 5 ? ',' : ' ');
    printf("...] (%d dim)\n", n);
}

static int has_nan_or_inf(const float *v, int n) {
    for (int i = 0; i < n; i++)
        if (isnan(v[i]) || isinf(v[i])) return 1;
    return 0;
}

static double vec_l2(const float *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)v[i] * v[i];
    return sqrt(sum);
}

/* ============ Self-test ============ */
static int run_selftest(Model *m) {
    int all_pass = 1;
    srand((unsigned)time(NULL));
    printf("=== V9v3 Self-Test ===\n");

    /* Boundary texts */
    const char *tests[] = {"", " ", "a", "Hello", "世界", "测试文本", "a\nb\nc"};
    const char *labels[] = {"空", "空格", "a", "Hello", "世界", "测试", "换行"};
    int nt = sizeof(tests) / sizeof(tests[0]);
    for (int i = 0; i < nt; i++) {
        float out[O];
        model_forward(m, tests[i], out, 0);
        int ni = has_nan_or_inf(out, O);
        double l2 = vec_l2(out, O);
        int empty = (strlen(tests[i]) == 0);
        int ok = !ni && (empty ? (l2 == 0.0) : (fabs(l2 - 1.0) < 0.01));
        printf("  [%s] %s L2=%.6f %s\n", ni ? "NAN" : "OK", labels[i], l2, ok ? "PASS" : "FAIL");
        if (!ok) all_pass = 0;
    }

    /* Determinism */
    float ref[O];
    model_forward(m, "确定性测试", ref, 0);
    int det_ok = 1;
    for (int i = 0; i < 50; i++) {
        float out[O];
        model_forward(m, "确定性测试", out, 0);
        for (int j = 0; j < O; j++)
            if (out[j] != ref[j]) { det_ok = 0; break; }
        if (!det_ok) break;
    }
    printf("  Determinism: %s\n", det_ok ? "PASS" : "FAIL");
    if (!det_ok) all_pass = 0;

    /* Stability */
    int stab_nan = 0;
    for (int i = 0; i < 500; i++) {
        float out[O];
        model_forward(m, "标准测试文本用于稳定性验证", out, 0);
        if (has_nan_or_inf(out, O)) stab_nan++;
    }
    printf("  Stability (500 runs): NaN/Inf=%d %s\n",
           stab_nan, stab_nan == 0 ? "PASS" : "FAIL");
    if (stab_nan) all_pass = 0;

    printf("=== Result: %s ===\n", all_pass ? "ALL PASS" : "FAILED");
    return all_pass ? 0 : -1;
}

/* ============ Main (renamed for openvela app link) ============ */
/* Renamed from main() so this translation unit links into the openvela
 * app, which supplies its own jianmu_main.c entry point. To build the
 * standalone CLI, compile v9v3_infer.c on its own (e.g. gcc -O2 -lm
 * v9v3_infer.c) and this symbol becomes main again via a -Dmain=... */
int v9v3_infer_main(int argc, char **argv) {
    const char *default_model = "v9v3_weights.baize";
    const char *default_text = "光合作用产生氧气";
    const char *model_path = default_model;
    const char *input_text = default_text;
    int selftest_mode = 0;

    if (argc >= 2) {
        if (strcmp(argv[1], "--test") == 0) {
            selftest_mode = 1;
            if (argc >= 3) model_path = argv[2];
        } else if (argc >= 3) {
            model_path = argv[1];
            input_text = argv[2];
        } else {
            input_text = argv[1];
        }
    }

    Model model;
    if (model_init(&model, model_path) != 0) return 1;

    if (selftest_mode) {
        int ret = run_selftest(&model);
        model_free(&model);
        return ret;
    }

    printf("=== V9v3 C Inference Engine ===\n");
    printf("Input: %s\n\n", input_text);

    float output[O];
    model_forward(&model, input_text, output, 0);

    clock_t start = clock();
    int iters = 100;
    for (int i = 0; i < iters; i++)
        model_forward(&model, input_text, output, 0);
    clock_t end = clock();
    double ms = (double)(end - start) / CLOCKS_PER_SEC * 1000 / iters;

    print_vec(output, O, "Output");
    printf("Time:  %.3f ms/emb (%d iters)\n", ms, iters);

    double l2 = vec_l2(output, O);
    printf("L2:    %.6f %s\n", l2, fabs(l2 - 1.0) < 0.001 ? "OK" : "OFF");

    printf("\nMemory:\n");
    printf("  Weights:  %.1f MB\n", model.n_floats * 4.0 / 1024 / 1024);
    printf("  Runtime:  %.1f KB\n",
           (N*E*2 + S*3 + (E > 3*E ? E : 3*E) + O) * 4.0 / 1024);
    printf("  Total:    %.1f MB\n",
           (model.n_floats * 4.0 + (N*E*2 + S*3 + (E > 3*E ? E : 3*E) + O) * 4.0) / 1024 / 1024);

    model_free(&model);
    return 0;
}

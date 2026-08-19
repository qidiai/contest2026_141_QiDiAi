/**
 * v10_infer.c — V10 Pure Source Pool C inference engine
 *
 * Architecture:
 *   tokens → Embed(384d) → PurePoolStack(6 layers)
 *   → bottleneck(384→144) → tanh → Proj(144→1024) → LayerNorm → L2
 *
 * Per-layer PureSourcePool:
 *   write_vals = tanh(write_gate(x))         [seq, sp_dim]
 *   pool = sum_t(decay^(T-1-t) * write[t]) * (1-decay)
 *   read = matvec(read_w, pool, read_b)       [embed_dim]
 *   concat = [fast_read; medium_read; slow_read]  [embed_dim*3]
 *   logits = softmax(fusion_gate(concat))     [3]
 *   mod = logits[0]*fast + logits[1]*med + logits[2]*slow
 *
 * No position encoding, no Linear+LayerNorm per token (pure pool).
 *
 * Weight file (.baize V10 format):
 *   [Header 68 bytes] [float32 weights, sequential]
 *   Header: BAIZ magic, version=10, n_floats, vocab, embed, layers, seq, out, sp, decays
 *
 * Compile:
 *   gcc -O2 -lm -o v10_infer v10_infer.c
 *   gcc -O2 -shared -fPIC -lm -o libv10.so v10_infer.c v10_api.c
 */

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "v10_token_map.h"

/* ============ Model constants ============ */
#define VOCAB   4958
#define E       384
#define S       48
#define L       6
#define N       96
#define O       1024
#define S3      (S * 3)   /* 144 */
#define E3      (E * 3)   /* 1152 */
#define EPS     1e-5f

/* ============ Embedded XIP weights (构建期可选) ============
 *
 * 当固件以 V10_XIP_EMBED 编译时，weights_blob.S 会把 v10_weights.baize
 * 作为 `.weights_blob` 段链入镜像，板级 ld.script 将其放在 XIP NOR flash
 * 中（紧随其余镜像内容之后）。这里只声明符号，运行时原地解析 ——
 * 权重的 malloc 与拷贝全部省掉，张量指针直接指向 flash。
 */
#ifdef V10_XIP_EMBED
extern const unsigned char v10_weights_blob_start[];
extern const unsigned char v10_weights_blob_end[];
#endif

/* ============ Model struct ============ */
typedef struct {
    int vocab_size, embed_dim, num_layers, max_seq_len, output_dim, sp_dim;
    float r1, r2, r3;   /* fast, medium, slow decay */

    const float *w_embed;       /* [VOCAB, E] */
    const float *skip_gates;     /* [L] */

    /* Per-layer weights (10 tensors per layer) */
    const float *l_write_gw[L];  /* [S, E] */
    const float *l_write_gb[L];  /* [S] */
    const float *l_fast_rw[L];   /* [E, S] */
    const float *l_fast_rb[L];   /* [E] */
    const float *l_med_rw[L];    /* [E, S] */
    const float *l_med_rb[L];    /* [E] */
    const float *l_slow_rw[L];   /* [E, S] */
    const float *l_slow_rb[L];   /* [E] */
    const float *l_fusion_w[L];  /* [3, E3] */
    const float *l_fusion_b[L];  /* [3] */

    const float *bottleneck_w;   /* [S3, E] */
    const float *bottleneck_b;   /* [S3] */
    const float *out_proj_w;     /* [O, S3] */
    const float *out_proj_b;     /* [O] */
    const float *out_norm_w;     /* [O] */
    const float *out_norm_b;     /* [O] */

    /* Runtime buffers */
    float *tok_emb;   /* [N * E] */
    float *hidden;    /* [N * E] */
    float *pool_f;    /* [S] */
    float *pool_m;    /* [S] */
    float *pool_s;    /* [S] */
    float *write_vals;/* [N * S] */
    float *fast_r;    /* [E] */
    float *med_r;     /* [E] */
    float *slow_r;    /* [E] */
    float *concat;    /* [E3] */
    float *mod;       /* [E] */
    float *bottleneck_out; /* [S3] */
    float *output;    /* [O] */

    float *weights;
    long n_floats;
} Model;

/* ============ .baize V10 loader ============ */
float *load_baize(const char *path, long *out_n_floats, int *out_version) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "V10: Cannot open: %s\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);

    uint8_t *buf = (uint8_t *)malloc(fsize);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, fsize, fp) != (size_t)fsize) {
        fprintf(stderr, "V10: Read failed\n");
        free(buf); fclose(fp); return NULL;
    }
    fclose(fp);

    if (buf[0] != 'B' || buf[1] != 'A' || buf[2] != 'I' || buf[3] != 'Z') {
        fprintf(stderr, "V10: Not a valid .baize file\n");
        free(buf); return NULL;
    }

    uint32_t *hdr = (uint32_t *)buf;
    int version = (int)hdr[1];
    long n_floats = (long)hdr[2];

    if (out_version) *out_version = version;
    if (out_n_floats) *out_n_floats = n_floats;

    /* Header is 68 bytes = 17 uint32s */
    long expected_floats = (fsize - 68) / 4;
    if (n_floats != expected_floats) {
        fprintf(stderr, "V10: Param count mismatch: header says %ld, file has %ld\n",
                n_floats, expected_floats);
        free(buf); return NULL;
    }

    /* Return pointer into the buffer (caller owns buf via weights pointer) */
    return (float *)(buf + 68);
}

/* ============ Zero-copy embedded loader ============ */
static float *embedded_weights(const unsigned char *buf,
                               const unsigned char *end,
                               long *out_n_floats,
                               int *out_version) {
    const uint32_t *hdr;
    long n;

    if (buf == NULL || end < buf + 68)
      {
        return NULL;
      }

    if (buf[0] != 'B' || buf[1] != 'A' || buf[2] != 'I' || buf[3] != 'Z')
      {
        return NULL;
      }

    hdr = (const uint32_t *)buf;
    if (hdr[1] != 10)   /* version */
      {
        return NULL;
      }

    n = (long)hdr[2];
    if ((const uint8_t *)(buf + 68) + (size_t)n * 4 > end)
      {
        fprintf(stderr, "V10: embedded blob shorter than header claims\n");
        return NULL;
      }

    if (out_version) *out_version = (int)hdr[1];
    if (out_n_floats) *out_n_floats = n;
    return (float *)(buf + 68);
}

/* ============ Model init / free ============ */
int model_init(Model *m, const char *path) {
    memset(m, 0, sizeof(*m));

    long n_floats;
    int version;
    int embedded = 0;
    float *data;

    if (path == NULL) {
        /* XIP embedded weights: tensor pointers go straight into flash */
#ifdef V10_XIP_EMBED
        data = embedded_weights(v10_weights_blob_start,
                                v10_weights_blob_end,
                                &n_floats, &version);
        if (data == NULL)
          {
            fprintf(stderr, "V10: embedded XIP weights unavailable or corrupt\n");
            return -1;
          }
        embedded = 1;
#else
        fprintf(stderr, "V10: built without embedded weights (V10_XIP_EMBED)\n");
        return -1;
#endif
    } else {
        data = load_baize(path, &n_floats, &version);
        if (!data) return -1;
    }

    if (version != 10) {
        fprintf(stderr, "V10: Wrong .baize version: %d (expected 10)\n", version);
        if (!embedded) free((void *)((uint8_t *)data - 68));
        return -1;
    }

    m->weights = data;
    m->n_floats = n_floats;
    m->vocab_size = VOCAB;
    m->embed_dim = E;
    m->num_layers = L;
    m->max_seq_len = N;
    m->output_dim = O;
    m->sp_dim = S;
    m->r1 = 0.85f;
    m->r2 = 0.95f;
    m->r3 = 0.995f;

    int off = 0;

    /* 1. embedding [VOCAB, E] */
    m->w_embed = data + off; off += VOCAB * E;

    /* 2. skip_gates [L] */
    m->skip_gates = data + off; off += L;

    /* Per-layer (10 tensors per layer) */
    for (int i = 0; i < L; i++) {
        m->l_write_gw[i]  = data + off; off += S * E;   /* [S, E] */
        m->l_write_gb[i]  = data + off; off += S;        /* [S] */
        m->l_fast_rw[i]   = data + off; off += E * S;    /* [E, S] */
        m->l_fast_rb[i]   = data + off; off += E;         /* [E] */
        m->l_med_rw[i]    = data + off; off += E * S;    /* [E, S] */
        m->l_med_rb[i]    = data + off; off += E;         /* [E] */
        m->l_slow_rw[i]   = data + off; off += E * S;    /* [E, S] */
        m->l_slow_rb[i]   = data + off; off += E;         /* [E] */
        m->l_fusion_w[i]  = data + off; off += 3 * E3;   /* [3, E3] */
        m->l_fusion_b[i]  = data + off; off += 3;         /* [3] */
    }

    /* 13. bottleneck [S3, E] */
    m->bottleneck_w = data + off; off += S3 * E;
    /* 14. bottleneck bias [S3] */
    m->bottleneck_b = data + off; off += S3;
    /* 15. output_proj [O, S3] */
    m->out_proj_w = data + off; off += O * S3;
    /* 16. output_proj bias [O] */
    m->out_proj_b = data + off; off += O;
    /* 17. output_norm weight [O] */
    m->out_norm_w = data + off; off += O;
    /* 18. output_norm bias [O] */
    m->out_norm_b = data + off; off += O;

    if (off != n_floats) {
        fprintf(stderr, "V10: Weight layout mismatch: consumed %d, expected %ld\n",
                off, n_floats);
        if (!embedded) free((void *)((uint8_t *)data - 68));
        return -1;
    }

    /* Allocate runtime buffers (weights themselves stay in flash — zero heap) */
    m->tok_emb       = (float *)calloc(N * E, sizeof(float));
    m->hidden        = (float *)calloc(N * E, sizeof(float));
    m->pool_f        = (float *)calloc(S, sizeof(float));
    m->pool_m        = (float *)calloc(S, sizeof(float));
    m->pool_s        = (float *)calloc(S, sizeof(float));
    m->write_vals    = (float *)calloc(N * S, sizeof(float));
    m->fast_r        = (float *)calloc(E, sizeof(float));
    m->med_r         = (float *)calloc(E, sizeof(float));
    m->slow_r        = (float *)calloc(E, sizeof(float));
    m->concat        = (float *)calloc(E3, sizeof(float));
    m->mod           = (float *)calloc(E, sizeof(float));
    m->bottleneck_out= (float *)calloc(S3, sizeof(float));
    m->output        = (float *)calloc(O, sizeof(float));

    if (!m->tok_emb || !m->hidden || !m->pool_f || !m->pool_m || !m->pool_s ||
        !m->write_vals || !m->fast_r || !m->med_r || !m->slow_r ||
        !m->concat || !m->mod || !m->bottleneck_out || !m->output) {
        fprintf(stderr, "V10: Memory allocation failed\n");
        return -1;
    }

    printf("[V10] %s%ld params, weights=%.1f MB, runtime=%.1f KB\n",
           embedded ? "[XIP] " : "",
           n_floats,
           n_floats * 4.0 / 1024 / 1024,
           (N*E*2 + S*3 + N*S + E*3 + E3 + E + S3 + O) * 4.0 / 1024);
    return 0;
}

void model_free(Model *m) {
    if (m->weights) {
#ifdef V10_XIP_EMBED
        if ((const unsigned char *)m->weights >= v10_weights_blob_start &&
            (const unsigned char *)m->weights <  v10_weights_blob_end)
          {
            /* Weights point into the XIP flash blob — nothing to free */
            m->weights = NULL;
          }
        else
#endif
        {
            /* weights points into the malloc'd buffer (data - 68 header bytes) */
            free((void *)((uint8_t *)m->weights - 68));
            m->weights = NULL;
        }
    }
    free(m->tok_emb); free(m->hidden);
    free(m->pool_f); free(m->pool_m); free(m->pool_s);
    free(m->write_vals);
    free(m->fast_r); free(m->med_r); free(m->slow_r);
    free(m->concat); free(m->mod);
    free(m->bottleneck_out);
    free(m->output);
    memset(m, 0, sizeof(*m));
}

/* ============ Tokenizer (char → tokenizer_id → remapped_id) ============
 * For the C engine we use a simplified approach: the tokenizer map is
 * embedded as a lookup table. For now we use ord(c) % VOCAB as fallback
 * — the real tokenizer map should be loaded from a file for production.
 * TODO: embed the 4958-entry char→id mapping.
 */
static int tokenize(const char *text, int *tokens, int max_len) {
    int len = 0;
    const unsigned char *p = (const unsigned char *)text;
    while (*p && len < max_len) {
        /* UTF-8 decode */
        uint32_t cp;
        if (p[0] < 0x80) {
            cp = p[0]; p += 1;
        } else if ((p[0] & 0xE0) == 0xC0) {
            cp = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F); p += 2;
        } else if ((p[0] & 0xF0) == 0xE0) {
            cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); p += 3;
        } else if ((p[0] & 0xF8) == 0xF0) {
            cp = ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F); p += 4;
        } else {
            cp = *p; p += 1;
        }
        /* Lookup in embedded char→token_id mapping (binary search) */
        tokens[len++] = v10_lookup_token(cp);
    }
    return len;
}

/* ============ Core math ============ */

/* y = W @ x + b   where W is [rows, cols], x is [cols], y is [rows] */
static void matvec(const float *W, const float *x, const float *b,
                   float *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = b ? b[i] : 0.0f;
        const float *w = W + i * cols;
        int j = 0;
        for (; j + 3 < cols; j += 4)
            sum += w[j]*x[j] + w[j+1]*x[j+1] + w[j+2]*x[j+2] + w[j+3]*x[j+3];
        for (; j < cols; j++)
            sum += w[j] * x[j];
        y[i] = sum;
    }
}

static void layernorm(float *x, const float *w, const float *b, int n) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + EPS);
    for (int i = 0; i < n; i++)
        x[i] = (x[i] - mean) * inv * w[i] + b[i];
}

static void softmax3(float *x) {
    float max = x[0];
    if (x[1] > max) max = x[1];
    if (x[2] > max) max = x[2];
    float s = expf(x[0]-max) + expf(x[1]-max) + expf(x[2]-max);
    float inv = 1.0f / s;
    x[0] = expf(x[0]-max) * inv;
    x[1] = expf(x[1]-max) * inv;
    x[2] = expf(x[2]-max) * inv;
}

static void l2norm(float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float inv = 1.0f / sqrtf(sum + 1e-12f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ============ PureSourcePool layer ============ */
static void pure_pool_layer(Model *m, const float *x_seq, int seq_len,
                            float *out_mod, int layer) {
    /* x_seq: [seq_len * E]  (token embeddings for this sequence) */

    /* 1. write_vals = tanh(write_gate(x))  → [seq_len, S] */
    for (int t = 0; t < seq_len; t++) {
        const float *x_t = x_seq + t * E;
        float *w_t = m->write_vals + t * S;
        matvec(m->l_write_gw[layer], x_t, m->l_write_gb[layer], w_t, S, E);
        for (int i = 0; i < S; i++)
            w_t[i] = tanhf(w_t[i]);
    }

    /* 2. Geometric decay pooling:
     *   pool = sum_t(decay^(T-1-t) * write[t]) * (1-decay) */
    /* fast pool */
    memset(m->pool_f, 0, S * sizeof(float));
    memset(m->pool_m, 0, S * sizeof(float));
    memset(m->pool_s, 0, S * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        float wf = powf(m->r1, seq_len - 1 - t) * (1.0f - m->r1);
        float wm = powf(m->r2, seq_len - 1 - t) * (1.0f - m->r2);
        float ws = powf(m->r3, seq_len - 1 - t) * (1.0f - m->r3);
        const float *wv = m->write_vals + t * S;
        for (int i = 0; i < S; i++) {
            m->pool_f[i] += wf * wv[i];
            m->pool_m[i] += wm * wv[i];
            m->pool_s[i] += ws * wv[i];
        }
    }

    /* 3. Read: fast_r, med_r, slow_r */
    matvec(m->l_fast_rw[layer], m->pool_f, m->l_fast_rb[layer], m->fast_r, E, S);
    matvec(m->l_med_rw[layer],  m->pool_m,  m->l_med_rb[layer],  m->med_r,  E, S);
    matvec(m->l_slow_rw[layer], m->pool_s, m->l_slow_rb[layer], m->slow_r, E, S);

    /* 4. Fusion: softmax over 3 reads */
    memcpy(m->concat, m->fast_r, E * sizeof(float));
    memcpy(m->concat + E, m->med_r, E * sizeof(float));
    memcpy(m->concat + 2*E, m->slow_r, E * sizeof(float));

    float logits[3];
    matvec(m->l_fusion_w[layer], m->concat, m->l_fusion_b[layer], logits, 3, E3);
    softmax3(logits);

    /* mod = logits[0]*fast + logits[1]*med + logits[2]*slow */
    for (int i = 0; i < E; i++)
        m->mod[i] = logits[0]*m->fast_r[i] + logits[1]*m->med_r[i] + logits[2]*m->slow_r[i];

    memcpy(out_mod, m->mod, E * sizeof(float));
}

/* ============ Forward pass ============ */
void model_forward(Model *m, const char *text, float *out_vec) {
    int tokens[N];
    int seq_len = tokenize(text, tokens, N);

    /* Empty input → zero vector */
    if (seq_len < 1) {
        memset(out_vec, 0, O * sizeof(float));
        return;
    }

    /* 1. Token embedding (no position encoding in V10) */
    for (int i = 0; i < seq_len; i++) {
        const float *emb = m->w_embed + tokens[i] * E;
        memcpy(m->tok_emb + i * E, emb, E * sizeof(float));
    }

    /* 2. PurePoolStack: 6 layers */
    float *buf_input = m->tok_emb;
    float *buf_output = m->hidden;

    /* residual = mean of token embeddings [E] */
    float residual[E];
    memset(residual, 0, E * sizeof(float));
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < E; j++)
            residual[j] += buf_input[i * E + j];
    float inv_len = 1.0f / seq_len;
    for (int j = 0; j < E; j++) residual[j] *= inv_len;

    /* Layer 0 */
    float mod[E];
    pure_pool_layer(m, buf_input, seq_len, mod, 0);
    /* current = mod */
    float current[E];
    memcpy(current, mod, E * sizeof(float));

    /* Layers 1..L-1 */
    for (int layer = 1; layer < L; layer++) {
        /* skip = residual * sigmoid(skip_gates[layer]) */
        float sg = 1.0f / (1.0f + expf(-m->skip_gates[layer]));
        float skip[E];
        for (int j = 0; j < E; j++)
            skip[j] = residual[j] * sg;

        /* pseudo_seq = x + current.unsqueeze(1)  → each token gets current added */
        for (int t = 0; t < seq_len; t++) {
            for (int j = 0; j < E; j++)
                buf_output[t * E + j] = buf_input[t * E + j] + current[j];
        }

        pure_pool_layer(m, buf_output, seq_len, mod, layer);

        /* current = mod + skip */
        for (int j = 0; j < E; j++)
            current[j] = mod[j] + skip[j];

        /* swap buffers for next layer's pseudo_seq */
        float *swap = buf_input;
        buf_input = buf_output;
        buf_output = swap;
    }

    /* 3. Bottleneck: 384→144, tanh */
    matvec(m->bottleneck_w, current, m->bottleneck_b, m->bottleneck_out, S3, E);
    for (int i = 0; i < S3; i++)
        m->bottleneck_out[i] = tanhf(m->bottleneck_out[i]);

    /* 4. Output projection: 144→1024 */
    matvec(m->out_proj_w, m->bottleneck_out, m->out_proj_b, m->output, O, S3);

    /* 5. Output LayerNorm */
    layernorm(m->output, m->out_norm_w, m->out_norm_b, O);

    /* 6. L2 normalization */
    l2norm(m->output, O);

    memcpy(out_vec, m->output, O * sizeof(float));
}

/* ============ Self-test main (compile standalone) ============ */
#ifdef V10_STANDALONE
int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "v10_weights.baize";
    const char *text = argc > 2 ? argv[2] : "光合作用产生氧气";

    Model model;
    if (model_init(&model, model_path) != 0) return 1;

    printf("=== V10 C Inference Engine ===\n");
    printf("Input: %s\n\n", text);

    float output[O];
    model_forward(&model, text, output);

    printf("Output [0..5]: %.6f %.6f %.6f %.6f %.6f\n",
           output[0], output[1], output[2], output[3], output[4]);

    double l2 = 0;
    for (int i = 0; i < O; i++) l2 += (double)output[i] * output[i];
    l2 = sqrt(l2);
    printf("L2: %.6f %s\n", l2, fabs(l2 - 1.0) < 0.001 ? "OK" : "OFF");

    model_free(&model);
    return 0;
}
#endif
/**
 * v10.c - BaizeEmbeddingV10 C Engine (Pure Source Pool)
 * Fixed: pool scaling, skip gate, residual, pseudo-sequence for layers > 0
 */
#include "v10.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BAIZE_MAGIC "BAIZE"
#define BAIZE_VERSION_V10 2

typedef struct {
    char name[52];
    uint32_t offset;
    uint32_t size;
} tensor_entry_t;

typedef struct {
    const float *write_gate_w, *write_gate_b;
    const float *fast_read_w, *fast_read_b;
    const float *medium_read_w, *medium_read_b;
    const float *slow_read_w, *slow_read_b;
    const float *fusion_gate_w, *fusion_gate_b;
} pool_layer_t;

struct v10_model {
    const float *embedding;
    const float *skip_gates;
    pool_layer_t layers[V10_NUM_LAYERS];
    const float *bottleneck_w, *bottleneck_b;
    const float *output_proj_w, *output_proj_b;
    const float *output_norm_w, *output_norm_b;
    void *raw_buf;
};

static v10_model_t *g_model = NULL;

static const float *find_tensor(const uint8_t *base, const tensor_entry_t *entries,
                                 int n, const char *suffix)
{
    for (int i = 0; i < n; i++) {
        if (strstr(entries[i].name, suffix)) return (const float *)(base + entries[i].offset);
    }
    return NULL;
}

int v10_init(const char *weight_path)
{
    FILE *f = fopen(weight_path, "rb");
    if (!f) { fprintf(stderr, "v10_init: cannot open %s\n", weight_path); return -1; }

    char magic[6] = {0};
    uint32_t version, num_tensors;
    fread(magic, 1, 5, f);
    fread(&version, 4, 1, f);
    fread(&num_tensors, 4, 1, f);
    if (memcmp(magic, BAIZE_MAGIC, 5) != 0 || version != BAIZE_VERSION_V10) {
        fprintf(stderr, "v10_init: bad magic/version\n"); fclose(f); return -1;
    }

    tensor_entry_t *entries = malloc(num_tensors * sizeof(tensor_entry_t));
    for (uint32_t i = 0; i < num_tensors; i++) {
        fread(entries[i].name, 1, 52, f);
        fread(&entries[i].offset, 4, 1, f);
        fread(&entries[i].size, 4, 1, f);
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    uint8_t *buf = malloc(file_size);
    fseek(f, 0, SEEK_SET);
    fread(buf, 1, file_size, f);
    fclose(f);

    v10_model_t *m = calloc(1, sizeof(v10_model_t));
    m->raw_buf = buf;
    m->embedding     = find_tensor(buf, entries, num_tensors, "embedding.weight");
    m->skip_gates    = find_tensor(buf, entries, num_tensors, "skip_gates");
    m->bottleneck_w  = find_tensor(buf, entries, num_tensors, "bottleneck.weight");
    m->bottleneck_b  = find_tensor(buf, entries, num_tensors, "bottleneck.bias");
    m->output_proj_w = find_tensor(buf, entries, num_tensors, "output_proj.weight");
    m->output_proj_b = find_tensor(buf, entries, num_tensors, "output_proj.bias");
    m->output_norm_w = find_tensor(buf, entries, num_tensors, "output_norm.weight");
    m->output_norm_b = find_tensor(buf, entries, num_tensors, "output_norm.bias");

    for (int i = 0; i < V10_NUM_LAYERS; i++) {
        char s[64];
        pool_layer_t *L = &m->layers[i];
        snprintf(s, sizeof(s), "layers.%d.write_gate.weight", i); L->write_gate_w = find_tensor(buf, entries, num_tensors, s);
        snprintf(s, sizeof(s), "layers.%d.write_gate.bias", i);   L->write_gate_b = find_tensor(buf, entries, num_tensors, s);
        snprintf(s, sizeof(s), "layers.%d.fast_read.weight", i);  L->fast_read_w = find_tensor(buf, entries, num_tensors, s);
        snprintf(s, sizeof(s), "layers.%d.fast_read.bias", i);    L->fast_read_b = find_tensor(buf, entries, num_tensors, s);
        snprintf(s, sizeof(s), "layers.%d.medium_read.weight", i); L->medium_read_w = find_tensor(buf, entries, num_tensors, s);
        snprintf(s, sizeof(s), "layers.%d.medium_read.bias", i);  L->medium_read_b = find_tensor(buf, entries, num_tensors, s);
        snprintf(s, sizeof(s), "layers.%d.slow_read.weight", i);  L->slow_read_w = find_tensor(buf, entries, num_tensors, s);
        snprintf(s, sizeof(s), "layers.%d.slow_read.bias", i);    L->slow_read_b = find_tensor(buf, entries, num_tensors, s);
        snprintf(s, sizeof(s), "layers.%d.fusion_gate.weight", i); L->fusion_gate_w = find_tensor(buf, entries, num_tensors, s);
        snprintf(s, sizeof(s), "layers.%d.fusion_gate.bias", i);  L->fusion_gate_b = find_tensor(buf, entries, num_tensors, s);
    }

    free(entries);
    g_model = m;
    printf("v10_init: loaded %u tensors\n", num_tensors);
    return 0;
}

int v10_dim(void) { return V10_OUTPUT_DIM; }

void v10_free(void)
{
    if (g_model) { free(g_model->raw_buf); free(g_model); g_model = NULL; }
}

/* ---- Math helpers ---- */
static void linear(const float *W, const float *bias, const float *x, float *y, int out_rows, int in_cols)
{
    for (int i = 0; i < out_rows; i++) {
        float s = bias ? bias[i] : 0.0f;
        const float *row = W + i * in_cols;
        for (int j = 0; j < in_cols; j++) s += row[j] * x[j];
        y[i] = s;
    }
}

static void softmax(float *x, int n)
{
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void layer_norm(float *x, int n, const float *weight, const float *bias)
{
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
}

static void l2_normalize(float *x, int n)
{
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += x[i] * x[i];
    norm = sqrtf(norm);
    if (norm > 1e-8f) for (int i = 0; i < n; i++) x[i] /= norm;
}

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---- Tokenizer: decode UTF-8 codepoint, then ord % 4096 ---- */
static int utf8_decode(const char **pp)
{
    const unsigned char *p = (const unsigned char *)*pp;
    int cp;
    if (*p < 0x80) { cp = *p++; }
    else if ((*p & 0xE0) == 0xC0) { cp = (*p++ & 0x1F) << 6; cp |= (*p++ & 0x3F); }
    else if ((*p & 0xF0) == 0xE0) { cp = (*p++ & 0x0F) << 12; cp |= (*p++ & 0x3F) << 6; cp |= (*p++ & 0x3F); }
    else if ((*p & 0xF8) == 0xF0) { cp = (*p++ & 0x07) << 18; cp |= (*p++ & 0x3F) << 12; cp |= (*p++ & 0x3F) << 6; cp |= (*p++ & 0x3F); }
    else { cp = *p++; }
    *pp = (const char *)p;
    return cp;
}

static int tokenize(const char *text, int *tokens, int max_len)
{
    int len = 0;
    while (*text && len < max_len) {
        int cp = utf8_decode(&text);
        tokens[len++] = cp % V10_VOCAB_SIZE;
    }
    return len;
}

/* ---- PureSourcePoolLayer forward ---- */
static void pool_layer_forward(const pool_layer_t *L,
                                const float *seq, int seq_len,
                                float *out)  /* out[embed_dim] */
{
    float write_buf[V10_MAX_SEQ_LEN * V10_SP_DIM];
    float fast_pool[V10_SP_DIM] = {0};
    float medium_pool[V10_SP_DIM] = {0};
    float slow_pool[V10_SP_DIM] = {0};

    /* 1. All tokens through write gate */
    for (int t = 0; t < seq_len; t++) {
        const float *x_t = seq + t * V10_EMBED_DIM;
        float *w_t = write_buf + t * V10_SP_DIM;
        linear(L->write_gate_w, L->write_gate_b, x_t, w_t, V10_SP_DIM, V10_EMBED_DIM);
        for (int i = 0; i < V10_SP_DIM; i++) w_t[i] = tanhf(w_t[i]);
    }

    /* 2. Vectorized pool accumulation: pool = (1-decay) * sum_t(decay^(T-1-t) * write[t]) */
    for (int t = 0; t < seq_len; t++) {
        float *w_t = write_buf + t * V10_SP_DIM;
        float fast_w   = powf(V10_FAST_DECAY,   (float)(seq_len - 1 - t));
        float medium_w = powf(V10_MEDIUM_DECAY, (float)(seq_len - 1 - t));
        float slow_w   = powf(V10_SLOW_DECAY,   (float)(seq_len - 1 - t));
        for (int i = 0; i < V10_SP_DIM; i++) {
            fast_pool[i]   += fast_w   * w_t[i];
            medium_pool[i] += medium_w * w_t[i];
            slow_pool[i]   += slow_w   * w_t[i];
        }
    }
    /* Scale by (1-decay) */
    for (int i = 0; i < V10_SP_DIM; i++) {
        fast_pool[i]   *= (1.0f - V10_FAST_DECAY);
        medium_pool[i] *= (1.0f - V10_MEDIUM_DECAY);
        slow_pool[i]   *= (1.0f - V10_SLOW_DECAY);
    }

    /* 3. Read from pools */
    float fast_r[V10_EMBED_DIM], medium_r[V10_EMBED_DIM], slow_r[V10_EMBED_DIM];
    linear(L->fast_read_w,   L->fast_read_b,   fast_pool,   fast_r,   V10_EMBED_DIM, V10_SP_DIM);
    linear(L->medium_read_w, L->medium_read_b,  medium_pool, medium_r, V10_EMBED_DIM, V10_SP_DIM);
    linear(L->slow_read_w,   L->slow_read_b,   slow_pool,   slow_r,   V10_EMBED_DIM, V10_SP_DIM);

    /* 4. Fusion gate */
    float fusion_in[3 * V10_EMBED_DIM];
    memcpy(fusion_in,                   fast_r,   sizeof(float) * V10_EMBED_DIM);
    memcpy(fusion_in + V10_EMBED_DIM,   medium_r, sizeof(float) * V10_EMBED_DIM);
    memcpy(fusion_in + 2*V10_EMBED_DIM, slow_r,   sizeof(float) * V10_EMBED_DIM);
    float fusion_w[3];
    linear(L->fusion_gate_w, L->fusion_gate_b, fusion_in, fusion_w, 3, 3 * V10_EMBED_DIM);
    softmax(fusion_w, 3);

    /* 5. Weighted combination */
    for (int i = 0; i < V10_EMBED_DIM; i++) {
        out[i] = fusion_w[0] * fast_r[i] + fusion_w[1] * medium_r[i] + fusion_w[2] * slow_r[i];
    }
}

/* ---- Forward pass ---- */
int v10_embed(const char *text, float *out, int out_dim)
{
    if (!g_model || !text || !out || out_dim < V10_OUTPUT_DIM) return -1;
    v10_model_t *m = g_model;

    int tokens[V10_MAX_SEQ_LEN];
    int seq_len = tokenize(text, tokens, V10_MAX_SEQ_LEN);
    if (seq_len == 0) { memset(out, 0, sizeof(float) * V10_OUTPUT_DIM); return 0; }

    /* Build token embedding sequence: seq[seq_len][embed_dim] */
    float seq[V10_MAX_SEQ_LEN * V10_EMBED_DIM];
    for (int t = 0; t < seq_len; t++) {
        memcpy(seq + t * V10_EMBED_DIM, m->embedding + tokens[t] * V10_EMBED_DIM,
               sizeof(float) * V10_EMBED_DIM);
    }

    /* residual = mean of input embeddings */
    float residual[V10_EMBED_DIM] = {0};
    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < V10_EMBED_DIM; i++) residual[i] += seq[t * V10_EMBED_DIM + i];
    }
    for (int i = 0; i < V10_EMBED_DIM; i++) residual[i] /= seq_len;

    /* Layer 0: process original sequence */
    float current[V10_EMBED_DIM];
    pool_layer_forward(&m->layers[0], seq, seq_len, current);

    /* Layers 1-5: pseudo_seq = x + current, add skip from residual */
    for (int layer = 1; layer < V10_NUM_LAYERS; layer++) {
        /* pseudo_seq[t] = seq[t] + current (broadcast) */
        float pseudo_seq[V10_MAX_SEQ_LEN * V10_EMBED_DIM];
        for (int t = 0; t < seq_len; t++) {
            for (int i = 0; i < V10_EMBED_DIM; i++) {
                pseudo_seq[t * V10_EMBED_DIM + i] = seq[t * V10_EMBED_DIM + i] + current[i];
            }
        }

        float layer_out[V10_EMBED_DIM];
        pool_layer_forward(&m->layers[layer], pseudo_seq, seq_len, layer_out);

        /* skip = residual * sigmoid(skip_gates[layer]) */
        float sg = sigmoidf(m->skip_gates[layer]);
        for (int i = 0; i < V10_EMBED_DIM; i++) {
            current[i] = layer_out[i] + residual[i] * sg;
        }
    }

    /* Bottleneck: [72] = W[72,384] @ current[384] + b */
    float bottleneck_out[V10_POOL_DIM];
    linear(m->bottleneck_w, m->bottleneck_b, current, bottleneck_out, V10_POOL_DIM, V10_EMBED_DIM);

    /* Output projection: [1024] = W[1024,72] @ bottleneck[72] + b */
    float proj_out[V10_OUTPUT_DIM];
    linear(m->output_proj_w, m->output_proj_b, bottleneck_out, proj_out, V10_OUTPUT_DIM, V10_POOL_DIM);

    /* LayerNorm */
    layer_norm(proj_out, V10_OUTPUT_DIM, m->output_norm_w, m->output_norm_b);

    /* L2 normalize */
    l2_normalize(proj_out, V10_OUTPUT_DIM);

    memcpy(out, proj_out, sizeof(float) * V10_OUTPUT_DIM);
    return 0;
}

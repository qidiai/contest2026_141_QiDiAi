/****************************************************************************
 * V10 on-device embedding engine — public API (建木)
 *
 * BaizeEmbeddingV10 — Pure Source Pool architecture.
 * 2.58M params, 1024-d output, 4958 vocab, FP16 weights 4.92MB.
 *
 * Anti-collapse: trained with hyperspherical repulsion + contraction loss.
 *
 * C engine files:
 *   v10_infer.c — forward pass (load_baize, model_init, model_forward)
 *   v10_api.c   — init/embed/free API (links against v10_infer.c)
 *   v10.h       — this header
 ****************************************************************************/

#ifndef V10_H
#define V10_H

#include <stddef.h>

#define V10_DIM      1024
#define V10_VOCAB    4958
#define V10_EMBED    384
#define V10_LAYERS   6
#define V10_SEQ      96
#define V10_SP       48
#define V10_SP3      144   /* sp_dim * 3 */

/* --- implemented in v10_api.c (links against v10_infer.c) --- */
int  v10_init(const char *model_path);
int  v10_embed(const char *text, float *out, int dim);
void v10_free(void);
int  v10_dim(void);

#endif /* V10_H */

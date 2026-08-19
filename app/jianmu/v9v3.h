/****************************************************************************
 * V9v3 on-device embedding engine — public API (建木)
 *
 * The real inference engine is v9v3_infer.c + v9v3_api.c, copied verbatim
 * from jianmu-os/worm_knowledge_engine/c_core. It is a self-contained,
 * libc + libm only C engine (no external deps), exposing:
 *
 *   v9v3_init(path)        load a .baize weight file (12.3MB, 3.15M params)
 *   v9v3_embed(text,out,d) text -> 1024-d L2-normalized vector
 *   v9v3_get_pools(...)    fast/med/slow 24-d semantic pool states
 *   v9v3_free()            release model
 *   v9v3_output_dim()=1024 / v9v3_pool_dim()=24
 *
 * v9v3_dim() is a thin alias kept for our index/demo layer.
 ****************************************************************************/

#ifndef V9V3_H
#define V9V3_H

#include <stddef.h>

/* Real V9v3 output dimension. Pools (fast/med/slow) are V9V3_SP_DIM. */
#define V9V3_DIM    1024
#define V9V3_SP_DIM 24

/* --- implemented in v9v3_api.c (links against v9v3_infer.c) --- */
int  v9v3_init(const char *model_path);
int  v9v3_embed(const char *text, float *out, int dim);
int  v9v3_get_pools(float *fast_pool, float *med_pool,
                    float *slow_pool, int dim);
void v9v3_free(void);
int  v9v3_output_dim(void);
int  v9v3_pool_dim(void);

/* Alias used by our index / demo layer (== v9v3_output_dim). */
int  v9v3_dim(void);

#endif /* V9V3_H */

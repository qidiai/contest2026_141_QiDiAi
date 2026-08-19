/****************************************************************************
 * V9v3 engine glue (建木)
 *
 * The actual inference lives in v9v3_infer.c + v9v3_api.c (copied verbatim
 * from jianmu-os/worm_knowledge_engine/c_core, only main() was renamed so it
 * links into this app). This file provides the single v9v3_dim() alias our
 * index / demo layer expects; everything else is supplied by the engine.
 ****************************************************************************/

#include "v9v3.h"

extern int v9v3_output_dim(void);

int v9v3_dim(void)
{
  return v9v3_output_dim();
}

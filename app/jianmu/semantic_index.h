/****************************************************************************
 * Local semantic index — offline cosine-similarity search over V10 vectors.
 *
 * Stores a small set of personal documents (contacts / notes / messages),
 * each pre-embedded with V10, and answers natural-language queries entirely
 * on-device (no network, no cloud). This is the core of 建木's offline
 * semantic search capability.
 *
 * NOTE: doc vectors are 1024-d (4 KB each). To keep the stack small the
 * doc array is heap-allocated in si_build() — call si_free() when done.
 ****************************************************************************/

#ifndef SEMANTIC_INDEX_H
#define SEMANTIC_INDEX_H

#include "v10.h"

#define SI_MAX_DOCS 32
#define SI_MAX_TEXT 128
#define SI_TOP_K    3

typedef struct
{
  char  text[SI_MAX_TEXT];
  float vec[V10_DIM];
} si_doc_t;

typedef struct
{
  si_doc_t *docs;   /* heap-allocated, idx->count entries (see si_build) */
  int       count;
} semantic_index_t;

/* Build the index from a list of text snippets (embeddings computed at
 * runtime via V10). Allocates idx->docs on the heap. */
void si_build(semantic_index_t *idx, const char *snippets[], int n);

/* Release the heap-allocated doc array. */
void si_free(semantic_index_t *idx);

/* Query the index; fills results[] (doc indices) and scores[] with the
 * top-k most similar documents, sorted by descending similarity.
 * Returns the number of results (<= SI_TOP_K). */
int si_search(const semantic_index_t *idx,
              const char *query,
              int results[SI_TOP_K],
              float scores[SI_TOP_K]);

#endif /* SEMANTIC_INDEX_H */

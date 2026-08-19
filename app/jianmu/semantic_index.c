/****************************************************************************
 * Local semantic index — implementation
 ****************************************************************************/

#include "semantic_index.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

static float cosine(const float *a, const float *b, int dim)
{
  float dot = 0.0f, na = 0.0f, nb = 0.0f;
  for (int i = 0; i < dim; i++)
    {
      dot += a[i] * b[i];
      na  += a[i] * a[i];
      nb  += b[i] * b[i];
    }
  if (na < 1e-8f || nb < 1e-8f)
    {
      return 0.0f;
    }
  return dot / (sqrtf(na) * sqrtf(nb));
}

void si_build(semantic_index_t *idx, const char *snippets[], int n)
{
  if (idx == NULL)
    {
      return;
    }
  idx->count = 0;
  idx->docs  = NULL;

  int cap = (n < SI_MAX_DOCS) ? n : SI_MAX_DOCS;
  if (cap <= 0)
    {
      return;
    }

  idx->docs = (si_doc_t *)calloc((size_t)cap, sizeof(si_doc_t));
  if (idx->docs == NULL)
    {
      return; /* OOM: index stays empty */
    }

  for (int i = 0; i < cap; i++)
    {
      strncpy(idx->docs[idx->count].text, snippets[i], SI_MAX_TEXT - 1);
      idx->docs[idx->count].text[SI_MAX_TEXT - 1] = '\0';
      v10_embed(snippets[i], idx->docs[idx->count].vec, V10_DIM);
      idx->count++;
    }
}

void si_free(semantic_index_t *idx)
{
  if (idx != NULL && idx->docs != NULL)
    {
      free(idx->docs);
      idx->docs  = NULL;
      idx->count = 0;
    }
}

int si_search(const semantic_index_t *idx,
              const char *query,
              int results[SI_TOP_K],
              float scores[SI_TOP_K])
{
  if (idx == NULL || idx->docs == NULL)
    {
      return 0;
    }

  float qvec[V10_DIM];
  if (v10_embed(query, qvec, V10_DIM) < 0)
    {
      return 0;
    }

  /* Keep the top-k results sorted by descending similarity. */
  for (int k = 0; k < SI_TOP_K; k++)
    {
      results[k] = -1;
      scores[k]  = -2.0f;
    }

  for (int i = 0; i < idx->count; i++)
    {
      float s = cosine(qvec, idx->docs[i].vec, V10_DIM);
      if (s <= scores[SI_TOP_K - 1])
        {
          continue; /* not better than current worst */
        }
      int pos = SI_TOP_K - 1;
      while (pos > 0 && s > scores[pos - 1])
        {
          results[pos] = results[pos - 1];
          scores[pos]  = scores[pos - 1];
          pos--;
        }
      results[pos] = i;
      scores[pos]  = s;
    }

  int n = 0;
  for (int k = 0; k < SI_TOP_K; k++)
    {
      if (results[k] >= 0)
        {
          n++;
        }
    }
  return n;
}

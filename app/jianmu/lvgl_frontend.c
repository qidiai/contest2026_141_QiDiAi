/****************************************************************************
 * lvgl_frontend.c — QiDiAi 建木 · LVGL v9 touch front-end (SF32LB52 DevKit
 *                    LCD 1.85" 390x450 RGB565 AMOLED + FT6146 touch)
 *
 * Two screens:
 *   compose — title / quick chips / query textarea / search·clear /
 *             built-in CJK soft keyboard (chars ⊆ embedded MiSans subset)
 *   result  — top-3 semantic hits: rank, similarity score, matched text
 *
 * All retrieval is on-device: v10_embed() + cosine similarity via
 * semantic_index (never leaves the board).  LVGL talks to NuttX through
 * the LVGL v9 NuttX port over /dev/lcd0 (CO5300) and /dev/input0 (FT6146).
 *
 * Entry point: jianmu_gui_run() — called by jianmu_main.c --gui
 ****************************************************************************/

#include <nuttx/config.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(CONFIG_BOARDCTL) && !defined(CONFIG_NSH_ARCHINIT)
#  include <sys/boardctl.h>
#  define JIANMU_NEED_BOARDINIT 1
#endif

#include <lvgl/lvgl.h>

#include "lvgl_frontend.h"
#include "v10.h"
#include "semantic_index.h"

LV_FONT_DECLARE(lv_font_misans_ui);

/* -------------------------------------------------------------------------
 * Screen / layout constants (390x450)
 * ---------------------------------------------------------------------- */
#define SCR_W           390
#define SCR_H           450

#define C_BG            0x191B22
#define C_PANEL         0x232634
#define C_PANEL2        0x2B3040
#define C_BORDER        0x3A4055
#define C_TEXT          0xEFF1F8
#define C_DIM           0x9AA3B8
#define C_ACCENT        0xF0A94C
#define C_ACCENT_BG     0x2B2438

#define TITLE_Y         8
#define SUBTITLE_Y      30
#define CHIPS1_Y        54
#define CHIPS2_Y        88
#define CHIP_W          116
#define CHIP_GAP        6
#define TA_Y            124
#define BTN_Y           176
#define STATUS_Y        212
#define KB_HDR_Y        240
#define KB_Y0           268
#define KB_KEY_W        34
#define KB_KEY_H        30
#define KB_GAP          2
#define KB_SPECIAL_Y    406
#define KB_SPECIAL_H    32

#define MARGIN          12
#define CONTENT_W       (SCR_W - MARGIN * 2)   /* 366 */

#define KB_COLS         10                     /* 10 keys per row */
#define KB_ROWS         4
#define KB_PAGE_CHARS   (KB_COLS * KB_ROWS)    /* 40 keys / page */
#define KB_PAGE_COUNT   3                      /* 121 chars -> 3 pages */
#define KB_X0           ((SCR_W - (KB_COLS * KB_KEY_W + (KB_COLS - 1) * KB_GAP)) / 2)

/* -------------------------------------------------------------------------
 * Textual data
 * ------------------------------------------------------------------------ */

static const char *k_corpus[6] =
{
  "上次聊装修的朋友 小王 微信",
  "周五的团队周会纪要 关于 openvela 移植",
  "妈妈生日提醒 下周二 买蛋糕",
  "健身计划 每周三跑步五公里",
  "装修预算表格 水电改造两万元",
  "围棋课报名 周末上午",
};

static const char *k_chips[6] =
{
  "装修预算", "妈妈生日", "每周跑步",
  "团队周会", "健身计划", "围棋课",
};

/* The 121 CJK characters baked into lv_font_misans_ui (must stay ⊆ font) */
static const char *const k_kb[121] =
{
  "…", "一", "万", "三", "上", "下", "与", "两", "中", "义",
  "买", "二", "于", "五", "会", "信", "修", "健", "元", "入",
  "全", "公", "关", "击", "分", "划", "加", "匹", "午", "去",
  "友", "名", "周", "回", "团", "围", "地", "型", "妈", "始",
  "完", "小", "就", "建", "开", "引", "得", "微", "快", "我",
  "或", "找", "报", "换", "捷", "提", "搜", "改", "文", "方",
  "无", "日", "朋", "木", "末", "本", "条", "果", "查", "格",
  "档", "棋", "植", "模", "次", "步", "每", "水", "法", "测",
  "清", "点", "王", "生", "电", "的", "离", "种", "移", "算",
  "糕", "索", "纪", "线", "结", "绪", "聊", "蛋", "表", "装",
  "要", "计", "试", "询", "语", "说", "课", "跑", "身", "车",
  "载", "输", "这", "选", "造", "配", "醒", "里", "队", "除",
  "预",
};

/* Special-key tokens (stored in key-button user_data) */
enum { SP_SPC = 1, SP_DEL = 2, SP_CLR = 3, SP_GO = 4 };

/* -------------------------------------------------------------------------
 * UI state
 * ------------------------------------------------------------------------ */

typedef struct
{
  lv_obj_t *scr;               /* active screen */
  lv_display_t *disp;          /* display handle from NuttX port */

  lv_obj_t *view_compose;
  lv_obj_t *view_result;

  lv_obj_t *ta;                /* query textarea */
  lv_obj_t *status;            /* compose status line */

  lv_obj_t *kb_pages[KB_PAGE_COUNT];  /* soft-keyboard page containers */
  lv_obj_t *kb_header;         /* "1 / 3" page indicator */

  lv_obj_t *q_echo;            /* result page: echoed query */
  lv_obj_t *empty_lbl;         /* result page: "no match" */
  lv_obj_t *r_rank[SI_TOP_K], *r_score[SI_TOP_K], *r_txt[SI_TOP_K],
           *r_card[SI_TOP_K];

  semantic_index_t idx;
  int kb_page_idx;             /* current keyboard page (0..2) */
  int ready;                   /* model + index built */
} ui_t;

static ui_t g_ui;

/* -------------------------------------------------------------------------
 * Small helpers
 * ------------------------------------------------------------------------ */

static void set_panel(lv_obj_t *o, uint32_t bg, uint32_t border,
                      int32_t r, int32_t bw)
{
  lv_obj_set_style_bg_color(o, lv_color_hex(bg), 0);
  lv_obj_set_style_bg_opa(o, LV_OPA_COVER, 0);
  lv_obj_set_style_border_color(o, lv_color_hex(border), 0);
  lv_obj_set_style_border_width(o, bw, 0);
  lv_obj_set_style_border_opa(o, bw > 0 ? LV_OPA_COVER : LV_OPA_TRANSP, 0);
  lv_obj_set_style_radius(o, r, 0);
}

static lv_obj_t *make_label(lv_obj_t *parent, const char *text,
                            uint32_t color, int32_t x, int32_t y,
                            int32_t w)
{
  lv_obj_t *l = lv_label_create(parent);
  lv_obj_set_style_text_font(l, &lv_font_misans_ui, 0);
  lv_obj_set_style_text_color(l, lv_color_hex(color), 0);
  if (text) lv_label_set_text(l, text);
  lv_obj_set_pos(l, x, y);
  if (w) lv_obj_set_width(l, w);
  return l;
}

static lv_obj_t *make_button(lv_obj_t *parent, const char *text,
                             int32_t w, int32_t h, uint32_t bg,
                             uint32_t border, uint32_t fg)
{
  lv_obj_t *b = lv_button_create(parent);
  lv_obj_remove_flag(b, LV_OBJ_FLAG_SCROLLABLE);
  lv_obj_set_size(b, w, h);
  set_panel(b, bg, border, 8, border ? 1 : 0);
  lv_obj_t *l = lv_label_create(b);
  lv_obj_set_style_text_font(l, &lv_font_misans_ui, 0);
  lv_obj_set_style_text_color(l, lv_color_hex(fg), 0);
  lv_label_set_text(l, text);
  lv_obj_center(l);
  return b;
}

/* Delete the last UTF-8 codepoint at the end of the textarea. */
static void textarea_backspace(void)
{
  const char *txt = lv_textarea_get_text(g_ui.ta);
  const char *p = txt + strlen(txt);
  while (p > txt && ((unsigned char)p[-1] & 0xC0) == 0x80)
    p--;
  if (p == txt)
    return;
  char buf[SI_MAX_TEXT];
  size_t n = (size_t)(p - txt);
  memcpy(buf, txt, n);
  buf[n] = '\0';
  lv_textarea_set_text(g_ui.ta, buf);
  lv_textarea_set_cursor_pos(g_ui.ta, (int32_t)n);
}

/* Decode one UTF-8 codepoint (LVGL wants code points, not bytes). */
static uint32_t utf8_codepoint(const char *s)
{
  unsigned char c = (unsigned char)s[0];
  if (c < 0x80)      return c;
  if ((c & 0xE0) == 0xC0)
    return ((uint32_t)(c & 0x1F) << 6) | (s[1] & 0x3F);
  if ((c & 0xF0) == 0xE0)
    return ((uint32_t)(c & 0x0F) << 12) |
           (((uint32_t)(s[1] & 0x3F)) << 6) | (s[2] & 0x3F);
  return c;
}

/* -------------------------------------------------------------------------
 * Event callbacks
 * ------------------------------------------------------------------------ */

static void run_search(const char *query);
static void show_result(const char *query, const int res[SI_TOP_K],
                        const float sc[SI_TOP_K], int n);

static void on_chip_clicked(lv_event_t *e)
{
  const char *q = (const char *)lv_event_get_user_data(e);
  lv_textarea_set_text(g_ui.ta, q);
  lv_textarea_set_cursor_pos(g_ui.ta, (int32_t)strlen(q));
  run_search(q);
}

static void on_char_clicked(lv_event_t *e)
{
  const char *ch = (const char *)lv_event_get_user_data(e);
  uint32_t cp = utf8_codepoint(ch);
  lv_textarea_add_char(g_ui.ta, cp);
  lv_textarea_set_cursor_pos(
      g_ui.ta, (int32_t)strlen(lv_textarea_get_text(g_ui.ta)));
}

static void on_special_clicked(lv_event_t *e)
{
  intptr_t tok = (intptr_t)lv_event_get_user_data(e);
  switch (tok)
    {
      case SP_SPC:
        lv_textarea_add_char(g_ui.ta, ' ');
        break;
      case SP_DEL:
        textarea_backspace();
        break;
      case SP_CLR:
        lv_textarea_set_text(g_ui.ta, "");
        break;
      case SP_GO:
        run_search(lv_textarea_get_text(g_ui.ta));
        break;
    }
}

static void kb_show_page(int page)
{
  if (page < 0 || page >= KB_PAGE_COUNT)
    return;
  g_ui.kb_page_idx = page;
  for (int i = 0; i < KB_PAGE_COUNT; i++)
    {
      if (i == page)
        lv_obj_remove_flag(g_ui.kb_pages[i], LV_OBJ_FLAG_HIDDEN);
      else
        lv_obj_add_flag(g_ui.kb_pages[i], LV_OBJ_FLAG_HIDDEN);
    }
  lv_label_set_text_fmt(g_ui.kb_header, "%d / %d", page + 1, KB_PAGE_COUNT);
}

static void on_page_prev(lv_event_t *e)
{
  (void)e;
  kb_show_page(g_ui.kb_page_idx - 1);
}

static void on_page_next(lv_event_t *e)
{
  (void)e;
  kb_show_page(g_ui.kb_page_idx + 1);
}

static void on_btn_search(lv_event_t *e)
{
  (void)e;
  run_search(lv_textarea_get_text(g_ui.ta));
}

static void on_btn_clear(lv_event_t *e)
{
  (void)e;
  lv_textarea_set_text(g_ui.ta, "");
}

static void on_btn_back(lv_event_t *e)
{
  (void)e;
  lv_obj_add_flag(g_ui.view_result, LV_OBJ_FLAG_HIDDEN);
  lv_obj_remove_flag(g_ui.view_compose, LV_OBJ_FLAG_HIDDEN);
  lv_refr_now(g_ui.disp);
}

/* -------------------------------------------------------------------------
 * Search + result rendering
 * ------------------------------------------------------------------------ */

static void run_search(const char *query)
{
  if (!g_ui.ready)
    {
      lv_label_set_text(g_ui.status, "搜索未就绪");
      return;
    }
  if (!query || query[0] == '\0')
    {
      lv_label_set_text(g_ui.status, "输查询或点快捷词");
      return;
    }

  lv_label_set_text(g_ui.status, "搜索中…");
  lv_refr_now(g_ui.disp);

  int   res[SI_TOP_K];
  float sc[SI_TOP_K];
  memset(res, 0, sizeof(res));
  memset(sc, 0, sizeof(sc));
  int n = si_search(&g_ui.idx, query, res, sc);

  show_result(query, res, sc, n);

  lv_obj_add_flag(g_ui.view_compose, LV_OBJ_FLAG_HIDDEN);
  lv_obj_remove_flag(g_ui.view_result, LV_OBJ_FLAG_HIDDEN);
  lv_refr_now(g_ui.disp);
}

static void show_result(const char *query, const int res[SI_TOP_K],
                        const float sc[SI_TOP_K], int n)
{
  char buf[SI_MAX_TEXT + 16];

  snprintf(buf, sizeof(buf), "查询: %s", query);
  lv_label_set_text(g_ui.q_echo, buf);

  for (int i = 0; i < SI_TOP_K; i++)
    {
      if (i < n)
        {
          lv_label_set_text_fmt(g_ui.r_rank[i], "%d", i + 1);
          lv_label_set_text_fmt(g_ui.r_score[i], "得分 %.3f",
                                (double)sc[i]);
          lv_label_set_text(g_ui.r_txt[i], g_ui.idx.docs[res[i]].text);
          lv_obj_remove_flag(g_ui.r_card[i], LV_OBJ_FLAG_HIDDEN);
        }
      else
        {
          lv_obj_add_flag(g_ui.r_card[i], LV_OBJ_FLAG_HIDDEN);
        }
    }

  if (n <= 0)
    lv_obj_remove_flag(g_ui.empty_lbl, LV_OBJ_FLAG_HIDDEN);
  else
    lv_obj_add_flag(g_ui.empty_lbl, LV_OBJ_FLAG_HIDDEN);
}

/* -------------------------------------------------------------------------
 * View construction
 * ------------------------------------------------------------------------ */

/* Builds one page of letter keys (40 chars) into `parent`.
 * `parent` is positioned at (0, KB_Y0), so keys use row offsets. */
static void build_kb_page(int page, lv_obj_t *parent)
{
  int start = page * KB_PAGE_CHARS;
  int end   = start + KB_PAGE_CHARS;
  if (end > (int)(sizeof(k_kb) / sizeof(k_kb[0])))
    end = (int)(sizeof(k_kb) / sizeof(k_kb[0]));

  for (int i = start; i < end; i++)
    {
      int col = (i - start) % KB_COLS;
      int row = (i - start) / KB_COLS;
      lv_obj_t *b = lv_button_create(parent);
      lv_obj_remove_flag(b, LV_OBJ_FLAG_SCROLLABLE);
      lv_obj_set_size(b, KB_KEY_W, KB_KEY_H);
      set_panel(b, C_PANEL2, 0x2E3448, 6, 0);
      lv_obj_set_pos(b, KB_X0 + col * (KB_KEY_W + KB_GAP),
                        row * (KB_KEY_H + KB_GAP));
      lv_obj_t *l = lv_label_create(b);
      lv_obj_set_style_text_font(l, &lv_font_misans_ui, 0);
      lv_obj_set_style_text_color(l, lv_color_hex(C_TEXT), 0);
      lv_label_set_text(l, k_kb[i]);
      lv_obj_center(l);
      lv_obj_set_user_data(b, (void *)k_kb[i]);
      lv_obj_add_event_cb(b, on_char_clicked, LV_EVENT_CLICKED, NULL);
    }
}

static void build_keyboard(lv_obj_t *parent)
{
  /* header: '<'  "1 / 3"  '>' */
  lv_obj_t *pp = make_button(parent, "<", 34, 24, C_PANEL, C_BORDER, C_DIM);
  lv_obj_set_pos(pp, KB_X0, KB_HDR_Y - 3);
  lv_obj_add_event_cb(pp, on_page_prev, LV_EVENT_CLICKED, NULL);

  lv_obj_t *pn = make_button(parent, ">", 34, 24, C_PANEL, C_BORDER, C_DIM);
  lv_obj_set_pos(pn, SCR_W - KB_X0 - 34, KB_HDR_Y - 3);
  lv_obj_add_event_cb(pn, on_page_next, LV_EVENT_CLICKED, NULL);

  g_ui.kb_header = make_label(parent, "1 / 3", C_DIM,
                              (SCR_W - 120) / 2, KB_HDR_Y, 120);
  lv_obj_set_style_text_align(g_ui.kb_header, LV_TEXT_ALIGN_CENTER, 0);

  /* keyboard pages: transparent containers sized to the key zone only,
   * so they never intercept touches aimed at the chips/textarea above. */
  const int kb_zone_h = KB_ROWS * (KB_KEY_H + KB_GAP) - KB_GAP;
  for (int p = 0; p < KB_PAGE_COUNT; p++)
    {
      lv_obj_t *page = lv_obj_create(parent);
      lv_obj_remove_flag(page, LV_OBJ_FLAG_SCROLLABLE);
      lv_obj_remove_flag(page, LV_OBJ_FLAG_CLICKABLE);
      lv_obj_set_pos(page, 0, KB_Y0);
      lv_obj_set_size(page, SCR_W, kb_zone_h);
      lv_obj_set_style_bg_opa(page, LV_OPA_TRANSP, 0);
      lv_obj_set_style_border_width(page, 0, 0);
      g_ui.kb_pages[p] = page;
      build_kb_page(p, page);
    }
}

static void build_special_row(lv_obj_t *parent)
{
  int  x = MARGIN;
  int  gap = 6;
  int  w[4] = { 120, 76, 76, 84 };
  const char *label[4] = { "SPC", "DEL", "CLR", "GO" };
  const intptr_t tok[4] = { SP_SPC, SP_DEL, SP_CLR, SP_GO };
  const uint32_t bg[4] = { C_PANEL2, C_PANEL2, C_PANEL2, C_ACCENT_BG };
  const uint32_t fg[4] = { C_TEXT, C_TEXT, C_TEXT, C_ACCENT };

  for (int i = 0; i < 4; i++)
    {
      lv_obj_t *b = lv_button_create(parent);
      lv_obj_remove_flag(b, LV_OBJ_FLAG_SCROLLABLE);
      lv_obj_set_size(b, w[i], KB_SPECIAL_H);
      set_panel(b, bg[i], 0x2E3448, 8, 1);
      lv_obj_set_pos(b, x, KB_SPECIAL_Y);
      lv_obj_t *l = lv_label_create(b);
      lv_obj_set_style_text_font(l, &lv_font_misans_ui, 0);
      lv_obj_set_style_text_color(l, lv_color_hex(fg[i]), 0);
      lv_label_set_text(l, label[i]);
      lv_obj_center(l);
      lv_obj_set_user_data(b, (void *)tok[i]);
      lv_obj_add_event_cb(b, on_special_clicked, LV_EVENT_CLICKED, NULL);
      x += w[i] + gap;
    }
}

static lv_obj_t *build_chip(lv_obj_t *parent, const char *text)
{
  lv_obj_t *c = lv_button_create(parent);
  lv_obj_remove_flag(c, LV_OBJ_FLAG_SCROLLABLE);
  lv_obj_set_size(c, CHIP_W, 30);
  set_panel(c, C_PANEL, C_BORDER, 15, 1);
  lv_obj_t *l = lv_label_create(c);
  lv_obj_set_style_text_font(l, &lv_font_misans_ui, 0);
  lv_obj_set_style_text_color(l, lv_color_hex(C_DIM), 0);
  lv_label_set_text(l, text);
  lv_obj_center(l);
  lv_obj_set_user_data(c, (void *)text);
  lv_obj_add_event_cb(c, on_chip_clicked, LV_EVENT_CLICKED, NULL);
  return c;
}

static void build_compose(void)
{
  lv_obj_t *scr = g_ui.scr;
  lv_obj_t *v = lv_obj_create(scr);
  lv_obj_remove_flag(v, LV_OBJ_FLAG_SCROLLABLE);
  lv_obj_set_pos(v, 0, 0);
  lv_obj_set_size(v, SCR_W, SCR_H);
  set_panel(v, C_BG, 0, 0, 0);
  g_ui.view_compose = v;

  /* header */
  make_label(v, "建木 离线语义搜索", C_TEXT, MARGIN, TITLE_Y, CONTENT_W);
  make_label(v, "完全离线 本地计算", C_DIM, MARGIN, SUBTITLE_Y, CONTENT_W);

  /* quick chips (two rows of three) */
  for (int i = 0; i < 3; i++)
    lv_obj_set_pos(build_chip(v, k_chips[i]),
                   MARGIN + i * (CHIP_W + CHIP_GAP), CHIPS1_Y);
  for (int i = 3; i < 6; i++)
    lv_obj_set_pos(build_chip(v, k_chips[i]),
                   MARGIN + (i - 3) * (CHIP_W + CHIP_GAP), CHIPS2_Y);

  /* query textarea */
  g_ui.ta = lv_textarea_create(v);
  lv_obj_set_size(g_ui.ta, CONTENT_W, 40);
  lv_obj_set_pos(g_ui.ta, MARGIN, TA_Y);
  lv_obj_set_style_text_font(g_ui.ta, &lv_font_misans_ui, 0);
  lv_obj_set_style_text_color(g_ui.ta, lv_color_hex(C_TEXT), 0);
  lv_obj_set_style_bg_color(g_ui.ta, lv_color_hex(C_PANEL), 0);
  lv_obj_set_style_border_color(g_ui.ta, lv_color_hex(C_BORDER), 0);
  lv_obj_set_style_border_width(g_ui.ta, 1, 0);
  lv_obj_set_style_radius(g_ui.ta, 8, 0);
  lv_textarea_set_placeholder_text(g_ui.ta, "输入查询…");

  /* search + clear */
  lv_obj_t *b_search =
      make_button(v, "搜索", 220, 32, C_ACCENT_BG, C_ACCENT, C_ACCENT);
  lv_obj_set_pos(b_search, MARGIN, BTN_Y);
  lv_obj_add_event_cb(b_search, on_btn_search, LV_EVENT_CLICKED, NULL);

  lv_obj_t *b_clear =
      make_button(v, "清除", CONTENT_W - 220 - 10, 32, C_PANEL, C_BORDER, C_DIM);
  lv_obj_set_pos(b_clear, MARGIN + 230, BTN_Y);
  lv_obj_add_event_cb(b_clear, on_btn_clear, LV_EVENT_CLICKED, NULL);

  /* status line */
  g_ui.status = make_label(v, "", C_ACCENT, MARGIN, STATUS_Y, CONTENT_W);

  build_keyboard(v);
  build_special_row(v);
}

static void build_result(void)
{
  lv_obj_t *scr = g_ui.scr;
  lv_obj_t *v = lv_obj_create(scr);
  lv_obj_remove_flag(v, LV_OBJ_FLAG_SCROLLABLE);
  lv_obj_set_pos(v, 0, 0);
  lv_obj_set_size(v, SCR_W, SCR_H);
  set_panel(v, C_BG, 0, 0, 0);
  g_ui.view_result = v;

  /* back button */
  lv_obj_t *back = make_button(v, "< 返回", 88, 34, C_PANEL, C_BORDER, C_TEXT);
  lv_obj_set_pos(back, MARGIN, 14);
  lv_obj_add_event_cb(back, on_btn_back, LV_EVENT_CLICKED, NULL);

  /* echoed query */
  g_ui.q_echo = make_label(v, "查询: ", C_DIM, MARGIN + 100, 22,
                          CONTENT_W - 100);
  lv_obj_set_style_text_font(g_ui.q_echo, &lv_font_misans_ui, 0);

  /* top-3 cards */
  const int card_h = 92;
  for (int i = 0; i < SI_TOP_K; i++)
    {
      int y = 64 + i * (card_h + 14);
      lv_obj_t *card = lv_obj_create(v);
      lv_obj_remove_flag(card, LV_OBJ_FLAG_SCROLLABLE);
      lv_obj_set_pos(card, MARGIN, y);
      lv_obj_set_size(card, CONTENT_W, card_h);
      set_panel(card, C_PANEL, C_BORDER, 10, 1);
      g_ui.r_card[i] = card;

      g_ui.r_rank[i] = make_label(card, "0", C_ACCENT, 14, 10, 30);
      g_ui.r_score[i] = make_label(card, "", C_DIM, 260, 10, 92);
      lv_obj_set_style_text_align(g_ui.r_score[i], LV_TEXT_ALIGN_RIGHT, 0);

      g_ui.r_txt[i] = make_label(card, "", C_TEXT, 14, 36, CONTENT_W - 28);
      lv_obj_set_style_text_font(g_ui.r_txt[i], &lv_font_misans_ui, 0);
    }

  /* empty state */
  g_ui.empty_lbl = make_label(v, "无匹配结果", C_DIM,
                              MARGIN, SCR_H / 2, CONTENT_W);
}

/* -------------------------------------------------------------------------
 * Entry
 * ------------------------------------------------------------------------ */

int jianmu_gui_run(void)
{
  memset(&g_ui, 0, sizeof(g_ui));
  g_ui.kb_page_idx = 0;

  lv_nuttx_dsc_t info;
  lv_nuttx_result_t result;

  if (lv_is_initialized())
    lv_deinit();
  lv_init();

#ifdef JIANMU_NEED_BOARDINIT
  boardctl(BOARDIOC_INIT, 0);
#endif

  lv_nuttx_dsc_init(&info);
#ifdef CONFIG_LV_USE_NUTTX_LCD
  info.fb_path = "/dev/lcd0";
#else
  info.fb_path = "/dev/fb0";
#endif
#if defined(CONFIG_LV_USE_NUTTX_TOUCHSCREEN) || \
    defined(CONFIG_INPUT_TOUCHSCREEN)
  info.input_path = "/dev/input0";   /* FT6146 capacitive touch */
#endif

  lv_nuttx_init(&info, &result);
  if (result.disp == NULL)
    {
      fprintf(stderr, "jianmu-gui: display init failed\n");
      return 1;
    }

  g_ui.disp = result.disp;
  g_ui.scr  = lv_screen_active();

  build_compose();
  build_result();
  kb_show_page(0);

  lv_obj_add_flag(g_ui.view_result, LV_OBJ_FLAG_HIDDEN);

  lv_label_set_text(g_ui.status, "加载模型中…");
  lv_refr_now(g_ui.disp);

  if (v10_init(NULL) != 0)
    {
      lv_label_set_text(g_ui.status, "无法载入模型");
      lv_refr_now(g_ui.disp);
    }
  else
    {
      si_build(&g_ui.idx, k_corpus, sizeof(k_corpus) / sizeof(k_corpus[0]));
      char buf[80];
      snprintf(buf, sizeof(buf), "就绪  %d 条 完全离线", g_ui.idx.count);
      lv_label_set_text(g_ui.status, buf);
      g_ui.ready = 1;
    }
  lv_refr_now(g_ui.disp);

  for (;;)
    {
      uint32_t idle = lv_timer_handler();
      idle = idle ? idle : 1;
      usleep(idle * 1000);
    }

  return 0;
}
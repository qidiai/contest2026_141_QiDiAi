/****************************************************************************
 * lvgl_frontend.h — QiDiAi 建木 LVGL 前端公共接口
 ****************************************************************************/

#ifndef LVGL_FRONTEND_H
#define LVGL_FRONTEND_H

/* Run the LVGL touch front-end (390x450 AMOLED).
 * Returns 0 on clean exit, non-zero on failure. Blocks forever once the
 * LVGL event loop starts. */
int jianmu_gui_run(void);

#endif /* LVGL_FRONTEND_H */
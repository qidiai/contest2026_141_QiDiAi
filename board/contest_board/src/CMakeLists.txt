# Contest 2026 team 141 - QiDiAi Jianmu board
nuttx_add_library(board)
target_sources(board PRIVATE board_boot.c)

if(CONFIG_BOARDCTL)
  target_sources(board PRIVATE board_appinit.c)
endif()

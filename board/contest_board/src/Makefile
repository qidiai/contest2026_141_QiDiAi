include $(TOPDIR)/Make.defs

CSRCS = board_boot.c

ifeq ($(CONFIG_BOARDCTL),y)
CSRCS += board_appinit.c
endif

include $(TOPDIR)/boards/Board.mk

#include <nuttx/config.h>
#include <nuttx/board.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <stdio.h>
#include <errno.h>

int board_app_initialize(uintptr_t arg)
{
  int ret;

  /* Create /data mount point */
  mkdir("/data", 0777);

  /* Mount VirtIO block device as FAT at /data
   * QEMU provides virtio-blk at /dev/vda
   */
  ret = mount("/dev/vda", "/data", "vfat", 0, NULL);
  if (ret < 0)
    {
      /* Try alternative device names */
      ret = mount("/dev/vd0a", "/data", "vfat", 0, NULL);
    }
  if (ret < 0)
    {
      ret = mount("/dev/mmcsd0", "/data", "vfat", 0, NULL);
    }

  return 0;
}

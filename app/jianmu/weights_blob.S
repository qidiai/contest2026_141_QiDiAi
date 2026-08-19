/****************************************************************************
 * weights_blob.S 鈥?V10 鏉冮噸鏂囦欢宓屽叆锛堝缓鏈?QiDiAi锛? *
 * 鎶?v10_weights.baize 浠ュ彧璇绘暟鎹褰㈠紡閾炬帴杩涘浐浠堕暅鍍忋€? * 鏉跨骇 ld.script 灏嗚 .weights_blob 娈垫斁鍦?XIP NOR flash 涓? * 锛堢揣闅忛暅鍍忓叾浣欓儴鍒嗕箣鍚庯級銆傝繍琛屾椂 V10 鍔犺浇鍣ㄥ師鍦拌В鏋愯 blob锛? * 寮犻噺鎸囬拡鐩存帴鎸囧悜 flash 鈥斺€?闆跺爢鍒嗛厤銆侀浂鎷疯礉銆? *
 * Path is resolved from the CMake build directory (cmake_out/vendor)
 * through the manifest linkfile to packages/demos/contest2026_141_jianmu/.
 ****************************************************************************/

.section .weights_blob,"a"
.align 4

.global v10_weights_blob_start
.type v10_weights_blob_start, %object
v10_weights_blob_start:
.incbin "../../packages/demos/contest2026_141_jianmu/v10_weights.baize"

.global v10_weights_blob_end
.type v10_weights_blob_end, %object
v10_weights_blob_end:
.balign 4
.size v10_weights_blob_start, . - v10_weights_blob_start
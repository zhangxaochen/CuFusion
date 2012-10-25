#include "../internal.h"

__device__ __forceinline__ static void
shift_tsdf_pointer(short2 ** value, pcl::gpu::tsdf_buffer buffer)
{
  ///Shift the pointer by (@origin - @start)
  *value += (buffer.tsdf_rolling_buff_origin - buffer.tsdf_memory_start);
  
  ///If we land outside of the memory, make sure to "modulo" the new value
  if(*value > buffer.tsdf_memory_end)
  {
    *value -= (buffer.tsdf_memory_end - buffer.tsdf_memory_start + 1);
  }       
}

__device__ __forceinline__ static void
shift_color_pointer(uchar4 ** value, pcl::gpu::tsdf_buffer buffer)
{
  ///Shift the pointer by (@origin - @start)
  *value += (buffer.color_rolling_buff_origin - buffer.color_memory_start);
  
  ///If we land outside of the memory, make sure to "modulo" the new value
  if(*value > buffer.color_memory_end)
  {
    *value -= (buffer.color_memory_end - buffer.color_memory_start + 1);
  }       
}

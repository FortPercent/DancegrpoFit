import torch
import torch.multiprocessing as mp
import os
import time

# --- 参数定义 ---
FILE_NAME = "large_tensor_10gb.pt"
NUM_GPUS = 4
TENSOR_SIZE = 10 * 1024**3 // 4  # 10GB 的 float32 张量，float32 占4字节

def create_tensor(file_name):
    """创建一个大张量并保存到磁盘"""
    print(f"🔧 正在创建一个大小为10GB的张量并保存到 '{file_name}' ...")
    tensor = torch.randn(TENSOR_SIZE, dtype=torch.float32)  # 随机初始化
    torch.save(tensor, file_name)
    print("✅ 张量创建并保存完毕。")

def worker_task(rank, file_name):
    """
    每个进程执行的任务：
    1. 设置并打印当前进程使用的GPU。
    2. 从磁盘加载数据到该进程的CPU内存。
    3. 计时并把数据从CPU传输到指定的GPU。
    """
    gpu_id = rank
    try:
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        print(f"[GPU {gpu_id}] 进程启动，将使用设备 {device}。")

        print(f"[GPU {gpu_id}] 正在从磁盘加载张量...")
        tensor_cpu = torch.load(file_name)
        print(f"[GPU {gpu_id}] 张量已加载到CPU内存。")

        print(f"[GPU {gpu_id}] 准备将数据传输到 {device}...")
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        starter.record()
        tensor_gpu = tensor_cpu.to(device)
        ender.record()

        torch.cuda.synchronize(device=device)

        transfer_duration_ms = starter.elapsed_time(ender)
        transfer_duration_s = transfer_duration_ms / 1000
        bandwidth = 10 / transfer_duration_s

        print(f"✅ [GPU {gpu_id}] 传输完成！CPU -> {device} 耗时: {transfer_duration_s:.4f} 秒 (带宽: {bandwidth:.2f} GB/s)")

        del tensor_cpu
        del tensor_gpu
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ [GPU {gpu_id}] 发生错误: {e}")
        print(f"❌ [GPU {gpu_id}] 请检查系统内存(RAM)和该卡的显存(VRAM)是否充足。")


if __name__ == "__main__":
    print("开始多GPU加载测试...")

    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到CUDA。")
    elif torch.cuda.device_count() < NUM_GPUS:
        print(f"❌ 错误: 需要 {NUM_GPUS} 张GPU，但只检测到 {torch.cuda.device_count()} 张。")
    else:
        if not os.path.exists(FILE_NAME):
            create_tensor(FILE_NAME)

        print(f"检测到 {torch.cuda.device_count()} 张GPU。将启动 {NUM_GPUS} 个并行进程进行测试。")
        print("-" * 50)

        mp.set_start_method('spawn', force=True)

        total_start_time = time.time()

        mp.spawn(worker_task,
                 args=(FILE_NAME,),
                 nprocs=NUM_GPUS,
                 join=True)

        total_end_time = time.time()
        print("-" * 50)
        print(f"所有 {NUM_GPUS} 个进程已执行完毕。")
        print(f"测试总耗时: {total_end_time - total_start_time:.2f} 秒。")

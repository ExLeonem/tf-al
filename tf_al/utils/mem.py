


def log_mem_usage(logger):
    import nvidia_smi
    import psutil
    from psutil._common import bytes2human

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    virtual_mem = psutil.virtual_memory()
    logger.info("(total) RAM: {}".format(bytes2human(virtual_mem.total)))
    logger.info("(free) RAM: {}".format(bytes2human(virtual_mem.free)))
    logger.info("(used) RAM: {}".format(bytes2human(virtual_mem.used)))

    logger.info("GPU----------")
    logger.info("Total GPU: {}".format(bytes2human(info.total)))
    logger.info("Free GPU: {}".format(bytes2human(info.free)))
    logger.info("Used GPU: {}".format(bytes2human(info.used)))
    nvidia_smi.nvmlShutdown()
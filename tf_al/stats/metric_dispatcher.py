

def prepend(prefix, name):
    if prefix is not None:
        return prefix + "_" + name
    
    return name


def is_loss_metric(metric_name):
    return "entropy" in metric_name \
            or "loss" in metric_name \
            or "hinge" in metric_name \
            or "error" in metric_name
        
   

def __get_stochastic_metric(metric_name):
    from .stochastic import Loss, Accuracy
    
    metric_name = metric_name.replace("stochastic_", "")
    # if metric_name == "auc":
    #     return AUC

    if "accuracy" in metric_name:
        return Accuracy(metric_name)
    
    elif is_loss_metric(metric_name):
        return Loss(metric_name)
    
    raise ValueError("Error in stats.get(). Couldn't find any metric for {}.".format(metric_name))


def __get_sampling_metric(metric_name):
    from .sampling import AUC, TruePositive, Loss, Accuracy

    metric_name = metric_name.replace("sampling_", "")
    if metric_name == "auc":
        return AUC
    
    elif "accuracy" in metric_name:
        return Accuracy(metric_name)
    
    elif is_loss_metric(metric_name):
        return Loss(metric_name)

    raise ValueError("Error in stats.get(). Couldn't find any metric for {}.".format(metric_name))
    

def get(prefix, metric_name):
    """
        Transform a metric if possible into a
        custom model metric used.
        
        If no metric was found try a fallback on regular tensorflow metric methods.
    """
    
    metric_name = prepend(prefix, metric_name)
    if "stochastic_" in metric_name:
        return __get_stochastic_metric(metric_name)

    if "sampling_" in metric_name:
        return __get_sampling_metric(metric_name)


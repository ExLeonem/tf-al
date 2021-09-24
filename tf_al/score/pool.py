from .. import Pool


def datapoints_in_use(pool: Pool) -> float:
    """
        Parameters:
            pool (Pool): The pool managing the labeled and unlabeled samples

        Returns:
            (float) percentage of used up total unlabeled pool size.
    """
    len_labeled = pool.get_length_labeled()
    len_unlabeled = pool.get_length_unlabeled()

    return len_labeled/(len_labeled+len_unlabeled)
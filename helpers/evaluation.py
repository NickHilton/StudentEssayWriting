import pandas as pd


def overlap(row: pd.Series) -> list:
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    Per: https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation

    :param row: (pd.Series) df row with [predictionstring_pred, predictionstring_gt]
    :return: (list) of the overlap
    """
    # Predictions
    set_pred = set(row.predictionstring_pred.split(" "))
    # Labels
    set_gt = set(row.predictionstring_gt.split(" "))

    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))

    # Overlap ratios
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred

    return [overlap_1, overlap_2]


def score_feedback_comp(df_pred: pd.DataFrame, df_true: pd.DataFrame) -> float:
    """
    Score for the kaggle feedback competition
    see: https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation

    :param df_pred:(pd.DataFrame) of predictions
    :param df_true: (pd.DataFrame) of labelled data
    :return:
    """
    df_true = (
        df_true[["id", "discourse_type", "predictionstring"]]
        .reset_index(drop=True)
        .copy()
    )

    df_pred = df_pred[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    df_pred["pred_id"] = df_pred.index
    df_true["gt_id"] = df_true.index

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = df_pred.merge(
        df_true,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )

    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)

    tp_pred_ids = (
        joined.query("potential_TP")
        .sort_values("max_overlap", ascending=False)
        .groupby(["id", "predictionstring_gt"])
        .first()["pred_id"]
        .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score

from keras import backend

def precision(y_true, y_pred):
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + backend.epsilon())
    return precision


def recall(y_true, y_pred):
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = backend.sum(backend.round(backend.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + backend.epsilon())
    return recall


def f1(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = 2 * ((pre * rec) / (pre + rec + backend.epsilon()))
    return f1
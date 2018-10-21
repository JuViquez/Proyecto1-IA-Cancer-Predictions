def l1_loss(y_test, y_pred):
    return abs(y_test - y_pred)


def l2_loss(y_test, y_pred):
    return (y_test - y_pred)**2


def l0_1_loss(y_test, y_pred):
    return 0 if y_test == y_pred else 1

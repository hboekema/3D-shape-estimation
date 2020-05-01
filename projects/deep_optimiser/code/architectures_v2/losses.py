


def delta_d_hat_mse(delta_d_NOGRAD, delta_d_hat):
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))

    return false_loss_delta_d_hat


def paramwise_sin_metric(delta_d_NOGRAD, delta_d_hat):
    false_sin_loss_delta_d_hat = get_sin_metric(delta_d_NOGRAD, delta_d_hat, average=False)
    false_sin_loss_delta_d_hat = Lambda(lambda x: x, name="delta_d_hat_sin_output")(false_sin_loss_delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    return false_sin_loss_delta_d_hat

def param_metric(delta_d_NOGRAD, delta_d_hat, sine_metric=False):
    if sine_metric:
        per_param_mse = Lambda(lambda x: K.square(K.sin(x[0] - x[1])))([delta_d_NOGRAD, delta_d_hat])
    else:
        per_param_mse = Lambda(lambda x: K.square(x[0] - x[1]))([delta_d_NOGRAD, delta_d_hat])
    per_param_mse = Reshape((85,), name="params_mse")(per_param_mse)

    return per_param_mse


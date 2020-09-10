
import numpy as np
from tools.model_helpers_v2 import get_trainable_layers



def learned_optimizer_gradient_updates(optlearner_model, x_test, y_test, param_trainable, cb, logging_cb, num_samples=5, epochs=50, mode="RODRIGUES", lr=1.0, BATCH_SIZE=1):
    # Prepare for predictions
    metrics_names = optlearner_model.metrics_names
    print("metrics_names: " + str(metrics_names))
    named_scores = {}
    output_names = [output.op.name.split("/")[0] for output in optlearner_model.outputs]
    print("output_names: " + str(output_names))
    pred_index = output_names.index("delta_d_hat")

    logging_cb.set_names(output_names=output_names)

    # Set number of data samples
    data_samples = x_test[0].shape[0]

    # Get trainable_params
    trainable_params = sorted([int(param.replace("param_", "")) for param, trainable in param_trainable.items() if trainable])

    # Get trainable layers for this model
    trainable_layers = get_trainable_layers(optlearner_model, param_trainable)
    #print(trainable_layers)

    # Set model for callback
    cb.set_model(optlearner_model)

    for epoch in range(epochs):
        print("Iteration: " + str(epoch + 1))
        print("----------------------")
        cb.on_epoch_begin(epoch=int(epoch), logs=named_scores)
        #print('emb layer weights'+str(emb_weights))
	#print('shape '+str(emb_weights[0].shape))
        test_samples = [arr[:num_samples] for arr in x_test]
        y_test_samples = [arr[:num_samples] for arr in y_test]

        # Get old optlearner parameters and actual predictions
        y_pred = optlearner_model.predict(test_samples)
        delta_d_hat = np.zeros((data_samples, 85))
        delta_d_hat[:num_samples, trainable_params] = y_pred[pred_index][:, trainable_params]
        old_params = y_pred[0][:num_samples]
        gradient_update = np.zeros((data_samples, 85))
        gradient_update[:num_samples, trainable_params] = y_pred[6][:, trainable_params]

        actual_update = np.divide(gradient_update[:num_samples], old_params, out=np.zeros_like(old_params), where=(old_params != 0))
        #print(actual_update[0])
        #print(delta_d_hat[0])
        assert np.allclose(actual_update, -delta_d_hat[:num_samples], rtol=1e-3)

        # Adjust weights
        train_history = optlearner_model.fit(test_samples, y_test_samples, batch_size=BATCH_SIZE, epochs=1)
        y_pred = optlearner_model.predict(test_samples)

        logging_cb.store_results(epoch, test_samples, y_pred)

        # Get new optlearner parameters
        y_pred = optlearner_model.predict(test_samples)
        new_params = y_pred[0][:num_samples]

        difference_weights = new_params - old_params
        diff_comp = difference_weights[0]
        delta_d_hat_comp = lr*delta_d_hat[0]
        #print("Prm\tChng\tPred\tDiff")
        #for param in range(85):
            #print("{:02d}   {:.05f}   {:.05f}   {:.06f}".format(param, diff_comp[param], delta_d_hat_comp[param], delta_d_hat_comp[param]-diff_comp[param]))
        #exit(1)
        #assert np.allclose(lr*delta_d_hat[:num_samples], difference_weights, rtol=1e-3), "predictions do not match"

        cb.on_epoch_end(epoch=int(epoch), logs=named_scores)
    cb.set_model(optlearner_model)


def learned_optimizer(optlearner_model, x_test, param_trainable, cb, num_samples=5, epochs=50, lr=0.5, mode="RODRIGUES", kinematic_levels=None):
    # Prepare for predictions
    metrics_names = optlearner_model.metrics_names
    print("metrics_names: " + str(metrics_names))
    named_scores = {}
    output_names = [output.op.name.split("/")[0] for output in optlearner_model.outputs]
    print("output_names: " + str(output_names))
    pred_index = output_names.index("delta_d_hat")

    # Set the kinematic levels, if not already done
    if kinematic_levels is None:
        # Apply predictions to all SMPL parameters at once
        kinematic_levels = [[i for i in range(85)], ]

    # Set number of data samples
    data_samples = x_test[0].shape[0]

    # Get trainable layers for this model
    trainable_layers = get_trainable_layers(optlearner_model, param_trainable)

    # Set model for callback
    cb.set_model(optlearner_model)

    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))
        print("----------------------")
        cb.on_epoch_begin(epoch=int(epoch), logs=named_scores)
        #print('emb layer weights'+str(emb_weights))
	#print('shape '+str(emb_weights[0].shape))
        test_samples = [arr[:num_samples] for arr in x_test]

        # Apply prediction down the specified kinematic tree
        for i, level in enumerate(kinematic_levels):
            print("Updating level {}".format(i))
            y_pred = optlearner_model.predict(test_samples)
            delta_d_hat = np.zeros((data_samples, 85))
            delta_d_hat[:num_samples] = y_pred[pred_index]

            # Update predictions in the model's embedding layers
            for emb_layer, param_num in trainable_layers.items():
                if "param_{:02d}".format(param_num) in level:
                    print("\tUpdating param_{:02d}".format(param_num))
                    emb_weights = emb_layer.get_weights()
                    emb_weights += lr * np.array(delta_d_hat[:, param_num]).reshape((data_samples, 1))
	            emb_layer.set_weights(emb_weights)

        cb.on_epoch_end(epoch=int(epoch), logs=named_scores)
    cb.set_model(optlearner_model)


def multinet_optimizer(optlearner_models, x_test, param_trainables, cb, num_samples=5, epochs=50, lr=0.5, mode="RODRIGUES"):
    # Prepare for predictions
    metrics_names = optlearner_models[0].metrics_names
    print("metrics_names: " + str(metrics_names))
    named_scores = {}
    output_names = [output.op.name.split("/")[0] for output in optlearner_models[0].outputs]
    print("output_names: " + str(output_names))
    pred_index = output_names.index("delta_d_hat")

    # Set number of data samples
    data_samples = x_test[0].shape[0]

    # Get param ids
    param_nos = [i for i in range(85)]
    param_ids = ["param_{:02d}".format(i) for i in param_nos]

    # List of lists of Boolean values indicating which parameters each model was trained on
    param_trainables_list = []
    for param_trainable in param_trainables:
        temp_list = [param_trainable[key] for key in sorted(param_trainable.keys(), key=lambda x: int(x[6:8]))]
        param_trainables_list.append(temp_list)

    # Set model for callback
    cb.set_model(optlearner_models[0])

    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))
        print("----------------------")
        cb.on_epoch_begin(epoch=int(epoch), logs=named_scores)
        test_samples = [arr[:num_samples] for arr in x_test]
        delta_d_hat = np.zeros((data_samples, 85))

        for i, optlearner_model in enumerate(optlearner_models):
            y_pred = optlearner_model.predict(test_samples)
            model_trainables = param_trainables_list[i]
            delta_d_hat[:num_samples][:, model_trainables] = y_pred[pred_index][:, model_trainables]

        # Apply this update to all models' weights
        for optlearner_model in optlearner_models:
            for param_num in param_nos:
    	        emb_layer = optlearner_model.get_layer(param_ids[param_num])
                emb_weights = emb_layer.get_weights()
                emb_weights += lr * np.array(delta_d_hat[:, param_num]).reshape((data_samples, 1))
    	        emb_layer.set_weights(emb_weights)

        cb.on_epoch_end(epoch=int(epoch), logs=named_scores)
    cb.set_model(optlearner_models[0])


def regular_optimizer(optlearner_model, x_test, y_test, cb, epochs=50):
    # Train the model
    optlearner_model.fit(
                x_test,
                y_test,
                batch_size=1,
                epochs=epochs,
                callbacks=[cb]
            )

{
  "ENV": {
    "CUDA_VISIBLE_DEVICES": "3",
    "CHANNEL_FORMAT": "channels_last",
    "USE_MULTIPROCESSING" : true,
    "WORKERS" : 6,
    "LOG_PATH" : "../logs/",
    "TRAINING_VIS_PATH" : "../training_meshes/",
    "TEST_VIS_PATH" : "../test_meshes/"
  },

  "DATA" : {
    "SOURCE" : {
      "TRAIN" : "../../../data/AMASS/train/",
      "VAL" : "../../../data/AMASS/val/",
      "TEST" : "../../../data/AMASS/test/"
    },
    "PARAM_INFO" : {},
    "SILH_INFO" : {"INPUT_WH" : [256, 256], "N_CHANNELS": 1},
    "PREPROCESSING" : {},
    "AUGMENTATIONS" : {}
  },

  "GENERATOR" : {
    "BATCH_SIZE" : 10,
    "SHUFFLE" : true
  },

  "MODEL" : {
    "EPOCHS" : 1,
    "STEPS_PER_EPOCH": 1,
    "VALIDATION_STEPS": 1,
    "SAVE_PERIOD": 5,
    "PRED_PERIOD": 1,
    "ARCHITECTURE" : {"ENCODER" : "", "DECODER" : ""},
    "LOSS_INFO" : {"LOSS_WEIGHTS" : [1.0], "LOSSES" : ["mean_squared_error"]},
    "METRICS" : ["accuracy"]
  }

}

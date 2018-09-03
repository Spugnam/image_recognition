import tensorflow as tf

# for example in tf.python_io.tf_record_iterator("../data/images/popsy_train_00000-of-00002.tfrecord"):
#    result = tf.train.Example.FromString(example)
#    print(result.features.feature['image/class/label'].int64_list.value)
#    print(result.features.feature['image/filename'].bytes_list.value)

# print("*************************")

i = 0
for example in tf.python_io.tf_record_iterator("../data/images/popsy_validation_00000-of-00002.tfrecord"):
    # print("type of tf_record_iterator: ", type(example))
    result = tf.train.Example.FromString(example)
    # print("type of result", type(result))
    # print("label: ", result.features.feature['image/class/label'].int64_list.value)
    # print("filename: ", result.features.feature['image/filename'].bytes_list.value)
    for key, value in result.features.feature.items():
        if 'encoded' not in key:
            print(key)
            print(value)
    i += 1
    if i >= 3:
        break

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/20 4:19 PM
# @Author  : Yugang.ji
# @Site    : 
# @File    : HEP_Aminer
# @Software: PyCharm


import tensorflow as tf

def build_col_defval_LP():
    cols_name = ["pid", 'label']
    cols_defval = [[-1], [-1]]
    return cols_name, cols_defval


def build_col_defval_EP(n_node_type):
    cols = ['node_id', 'node_type', "negs"]
    cols_defval = [[-1], [-1], [""]]
    nbr = []
    edge_features = []
    edge_type = []
    nbr_defval = []
    edge_features_defval = []
    edge_type_defval = []

    for i in range(n_node_type):
        nbr.append("nbrs_" + str(i))
        edge_features.append("edges_features" + str(i))
        edge_type.append('edge_type_' + str(i))
        nbr_defval.append([''])
        edge_features_defval.append([''])
        edge_type_defval.append([-1])
    cols = cols + nbr + edge_features + edge_type
    cols_defval = cols_defval + nbr_defval + edge_features_defval + edge_type_defval
    return cols, cols_defval


class model:

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        num_nodes = 41525
        num_train_pairs = 9232

        min_size = min(num_nodes, num_train_pairs)
        self.batch_size1 = int(num_train_pairs / min_size) * FLAGS.batch_size
        self.batch_size2 = int(num_nodes / min_size) * FLAGS.batch_size

    def run_train(self):
        self.input1 = self.input_LP(self.FLAGS, self.batch_size1)
        self.input2 = self.input_EP(self.FLAGS, self.batch_size2)
        global_step = tf.train.get_or_create_global_step()

        train_model = hep(FLAGS, inputs_L1, inputs_L2, global_step, batch_size1, batch_size2, num_nodes, "train")

        hooks = [tf.train.StopAtStepHook(last_step=8000000000)]
        scaffold = tf.train.Scaffold(saver=train_model.saver,
                                     init_op=tf.global_variables_initializer())

        starttime = datetime.datetime.now()
        create_directory(store_dir)
        with tf.train.MonitoredTrainingSession(scaffold=scaffold, checkpoint_dir=store_dir, save_checkpoint_secs=3600,
                                               hooks=hooks) as mon_sess:
            step = 0
            writer = tf.summary.FileWriter(store_dir + "logs", mon_sess.graph)
            while not mon_sess.should_stop():
                step += 1
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, rs = mon_sess.run(train_model.train_op, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % step)
                writer.add_summary(rs, step)

        endtime = datetime.datetime.now()
        times = (endtime - starttime).seconds
        print("{}\t{}\t{}".format(times, times, times))

    def input_LP(self, FLAGS, batch_size, slice_id=None, slice_count=None, is_dict=False):
        #  for training
        cols_name, cols_defval = build_col_defval_LP()
        input_file = FLAGS.input_path + "train.txt".format(FLAGS.sv_suffix)
        num_epochs = FLAGS.num_epochs
        shuffle = FLAGS.shuffle
        min_after_dequeue = 32 * batch_size
        capacity = 64 * batch_size

        filename_queue = tf.train.string_input_producer([input_file], num_epochs=num_epochs, shuffle=shuffle)

        if FLAGS.file_reader == "textline":
            reader = tf.TextLineReader()
        else:
            reader = tf.TableRecordReader(csv_delimiter=FLAGS.col_delim1, slice_count=slice_count, slice_id=slice_id)
        _, value = reader.read_up_to(filename_queue, batch_size)
        value = tf.train.shuffle_batch(
            [value],
            batch_size=batch_size,
            num_threads=24,
            capacity=capacity,
            enqueue_many=True,
            min_after_dequeue=min_after_dequeue)

        features = tf.decode_csv(
            value, record_defaults=cols_defval, field_delim=FLAGS.col_delim1, use_quote_delim=False)

        pids, labels = features
        if is_dict:
            return {
                "pid": pids,
                "label": labels,
            }
        else:
            return pids, labels

    def input_EP(self, FLAGS, batch_size, slice_id=None, slice_count=None, is_dict=False):
        cols, cols_defval = build_col_defval_EP(FLAGS.n_node_type)
        input_file = FLAGS.input_path + "graph.txt"
        num_epochs = FLAGS.num_epochs
        n_node_type = FLAGS.n_node_type

        shuffle = FLAGS.shuffle
        min_after_dequeue = 32 * batch_size
        capacity = 64 * batch_size

        filename_queue = tf.train.string_input_producer([input_file], num_epochs=num_epochs, shuffle=shuffle)

        if FLAGS.file_reader == "textline":
            reader = tf.TextLineReader()
        else:
            reader = tf.TableRecordReader(csv_delimiter=FLAGS.col_delim1, slice_count=slice_count, slice_id=slice_id)
        _, value = reader.read_up_to(filename_queue, batch_size)
        value = tf.train.shuffle_batch(
            [value],
            batch_size=batch_size,
            num_threads=24,
            capacity=capacity,
            enqueue_many=True,
            min_after_dequeue=min_after_dequeue)

        features = tf.decode_csv(
            value, record_defaults=cols_defval, field_delim=FLAGS.col_delim1, use_quote_delim=False)

        node_ids = features[0]
        node_types = features[1]
        nodes_nbrs_array = []
        nbr_segment_array = []

        edge_features_array = []
        edge_type_array = []

        base_l = 3

        negs = extract_negative_nodes(features[base_l - 1], FLAGS.col_delim2)

        for i in range(n_node_type):
            nbr_segment, nbr_nodes = extract_neighbor_nodes(features[base_l + i], FLAGS.col_delim2)
            edge_features = extract_features(features[base_l + n_node_type + i], FLAGS.col_delim2)
            edge_type = features[base_l + 2 * n_node_type + i]

            nodes_nbrs_array.append(nbr_nodes)
            nbr_segment_array.append(nbr_segment)
            edge_features_array.append(edge_features)
            edge_type_array.append(edge_type)

        if is_dict:
            return {
                "node_ids": node_ids,
                "node_types": node_types,
                "nodes_nbrs_array": nodes_nbrs_array,
                "nbr_segment_array": nbr_segment_array,
                "edge_type_array": edge_type_array,
                "edge_features_array": edge_features_array,
                "negs": negs,
            }
        else:
            return node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array, edge_features_array


def extract_negative_nodes(string_tensor, sp):
    split = tf.string_split(string_tensor, sp)
    return tf.sparse_to_dense(
        sparse_indices=split.indices,
        output_shape=split.dense_shape,
        sparse_values=tf.string_to_number(split.values, out_type=tf.int32),
    )


def extract_neighbor_nodes(string_tensor, sp):
    split = tf.string_split(string_tensor, sp)
    seg_ids = split.indices[:, 0]
    values = tf.string_to_number(split.values, out_type=tf.int32)
    return seg_ids, values



def extract_features(string_tensor, sp):
    split = tf.string_split(string_tensor, sp)
    values = tf.string_to_number(split.values, out_type=tf.float32)
    return values
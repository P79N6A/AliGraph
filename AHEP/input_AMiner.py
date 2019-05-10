#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/20 4:38 PM
# @Author  : Yugang.ji
# @Site    : 
# @File    : input_AMiner
# @Software: PyCharm

from sampling_AMiner import *


def build_col_defval_LP():
    cols_name = ['uid', 'sid', 'label']
    cols_defval = [[-1], [-1], [-1]]
    return cols_name, cols_defval


def input_LP(FLAGS, batch_size, slice_id=None, slice_count=None, is_dict=False):
    #  for training
    cols_name, cols_defval = build_col_defval_LP()
    input_file = FLAGS.input_path + "train_ali.csv"
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

    uids, sids, labels = features
    if is_dict:
        return {
            "mid": uids,
            "pid": sids,
            "label": labels,
        }
    else:
        return uids, sids, labels



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


def input_EP(FLAGS, batch_size, slice_id=None, slice_count=None, is_dict=False):
    cols, cols_defval = build_col_defval_EP(FLAGS.n_node_type)
    input_file = FLAGS.input_path + "graphs_ali.csv"
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
        edge_features = extract_features(features[base_l + n_node_type + i], FLAGS)
        edge_type = features[base_l + 2 * n_node_type + i]

        nodes_nbrs_array.append(nbr_nodes)
        nbr_segment_array.append(nbr_segment)
        edge_features_array.append(edge_features)
        edge_type_array.append(edge_type)
    with tf.name_scope("sampling"):
        if FLAGS.model == "HEP":
            return no_sampling(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                               edge_features_array)
        elif FLAGS.model == "Batch_RS_Type_IS":
            results = Batch_RS_Type_IS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                                    edge_features_array)
        elif FLAGS.model =="Batch_Fast_Type_IS":
           results = Batch_Fast_Type_IS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                               edge_features_array)
        elif FLAGS.model =="Batch_RS_Type_SNIS":
           results = Batch_RS_Type_SNIS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                               edge_features_array)
        elif FLAGS.model =="Batch_Fast_Type_SNIS":
           results = Batch_Fast_Type_SNIS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                               edge_features_array)
        elif FLAGS.model =="Batch_Fast_Typeless_SNIS":
           results = Batch_Fast_Typeless_SNIS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                               edge_features_array)
        elif FLAGS.model =="Batch_Fast_Typeless_IS":
           results = Batch_Fast_Typeless_IS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                               edge_features_array)
        elif FLAGS.model =="Batch_RS_Typeless_IS":
           results = Batch_RS_Typeless_IS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                               edge_features_array)
        elif FLAGS.model =="Batch_RS_Typeless_SNIS":
           results = Batch_RS_Typeless_SNIS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                               edge_features_array)

        node_ids, node_types, negs, sampled_nbrs_array, sampled_segs, edge_type_array, sampled_features, weight_ab, num_nbrs, num_edges = results
        sampled_node_ids, sampled_node_types, sampled_negs, sampled_segs2 = reconstruct_data(node_ids, node_types, negs, sampled_segs, FLAGS.n_node_type)
        return sampled_node_ids, sampled_node_types, sampled_negs, sampled_nbrs_array, sampled_segs2, edge_type_array, sampled_features, weight_ab,num_nbrs, num_edges

def reconstruct_data(node_ids, node_type, negs, sampled_segs, n_node_type):
    unique_segments = tf.cast(tf.contrib.framework.sort(tf.unique(tf.concat(sampled_segs, axis=0)).y),tf.int64)
    sampled_node_ids = tf.gather(tf.range(tf.size(node_ids)), unique_segments)
    sampled_type = tf.gather(node_type, unique_segments)
    sampled_negs = tf.gather(negs, unique_segments)
    values = tf.range(tf.size(unique_segments), dtype=tf.int32)
    indices = tf.reshape(unique_segments, [-1,1])
    maps = tf.sparse_to_dense(sparse_indices=indices, output_shape=tf.cast(tf.shape(node_ids), tf.int64), sparse_values=values)

    new_segments = []
    for i in range(n_node_type):
        new_segments.append(tf.gather(maps, sampled_segs[i]))
    return sampled_node_ids, sampled_type, sampled_negs, new_segments


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


def extract_features(string_tensor, FLAGS):
    node_feature_list = tf.string_split(string_tensor, FLAGS.col_delim2)

    element_feature_list = tf.string_split(node_feature_list.values, FLAGS.col_delim3)

    dense_matrix = tf.sparse_to_dense(element_feature_list.indices, element_feature_list.dense_shape,
                                      tf.string_to_number(element_feature_list.values, tf.float32))

    return dense_matrix

def calculate_sampled_size(unique_nbrs, n_node_type, edge_type_array, FLAGS):
    rates = []
    conditions = [tf.reduce_all(tf.math.equal(edge_type_array[i], -1)) for i in range(FLAGS.n_node_type)]
    for i in range(n_node_type):
        size = tf.cond(conditions[i], true_fn=lambda: tf.zeros(1),
                       false_fn=lambda: tf.cast(tf.size(unique_nbrs[i].y), tf.float32))
        rates.append(size)
    sums = tf.reduce_sum(rates)
    return [tf.cast(rates[i] * FLAGS.num_sample / sums, dtype=tf.int32) + 1 for i in range(n_node_type)]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/20 4:40 PM
# @Author  : Yugang.ji
# @Site    : 
# @File    : sampling_AMiner.py
# @Software: PyCharm

import tensorflow as tf


def no_sampling(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                edge_features_array):
    weight_ab = []
    for i in range(FLAGS.n_node_type):
        unique_nbrs = tf.unique_with_counts(nbr_segment_array[i])
        p1 = 1.0 / tf.cast(tf.gather(unique_nbrs.count, unique_nbrs.idx), tf.float32)

        weight_ab.append(p1)
    num_nodes = tf.size(node_ids)
    num_edges = 0
    num_edges += tf.reduce_sum([tf.size(nodes_nbrs_array[i]) for i in range(FLAGS.n_node_type)])

    return node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array, edge_features_array, weight_ab, num_nodes, num_edges


def Batch_RS_Type_IS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                     edge_features_array):
    weight_ab = []
    samples_num = calculate_sample_size_type(nodes_nbrs_array, FLAGS.num_sample, FLAGS.n_node_type)
    num_nbrs = 0
    num_edges = 0
    sampled_nbrs_array = []
    sampled_segs = []
    sampled_features = []
    conditions = [tf.reduce_all(tf.math.equal(edge_type_array[i], -1)) for i in range(FLAGS.n_node_type)]
    for i in range(FLAGS.n_node_type):
        weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, fs = tf.cond(
            conditions[i],
            false_fn=lambda: sampling_typed_IS_rs(nodes_nbrs_array[i], nbr_segment_array[i], edge_features_array[i],
                                                  samples_num[i]),
            true_fn=lambda: [tf.zeros([1]), tf.zeros(1, tf.int32), tf.zeros(1, tf.int32), tf.ones([1], dtype=tf.int32),
                             tf.ones([1], dtype=tf.int32), tf.zeros([1, FLAGS.edge_dim])])
        num_nbrs += num_sampled_nbrs
        num_edges += num_sampled_edges
        weight_ab.append(weight)
        sampled_nbrs_array.append(sampled_nbrs)
        sampled_segs.append(sampled_segments)
        sampled_features.append(fs)

    return node_ids, node_types, negs, sampled_nbrs_array, sampled_segs, edge_type_array, sampled_features, weight_ab, num_nbrs, num_edges


def sampling_typed_IS_rs(nodes_nbrs, nbr_segment, edge_features, num_sample):
    unique_nbrs = tf.unique_with_counts(nbr_segment)
    p = 1.0 / tf.cast(tf.gather(unique_nbrs.count, unique_nbrs.idx), tf.float32)

    num_nbrs = tf.size(unique_nbrs.y)

    q = tf.gather(tf.ones(num_nbrs) / tf.cast(num_nbrs, dtype=tf.float32), unique_nbrs.idx)

    samples = tf.unique(tf.cast(tf.multinomial(tf.log([q]), num_sample)[0], tf.int32)).y

    infos = tf.sparse_to_dense(tf.reshape(tf.contrib.framework.sort(samples), [-1, 1]),
                               output_shape=tf.shape(unique_nbrs.idx),
                               sparse_values=tf.ones_like(samples, dtype=tf.int32))

    partitions = tf.gather(infos, unique_nbrs.idx)

    samples_to_gather = tf.cast(tf.dynamic_partition(tf.range(tf.size(partitions)), partitions, 2)[1],
                                tf.int32)

    sampled_p = tf.gather(p, samples_to_gather)
    sampled_q = tf.gather(tf.gather(q, unique_nbrs.idx), samples_to_gather)

    sampled_unique_nodes = tf.unique_with_counts(tf.gather(nbr_segment, samples_to_gather))

    weight1 = tf.cast(tf.gather(sampled_unique_nodes.count, sampled_unique_nodes.idx), dtype=tf.float32)

    weight = sampled_p / (sampled_q * weight1)
    num_sampled_edges = tf.size(samples_to_gather)
    num_sampled_nbrs = tf.size(samples)
    sampled_nbrs = tf.gather(nodes_nbrs, samples_to_gather)
    sampled_segments = tf.cast(tf.gather(nbr_segment, samples_to_gather), tf.int32)
    sampled_features = tf.gather(edge_features, samples_to_gather)
    return [weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, sampled_features]


def Batch_Fast_Type_IS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                       edge_features_array):
    weight_ab = []
    samples_num = calculate_sample_size_type(nodes_nbrs_array, FLAGS.num_sample, FLAGS.n_node_type)
    num_nbrs = 0
    num_edges = 0
    sampled_nbrs_array = []
    sampled_segs = []
    sampled_features = []
    conditions = [tf.reduce_all(tf.math.equal(edge_type_array[i], -1)) for i in range(FLAGS.n_node_type)]
    for i in range(FLAGS.n_node_type):
        weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, fs = tf.cond(
            conditions[i],
            false_fn=lambda: sampling_typed_IS_fast(nodes_nbrs_array[i], nbr_segment_array[i], edge_features_array[i],
                                                    samples_num[i]),
            true_fn=lambda: [tf.zeros([1]), tf.zeros(1, tf.int32), tf.zeros(1, tf.int32), tf.ones([1], dtype=tf.int32),
                             tf.ones([1], dtype=tf.int32), tf.zeros([1,FLAGS.edge_dim])])
        num_nbrs += num_sampled_nbrs
        num_edges += num_sampled_edges
        weight_ab.append(weight)
        sampled_nbrs_array.append(sampled_nbrs)
        sampled_segs.append(sampled_segments)
        sampled_features.append(fs)

    return node_ids, node_types, negs, sampled_nbrs_array, sampled_segs, edge_type_array, sampled_features, weight_ab, num_nbrs, num_edges


def sampling_typed_IS_fast(nodes_nbrs, nbr_segment, edge_features, num_sample):
    unique_nbrs = tf.unique_with_counts(nbr_segment)
    p = 1.0 / tf.cast(tf.gather(unique_nbrs.count, unique_nbrs.idx), tf.float32)

    q0 = tf.unsorted_segment_sum(p * p, unique_nbrs.idx, tf.size(unique_nbrs.y))
    q = q0 / tf.reduce_sum(q0)

    samples = tf.unique(tf.cast(tf.multinomial(tf.log([q]), num_sample)[0], tf.int32)).y

    infos = tf.sparse_to_dense(tf.reshape(tf.contrib.framework.sort(samples), [-1, 1]),
                               output_shape=tf.shape(unique_nbrs.idx),
                               sparse_values=tf.ones_like(samples, dtype=tf.int32))

    partitions = tf.gather(infos, unique_nbrs.idx)

    samples_to_gather = tf.cast(tf.dynamic_partition(tf.range(tf.size(partitions)), partitions, 2)[1],
                                tf.int32)

    sampled_p = tf.gather(p, samples_to_gather)
    sampled_q = tf.gather(tf.gather(q, unique_nbrs.idx), samples_to_gather)

    sampled_unique_nodes = tf.unique_with_counts(tf.gather(nbr_segment, samples_to_gather))

    weight1 = tf.cast(tf.gather(sampled_unique_nodes.count, sampled_unique_nodes.idx), dtype=tf.float32)

    weight = sampled_p / (sampled_q * weight1)
    num_sampled_edges = tf.size(samples_to_gather)
    num_sampled_nbrs = tf.size(samples)
    sampled_nbrs = tf.gather(nodes_nbrs, samples_to_gather)
    sampled_segments = tf.cast(tf.gather(nbr_segment, samples_to_gather), tf.int32)
    sampled_features = tf.gather(edge_features, samples_to_gather)
    return [weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, sampled_features]


def Batch_RS_Type_SNIS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                       edge_features_array):
    weight_ab = []
    samples_num = calculate_sample_size_type(nodes_nbrs_array, FLAGS.num_sample, FLAGS.n_node_type)
    num_nbrs = 0
    num_edges = 0
    sampled_nbrs_array = []
    sampled_segs = []
    sampled_features = []
    conditions = [tf.reduce_all(tf.math.equal(edge_type_array[i], -1)) for i in range(FLAGS.n_node_type)]
    for i in range(FLAGS.n_node_type):
        weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, fs = tf.cond(
            conditions[i],
            false_fn=lambda: sampling_typed_SNIS_rs(nodes_nbrs_array[i], nbr_segment_array[i], edge_features_array[i],
                                                    samples_num[i]),
            true_fn=lambda: [tf.zeros([1]), tf.zeros(1, tf.int32), tf.zeros(1, tf.int32), tf.ones([1], dtype=tf.int32),
                             tf.ones([1], dtype=tf.int32), tf.zeros([1,FLAGS.edge_dim])])
        num_nbrs += num_sampled_nbrs
        num_edges += num_sampled_edges
        weight_ab.append(weight)
        sampled_nbrs_array.append(sampled_nbrs)
        sampled_segs.append(sampled_segments)
        sampled_features.append(fs)

    return node_ids, node_types, negs, sampled_nbrs_array, sampled_segs, edge_type_array, sampled_features, weight_ab, num_nbrs, num_edges


def sampling_typed_SNIS_rs(nodes_nbrs, nbr_segment, edge_features, num_sample):
    unique_nbrs = tf.unique_with_counts(nbr_segment)
    p = 1.0 / tf.cast(tf.gather(unique_nbrs.count, unique_nbrs.idx), tf.float32)

    num_nbrs = tf.size(unique_nbrs.y)

    q = tf.gather(tf.ones(num_nbrs) / tf.cast(num_nbrs, dtype=tf.float32), unique_nbrs.idx)

    samples = tf.unique(tf.cast(tf.multinomial(tf.log([q]), num_sample)[0], tf.int32)).y

    infos = tf.sparse_to_dense(tf.reshape(tf.contrib.framework.sort(samples), [-1, 1]),
                               output_shape=tf.shape(unique_nbrs.idx),
                               sparse_values=tf.ones_like(samples, dtype=tf.int32))

    partitions = tf.gather(infos, unique_nbrs.idx)

    samples_to_gather = tf.cast(tf.dynamic_partition(tf.range(tf.size(partitions)), partitions, 2)[1],
                                tf.int32)

    sampled_p = tf.gather(p, samples_to_gather)
    sampled_q = tf.gather(tf.gather(q, unique_nbrs.idx), samples_to_gather)

    sampled_unique_nodes = tf.unique_with_counts(tf.gather(nbr_segment, samples_to_gather))

    # weight1 = tf.cast(tf.gather(sampled_unique_nodes.count, sampled_unique_nodes.idx), dtype=tf.float32)

    w0 = sampled_p / sampled_q
    wpq = w0 / tf.gather(tf.segment_sum(w0, sampled_unique_nodes.idx), sampled_unique_nodes.idx)

    weight = wpq
    num_sampled_edges = tf.size(samples_to_gather)
    num_sampled_nbrs = tf.size(samples)
    sampled_nbrs = tf.gather(nodes_nbrs, samples_to_gather)
    sampled_segments = tf.cast(tf.gather(nbr_segment, samples_to_gather), tf.int32)
    sampled_features = tf.gather(edge_features, samples_to_gather)
    return [weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, sampled_features]


def Batch_Fast_Type_SNIS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                         edge_features_array):
    weight_ab = []
    samples_num = calculate_sample_size_type(nodes_nbrs_array, FLAGS.num_sample, FLAGS.n_node_type)
    num_nbrs = 0
    num_edges = 0
    sampled_nbrs_array = []
    sampled_segs = []
    sampled_features = []
    conditions = [tf.reduce_all(tf.math.equal(edge_type_array[i], -1)) for i in range(FLAGS.n_node_type)]
    for i in range(FLAGS.n_node_type):
        weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, fs = tf.cond(
            conditions[i],
            false_fn=lambda: sampling_typed_SNIS_fast(nodes_nbrs_array[i], nbr_segment_array[i], edge_features_array[i],
                                                      samples_num[i]),
            true_fn=lambda: [tf.zeros([1]), tf.zeros(1, tf.int32), tf.zeros(1, tf.int32), tf.ones([1], dtype=tf.int32),
                             tf.ones([1], dtype=tf.int32), tf.zeros([1,FLAGS.edge_dim])])
        num_nbrs += num_sampled_nbrs
        num_edges += num_sampled_edges
        weight_ab.append(weight)
        sampled_nbrs_array.append(sampled_nbrs)
        sampled_segs.append(sampled_segments)
        sampled_features.append(fs)

    return node_ids, node_types, negs, sampled_nbrs_array, sampled_segs, edge_type_array, sampled_features, weight_ab, num_nbrs, num_edges


def sampling_typed_SNIS_fast(nodes_nbrs, nbr_segment, edge_features, num_sample):
    unique_nbrs = tf.unique_with_counts(nbr_segment)
    p = 1.0 / tf.cast(tf.gather(unique_nbrs.count, unique_nbrs.idx), tf.float32)

    q0 = tf.unsorted_segment_sum(p * p, unique_nbrs.idx, tf.size(unique_nbrs.y))
    q = q0 / tf.reduce_sum(q0)

    samples = tf.unique(tf.cast(tf.multinomial(tf.log([q]), num_sample)[0], tf.int32)).y

    infos = tf.sparse_to_dense(tf.reshape(tf.contrib.framework.sort(samples), [-1, 1]),
                               output_shape=tf.shape(unique_nbrs.idx),
                               sparse_values=tf.ones_like(samples, dtype=tf.int32))

    partitions = tf.gather(infos, unique_nbrs.idx)

    samples_to_gather = tf.cast(tf.dynamic_partition(tf.range(tf.size(partitions)), partitions, 2)[1],
                                tf.int32)

    sampled_p = tf.gather(p, samples_to_gather)
    sampled_q = tf.gather(tf.gather(q, unique_nbrs.idx), samples_to_gather)

    sampled_unique_nodes = tf.unique_with_counts(tf.gather(nbr_segment, samples_to_gather))

    # weight1 = tf.cast(tf.gather(sampled_unique_nodes.count, sampled_unique_nodes.idx), dtype=tf.float32)
    w0 = sampled_p / sampled_q
    wpq = w0 / tf.gather(tf.segment_sum(w0, sampled_unique_nodes.idx), sampled_unique_nodes.idx)

    weight = wpq
    num_sampled_edges = tf.size(samples_to_gather)
    num_sampled_nbrs = tf.size(samples)
    sampled_nbrs = tf.gather(nodes_nbrs, samples_to_gather)
    sampled_segments = tf.cast(tf.gather(nbr_segment, samples_to_gather), tf.int32)
    sampled_features = tf.gather(edge_features, samples_to_gather)
    return [weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, sampled_features]


def Batch_Fast_Typeless_SNIS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                             edge_features_array):
    conditions = [tf.reduce_all(tf.math.equal(edge_type_array[i], -1)) for i in range(FLAGS.n_node_type)]


    P = []
    partition_type = []
    for i in range(FLAGS.n_node_type):
        type_ps = tf.cond(conditions[i], false_fn=lambda: tf.ones(tf.size(nodes_nbrs_array[i]), dtype=tf.int32) * i,
                          true_fn=lambda: tf.Variable([], dtype=tf.int32))
        type_p = tf.cond(conditions[i], false_fn=lambda: temp1(nbr_segment_array[i]),
                         true_fn=lambda: tf.Variable([], dtype=tf.float32))
        P.append(type_p)
        partition_type.append(type_ps)
    partition_concat = tf.concat(partition_type, axis=0)
    nbrs_all = tf.concat(nodes_nbrs_array, axis=0)
    segs_all = tf.concat(nbr_segment_array, axis=0)
    P_all = tf.concat(P, axis=0)

    concat_features = tf.concat(edge_features_array, axis=0)

    weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, fs, edge_type_array = sampling_typeless_SNIS_fast(P_all,
        partition_concat, nbrs_all, segs_all, concat_features, FLAGS.num_sample, FLAGS.n_node_type, edge_type_array)

    return node_ids, node_types, negs, sampled_nbrs, sampled_segments, edge_type_array, fs, weight, num_sampled_edges, num_sampled_nbrs


def sampling_typeless_SNIS_fast(p, parten, nodes_nbrs, nbr_segment, edge_features, num_sample, n_node_type,
                                edge_type_array):
    unique_nbrs = tf.unique_with_counts(nbr_segment)
    # p = 1.0 / tf.cast(tf.gather(unique_nbrs.count, unique_nbrs.idx), tf.float32)

    q0 = tf.unsorted_segment_sum(p * p, unique_nbrs.idx, tf.size(unique_nbrs.y))
    q = q0 / tf.reduce_sum(q0)

    samples = tf.unique(tf.cast(tf.multinomial(tf.log([q]), num_sample)[0], tf.int32)).y

    infos = tf.sparse_to_dense(tf.reshape(tf.contrib.framework.sort(samples), [-1, 1]),
                               output_shape=tf.shape(unique_nbrs.idx),
                               sparse_values=tf.ones_like(samples, dtype=tf.int32))

    partitions = tf.gather(infos, unique_nbrs.idx)

    samples_to_gather = tf.dynamic_partition(tf.range(tf.size(partitions), dtype=tf.int32), partitions, 2)[1],

    sampled_p = tf.gather(p, samples_to_gather)
    sampled_q = tf.gather(tf.gather(q, unique_nbrs.idx), samples_to_gather)

    sampled_parten = tf.gather(parten, samples_to_gather)
    sampled_nbrs = tf.gather(nodes_nbrs, samples_to_gather)

    nbrset = tf.dynamic_partition(sampled_nbrs, sampled_parten, n_node_type)
    segset = tf.dynamic_partition(tf.gather(nbr_segment, samples_to_gather), sampled_parten, n_node_type)
    edge_f_set = []
    feature_ids = tf.dynamic_partition(tf.gather(tf.range(tf.size(nbr_segment)), samples_to_gather), sampled_parten,
                                       n_node_type)
    for i in range(n_node_type):
        edge_f_set.append(tf.gather(edge_features, feature_ids[i]))

    sampled_ps = tf.dynamic_partition(sampled_p, sampled_parten, n_node_type)
    sampled_qs = tf.dynamic_partition(sampled_q, sampled_parten, n_node_type)

    condition2 = [tf.reduce_all(tf.math.greater(tf.size(nbrset[i]), 0)) for i in range(n_node_type)]
    all_weight = []
    for i in range(n_node_type):
        weights = tf.cond(condition2[i], false_fn=lambda: [tf.zeros(0)], true_fn=
        lambda: calculate_pq_SNIS(segset[i], nbrset[i], edge_f_set[i], sampled_ps[i], sampled_qs[i])
                          )
        all_weight.append(weights)
        print(edge_type_array[i])
        edge_type_array[i] = tf.cond(condition2[i], true_fn=lambda: edge_type_array[i],
                                     false_fn=lambda: -tf.ones(tf.size(edge_type_array[i]), tf.int32))
    num_sampled_edges = tf.size(samples_to_gather)
    num_sampled_nbrs = tf.size(samples)

    return [all_weight, num_sampled_edges, num_sampled_nbrs, nbrset, segset, edge_f_set, edge_type_array]


def calculate_pq_SNIS(segment, nbrs, features, p, q):
    #
    unique_nodes = tf.unique_with_counts(segment)

    # weight1 = tf.cast(tf.gather(unique_nodes.count, unique_nodes.idx), dtype=tf.float32)
    w0 = p / q
    wpq = w0 / tf.gather(tf.segment_sum(w0, unique_nodes.idx), unique_nodes.idx)

    weight = wpq
    return weight


def temp1(nbr_segments):
    unique_nbrs = tf.unique_with_counts(nbr_segments)
    p = 1.0 / tf.cast(tf.gather(unique_nbrs.count, unique_nbrs.idx), tf.float32)
    return p

def Batch_Fast_Typeless_IS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                           edge_features_array):
    conditions = [tf.reduce_all(tf.math.equal(edge_type_array[i], -1)) for i in range(FLAGS.n_node_type)]

    partition_type = []
    P = []
    for i in range(FLAGS.n_node_type):
        type_ps = tf.cond(conditions[i], false_fn=lambda: tf.ones(tf.size(nodes_nbrs_array[i]), dtype=tf.int32) * i,
                          true_fn=lambda: tf.Variable([], dtype=tf.int32))

        type_p = tf.cond(conditions[i], false_fn=lambda: temp1(nbr_segment_array[i]),
                          true_fn=lambda: tf.Variable([], dtype=tf.float32))
        partition_type.append(type_ps)
        P.append(type_p)
    partition_concat = tf.concat(partition_type, axis=0)
    nbrs_all = tf.concat(nodes_nbrs_array, axis=0)
    segs_all = tf.concat(nbr_segment_array, axis=0)

    concat_features = tf.concat(edge_features_array, axis=0)
    P_all = tf.concat(P, axis=0)

    weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, fs, edge_type_array = sampling_typeless_IS_fast(P_all,
        partition_concat, nbrs_all, segs_all, concat_features, FLAGS.num_sample, FLAGS.n_node_type, edge_type_array)

    return node_ids, node_types, negs, sampled_nbrs, sampled_segments, edge_type_array, fs, weight, num_sampled_edges, num_sampled_nbrs


def sampling_typeless_IS_fast(p, parten, nodes_nbrs, nbr_segment, edge_features, num_sample, n_node_type, edge_type_array):
    unique_nbrs = tf.unique_with_counts(nbr_segment)
    # p = 1.0 / tf.cast(tf.gather(unique_nbrs.count, unique_nbrs.idx), tf.float32)

    q0 = tf.unsorted_segment_sum(p * p, unique_nbrs.idx, tf.size(unique_nbrs.y))
    q = q0 / tf.reduce_sum(q0)

    samples = tf.unique(tf.cast(tf.multinomial(tf.log([q]), num_sample)[0], tf.int32)).y

    infos = tf.sparse_to_dense(tf.reshape(tf.contrib.framework.sort(samples), [-1, 1]),
                               output_shape=tf.shape(unique_nbrs.idx),
                               sparse_values=tf.ones_like(samples, dtype=tf.int32))

    partitions = tf.gather(infos, unique_nbrs.idx)

    samples_to_gather = tf.dynamic_partition(tf.range(tf.size(partitions), dtype=tf.int32), partitions, 2)[1],

    sampled_p = tf.gather(p, samples_to_gather)
    sampled_q = tf.gather(tf.gather(q, unique_nbrs.idx), samples_to_gather)

    sampled_parten = tf.gather(parten, samples_to_gather)
    sampled_nbrs = tf.gather(nodes_nbrs, samples_to_gather)

    nbrset = tf.dynamic_partition(sampled_nbrs, sampled_parten, n_node_type)
    segset = tf.dynamic_partition(tf.gather(nbr_segment, samples_to_gather), sampled_parten, n_node_type)

    edge_f_set = []
    feature_ids = tf.dynamic_partition(tf.gather(tf.range(tf.size(nbr_segment)), samples_to_gather), sampled_parten,
                                       n_node_type)
    for i in range(n_node_type):
        edge_f_set.append(tf.gather(edge_features, feature_ids[i]))

    sampled_ps = tf.dynamic_partition(sampled_p, sampled_parten, n_node_type)
    sampled_qs = tf.dynamic_partition(sampled_q, sampled_parten, n_node_type)

    condition2 = [tf.reduce_all(tf.math.greater(tf.size(nbrset[i]), 0)) for i in range(n_node_type)]
    all_weight = []
    for i in range(n_node_type):
        weights = tf.cond(condition2[i], false_fn=lambda: [tf.zeros(0)], true_fn=
        lambda: calculate_pq_IS(segset[i], nbrset[i], edge_f_set[i], sampled_ps[i], sampled_qs[i])
                          )
        all_weight.append(weights)
        print(edge_type_array[i])
        edge_type_array[i] = tf.cond(condition2[i], true_fn=lambda: edge_type_array[i],
                                     false_fn=lambda: -tf.ones(tf.size(edge_type_array[i]), tf.int32))
    num_sampled_edges = tf.size(samples_to_gather)
    num_sampled_nbrs = tf.size(samples)

    return [all_weight, num_sampled_edges, num_sampled_nbrs, nbrset, segset, edge_f_set, edge_type_array]


def calculate_pq_IS(segment, nbrs, features, p, q):
    #
    unique_nodes = tf.unique_with_counts(segment)

    weight1 = tf.cast(tf.gather(unique_nodes.count, unique_nodes.idx), dtype=tf.float32)

    weight = p / (q * weight1)
    return weight


def Batch_RS_Typeless_IS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                         edge_features_array):
    conditions = [tf.reduce_all(tf.math.equal(edge_type_array[i], -1)) for i in range(FLAGS.n_node_type)]

    P = []
    partition_type = []
    for i in range(FLAGS.n_node_type):
        type_ps = tf.cond(conditions[i], false_fn=lambda: tf.ones(tf.size(nodes_nbrs_array[i]), dtype=tf.int32) * i,
                          true_fn=lambda: tf.Variable([], dtype=tf.int32))
        type_p = tf.cond(conditions[i], false_fn=lambda: temp1(nbr_segment_array[i]),
                         true_fn=lambda: tf.Variable([], dtype=tf.float32))
        P.append(type_p)
        partition_type.append(type_ps)
    P_all = tf.concat(P, axis=0)
    partition_concat = tf.concat(partition_type, axis=0)
    nbrs_all = tf.concat(nodes_nbrs_array, axis=0)
    segs_all = tf.concat(nbr_segment_array, axis=0)

    concat_features = tf.concat(edge_features_array, axis=0)

    weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, fs, edge_type_array = sampling_typeless_IS_rs(
        P_all, partition_concat, nbrs_all, segs_all, concat_features, FLAGS.num_sample, FLAGS.n_node_type, edge_type_array)

    return node_ids, node_types, negs, sampled_nbrs, sampled_segments, edge_type_array, fs, weight, num_sampled_edges, num_sampled_nbrs


def sampling_typeless_IS_rs(p, parten, nodes_nbrs, nbr_segment, edge_features, num_sample, n_node_type, edge_type_array):
    unique_nbrs = tf.unique_with_counts(nbr_segment)

    num_nbrs = tf.size(unique_nbrs.y)

    q = tf.gather(tf.ones(num_nbrs) / tf.cast(num_nbrs, dtype=tf.float32), unique_nbrs.idx)

    samples = tf.unique(tf.cast(tf.multinomial(tf.log([q]), num_sample)[0], tf.int32)).y

    infos = tf.sparse_to_dense(tf.reshape(tf.contrib.framework.sort(samples), [-1, 1]),
                               output_shape=tf.shape(unique_nbrs.idx),
                               sparse_values=tf.ones_like(samples, dtype=tf.int32))

    partitions = tf.gather(infos, unique_nbrs.idx)

    samples_to_gather = tf.dynamic_partition(tf.range(tf.size(partitions), dtype=tf.int32), partitions, 2)[1],

    sampled_p = tf.gather(p, samples_to_gather)
    sampled_q = tf.gather(tf.gather(q, unique_nbrs.idx), samples_to_gather)

    sampled_parten = tf.gather(parten, samples_to_gather)
    sampled_nbrs = tf.gather(nodes_nbrs, samples_to_gather)

    nbrset = tf.dynamic_partition(sampled_nbrs, sampled_parten, n_node_type)
    segset = tf.dynamic_partition(tf.gather(nbr_segment, samples_to_gather), sampled_parten, n_node_type)

    edge_f_set = []
    feature_ids = tf.dynamic_partition(tf.gather(tf.range(tf.size(nbr_segment)), samples_to_gather), sampled_parten,
                                       n_node_type)
    for i in range(n_node_type):
        edge_f_set.append(tf.gather(edge_features, feature_ids[i]))

    sampled_ps = tf.dynamic_partition(sampled_p, sampled_parten, n_node_type)
    sampled_qs = tf.dynamic_partition(sampled_q, sampled_parten, n_node_type)

    condition2 = [tf.reduce_all(tf.math.greater(tf.size(nbrset[i]), 0)) for i in range(n_node_type)]
    all_weight = []
    for i in range(n_node_type):
        weights = tf.cond(condition2[i], false_fn=lambda: [tf.zeros(0)], true_fn=
        lambda: calculate_pq_IS(segset[i], nbrset[i], edge_f_set[i], sampled_ps[i], sampled_qs[i])
                          )
        all_weight.append(weights)
        print(edge_type_array[i])
        edge_type_array[i] = tf.cond(condition2[i], true_fn=lambda: edge_type_array[i],
                                     false_fn=lambda: -tf.ones(tf.size(edge_type_array[i]), tf.int32))
    num_sampled_edges = tf.size(samples_to_gather)
    num_sampled_nbrs = tf.size(samples)

    return [all_weight, num_sampled_edges, num_sampled_nbrs, nbrset, segset, edge_f_set, edge_type_array]


def Batch_RS_Typeless_SNIS(FLAGS, node_ids, node_types, negs, nodes_nbrs_array, nbr_segment_array, edge_type_array,
                             edge_features_array):
    conditions = [tf.reduce_all(tf.math.equal(edge_type_array[i], -1)) for i in range(FLAGS.n_node_type)]

    partition_type = []

    P = []

    for i in range(FLAGS.n_node_type):
        type_ps = tf.cond(conditions[i], false_fn=lambda: tf.ones(tf.size(nodes_nbrs_array[i]), dtype=tf.int32) * i,
                          true_fn=lambda: tf.Variable([], dtype=tf.int32))
        type_p = tf.cond(conditions[i], false_fn=lambda: temp1(nbr_segment_array[i]),
                         true_fn=lambda: tf.Variable([], dtype=tf.float32))
        P.append(type_p)
        partition_type.append(type_ps)
    partition_concat = tf.concat(partition_type, axis=0)
    nbrs_all = tf.concat(nodes_nbrs_array, axis=0)
    segs_all = tf.concat(nbr_segment_array, axis=0)

    concat_features = tf.concat(edge_features_array, axis=0)
    P_all = tf.concat(P, axis=0)

    weight, num_sampled_edges, num_sampled_nbrs, sampled_nbrs, sampled_segments, fs, edge_type_array = sampling_typeless_SNIS_rs(P_all,
        partition_concat, nbrs_all, segs_all, concat_features, FLAGS.num_sample, FLAGS.n_node_type, edge_type_array)

    return node_ids, node_types, negs, sampled_nbrs, sampled_segments, edge_type_array, fs, weight, num_sampled_edges, num_sampled_nbrs


def sampling_typeless_SNIS_rs(p, parten, nodes_nbrs, nbr_segment, edge_features, num_sample, n_node_type,
                                edge_type_array):
    unique_nbrs = tf.unique_with_counts(nbr_segment)
    num_nbrs = tf.size(unique_nbrs.y)

    q = tf.gather(tf.ones(num_nbrs) / tf.cast(num_nbrs, dtype=tf.float32), unique_nbrs.idx)

    samples = tf.unique(tf.cast(tf.multinomial(tf.log([q]), num_sample)[0], tf.int32)).y

    infos = tf.sparse_to_dense(tf.reshape(tf.contrib.framework.sort(samples), [-1, 1]),
                               output_shape=tf.shape(unique_nbrs.idx),
                               sparse_values=tf.ones_like(samples, dtype=tf.int32))

    partitions = tf.gather(infos, unique_nbrs.idx)

    samples_to_gather = tf.dynamic_partition(tf.range(tf.size(partitions), dtype=tf.int32), partitions, 2)[1],

    sampled_p = tf.gather(p, samples_to_gather)
    sampled_q = tf.gather(tf.gather(q, unique_nbrs.idx), samples_to_gather)

    sampled_parten = tf.gather(parten, samples_to_gather)
    sampled_nbrs = tf.gather(nodes_nbrs, samples_to_gather)

    nbrset = tf.dynamic_partition(sampled_nbrs, sampled_parten, n_node_type)
    segset = tf.dynamic_partition(tf.gather(nbr_segment, samples_to_gather), sampled_parten, n_node_type)

    edge_f_set = []
    feature_ids = tf.dynamic_partition(tf.gather(tf.range(tf.size(nbr_segment)), samples_to_gather), sampled_parten,
                                       n_node_type)
    for i in range(n_node_type):
        edge_f_set.append(tf.gather(edge_features, feature_ids[i]))

    sampled_ps = tf.dynamic_partition(sampled_p, sampled_parten, n_node_type)
    sampled_qs = tf.dynamic_partition(sampled_q, sampled_parten, n_node_type)

    condition2 = [tf.reduce_all(tf.math.greater(tf.size(nbrset[i]), 0)) for i in range(n_node_type)]
    all_weight = []
    for i in range(n_node_type):
        weights = tf.cond(condition2[i], false_fn=lambda: [tf.zeros(0)], true_fn=
        lambda: calculate_pq_SNIS(segset[i], nbrset[i], edge_f_set[i], sampled_ps[i], sampled_qs[i])
                          )
        all_weight.append(weights)
        print(edge_type_array[i])
        edge_type_array[i] = tf.cond(condition2[i], true_fn=lambda: edge_type_array[i],
                                     false_fn=lambda: -tf.ones(tf.size(edge_type_array[i]), tf.int32))
    num_sampled_edges = tf.size(samples_to_gather)
    num_sampled_nbrs = tf.size(samples)

    return [all_weight, num_sampled_edges, num_sampled_nbrs, nbrset, segset, edge_f_set, edge_type_array]


def calculate_sample_size_type(nodes_nbrs_array, sample_size, n_node_type):
    numbers = []

    for i in range(n_node_type):
        numbers.append(tf.size(nodes_nbrs_array[i]))
    numbers = tf.cast(numbers, tf.float32)
    numbers = tf.cast(numbers * sample_size / tf.reduce_sum(numbers), tf.int32) + 1
    return numbers

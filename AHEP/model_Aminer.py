#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/21 12:28 AM
# @Author  : Yugang.ji
# @Site    : 
# @File    : model_Aminer.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import math


class hep(object):

    def __init__(self, FLAGS, global_step, batch_size1, batch_size2, num_node, model="train"):

        self.global_step = global_step
        self.batch_size1 = batch_size1
        self.FLAGS = FLAGS

        alpha = FLAGS.alpha
        beta = FLAGS.beta
        gamma = FLAGS.gamma

        learning_rate = FLAGS.learning_rate
        learning_algo = FLAGS.learning_algo
        n_node_type = FLAGS.n_node_type
        n_edge_type = FLAGS.n_edge_type
        edge_dim = FLAGS.edge_dim

        self.embedding_dim = FLAGS.embedding_dim

        init_range = np.sqrt(3.0 / (num_node + self.embedding_dim))

        with tf.name_scope("parameters"):
            if FLAGS.distributed_run:
                self.embedding_table = tf.get_variable('hep', [num_node, self.embedding_dim],
                                                       initializer=tf.random_uniform([num_node, self.embedding_dim],
                                                                                     minval=-init_range,
                                                                                     maxval=init_range,
                                                                                     dtype=tf.float32),
                                                       partitioner=tf.min_max_variable_partitioner(
                                                           max_partitions=len(FLAGS.ps_hosts.split(","))))
            else:
                self.embedding_table = tf.get_variable('hep', [num_node, self.embedding_dim],
                                                       initializer=tf.random_uniform([num_node, self.embedding_dim],
                                                                                     minval=-init_range,
                                                                                     maxval=init_range,
                                                                                     dtype=tf.float32))





        self.ids = tf.placeholder(dtype=tf.int32, shape=[None], name="ids")
        self.negs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.n_negs], name="Negs")
        self.id_types = tf.placeholder(dtype=tf.int32, shape=[None], name="id_types")
        self.nbrs = [tf.placeholder(dtype=tf.int32, shape=[None], name="nbrs_{}".format(i)) for i in
                     range(FLAGS.n_node_type)]
        self.segments = [tf.placeholder(dtype=tf.int32, shape=[None], name="segs_{}".format(i)) for i in
                         range(FLAGS.n_node_type)]
        self.edge_types = [tf.placeholder(dtype=tf.int32, shape=[None], name="types_{}".format(i)) for i in
                           range(FLAGS.n_node_type)]
        self.edge_features = [tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.edge_dim], name="features_{}".format(i)) for i in range(FLAGS.n_node_type)]
        self.weight_ab = [tf.placeholder(dtype=tf.float32, shape=[None], name="fweight_{}".format(i)) for i in
                          range(FLAGS.n_node_type)]
        # self.sampled_nodes = tf.placeholder(dtype=tf.float32, name="sampled_nodes")
        # self.sampled_edges = tf.placeholder(dtype=tf.float32, name="sampled_edges")

        self.uids = tf.placeholder(dtype=tf.int32, shape=[None], name="uids")
        self.sids = tf.placeholder(dtype=tf.int32, shape=[None], name="sids")
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
        self.batch_size2 = tf.size(self.ids)

        with tf.name_scope("LP"):
            # pids, labels = inputs1

            self.W = tf.get_variable("w", [self.embedding_dim, self.embedding_dim], dtype=tf.float32)
            self.b = tf.get_variable("b", [1], dtype=tf.float32)
            id_left_lookup = tf.gather(self.embedding_table, self.uids)
            id_right_lookup = tf.gather(self.embedding_table, self.sids)
            self.l1_loss = self.L1(id_left_lookup, id_right_lookup, self.labels,self.W)

        if model == "train":

            with tf.name_scope("EP"):

                # self.edge_weight = tf.get_variable("edge_weight", [n_edge_type, edge_dim], dtype=tf.float32)
                # self.edge_b = tf.get_variable("edge_b", [n_edge_type], dtype=tf.float32)
                self.edge_weight = tf.get_variable("edge_weight", [edge_dim, 1], dtype=tf.float32)
                self.edge_b = tf.get_variable("edge_b", [1], dtype=tf.float32)
                self.node_W = tf.get_variable("node_W", [n_node_type * self.embedding_dim, self.embedding_dim],
                                              dtype=tf.float32)
                self.node_b = tf.get_variable("node_b", [self.embedding_dim], dtype=tf.float32)

                embs_lookup = tf.nn.embedding_lookup(self.embedding_table, self.ids)
                negs_lookup = tf.nn.embedding_lookup(self.embedding_table, self.negs)
                self.l2_loss = self.L2(embs_lookup, negs_lookup, self.nbrs, self.segments, self.edge_features,
                                       self.edge_types,
                                       FLAGS.n_node_type, self.weight_ab)

            with tf.name_scope("Regu"):
                omega = tf.reduce_mean(tf.multiply(self.W, self.W)) + tf.reduce_mean(
                    tf.multiply(self.node_W, self.node_W)) + tf.reduce_mean(
                    tf.multiply(self.node_b, self.node_b)) + tf.reduce_mean(
                    tf.multiply(self.edge_weight, self.edge_weight)) + tf.reduce_mean(
                    tf.multiply(self.edge_b, self.edge_b)) + tf.reduce_mean(tf.multiply(self.b, self.b))

            with tf.name_scope("loss"):
                if learning_algo == "adam":
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                elif learning_algo == "sgd":
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                else:
                    self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
                self.loss = self.l1_loss + alpha * self.l2_loss + beta * (omega)

            # self.sampled_nodes = tf.size(tf.unique(nbrs[0]).y) + tf.size(tf.unique(nbrs[1]).y) + tf.size(tf.unique(nbrs[2]).y)
            # self.sampled_edges = tf.size(nbrs[0]) + tf.size(nbrs[1]) + tf.size(nbrs[2])

            with tf.name_scope("opt"):
                self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                self.merged = tf.summary.merge_all()
                self.train_op = [self.opt_op, self.merged]

        with tf.name_scope("outs"):
            self.output = tf.reduce_sum(tf.multiply(tf.matmul(id_left_lookup, self.W), id_right_lookup), 1)

        with tf.name_scope("init"):
            self.init_saver()

    def L1(self, id_left_lookup, id_right_lookup, labels, W):
        logit = tf.reduce_sum(tf.multiply(tf.matmul(id_left_lookup, W), id_right_lookup), 1)
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.reshape(labels, [-1, 1]),
                                               logits=tf.reshape(logit, [-1, 1]))
        return loss

    def L2(self, embs, negs_lookup, nbrs, segment_ids, edge_features, edge_types, n_node_type, weight_ab):
        encode_emb = []

        for i in range(n_node_type):
            encode_emb.append(
                self.emb_type(edge_features[i], nbrs[i], edge_types[i], segment_ids[i], weight_ab[i])
            )
        concat_emb = tf.concat(encode_emb, 1)
        h_v = tf.sigmoid(tf.matmul(concat_emb, self.node_W) + self.node_b)

        pi_neg = self.batch_distance_neg(h_v, negs_lookup)
        pi_pos = self.batch_distance(h_v, embs)
        l = self.batch_distance_pair(pi_pos, pi_neg, self.FLAGS)
        loss = tf.reduce_mean(l)
        return loss

    def emb_type(self, features, nbr, edge_type, segment_id, w):
        return tf.cond(tf.reduce_all(tf.math.equal(edge_type, -1)),
                       true_fn=lambda: tf.zeros(shape=[self.batch_size2, self.embedding_dim], dtype=tf.float32),
                       false_fn=lambda: self._emb_nonempty_type(features, nbr, edge_type, segment_id, w))

    def _emb_nonempty_type(self, edge_feature, nbr, edge_type_in, segment_id, w):

        w1 = tf.nn.relu(tf.reshape(tf.matmul(edge_feature, self.edge_weight) + self.edge_b,[-1]))
        w2 = w * w1
        nbr_lookup = tf.nn.embedding_lookup(self.embedding_table, nbr)
        weight_emb = tf.multiply(nbr_lookup, tf.reshape(w2, [-1, 1]))
        nonsafe = tf.unsorted_segment_sum(weight_emb, segment_id, self.batch_size2)
        return nonsafe


    def batch_distance(self, embedding, nbr_embedding):
        red = embedding - nbr_embedding
        distance = tf.norm(red, axis=1, ord=2) / math.sqrt(self.FLAGS.embedding_dim)
        return distance

    def batch_distance_neg(self, embedding, nbr_embedding):
        red = nbr_embedding - tf.expand_dims(embedding, 1)
        distance = tf.norm(red, axis=2, ord=2) / math.sqrt(self.FLAGS.embedding_dim)
        return distance

    def batch_distance_pair(self, pos, neg, FLAGS):
        return tf.reduce_sum(tf.nn.relu(tf.expand_dims(pos, 1) - neg + self.FLAGS.gamma), axis=1)

    def init_saver(self):
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        bn_moving_vars += [g for g in g_list if 'global_step' in g.name]
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot_initializer(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def sparse_to_dense(sparse_tensor):
    return tf.sparse_to_dense(sparse_indices=sparse_tensor.indices, output_shape=sparse_tensor.dense_shape,
                              sparse_values=sparse_tensor.values)

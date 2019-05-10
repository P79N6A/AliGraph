#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/20 3:55 PM
# @Author  : Yugang.ji
# @Site    : 
# @File    : main_Aminer.py
# @Software: PyCharm


import tensorflow as tf
import numpy as np

import datetime
import os
from input_AMiner import *
from model_Aminer import *

from sklearn.metrics import f1_score
from sklearn import metrics
from tensorflow.python.client import timeline
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
flags.DEFINE_integer("task_index", None, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_boolean('distributed_run', True, 'Whether to use distributed in pai')

flags.DEFINE_string('input_path', "./data/", "input path")
flags.DEFINE_string("sv_suffix", "3", "the su")
flags.DEFINE_bool("GG", True, "generate graph")
flags.DEFINE_bool("GSF", True, "generate sparse_features")

flags.DEFINE_string("col_delim1", "\t", "col delim char")
flags.DEFINE_string("col_delim2", ";", "col delim char")
flags.DEFINE_string("col_delim3", ":", "col delim char")

flags.DEFINE_string("opt", "train", "opt")
flags.DEFINE_float("alpha", 0.4, "alpha")
flags.DEFINE_float("beta", 0.1, "beta")
flags.DEFINE_float("gamma", 0.1, "gamma")
flags.DEFINE_float("psi", 0.1, "psi")
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string('learning_algo', 'adam', '')
flags.DEFINE_string('model_name', '', '')
flags.DEFINE_integer("seed", 1, "random seed")
flags.DEFINE_integer("lsize", 1, "0,1,2")

flags.DEFINE_integer('num_epochs', 10, 'num_epochs')
flags.DEFINE_boolean('shuffle', True, 'shuffle')
flags.DEFINE_integer('batch_size', 4096, 'batch_size')
flags.DEFINE_string('file_reader', 'textline', 'file_reader')

flags.DEFINE_integer('n_node_type', 2, 'num_node_type')
flags.DEFINE_integer('n_edge_type', 1, 'num_edge_type')
flags.DEFINE_integer('edge_dim', 30, 'edge_dim')
flags.DEFINE_integer('embedding_dim', 128, 'embedding_dim')
flags.DEFINE_string("buckets", './buckets/', "buckets")
flags.DEFINE_integer("num_sample", 100, "sample number")
flags.DEFINE_integer("n_negs", 5, "n_negs")

flags.DEFINE_string("model", "HEP", "the sampling model")
tf.set_random_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)


def create_directory(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print("Complete create directory:\t{}".format(pathname))
    else:
        print("The directory:\t {} has been created".format(pathname))
    return 0


def train(store_dir, num_nodes, num_train_pairs, batch_size1, batch_size2):

    inputs_L1 = input_LP(FLAGS, batch_size1)
    inputs_L2 = input_EP(FLAGS, batch_size2)

    global_step = tf.train.get_or_create_global_step()

    train_model = hep(FLAGS, global_step, batch_size1, batch_size2, num_nodes, "train")

    hooks = [tf.train.StopAtStepHook(last_step=8000000000)]
    scaffold = tf.train.Scaffold(saver=train_model.saver,
                                 init_op=tf.global_variables_initializer())

    starttime = datetime.datetime.now()
    create_directory(store_dir)
    www = open(store_dir + "times.txt", "w")
    with tf.train.MonitoredTrainingSession(scaffold=scaffold, checkpoint_dir=store_dir, save_checkpoint_secs=3600,
                                           hooks=hooks) as mon_sess:
        step = 0
        writer = tf.summary.FileWriter(store_dir + "logs", mon_sess.graph)
        sum1 = 0
        sum2 = 0
        t1 = datetime.datetime.now()
        while not mon_sess.should_stop():
            step += 1
            t2 = datetime.datetime.now()
            input1 = mon_sess.run(inputs_L1)

            input2 = mon_sess.run(inputs_L2)
            t21 = datetime.datetime.now()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            feed_dicts = create_placeholders(input1, train_model, input2)
            t3 = datetime.datetime.now()
            _, rs = mon_sess.run(train_model.train_op, options=run_options, run_metadata=run_metadata,
                                 feed_dict=feed_dicts)
            t4= datetime.datetime.now()
            writer.add_run_metadata(run_metadata, 'step%d' % step)
            writer.add_summary(rs, step)

            sum1 += (t21-t2).seconds
            sum2 += (t4-t3).seconds
            www.write("{}\t{}\t{}\t{}\t{}\n".format((t4-t1).seconds, sum1, sum2, (t21-t2).seconds, (t4-t3).seconds))
            # print("{}\t{}\t{}\n".format((t4-t1).seconds, sum1, sum2))
            if step % 50 == 0:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(store_dir + 'timeline_01_{}.json'.format(step), 'w') as f2:
                    f2.write(chrome_trace)

def create_placeholders(inputs1, model, inputs2):
    ids, id_types, negs, nbrs, segments, edge_types, edge_features, weight_ab, sampled_nodes, sampled_edges = inputs2
    feed_dicts = {
        model.ids: ids,
        model.id_types: id_types,
        model.negs: negs,
        model.uids:inputs1[0],
        model.sids:inputs1[1],
        model.labels:inputs1[2],
    }
    # print(negs)
    for i in range(FLAGS.n_node_type):
        feed_dicts[model.nbrs[i]] = nbrs[i]
        feed_dicts[model.edge_types[i]] = edge_types[i]
        feed_dicts[model.segments[i]] = segments[i]
        feed_dicts[model.weight_ab[i]] = weight_ab[i]
        feed_dicts[model.edge_features[i]] = edge_features[i]
    return feed_dicts


def test(store_dir, path, num_node, filename, output_name):
    test_file = FLAGS.input_path + filename
    create_directory(path)

    uids = []
    sids = []
    labels = []
    with open(test_file) as f:
        while 1:
            line = f.readline()
            if not line:
                break
            info = line.strip().split(FLAGS.col_delim1)
            uids.append(int(info[0]))
            sids.append(int(info[1]))
            labels.append(int(info[2]))

    inputs_L1 = [uids, sids, labels]

    global_step = tf.train.get_or_create_global_step()

    test_model = hep(FLAGS,  global_step, 0, 0, num_node, "test")


    with tf.Session() as sess:
        test_model.saver.restore(sess, store_dir + FLAGS.model_name)
        data2 = sess.run(test_model.output, feed_dict={test_model.uids:uids, test_model.sids:sids, test_model.labels:labels})


        predinfo = (data2 > 0.5) + 0
        trueinfo = (np.array(inputs_L1[2]) > 0.5) + 0
        with open(store_dir + "f1_score.txt", "w") as w:
            r = metrics.recall_score(trueinfo, predinfo)
            p = metrics.precision_score(trueinfo, predinfo)

            f1 = (2 * p * r) / (p + r)
            auc = calculate_AUC(np.array(inputs_L1[2]), data2)

            w.write("{}\t{}\t{}\t{}\n".format(p, r, f1, auc))

def main(argv=None):
    store_dir = FLAGS.buckets + "{}{}_alpha_{}_seed_{}_sample_{}_{}/".format(FLAGS.input_path, FLAGS.model,
                                                                          str(FLAGS.alpha), str(FLAGS.seed),
                                                                          str(FLAGS.num_sample), str(FLAGS.lsize))

    num_nodes = 4769654
    num_train_pairs = 1698375

    min_size = 1698375
    basic_size = FLAGS.batch_size + FLAGS.lsize * 1024
    batch_size1 = int((num_train_pairs / min_size) * basic_size)
    batch_size2 = int((num_nodes / min_size) * basic_size)

    if FLAGS.opt == "train":
        train(store_dir, num_nodes, num_train_pairs, batch_size1, batch_size2)
    else:
        output_filename = "{}_alpha_{}_seed_{}_sample_{}_{}/".format(FLAGS.model,
                                                                          str(FLAGS.alpha), str(FLAGS.seed),
                                                                          str(FLAGS.num_sample), str(FLAGS.lsize))

        test(store_dir, "./Aminerdt/", num_nodes, FLAGS.opt + "_ali.csv", output_filename)


def calculate_AUC(label, preds):
    ranked_ids = np.argsort(preds)

    p = preds[ranked_ids]
    t = np.array(label)[ranked_ids]

    fpr, tpr, thresholds = metrics.roc_curve(t, p, pos_label=1)
    return metrics.auc(fpr, tpr)


if __name__ == '__main__':
    tf.app.run()

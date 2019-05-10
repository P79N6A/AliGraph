import csv
from odps import ODPS
from odps.tunnel import TableTunnel


def download_infos(tablename, storename, keys):
    o = ODPS("LTAIWt3hG5GvYBhX", "RriedkAIENmPvXvRmQcy9wRqOYx3QV", 'graph_embedding_intern_dev',
             endpoint='http://service-corp.odps.aliyun-inc.com/api')

    project = o.get_project()
    csv_file = open(storename, mode='w')
    writer = csv.writer(csv_file, delimiter='\t')

    tunnel = TableTunnel(o)
    download_session = tunnel.create_download_session(tablename)
    with download_session.open_record_reader(0, download_session.count) as reader:
        for record in reader:
            info = [record[key] for key in keys]
            writer.writerow(info)
    print("complete storing {}".format(storename))


if __name__ == '__main__':
    store_path = "./data/"

    # tablename = "graphs_final_format"
    # storename = store_path + tablename + ".csv"
    # download_infos(tablename, storename, ["id1","ids2","neg_nodes","feature"])

    # tablename = "labels_info"
    # storename = store_path + tablename + ".csv"
    # keys = ['uid', 'sids', "label"]
    # download_infos(tablename, storename, keys)
    #
    # tablename = "graph_user"
    # storename = store_path + tablename + ".csv"
    # keys = ['user_id', 'item_ids', 'edge_types_array', 'fids']
    # download_infos(tablename, storename, keys)
    #
    # tablename = "graph_item"
    # storename = store_path + tablename + ".csv"
    # keys = ['user_ids', 'item_id', 'edge_types_array', 'fids']
    # download_infos(tablename, storename, keys)
    #
    tablename = "yg_graph_hep_with_negs_v2"
    storename = store_path + tablename + ".csv"
    keys = ['info']
    download_infos(tablename, storename, keys)
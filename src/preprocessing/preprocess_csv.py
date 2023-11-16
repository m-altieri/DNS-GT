# -*- coding: utf-8 -*-
import datetime
import dill
import pandas as pd
import sys

from tacks.utils.base import Workspace, get_argparser, get_config


#######################################################################################
# Argument Parser and Workspace

argparser = get_argparser(
    sys.modules[__name__].__doc__,
    flags=['debug', 'silent'],
)

args = argparser.parse_args()
args.force = True

workspace = Workspace(name='PCAP_CSV', instance_name='Preprocessing', args=args)

config = get_config()

#######################################################################################
# Processing of CSV

data_path = config.get_path('paths', 'data') / 'TI-2016'

csv_path = data_path / 'csv'

if not csv_path.exists():
    err_msg = 'Cannot find csv files at `{}``.'
    raise FileNotFoundError(err_msg.format(csv_path))

csvpp_path = data_path / 'csv_pp'
csvpp_path.mkdir(exist_ok=True)

hosts = set()
domains = set()
queries_rem = list()
queries_ts = list()
responses_rem = list()
pp_pkts = list()

for csv_path in csv_path.glob('*.csv'):
    csv_path = csv_path.parent / '20160424_055409.csv'
    workspace.logger.info(f'Loading {csv_path}...')

    # -- Openning the CSV
    filename = csv_path.name

    if (csvpp_path / filename).exists():
        workspace.logger.info('Already preprocessed. Skipped.')
        continue

    packets = pd.read_csv(str(csv_path), sep=';')

    # -- Cleaning

    # renaming column names
    packets = packets.rename(
        columns={
            'frame.number': 'id',
            'frame.time_epoch': 'timestamp',
            'ip.src': 'ip_src',
            'ip.dst': 'ip_dst',
            'udp.srcport': 'port_src',
            'udp.dstport': 'port_dst',
            'dns.retransmission': 'is_retransmission',
            'dns.qry.name': 'qry_name',
            'dns.qry.type': 'qry_type',
            'dns.flags.response': 'is_response',
            'dns.resp.name': 'resp_names',
            'dns.resp.type': 'resp_types',
            'dns.flags.rcode': 'rcode',
        }
    )

    # fill missing values
    packets = packets.fillna(
        {'port_src': -1, 'port_dst': -1, 'qry_type': -1, 'is_retransmission': 0}
    )
    packets = packets.astype(
        {
            'port_src': int,
            'port_dst': int,
            'qry_type': int,
            'is_response': bool,
            'is_retransmission': bool,
        }
    )

    # set the time origin to Day 0 00:00
    packets['timestamp'] -= datetime.datetime(2016, 4, 24, 0, 0, 0).timestamp()

    # convert port as int and fill missing values
    packets['port_src'] = packets['port_src'].astype(int)
    packets['port_dst'] = packets['port_dst'].astype(int)

    # remove queries with missing name
    packets.dropna(subset=['qry_name'], inplace=True)

    # convert query types
    packets['qry_type'] = packets['qry_type'].astype(int)
    packets['qry_type'].replace(
        to_replace={
            -1: 'NA',
            0: 'NA',
            1: 'A',
            2: 'NS',
            5: 'CNAME',
            6: 'SOA',
            12: 'PTR',
            15: 'MX',
            16: 'TXT',
            28: 'AAAA',
            33: 'SRV',
            43: 'DS',
            48: 'DNSKEY',
            255: '*',
        },
        inplace=True,
    )

    # -- Splitting between queries and responses
    # grouped = packets.groupby('is_response')
    # queries = grouped.get_group(False)
    # responses = grouped.get_group(True)

    # # drop useless columns
    # queries.drop(
    #     columns=['is_response', 'resp_names', 'resp_types', 'rcode'], inplace=True
    # )
    # responses.drop(columns=['is_response'], inplace=True)

    # create DNS chains

    for idp in range(packets.size):
        pkt = packets.loc[idp]

        if pkt['is_response']:
            pkt_sig = (
                pkt['ip_dst'],
                pkt['port_dst'],
                pkt['ip_src'],
                pkt['port_src'],
                pkt['qry_name'],
            )

            if pkt_sig in queries_rem:
                queries_rem.pop(pkt_sig)
                idq = queries_rem.index(pkt_sig)
                pp_pkts.append([queries_ts[idq]] + [packets['timestamp']] + pkt_sig)
            else:
                responses_rem.append(pkt_sig)

        else:
            pkt_sig = (
                pkt['ip_src'],
                pkt['port_src'],
                pkt['ip_dst'],
                pkt['port_dst'],
                pkt['qry_name'],
            )
            queries_rem.append(pkt_sig)
            queries_ts.append(pkt['timestamp'])

    raise

    # save the preprocessed csv
    workspace.logger.info('Saving preprocessed CSV.')
    packets.to_csv(csvpp_path / filename, sep=';')

    # update list of domains and hosts
    hosts.update(packets['ip_src'])
    hosts.update(packets['ip_dst'])
    domains.update(packets['qry_name'])


# save hosts
workspace.logger.info('Saving hosts.')
with open(csvpp_path / 'hosts', 'w') as outfile:
    dill.dump(hosts)

workspace.logger.info('Saving domains.')
with open(csvpp_path / 'domains', 'w') as outfile:
    dill.dump(domains)
    dill.dump(domains)
    dill.dump(domains)

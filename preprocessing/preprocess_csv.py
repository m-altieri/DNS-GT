# -*- coding: utf-8 -*-
"""Preprocessing PCAP files.

Author: Massimiliano Altieri <massimiliano.altieri@ec.europa.eu>
Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
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

data_path = config.get_path('paths', 'data') / 'TI-2016-Partial'

csv_path = data_path / 'csv'

csvpp_path = data_path / 'csv_pp'
csvpp_path.mkdir(exist_ok=True)

hosts = set()
domains = set()

for csv_path in csv_path.glob('*.csv'):
    workspace.logger.info(f'Preprocessing {csv_path}...')

    filename = csv_path.name

    if (csvpp_path / filename).exists():
        workspace.logger.info('Already preprocessed. Skipped.')
        continue

    packets = pd.read_csv(str(csv_path), sep=';')

    # rename columns
    packets = packets.rename(
        columns={
            'frame.time_epoch': 'timestamp',
            'ip.src': 'ip_src',
            'ip.dst': 'ip_dst',
            'udp.srcport': 'port_src',
            'udp.dstport': 'port_dst',
            'dns.count.queries': 'n_queries',
            'dns.qry.name': 'qry_name',
            'dns.qry.type': 'qry_type',
            'dns.flags.response': 'is_response',
            'dns.resp.name': 'resp_names',
            'dns.resp.type': 'resp_types',
        }
    )

    # set the time origin to Day 0 00:00
    packets['timestamp'] -= datetime.datetime(2016, 4, 24, 0, 0, 0).timestamp()

    # convert query types
    packets['qry_type'] = packets['qry_type'].astype(float)

    # packets = packets.fillna({'qry_type': -1.0})
    packets['qry_type'].replace(
        to_replace={
            -1.0: 'NA',
            0.0: 'NA',
            1.0: 'A',
            2.0: 'NS',
            5.0: 'CNAME',
            6.0: 'SOA',
            12.0: 'PTR',
            15.0: 'MX',
            16.0: 'TXT',
            28.0: 'AAAA',
            33.0: 'SRV',
            43.0: 'DS',
            48.0: 'DNSKEY',
            255.0: '*',
        },
        inplace=True,
    )

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

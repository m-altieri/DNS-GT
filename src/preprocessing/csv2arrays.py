import numpy as np
import pandas as pd
import os
import argparse
import re

argparser = argparse.ArgumentParser()
argparser.add_argument('input_folder', help='Folder containing processed PCAPs in .csv format')
argparser.add_argument('output_folder', help='Folder to save output arrays to')
argparser.add_argument('--from-responses', action='store_const', const=True, default=False, help='Whether to extract query information from requests or responses. If True and some queries have no response, it will throw an error.')
args = argparser.parse_args()


def clean_urls(url):
    url = re.sub('[^!-~]+', '', url).lower()
    #match = re.match('^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$', url)
    #    if match:
    #        url = match[0]
    return url
    
    
# check input folder is valid
if not os.path.exists(args.input_folder) or len(os.listdir(args.input_folder)) == 0:
    raise FileNotFoundError(f'{args.input_folder} is not a valid input folder.')

# create output folder if not exists
if not os.path.exists(os.path.join(args.output_folder, 'hosts')):
    os.makedirs(os.path.join(args.output_folder, 'hosts'))
if not os.path.exists(os.path.join(args.output_folder, 'domains')):
    os.makedirs(os.path.join(args.output_folder, 'domains'))
if not os.path.exists(os.path.join(args.output_folder, 'queries')):
    os.makedirs(os.path.join(args.output_folder, 'queries'))

for filename in os.listdir(args.input_folder):
    # read processed pcap
    df = pd.read_csv(os.path.join(args.input_folder, filename))

    # get host and domain columns
    hosts = df['req_src_ip'] if not args.from_responses else df['res_dst_ip']
    domains = df['req_qry_domain'] if not args.from_responses else df['res_qry_domain']

    # remove NaNs (responses with no query)
    hosts = hosts.dropna()
    domains = domains.dropna()

    # clean domains (now they are in the form b'something')
    assert all(domains.str.startswith("b'")) or not any(domains.str.startswith("b'"))
    if all(domains.str.startswith("b'")):
        domains = domains.map(lambda x: str(x).split("'")[1])
        
    hosts = hosts.map(clean_urls)
    domains = domains.map(clean_urls)

    # save array of communications
    np.save(os.path.join(args.output_folder, 'queries', f'queries-{filename}.npy'), list(zip(hosts, domains)))

    # save arrays of unique hosts and domains
    np.save(os.path.join(args.output_folder, 'hosts', f'hosts-{filename}.npy'), hosts.unique())
    np.save(os.path.join(args.output_folder, 'domains', f'domains-{filename}.npy'), domains.unique())



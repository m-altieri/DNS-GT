import numpy as np
import os
import argparse
import re

argparser = argparse.ArgumentParser()
argparser.add_argument('--hosts_folder', default='arrays/small/hosts/', help='Folder containing the hosts arrays')
argparser.add_argument('--domains_folder', default='arrays/small/domains/', help='Folder containing the domains arrays')
argparser.add_argument('--output_folder', default='vocabs/small/', help='Folder to output the vocabularies to')
args = argparser.parse_args()

hosts, domains = [], []

for f in os.listdir(args.hosts_folder):
    fpath = os.path.join(args.hosts_folder, f)
    new_hosts = np.load(fpath, allow_pickle=True, encoding='ASCII')
    #for i in range(len(new_hosts)):
    #    new_hosts[i] = re.sub('[^!-~]+', '', new_hosts[i]).lower()
    #    match = re.match('^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$', new_hosts[i])
    #    if match:
    #        new_hosts[i] = match[0]
    new_hosts = list(map(lambda h: re.sub('[^!-~\\n]+', '', h), new_hosts))
    hosts = np.unique(np.concatenate((hosts, new_hosts)))
    print(f'{f} processed')
hosts = np.unique(hosts)
    
for f in os.listdir(args.domains_folder):
    fpath = os.path.join(args.domains_folder, f)
    new_domains = np.load(fpath, allow_pickle=True, encoding='ASCII')
    #for i in range(len(new_domains)):
    #    new_domains[i] = re.sub('[^!-~]+', '', new_domains[i]).lower()
    #    match = re.match('^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$', new_domains[i])
    #    if match:
    #        new_domains[i] = match[0]
    new_domains = list(map(lambda d: re.sub('[^!-~\\n]+', '', d), new_domains))
    domains = np.unique(np.concatenate((domains, new_domains)))
    print(f'{f} processed')
domains = np.unique(domains)

hosts_vocab = '\n'.join(hosts)#.replace('[', '').replace(']', '')#.encode('ascii', errors='ignore').decode()
domains_vocab = '\n'.join(domains)#.replace('[', '').replace(']', '')#.encode('ascii', errors='ignore').decode()

#hosts_vocab = re.sub('[^!-~\\n]+', '', hosts_vocab)
#domains_vocab = re.sub('[^!-~\\n]+', '', domains_vocab)

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

with open(os.path.join(args.output_folder, 'hosts_vocab.txt'), 'w') as f:
    f.write('<START>\n')
    f.write('<PAD>\n')
    f.write('<MASK>\n')
    f.write(hosts_vocab)

with open(os.path.join(args.output_folder, 'domains_vocab.txt'), 'w') as f:
    f.write('<START>\n')
    f.write('<PAD>\n')
    f.write('<MASK>\n')
    f.write(domains_vocab)
    

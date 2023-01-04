
# TI-2016 Scapy Analysis Report

PCAP File `20160424_055409.pcap`
___

## General PCAP info
```python
>>> pcap = rdpcap('~/datasets/TI-2016-Partial/Day0_24_04_2016/20160424_055409.pcap')
```

**Total number of packets:** 348,481
```python
>>> len(pcap)
```
```python
348481
```

## Classification of packets based on direction of query

**Query requests:** 143,197 (41.09%)
```python
>>> reqs = [p for p in pcap if 'UDP' in p and p['UDP'].dport == 53]
>>> print(f'{len(reqs)} ({100 * len(reqs) / len(pcap):.2f}%)')
```
```python
143197 (0.41%)
```

**Query responses:** 203,650 (58.44%)
```python
>>> resp = [p for p in pcap if 'UDP' in p and p['UDP'].sport == 53]
>>> print(f'{len(resp)} ({100 * len(resp) / len(pcap):.2f}%)')
```
```python
203650 (58.44%)
```

**Non-UDP packets:** 1,634 (0.47%)
```python
>>> nonudp = len([p for p in pcap if 'UDP' not in p])
>>> print(f'{nonudp} ({100 * nonudp / len(pcap):.2f}%)')
```
```python
1634 (0.47%)
```

## DNS servers

>__The number of source DNS servers is different from the number of unique destination DNS servers. Why?__

**Unique source DNS servers: (IPs that appear as source in at least one query response with port 53)**
```python
>>> src_servers = [p['IP'].src for p in pcap if 'UDP' in p and p['UDP'].sport == 53]
>>> set(src_servers)
```

**Source DNS servers list**
```python
{'104.192.108.109',
 '104.192.108.114',
 '104.197.191.4',
 '119.81.212.83',
 '154.53.224.130',
 '154.53.224.134',
 '169.54.166.72',
 '169.55.26.189',
 '169.57.1.214',
 '172.31.1.6',
 '172.31.1.7',
 '172.31.1.8',
 '172.31.3.121',
 '177.47.27.159',
 '178.18.201.98',
 '198.41.222.254',
 '208.67.220.123',
 '208.67.220.220',
 '208.67.222.123',
 '208.67.222.222',
 '209.244.0.3',
 '209.244.0.4',
 '4.2.2.1',
 '4.2.2.2',
 '4.2.2.3',
 '4.2.2.4',
 '5.133.8.122',
 '54.251.106.38',
 '62.210.142.187',
 '64.228.201.151',
 '64.68.192.10',
 '64.68.196.10',
 '69.171.239.13',
 '74.82.42.42',
 '77.234.40.92',
 '77.234.42.114',
 '77.234.43.63',
 '77.234.45.53',
 '8.8.4.4',
 '8.8.8.8'}
```

**Number of source DNS servers:** 40
```python
>>> print(f'DNS servers: {len(set(src_servers))}')
```
```python
DNS servers: 40
```

**Unique destination DNS servers: (IPs that appear as destination in at least one query request with port 53)**
```python
>>> dst_servers = [p['IP'].dst for p in pcap if 'UDP' in p and p['UDP'].dport == 53]
>>> set(dst_servers)
```

**Destination DNS servers list**
```python
{'115.249.164.142',
 '125.19.40.90',
 '172.31.1.6',
 '172.31.1.8',
 '172.31.19.11',
 '172.31.19.13',
 '172.31.19.9',
 '172.31.3.105',
 '172.31.3.121',
 '172.31.3.128',
 '172.31.3.129',
 '172.31.3.150',
 '172.31.4.13',
 '172.31.4.40',
 '172.31.7.251',
 '192.82.134.30',
 '193.0.9.10',
 '199.180.180.63',
 '199.212.0.63',
 '199.249.117.1',
 '199.249.125.1',
 '199.254.59.1',
 '199.254.60.1',
 '199.254.61.1',
 '199.254.62.1',
 '199.71.0.63',
 '204.61.216.50',
 '64.228.201.151',
 '8.8.4.4',
 '8.8.8.8'}
```

**Number of destination DNS servers:** 30
```python
>>> print(f'DNS servers: {len(set(dst_servers))}')
```
```python
DNS servers: 30
```

> I don't know why, but most DNS servers work as either source or destination, not both.
> In fact, there are 40 source DNS servers and 30 destination DNS servers, but of these two sets are mostly disjoint. If $S$ is the set of source servers and $D$ the set of destination servers:

- $|S| = 40$
- $|D| = 30$
- $|S\cap D| = 6$
- $|S\cup D| = 64$
- $|S \setminus D| = 34$
- $|D \setminus S| = 24$
> The DNS servers that both receive queries and send responses are:
- 172.31.1.8 (sent: 11113, received: 235)
- 172.31.3.121
- 8.8.8.8 (sent: 17963, received: 17)
- 8.8.4.4 (sent: 7790, received: 6)
- 172.31.1.6
- 64.228.201.151

```python
>>> print(f'Sent: {len([p for p in pcap if 'UDP' in p and p['IP'].src=='8.8.8.8']}')
>>> print(f'Received: {len([p for p in pcap if 'UDP' in p and p['IP'].dst=='8.8.8.8']}')
```

## Local DNS Servers

The amount of DNS servers that in turn make queries to other DNS servers or receive responses from other DNS servers are intermediate resolvers.
We assume that the dataset only contains packets related to intermediate resolvers that are within the campus network.
We can calculate how many intermediate local resolvers are in the campus network, as the amount of DNS servers that make queries or receive responses from other DNS servers.
```python
>>> dns_servers = src_servers | dst_servers
```
**Number of local querying intermediate resolvers: 6**
```python
>>> request_makers = set([p['IP'].src for p in pcap if 'UDP' in p and p['UDP'].dport == 53])
>>> len(request_makers & dns_servers)
6
```
**List of local querying intermediate resolvers:**
```python
>>> request_makers & dns_servers
{'172.31.1.6',
 '172.31.19.11',
 '172.31.19.9',
 '172.31.3.105',
 '172.31.3.128',
 '172.31.3.150'}
```
**Number of local response-receiving intermediate resolvers: 8**
```python
>>> response_receivers = set([p['IP'].dst for p in pcap if 'UDP' in p and p['UDP'].sport == 53])
>>> len(response_receivers & dns_servers)
8
```
**List of local response-receiving intermediate resolvers:**
```python
>>> response_receivers & dns_servers
{'172.31.1.6',
 '172.31.19.11',
 '172.31.19.9',
 '172.31.3.105',
 '172.31.3.121',
 '172.31.3.128',
 '172.31.3.150',
 '172.31.4.40'}
```
The two sets are very similar, with the exception of 172.31.3.121 and 172.31.4.40 which receive responses but don't make any query.

We can also calculate the amount of queries made by the querying local DNS servers and the amount of responses received by the receiving local DNS servers.
**Queries made by each local querying intermediate resolver:**
```python
>>> for ip in sorted(request_makers & dns_servers):
...:     print(f"{ip:<16}: {len([p for p in pcap if p['IP'].src == ip and 'UDP' in p and p['UDP'].dport == 53])}")
...: 
172.31.1.6      : 54127
172.31.19.11    : 194
172.31.19.9     : 33
172.31.3.105    : 182
172.31.3.128    : 118
172.31.3.150    : 1
```
**Responses received by each local receiving intermediate resolver:**
```python
>>> for ip in sorted(response_receivers & dns_servers):
...:     print(f"{ip:<16}: {len([p for p in pcap if p['IP'].dst == ip and 'UDP' in p and p['UDP'].sport == 53])}")
...: 
172.31.1.6      : 54324
172.31.19.11    : 125
172.31.19.9     : 31
172.31.3.105    : 182
172.31.3.121    : 76787
172.31.3.128    : 118
172.31.3.150    : 1
172.31.4.40     : 24
```

We can see that the main querying local DNS server is `172.31.1.6` and the main receiving local DNS servers are `172.31.3.121` and again `172.31.1.6`.


## Query Types
```python
>>> set([p['DNSQR'].qtype for p in pcap if 'DNSQR' in p])
```
**List of query types that appear in the pcap:**
- 0 (unused)
- 1 (A): Address record
- 2 (NS): Name Server record
- 6 (SOA): Start Of a zone of Authority
- 12 (PTR): domain name PoinTeR
- 15 (MX): Mail eXchange
- 16 (TXT): TeXT strings
- 28 (AAAA): IPv6 Address
- 33 (SRV): SeRVer selection
- 43 (DS): Delegation Signer
- 48 (DNSKEY): DNS public KEY
- 255 (*): any records in the cache
- ~~37420: malformed packets~~
- ~~42789: malformed packets~~

**Number of queries for each query type:**
```python
>>> for t in sorted(unique_types):
...:     print(f"Type {t:<6}: {len([p for p in pcap if 'DNSQR' in p and p['DNSQR'].qtype == t])}")
```
```python
Type 0     : 1
Type 1     : 293149
Type 2     : 2492
Type 6     : 3555
Type 12    : 4248
Type 15    : 6
Type 16    : 622
Type 28    : 17750
Type 33    : 456
Type 43    : 24055
Type 48    : 324
Type 255   : 89
Type 37420 : 2
Type 42789 : 1
```
The most common type of query is type A by far, followed by DS, AAAA, PTR, SOA and NS, in this order.

**Number of <ins>request</ins> queries for each query type:**
```python
>>> for t in sorted(unique_types):
...:     print(f"Type {t:<6}: {len([p for p in pcap if 'DNSQR' in p and p['DNSQR'].qtype == t and p['IP'].dport == 53])}")
```
```python
Type 0     : 1
Type 1     : 133392
Type 2     : 29
Type 6     : 2062
Type 12    : 2045
Type 15    : 0
Type 16    : 321
Type 28    : 5083
Type 33    : 176
Type 43    : 1
Type 48    : 0
Type 255   : 48
Type 37420 : 0
Type 42789 : 0
```

**Number of <ins>response</ins> queries for each query type:**
```python
>>> for t in sorted(unique_types):
...:     print(f"Type {t:<6}: {len([p for p in pcap if 'DNSQR' in p and p['DNSQR'].qtype == t and p['IP'].sport == 53])}")
```
```python
Type 0     : 0
Type 1     : 159757
Type 2     : 2463
Type 6     : 1493
Type 12    : 2203
Type 15    : 6
Type 16    : 301
Type 28    : 12667
Type 33    : 280
Type 43    : 24054
Type 48    : 324
Type 255   : 41
Type 37420 : 2
Type 42789 : 1
```
It's noteworthy that some query types (especially 2 (NS), 43 (DNS) and 48 (DNSKEY)) appear almost exclusively in response queries.
Moreover, only response queries contain malformed query types.
The other query types are more or less balanced between requests and responses.

## Hosts

Hosts are considered as the IPs that make DNS requests but are not DNS servers themselves (they appear at the start of the DNS query chain),
or the IPs that receive DNS responses but are not DNS servers themselves (they appear at the end of the DNS resolution chain).

To obtain the first group of hosts, let's take all request makers and then remove the DNS servers from there.
To obtain the second group of hosts, let's take all response receivers and then remove the DNS servers from there.
To obtain all hosts, let's take the union of these two sets.
```python
>>> dst_dns_servers = set([p['IP'].dst for p in pcap if 'UDP' in p and p['UDP'].dport == 53])
>>> src_dns_servers = set([p['IP'].src for p in pcap if 'UDP' in p and p['UDP'].sport == 53])
>>> dns_servers = src_dns_servers + dst_dns_servers
```
**Number of IPs that send DNS requests: 1472**
```python
>>> request_makers = set([p['IP'].src for p in pcap if 'UDP' in p and p['UDP'].dport == 53])
>>> len(request_makers)
1472
```
**Number of IPs that send DNS requests but are not DNS servers (first-group hosts): 1466**
```python
>>> len(request_maker - dns_servers)
1466
```
**Number of IPs that receive DNS responses: 1037**
```python
>>> response_receivers = set([p['IP'].dst for p in pcap if 'UDP' in p and p['UDP'].sport == 53])
>>> len(response_receivers)
1037
```
**Number of IPs that receive DNS responses but are not DNS servers (second-group hosts): 1029**
```python
>>> len(response_receivers - dns_servers)
1029
```
**All actual hosts: 2197**
```python
>>> len(request_makers | response_receivers)
2197
```
We can notice that most hosts act either as a sender or as a receiver, rarely both.
**Number of hosts that are both senders and receivers: 312**
```python
>>> len(request_makers & response_receivers)
312
```
**Jaccard Similarity between sending and receiving hosts sets: 0.142**
```python
>>> print(f'{len(request_maker & response_receivers) / len(request_maker | response_receivers):.3f}')
0.142
```

fare numero medio di query per host










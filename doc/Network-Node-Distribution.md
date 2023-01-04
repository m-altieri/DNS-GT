```python
>>> Aqs = set([p['IP'].src for p in pcap if 'UDP' in p and p['UDP'].dport == 53])
>>> len(Aqs)
1472
```

```python
>>> Arr = set([p['IP'].dst for p in pcap if 'UDP' in p and p['UDP'].sport == 53])
>>> len(Arr)
1037
```

```python
>>> Aqr = set([p['IP'].dst for p in pcap if 'UDP' in p and p['UDP'].dport == 53])
>>> len(Aqr)
30
```

```python
>>> Ars = set([p['IP'].src for p in pcap if 'UDP' in p and p['UDP'].sport == 53])
>>> len(Ars)
40
```

```python
>>> Servers = Aqr | Ars
```

```python
>>> Hqs = Aqs - Servers
>>> len(Hqs)
1466
```

```python
>>> Hrr = Arr - Servers
>>> len(Hrr)
1029
```

```python
>>> Eqr = set([p['IP'].dst for p in pcap if 'UDP' in p and p['UDP'].dport == 53 and p['IP'].dst not in Aqs | Arr])
>>> len(Eqr)
22
```

```python
>>> Ers = set([p['IP'].src for p in pcap if 'UDP' in p and p['UDP'].sport == 53 and p['IP'].src not in Arr | Aqs])
>>> len(Ers)
38
```

List of IPs that sent at least a direct query from Hqs to Eqr (in Hqs):
```python
>>> set([p['IP'].src for p in pcap if 'UDP' in p and p['UDP'].dport == 53 and p['IP'].src in Hqs and p['IP'].dst in Eqr])
{'172.31.1.4',
 '172.31.12.119',
 '172.31.12.12',
 '172.31.12.228',
 '172.31.12.36',
 '172.31.13.127',
 '172.31.13.9',
 '204.42.253.2',
 '218.60.5.146'}
```
List of IPs that have received at least a direct query from Hqs to Eqr (in Eqr):
```python
>>> set([p['IP'].dst for p in pcap if 'UDP' in p and p['UDP'].dport == 53 and p['IP'].src in Hqs and p['IP'].dst in Eqr])
{'115.249.164.142',
 '125.19.40.90',
 '172.31.1.8',
 '172.31.3.129',
 '172.31.4.13',
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
 '8.8.4.4',
 '8.8.8.8'}
```
Number of direct queries from Hqs to Eqr:
```python
>>> len([p.time for p in pcap if 'UDP' in p and p['UDP'].dport == 53 and p['IP'].src in Hqs and p['IP'].dst in Eqr])
1188
```


Number of IPs in Hrr that have received at least a direct response from Ers
```python
>>> len(set([p['IP'].dst for p in pcap if 'UDP' in p and p['UDP'].sport == 53 and p['IP'].src in Ers and p['IP'].dst in Hrr]))
363
```
Number of IPs in Ers that have sent at least a direct response to Hrr
```python
>>> len(set([p['IP'].src for p in pcap if 'UDP' in p and p['UDP'].sport == 53 and p['IP'].src in Ers and p['IP'].dst in Hrr]))
27
```
Number of packets that have been sent directly from Ers to Hrr
```python
>>> len([p['IP'].time for p in pcap if 'UDP' in p and p['UDP'].sport == 53 and p['IP'].src in Ers and p['IP'].dst in Hrr])
22211
```



from scapy.all import rdpcap

pcap = rdpcap('../datasets/TI-2016-Partial/Day0_24_04_2016/20160424_055409.pcap')
pcap.nsummary()


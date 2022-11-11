import pyshark

pcap_path = '../dataset/TI-2016-Partial/Day0_24_04_2016/20160424_055409.pcap')

cap = pyshark.FileCapture(pcap_path)


cap[0].pretty_print()

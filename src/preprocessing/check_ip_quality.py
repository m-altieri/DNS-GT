import os
import pandas as pd
import scapy.all as scapy


# Path containing pcap files
INPUT_PATH = "/mnt/storage15/TI-2016"

# Initialize dictionary to hold IP addresses and their DNS requests and responses
ip_info = {}

for root, dirs, files in os.walk(INPUT_PATH):
    for pcap_file in files:
        if not pcap_file.endswith(".pcap"):
            continue

        print(f"Analyzing {pcap_file}...", end="", flush=True)

        # Open PCAP file
        packets = scapy.rdpcap(os.path.join(root, pcap_file))

        # Filter for DNS packets
        dns_packets = [packet for packet in packets if packet.haslayer(scapy.DNS)]

        resolvers = []

        for packet in dns_packets:
            # Check if packet is a DNS request
            if "DNS" in packet and "UDP" in packet and packet[scapy.UDP].dport == 53:
                # if packet.haslayer(scapy.DNSQR) and not packet.haslayer(scapy.DNSRR):
                # Get source IP address
                src_ip = packet[scapy.IP].src

                # Increment DNS request count for source IP address
                if src_ip in ip_info:
                    ip_info[src_ip]["requests"] += 1
                else:
                    ip_info[src_ip] = {"requests": 1, "responses": 0, "resolver": False}

                dst_ip = packet[scapy.IP].dst
                if dst_ip in ip_info:
                    ip_info[dst_ip]["resolver"] |= True
                else:
                    ip_info[dst_ip] = {"requests": 0, "responses": 0, "resolver": False}

            # Check if packet is a DNS response
            elif "DNS" in packet and "UDP" in packet and packet[scapy.UDP].sport == 53:
                # elif packet.haslayer(scapy.DNSRR) and not packet.haslayer(scapy.DNSQR):
                # Get destination IP address
                dst_ip = packet[scapy.IP].dst

                # Increment DNS response count for destination IP address
                if dst_ip in ip_info:
                    ip_info[dst_ip]["responses"] += 1
                else:
                    ip_info[dst_ip] = {"requests": 0, "responses": 1, "resolver": False}

                src_ip = packet[scapy.IP].src
                if src_ip in ip_info:
                    ip_info[src_ip]["resolver"] |= True
                else:
                    ip_info[src_ip] = {"requests": 0, "responses": 0, "resolver": False}

        print(f"analyzed {len(dns_packets)} DNS packets.")

# Print DNS requests and responses for each IP address
for ip, dns_counts in ip_info.items():
    print("IP address:", ip)
    print("DNS requests:", dns_counts["requests"])
    print("DNS responses:", dns_counts["responses"])
    print("Is Resolver:", dns_counts["resolver"])
    print()

ip_info = pd.DataFrame.from_dict(
    ip_info, orient="index", columns=["requests", "responses", "resolver"]
)
ip_info.index.rename("ip", inplace=True)
ip_info["ratio"] = ip_info["requests"] / ip_info["responses"]
print(ip_info)

ip_info.sort_values("requests", ascending=False, inplace=True)
ip_info.to_csv("ip_info.csv")

# Select suitable IPs and save them
ip_info = ip_info[
    (ip_info["ratio"] > 0.985)
    & (ip_info["ratio"] < 1.015)
    & (ip_info["resolver"] == False)
    & (ip_info["requests"] > 1000)
]
print(ip_info)
ip_info.to_csv("good_ips.csv")

#!/bin/sh
# Preprocessing of PCAP files


# Column names


echo "Conversion of PCAP into CSV"


mkdir -p $1/csv
for infilename in `find $1 -name '*.pcap' -type f`;
do
    echo Processing $infilename...
    outfilename=$(basename "$infilename" .pcap).csv

    tshark -r $infilename -Y "dns and dns.count.queries == 1" \
    -T fields -E header='y' -E separator=';' \
     -e frame.time_epoch \
     -e ip.src \
     -e ip.dst \
     -e udp.srcport \
     -e udp.dstport \
     -e dns.qry.name \
     -e dns.qry.type \
     -e dns.flags.response \
     -e dns.resp.name \
     -e dns.resp.type > $1/csv/$outfilename

    echo Saved as $1/csv/$outfilename
done


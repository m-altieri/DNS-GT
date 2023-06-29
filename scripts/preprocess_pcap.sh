#!/bin/sh
# Preprocessing of PCAP files
#
# Author: Massimiliano Altieri <massimiliano.altieri@ec.europa.eu>
# Author: Ronan Hamon <ronan.hamon@ec.europa.eu>

echo "Conversion of PCAP into CSV"

mkdir -p $1/csv
for infilename in `find $1 -name '*.pcap' -type f`;
do
    echo Processing $infilename...
    outfilename=$(basename "$infilename" .pcap).csv

    tshark -r $infilename -Y "dns and not _ws.malformed" \
    -T fields -E header='y' -E separator=';' -E quote='d' \
     -e frame.number \
     -e frame.time_epoch \
     -e ip.src \
     -e ip.dst \
     -e udp.srcport \
     -e udp.dstport \
     -e dns.id \
     -e dns.retransmission \
     -e dns.qry.name \
     -e dns.qry.type \
     -e dns.flags.response \
     -e dns.resp.name \
     -e dns.resp.type  \
     -e dns.flags.rcode \
    > $1/csv/$outfilename

    echo Saved as $1/csv/$outfilename
done

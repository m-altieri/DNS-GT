#!/bin/sh
# Preprocessing of PCAP files using tshark
#
# This scripts extract all meaningful information and save them as CSV files with a
# similar folder structure.
#
# Author: Massimiliano Altieri <massimiliano.altieri@ec.europa.eu>
# Author: Ronan Hamon <ronan.hamon@ec.europa.eu>

echo "Conversion of PCAP files into CSV"

# Test the absence of argument
if [ -z "$1" ]; then
  echo "A folder should be given as argument."
  exit 1
fi


# Remove trailing slash
datapath="${1%/}"
outpath="$datapath/tcsv"

# Create the forder if missing
mkdir -p $datapath/tcsv

for infilename in `find $datapath -name '*.pcap' -type f`;
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
    > $outpath/$outfilename

    echo Saved as $outpath/$outfilename
done

#!/bin/bash
# Downloading of blacklists from firebog.net and their categorization

echo "Downloading blacklists from firebog.net"

# Test the absence of argument
if [ -z "$1" ]; then
  echo "A path should be provided as argument."
  exit 1
fi

blacklists_path="${1%/}/blacklists"

# Define blacklists URLs
suspicious_good=(
    "https://raw.githubusercontent.com/PolishFiltersTeam/KADhosts/master/KADhosts.txt" 
    "https://raw.githubusercontent.com/FadeMind/hosts.extras/master/add.Spam/hosts" 
    "https://v.firebog.net/hosts/static/w3kbl.txt"
)
suspicious_ok=(
    "https://raw.githubusercontent.com/matomo-org/referrer-spam-blacklist/master/spammers.txt"
    "https://someonewhocares.org/hosts/zero/hosts"
    "https://raw.githubusercontent.com/VeleSila/yhosts/master/hosts"
    "https://winhelp2002.mvps.org/hosts.txt"
    "https://v.firebog.net/hosts/neohostsbasic.txt"
    "https://raw.githubusercontent.com/RooneyMcNibNug/pihole-stuff/master/SNAFU.txt"
    "https://paulgb.github.io/BarbBlock/blacklists/hosts-file.txt"
)
advertising_good=(
    "https://adaway.org/hosts.txt"
    "https://v.firebog.net/hosts/AdguardDNS.txt"
    "https://v.firebog.net/hosts/Admiral.txt"
    "https://raw.githubusercontent.com/anudeepND/blacklist/master/adservers.txt"
    "https://s3.amazonaws.com/lists.disconnect.me/simple_ad.txt"
    "https://v.firebog.net/hosts/Easylist.txt"
    "https://pgl.yoyo.org/adservers/serverlist.php?hostformat=hosts&showintro=0&mimetype=plaintext"
    "https://raw.githubusercontent.com/FadeMind/hosts.extras/master/UncheckyAds/hosts"
    "https://raw.githubusercontent.com/bigdargon/hostsVN/master/hosts"
)
advertising_ok=(
    "https://raw.githubusercontent.com/jdlingyu/ad-wars/master/hosts"
)
tracking_good=(
    "https://v.firebog.net/hosts/Easyprivacy.txt"
    "https://v.firebog.net/hosts/Prigent-Ads.txt"
    "https://raw.githubusercontent.com/FadeMind/hosts.extras/master/add.2o7Net/hosts"
    "https://raw.githubusercontent.com/crazy-max/WindowsSpyBlocker/master/data/hosts/spy.txt"
    "https://hostfiles.frogeye.fr/firstparty-trackers-hosts.txt"
)
tracking_ok=(
    "https://www.github.developerdan.com/hosts/lists/ads-and-tracking-extended.txt"
    "https://raw.githubusercontent.com/Perflyst/PiHoleBlocklist/master/android-tracking.txt"
    "https://raw.githubusercontent.com/Perflyst/PiHoleBlocklist/master/SmartTV.txt"
    "https://raw.githubusercontent.com/Perflyst/PiHoleBlocklist/master/AmazonFireTV.txt"
    "https://gitlab.com/quidsup/notrack-blocklists/raw/master/notrack-blocklist.txt"
)
malicious_good=(
    "https://raw.githubusercontent.com/DandelionSprout/adfilt/master/Alternate%20versions%20Anti-Malware%20List/AntiMalwareHosts.txt"
    "https://osint.digitalside.it/Threat-Intel/lists/latestdomains.txt"
    "https://s3.amazonaws.com/lists.disconnect.me/simple_malvertising.txt"
    "https://v.firebog.net/hosts/Prigent-Crypto.txt"
    "https://raw.githubusercontent.com/FadeMind/hosts.extras/master/add.Risk/hosts"
    "https://bitbucket.org/ethanr/dns-blacklists/raw/8575c9f96e5b4a1308f2f12394abd86d0927a4a0/bad_lists/Mandiant_APT1_Report_Appendix_D.txt"
    "https://phishing.army/download/phishing_army_blocklist_extended.txt"
    "https://gitlab.com/quidsup/notrack-blocklists/raw/master/notrack-malware.txt"
    "https://v.firebog.net/hosts/RPiList-Malware.txt"
    "https://v.firebog.net/hosts/RPiList-Phishing.txt"
    "https://raw.githubusercontent.com/Spam404/lists/master/main-blacklist.txt"
    "https://raw.githubusercontent.com/AssoEchap/stalkerware-indicators/master/generated/hosts"
    "https://urlhaus.abuse.ch/downloads/hostfile/"
)
malicious_ok=(
    "https://malware-filter.gitlab.io/malware-filter/phishing-filter-hosts.txt"
    "https://v.firebog.net/hosts/Prigent-Malware.txt"
)
other_good=(
    "https://zerodot1.gitlab.io/CoinBlockerLists/hosts_browser"
)
other_ok=(
    "https://raw.githubusercontent.com/chadmayfield/my-pihole-blocklists/master/lists/pi_blocklist_porn_top1m.list"
    "https://v.firebog.net/hosts/Prigent-Adult.txt"
    "https://raw.githubusercontent.com/anudeepND/blacklist/master/facebook.txt"
)

# Create blacklists folders
mkdir -p $blacklists_path
mkdir -p $blacklists_path/suspicious/good
mkdir -p $blacklists_path/suspicious/ok
mkdir -p $blacklists_path/advertising/good
mkdir -p $blacklists_path/advertising/ok
mkdir -p $blacklists_path/tracking/good
mkdir -p $blacklists_path/tracking/ok
mkdir -p $blacklists_path/malicious/good
mkdir -p $blacklists_path/malicious/ok
mkdir -p $blacklists_path/other/good
mkdir -p $blacklists_path/other/ok

# Download blacklists into the correct folders
for url in "${suspicious_good[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/suspicious/good/${filename}" "$url"
done

for url in "${suspicious_ok[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/suspicious/ok/${filename}" "$url"
done

for url in "${advertising_good[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/advertising/good/${filename}" "$url"
done

for url in "${advertising_ok[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/advertising/ok/${filename}" "$url"
done

for url in "${tracking_good[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/tracking/good/${filename}" "$url"
done

for url in "${tracking_ok[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/tracking/ok/${filename}" "$url"
done

for url in "${malicious_good[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/malicious/good/${filename}" "$url"
done

for url in "${malicious_ok[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/malicious/ok/${filename}" "$url"
done

for url in "${other_good[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/other/good/${filename}" "$url"
done

for url in "${other_ok[@]}"; do
    filename=$(echo ${url} | cut -d '/' -f4- | tr '/' '_')
    wget -O "${blacklists_path}/other/ok/${filename}" "$url"
done

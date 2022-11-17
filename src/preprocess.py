from collections import defaultdict, Counter
from operator import itemgetter
import os
import sys
import logging
from scapy.all import rdpcap, Scapy_Exception
import time
import argparse
from colorama import Fore, Style
from pprint import pprint
import pandas as pd
import traceback

argparser = argparse.ArgumentParser()
argparser.add_argument('files', action='append')
argparser.add_argument('-v', '--verbose', action='store_const', const=True, required=False, dest='verbose')
argparser.add_argument('-o', '--output-folder', action='store', required=False, default='preprocessed', dest='output_folder')
argparser.add_argument('--file-log', action='store_const', const=True, required=False, default=False, dest='file_log')
args = argparser.parse_args()


def addLogLevel(level_name, level_num):
   """Create a new logging level for the logger.

   Args:
      level_name (str): Name of the new level.
      level_num (int): Logging priority of the new level. For reference, consult https://docs.python.org/3/library/logging.html#logging-levels.
   """
   logging.Logger.trace = lambda self, msg, *args, **kws: logger._log(level_num, msg, args, **kws)
   setattr(logging, level_name, level_num)
   logging.addLevelName(level_num, level_name)


logger = logging.getLogger(__name__)
addLogLevel('TRACE', 5)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("[%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel('DEBUG')
logger.addHandler(consoleHandler)
fileHandler = logging.FileHandler('log.txt')
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.TRACE)
if args.file_log:
   logger.addHandler(fileHandler)
if args.verbose:
   logger.setLevel(logging.DEBUG)


def indent(depth: int, length: int = 2) -> str:
   """Create an indentation string based on the desired depth.

   Args:
      depth (int): Indentation depth. 

   Returns:
      str: The indentation string.
   """
   s = ''
   if depth > 0:
      for d in range((depth-1) * length): s+='-'
      s+='> '
   return s


def getPCAPFiles(paths: list[str], **kwargs) -> list[str]:
   """Recursively retrieve all `.pcap` files in the given directories.

   Args:
      paths (list of str): List of relative paths to search for `.pcap` files.

   Returns:
      list of str: 1-D list of relative `.pcap` file paths. 
   """
   depth = kwargs.get('depth', 0)

   pcap_files = []
   for path in paths:
      if os.path.isfile(path) and path.endswith('.pcap'):
         logger.info(f'{Fore.GREEN}{indent(depth)}{path} is a PCAP file{Style.RESET_ALL}')
         pcap_files.append(path)
      elif os.path.isdir(path):
         logger.info(f'{indent(depth)}{path} is a folder ({len(os.listdir(path))} items)')
         pcaps = getPCAPFiles([os.path.join(path, p) for p in os.listdir(path)], depth=depth+1)
         pcap_files.extend(pcaps)
      else:
         logger.info(f'{Fore.RED}{indent(depth)}{path} is NOT a PCAP file{Style.RESET_ALL}')

   return pcap_files


def extract_info(p):
   """Extract useful query information from the given packet.
   The packet is actively checked for integrity, so if this function terminates with no exception,
   the packet can be considered relatively clean (e.g. if the packet is a query response (source port 53), it must contain the resolved IP).
   It also throws an exception for non-A type queries.
   Args:
      p: The packet.

   Returns:
      Dictionary containing all main information that could be retrieved from the packet, including:
      * qry_id: query ID
      * qry_domain: query domain
      * src_ip: source IP
      * dst_ip: destination IP
      * src_port: source port
      * dst_port: destination port
      * ip: resolved ip 
   """
   # Check basic packet structure
   assert 'DNS' in p
   assert 'IP' in p
   assert 'UDP' in p
   
   info = {}
   
   # Extract basic information
   if hasattr(p, 'time'): info['ts'] = int(p.time * 1000000) # microseconds (integer)
   if 'IP' in p:
      if hasattr(p['IP'], 'src'): info['src_ip'] = p['IP'].src
      if hasattr(p['IP'], 'dst'): info['dst_ip'] = p['IP'].dst
   if 'UDP' in p:
      if hasattr(p['UDP'], 'sport'): info['src_port'] = p['UDP'].sport
      if hasattr(p['UDP'], 'dport'): info['dst_port'] = p['UDP'].dport
   if 'DNS' in p:
      if hasattr(p['DNS'], 'id'): info['qry_id'] = p['DNS'].id
      # Counts are useful for malformed packet check. They can change in the response and that's not checked here
      if hasattr(p['DNS'], 'qdcount'): info['qdcount'] = p['DNS'].qdcount
      if hasattr(p['DNS'], 'ancount'): info['ancount'] = p['DNS'].ancount
      if hasattr(p['DNS'], 'nscount'): info['nscount'] = p['DNS'].nscount
      if hasattr(p['DNS'], 'arcount'): info['arcount'] = p['DNS'].arcount
      if 'DNSQR' in p: # Extract request info (@TODO i'm assuming a packet will never have more than 1 DNSQR layer because so far the ones with qdcount>1 are all malformed)
         if hasattr(p['DNSQR'], 'qname'): info['qry_domain'] = p['DNSQR'].qname#.decode('utf-8').rstrip('.')
         if hasattr(p['DNSQR'], 'qtype'): info['qtype'] = p['DNSQR'].qtype
      if 'DNSRR' in p: # Extract response info
         info['rtypes'], info['rdatas'] = [], []
         for l in p['DNSRR'].layers():
            if hasattr(p['DNSRR'][l], 'type'): info['rtypes'].append(p['DNSRR'][l].type)
            if hasattr(p['DNSRR'][l], 'rdata'): info['rdatas'].append(p['DNSRR'][l].rdata) # and p['DNSRR'][l].type == 1: # only save IP if type is 1 (1 is the code for type A)
            
   return info


# def sort_ip(ip):
#    """Reduce the IP address to a single integer usable for comparison, acting as a sorting key based on octets.
# 
#    Args:
#       ip (str): IP address.
# 
#    Returns:
#       int: Integer value usable for IP address comparing and sorting.
#    """
#    ip = ip.split('.')
#    total = 0
# 
#    for i, octet in enumerate(ip[::-1]):
#       total += int(octet) << i * 8
# 
#    return total


def dict_filter(dicts_list, cond, require_unique=False, require_keys=False):
   """Search dicts_list for the dictionaries that satisfy the given condition.
   
   Args:
      dicts_list (list of dicts): The list of dictionaries.
      cond (bool function): The condition to satisfy.
      require_unique (bool, default False): Require that the condition is satisfied by exactly one dict, otherwise throw an exception.
      require_keys (bool, default False): Require that, if the condition uses keys to directly access the dict, they must be present in all dicts, otherwise throw an exception.
      
   Returns:
      list of dicts: The dictionaries that satisfy the given condition.
      
   Examples:
      Search for the unique dict with the specified `id`:
      >>> dict_filter(dicts, lambda x: x['id'] == 42, require_unique=True)
      
      Search for all dicts where attribute `eta` is less than 5:
      >>> dict_filter(dicts, lambda x: x['eta'] < 5)
   """
   if not all([isinstance(d, dict) for d in dicts_list]):
      raise ValueError('dicts_list must be a list of dicts.')
   if not callable(cond):
      raise ValueError('cond must be a boolean function.')
   
   res = []
   for item in dicts_list:
      try:
         satisfied = cond(item)
      except KeyError as e:
         satisfied = False
         if require_keys:
            raise KeyError(f'Some condition key is not present in {item} but require_keys is True.')
      if satisfied:
         res.append(item)
         
   if require_unique:
      if len(res) == 0:
         raise AssertionError('No dict found for the given condition.')
      if len(res) > 1:
         raise AssertionError('More than one dict satisfy the given condition.')
   return res
   

# Convenience function, don't use. If you want to skip packet p, throw an exception
def check_packet(p): 
   # Check for basic fieldspcap
   if 'qry_id' not in p:
      raise KeyError('Could not extract the query ID.')
   if 'qry_domain' not in p:
      raise KeyError('Could not extract the query domain.')
   if 'src_ip' not in p:
      raise KeyError('Could not extract the source IP.')
   if 'dst_ip' not in p:
      raise KeyError('Could not extract the destination IP.')
   if 'src_port' not in p:
      raise KeyError('Could not extract the source port.')
   if 'dst_port' not in p:
      raise KeyError('Could not extract the destination port.')
   # Check for non-A type requests
   # if 'qtype' not in p or p['qtype'] != 1:
   #    raise KeyError('Query request is not of type A.')
   # Check for non-A type responses and unresolved IPs
   # if info['src_port'] == 53:
      # if 'rtypes' not in p or 1 not in p['rtypes']: # it didn't find any A-type RR
      #    raise KeyError('Could not find any A-type RR in response packet.')
      # if 'ip' not in p:
      #    raise KeyError('Response should be A-type but contains no IP.')
   # Check for malformed (nonsensical) packet
   if p['qdcount'] > 1000 or p['ancount'] > 1000 or p['nscount'] > 1000 or p['arcount'] > 1000:
      raise AssertionError('Packet is malformed.')
   
   # ===============
   # Put all assumptions about counts here: if they are not catched earlier, these assumptions are probably wrong
   if p['qdcount'] > 1:
      sys.exit(f'{Fore.RED}{Style.BRIGHT}Critical error: assumptions may not be correct.{Style.RESET_ALL}')
   # ===============
   
   
def main(argv):
   start_time = time.time()

   logger.info(f'\n{Style.BRIGHT}Retrieving PCAP files from {", ".join(args.files)}{Style.RESET_ALL}')
   pcap_files = getPCAPFiles(args.files)

   # all_queries = defaultdict(list)

   for fname in pcap_files:
      try:
         logger.info(f'\n{Style.BRIGHT}Analyzing {fname}...{Style.RESET_ALL}')
         logger.debug(f'Loading {fname}...')
         pcap = rdpcap(fname)
         logger.debug(f'Loaded {fname}.')
      except Scapy_Exception as e:
         logger.error(f'{Fore.RED}{Style.BRIGHT}Scapy_Exception: {fname} could not be read and will be skipped.{Style.RESET_ALL}')
         logger.debug(f'{Fore.RED}Cause of the exception: {e}{Style.RESET_ALL}')
         continue

      resolved, unresolved = [], [] # using a specific list to find unresolved queries should be faster
      stats = {'requests': 0, 'responses': 0, 'matched_responses': 0, 'bad_packets': 0}
      for i, pkt in enumerate(pcap):
         if pkt.getlayer('DNS') and pkt.getlayer('DNS').opcode == 0:
            try:
               logger.debug(f'{Fore.GREEN}Extracting Packet {i}...{Style.RESET_ALL}')
               info = extract_info(pkt)
               check_packet(info) # if this fails, it will throw an exception and the packet will be skipped
               logger.debug(f'''{Fore.GREEN}{Style.DIM}{indent(1)}Query #{f"{info['qry_id']}:":<8} {info["src_ip"]}:{info["src_port"]} -> {info["dst_ip"]}:{info["dst_port"]}{Style.RESET_ALL}''')
            except (AssertionError, KeyError, Exception) as e: # @TODO this should be the same as just Exception
               stats['bad_packets'] += 1
               logger.error(f'{Fore.RED}{Style.BRIGHT}Packet {i}: Could not extract all relevant information; skipped: {e}{Style.RESET_ALL}')
               logger.error(f'{Fore.RED}{Style.BRIGHT}{traceback.format_exc()}{Style.RESET_ALL}')
               logger.debug(f'{Fore.RED}Cause of the exception: {e}{Style.RESET_ALL}')
               continue
               
            if info['dst_port'] == 53: # it is a request, right? so it must be a new query
               stats['requests'] += 1
               logger.debug(f'{Fore.GREEN}{Style.DIM}{indent(1)}Packet {i} is a request. Appending it to the unresolved list...{Style.RESET_ALL}')
               try:
                  unresolved.append({f'req_{i}': info[i] for i in info}) # prefix each key in info with 'req_'
                  '''
                  unresolved.append({'qry_id': info['qry_id'], 
                                     'req_ts': info['ts'],
                                     'req_src_ip': info['src_ip'], 
                                     'req_src_port': info['src_port'], 
                                     'req_dst_ip': info['dst_ip'], 
                                     'req_dst_port': info['dst_port'], 
                                     'qtype': info['qtype'],
                                     'domain': info['qry_domain'],
                                     'req_qdcount': info['qdcount'],
                                     'req_ancount': info['ancount'],
                                     'req_nscount': info['nscount'],
                                     'req_arcount': info['arcount']})
                  '''
               except KeyError as e:
                  logger.error(f'{Fore.RED}{Style.BRIGHT}Packet {i}: Some fields could not be found in request query.{Style.RESET_ALL}')
                  logger.debug(f'{Fore.RED}Cause of the exception: {e}{Style.RESET_ALL}')
               
            elif info['src_port'] == 53: # this means it's a response, right? so there must be a preexisting unresolved query
               stats['responses'] += 1
               logger.debug(f'{Fore.GREEN}{Style.DIM}{indent(1)}Packet is a response. Matching it with its respective unresolved request...{Style.RESET_ALL}')
               
               try: # look for the preexisting unresolved query
                  cond = lambda q: q['qry_id'] == info['qry_id'] and \
                                   q['req_src_ip'] == info['dst_ip'] and q['req_src_port'] == info['dst_port'] and \
                                   q['req_dst_ip'] == info['src_ip'] and q['req_dst_port'] == info['src_port']
                  qry = dict_filter(unresolved, cond, require_unique=True)[0]
               except Exception as e:
                  logger.error(f'{Fore.RED}{Style.BRIGHT}Packet {i}: Query response doesn\'t match any pre-existing unresolved query.{Style.RESET_ALL}')
                  logger.error(f'{Fore.RED}Cause of the exception: {e}{Style.RESET_ALL}')
                  continue # @TODO actually handle exception. What if the query it refers to was actually present, and I didn't match it correctly?
                  
               try: # add the response field to the found query
                  qry = qry | {f'res_{i}': info[i] for i in info} # prefix each key in info with 'res_' and pour it into qry
                  '''
                  qry['res_ts'] = info['ts']
                  qry['res_src_ip'] = info['src_ip']
                  qry['res_src_port'] = info['src_port']
                  qry['res_dst_ip'] = info['dst_ip']
                  qry['res_dst_port'] = info['dst_port']
                  if 'ip' in info: qry['ip'] = info['ip']
                  if 'qdcount' in info: qry['res_qdcount'] = info['qdcount']
                  if 'ancount' in info: qry['res_ancount'] = info['ancount']
                  if 'nscount' in info: qry['res_nscount'] = info['nscount']
                  if 'arcount' in info: qry['res_arcount'] = info['arcount']
                  '''
               except KeyError as e:
                  logger.error(f'{Fore.RED}{Style.BRIGHT}Packet {i}: Some fields could not be found in response query.{Style.RESET_ALL}')
                  logger.debug(f'{Fore.RED}Cause of the exception: {e}{Style.RESET_ALL}')
                  continue
                  
               resolved.append(qry) # add the now-resolved query to the resolved list ...
               unresolved.remove(qry) # ... and remove it from the unresolved list
               stats['matched_responses'] += 1
               logger.debug(f'{Fore.GREEN}{Style.DIM}{indent(1)}Matched: the pre-existing query has been successfully resolved.{Style.RESET_ALL}')
               
            # all_queries[info['src_ip']].append(info['qry_domain'])

         elif pkt.getlayer('DNS') and pkt.getlayer('DNS').opcode != 0:
            # @TODO What does opcode != 0 mean? What do I have to do in this case?
            stats['bad_packets'] += 1
            logger.error(f'{Fore.RED}{Style.BRIGHT}Packet {i} is a DNS Query but has a non-zero opcode: {pkt.getlayer("DNS").opcode}{Style.RESET_ALL}')
         else:
            stats['bad_packets'] += 1
            logger.error(f'{Fore.RED}Packet {i} is NOT a DNS packet.{Style.RESET_ALL}')
      try: # @TODO check if it works, then remove try block
         logger.info(f"{Style.BRIGHT}Finished analyzing {fname}:{Style.RESET_ALL}")
         logger.info(f"{f'{indent(1)}Total packets:':<25} {i+1}\n{f'{indent(1)}Resolved packets:':<25} {len(resolved)} (x2)\n{f'{indent(1)}Unresolved requests:':<25} {len(unresolved)}\n{f'{indent(1)}Skipped packets:':<25} {i+1 - 2*len(resolved) - len(unresolved)}")
         try:
            logger.info(f"\nRequests: {stats['requests']}\nResponses: {stats['responses']}\nMatched responses: {stats['matched_responses']}\nBad packets: {stats['bad_packets']}")
         except Exception as e:
            sys.exit('Tutto ok, il logging è rotto, fixa il bug') # @TODO if it works, delete this try block
      except Exception as e:
         logger.info(f'It finished, but fix this bug\n{e}')
      # create DataFrame and save it to file
      logger.info('\nCreating DataFrame from resolved queries...')
      df = pd.DataFrame(resolved)
      logger.info('DataFrame created.')
      logger.info(f'\nSaving DataFrame to {args.output_folder}...')
      if not os.path.isdir(args.output_folder):
         os.makedirs(args.output_folder)
      output_file = os.path.join(args.output_folder, fname) + '.csv'
      df.to_csv(output_file)
      logger.info(f'DataFrame saved as {output_file}.')
      
   # logging query summary to file (not needed for now)
   # for src_ip in sorted(all_queries, key=sort_ip):
   #    sub_queries = Counter(all_queries[src_ip])
   #    for i, query in enumerate(sorted(sub_queries.items(), key=itemgetter(1, 0), reverse=True)):
   #       if i == 0:
   #           logger.trace(f'{src_ip:<15} {query[1]:>7}  {query[0]:<}')
   #       else:
   #           logger.trace(f'{"":15} {query[1]:>7}  {query[0]:<}')

   logger.info(f'\n{Style.DIM}Execution time: {time.time() - start_time:.2f} seconds.{Style.RESET_ALL}')
   
   
if __name__ == '__main__':
   main(sys.argv)
   

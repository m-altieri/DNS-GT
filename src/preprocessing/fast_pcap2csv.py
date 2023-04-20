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
import numpy as np
import matplotlib.pyplot as plt


def _logger_setup(log_to_file=False, verbose=False):
    """Create a new global logger for this file.

    `logger.critical()`, `logger.error()`, `logger.warning()` and `logger.info()` will always log the message.
    `logger.debug()` will log the message if `verbose=True`.
    `logger.trace()` will log the message *only to file* if `log_to_file=True`.

    Args:
       log_to_file (bool, default=False): Whether you want to also log everything to file `log.txt`. Note that *all* logs are logged to file, even trace-level ones.
       verbose (bool, default=False): If True, also log debug-level logs.
    """
    global logger
    logger = logging.getLogger(__name__)

    def _addLogLevel(level_name, level_num):
        """Create a new logging level for the logger.

        Args:
           level_name (str): Name of the new level.
           level_num (int): Logging priority of the new level. For reference, consult https://docs.python.org/3/library/logging.html#logging-levels.
        """
        logging.Logger.trace = lambda self, msg, *args, **kws: logger._log(
            level_num, msg, args, **kws
        )
        setattr(logging, level_name, level_num)
        logging.addLevelName(level_num, level_name)

    _addLogLevel("TRACE", 5)
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel("DEBUG")
    logger.addHandler(consoleHandler)
    if verbose:
        logger.setLevel(logging.DEBUG)


def indent(depth: int, length: int = 2) -> str:
    """Create an indentation string based on the desired depth.

    Args:
       depth (int): Indentation depth.

    Returns:
       str: The indentation string.
    """
    s = ""
    if depth > 0:
        for d in range((depth - 1) * length):
            s += "-"
        s += "> "
    return s


def getPCAPFiles(paths: list[str], **kwargs) -> list[str]:
    """Recursively retrieve all `.pcap` files in the given directories.

    Args:
       paths (list of str): List of relative paths to search for `.pcap` files.

    Returns:
       list of str: 1-D list of relative `.pcap` file paths.
    """
    depth = kwargs.get("depth", 0)

    pcap_files = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".pcap"):
            logger.info(
                f"{Fore.GREEN}{indent(depth)}{path} is a PCAP file{Style.RESET_ALL}"
            )
            pcap_files.append(path)
        elif os.path.isdir(path):
            logger.info(
                f"{indent(depth)}{path} is a folder ({len(os.listdir(path))} items)"
            )
            pcaps = getPCAPFiles(
                [os.path.join(path, p) for p in os.listdir(path)], depth=depth + 1
            )
            pcap_files.extend(pcaps)
        else:
            logger.info(
                f"{Fore.RED}{indent(depth)}{path} is NOT a PCAP file{Style.RESET_ALL}"
            )

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
    assert "DNS" in p
    assert "IP" in p
    assert "UDP" in p

    info = {}

    # Extract basic information
    if hasattr(p, "time"):
        info["ts"] = int(p.time * 1000000)  # microseconds (integer)
    if "IP" in p:
        if hasattr(p["IP"], "src"):
            info["src_ip"] = p["IP"].src
        if hasattr(p["IP"], "dst"):
            info["dst_ip"] = p["IP"].dst
    if "UDP" in p:
        if hasattr(p["UDP"], "sport"):
            info["src_port"] = p["UDP"].sport
        if hasattr(p["UDP"], "dport"):
            info["dst_port"] = p["UDP"].dport
    if "DNS" in p:
        if hasattr(p["DNS"], "id"):
            info["qry_id"] = p["DNS"].id
        # Counts are useful for malformed packet check. They can change in the response and that's not checked here
        if hasattr(p["DNS"], "qdcount"):
            info["qdcount"] = p["DNS"].qdcount
        if hasattr(p["DNS"], "ancount"):
            info["ancount"] = p["DNS"].ancount
        if hasattr(p["DNS"], "nscount"):
            info["nscount"] = p["DNS"].nscount
        if hasattr(p["DNS"], "arcount"):
            info["arcount"] = p["DNS"].arcount
        if (
            "DNSQR" in p
        ):  # Extract request info (@TODO i'm assuming a packet will never have more than 1 DNSQR layer because so far the ones with qdcount>1 are all malformed)
            if hasattr(p["DNSQR"], "qname"):
                info["qry_domain"] = p["DNSQR"].qname.decode("utf-8").rstrip(".")
            if hasattr(p["DNSQR"], "qtype"):
                info["qtype"] = p["DNSQR"].qtype
        if "DNSRR" in p:  # Extract response info
            info["rtypes"], info["rdatas"] = [], []
            for l in p["DNSRR"].layers():
                if hasattr(p["DNSRR"][l], "type"):
                    info["rtypes"].append(p["DNSRR"][l].type)
                if hasattr(p["DNSRR"][l], "rdata"):
                    info["rdatas"].append(
                        p["DNSRR"][l].rdata
                    )  # and p['DNSRR'][l].type == 1: # only save IP if type is 1 (1 is the code for type A)

    return info


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
        raise ValueError("dicts_list must be a list of dicts.")
    if not callable(cond):
        raise ValueError("cond must be a boolean function.")

    res = []
    for item in dicts_list:
        try:
            satisfied = cond(item)
        except KeyError as e:
            satisfied = False
            if require_keys:
                raise KeyError(
                    f"Some condition key is not present in {item} but require_keys is True."
                )
        if satisfied:
            res.append(item)

    if require_unique:
        if len(res) == 0:
            raise AssertionError("No dict found for the given condition.")
        if len(res) > 1:
            raise AssertionError("More than one dict satisfy the given condition.")
    return res


# Convenience function, don't use. If you want to skip packet p, throw an exception
def check_packet(p):
    # Check for basic fieldspcap
    if "qry_id" not in p:
        raise KeyError("Could not extract the query ID.")
    if "qry_domain" not in p:
        raise KeyError("Could not extract the query domain.")
    if "src_ip" not in p:
        raise KeyError("Could not extract the source IP.")
    if "dst_ip" not in p:
        raise KeyError("Could not extract the destination IP.")
    if "src_port" not in p:
        raise KeyError("Could not extract the source port.")
    if "dst_port" not in p:
        raise KeyError("Could not extract the destination port.")
    if (
        p["qdcount"] > 1000
        or p["ancount"] > 1000
        or p["nscount"] > 1000
        or p["arcount"] > 1000
    ):
        raise AssertionError("Packet is malformed.")

    # ===============
    # Put all assumptions about counts here: if they are not catched earlier, these assumptions are probably wrong
    if p["qdcount"] > 1:
        sys.exit(
            f"{Fore.RED}{Style.BRIGHT}Critical error: assumptions may not be correct.{Style.RESET_ALL}"
        )
    # ===============


def main(argv):

    argparser = argparse.ArgumentParser()
    argparser.add_argument("files", action="append")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=True,
        required=False,
        dest="verbose",
    )
    argparser.add_argument(
        "-o",
        "--output-folder",
        action="store",
        required=False,
        default="../../outputs/CSVs",
        dest="output_folder",
    )
    argparser.add_argument(
        "-r",
        "--response-only",
        action="store_const",
        const=True,
        required=False,
        default=False,
        dest="response_only",
    )
    argparser.add_argument(
        "--file-log",
        action="store_const",
        const=True,
        required=False,
        default=False,
        dest="file_log",
    )
    args = argparser.parse_args()

    _logger_setup(args.file_log, args.verbose)

    start_time = time.time()

    logger.info(
        f'\n{Style.BRIGHT}Retrieving PCAP files from {", ".join(args.files)}{Style.RESET_ALL}'
    )
    pcap_files = getPCAPFiles(args.files)

    for fname in pcap_files:
        try:
            logger.info(f"\n{Style.BRIGHT}Analyzing {fname}...{Style.RESET_ALL}")
            pcap = rdpcap(fname)

        except Scapy_Exception as e:
            logger.error(
                f"{Fore.RED}{Style.BRIGHT}Scapy_Exception: {fname} could not be read and will be skipped.{Style.RESET_ALL}"
            )
            logger.debug(f"{Fore.RED}Cause of the exception: {e}{Style.RESET_ALL}")
            continue

        unresolved = []

        # requests, responses, matched_responses = [], [], []

        for i, pkt in enumerate(pcap):
            if pkt.getlayer("DNS") and pkt.getlayer("DNS").opcode == 0:
                try:
                    logger.debug(
                        f"{Fore.GREEN}Extracting Packet {i}...{Style.RESET_ALL}"
                    )
                    info = extract_info(pkt)
                    # check_packet(info) # if this fails, it will throw an exception and the packet will be skipped
                    logger.debug(
                        f"""{Fore.GREEN}{Style.DIM}{indent(1)}Query #{f"{info['qry_id']}:":<8} {info["src_ip"]}:{info["src_port"]} -> {info["dst_ip"]}:{info["dst_port"]}{Style.RESET_ALL}"""
                    )
                except (
                    AssertionError,
                    KeyError,
                    Exception,
                ) as e:  # @TODO this should be the same as just Exception
                    logger.error(
                        f"{Fore.RED}{Style.BRIGHT}Packet {i}: Could not extract all relevant information; skipped: {e}{Style.RESET_ALL}"
                    )
                    continue

                if (
                    info["dst_port"] == 53
                ):  # it is a request, right? so it must be a new query
                    logger.debug(
                        f"{Fore.GREEN}{Style.DIM}{indent(1)}Packet {i} is a request. Appending it to the unresolved list...{Style.RESET_ALL}"
                    )
                    try:
                        unresolved.append(
                            {f"req_{i}": info[i] for i in info}
                        )  # prefix each key in info with 'req_'
                    except KeyError as e:
                        logger.error(
                            f"{Fore.RED}{Style.BRIGHT}Packet {i}: Some fields could not be found in request query.{Style.RESET_ALL}"
                        )
                        logger.debug(
                            f"{Fore.RED}Cause of the exception: {e}{Style.RESET_ALL}"
                        )

                elif (
                    info["src_port"] == 53
                ):  # this means it's a response, right? so there must be a preexisting unresolved query
                    logger.info(f"Skipping query response.")
                    continue
                    """
               qry = {}
               if not args.response_only: # look for the preexisting unresolved query
                  logger.debug(f'{Fore.GREEN}{Style.DIM}{indent(1)}Packet is a response. Matching it with its respective unresolved request...{Style.RESET_ALL}')
                  try: # try to match; if you can't, doesn't matter, use what you have
                     cond = lambda q: q['req_qry_id'] == info['qry_id'] and \
                                      q['req_src_ip'] == info['dst_ip'] and q['req_src_port'] == info['dst_port'] and \
                                      q['req_dst_ip'] == info['src_ip'] and q['req_dst_port'] == info['src_port']
                     qry = dict_filter(unresolved, cond, require_unique=True)[0]
                     
                     unresolved.remove(qry) # remove the now-resolved query from the unresolved list ...
                     
                     stats['matched_responses'] += 1
                     matched_responses.append(1)
                     logger.debug(f'{Fore.GREEN}{Style.DIM}{indent(1)}Matched: the pre-existing query has been successfully resolved.{Style.RESET_ALL}')
                     
                  except Exception as e:
                     logger.error(f'{Fore.RED}{Style.BRIGHT}Packet {i}: Query response doesn\'t match any pre-existing unresolved query.{Style.RESET_ALL}')
                     logger.debug(f'{Fore.RED}Cause of the exception: {e}{Style.RESET_ALL}')   
                    
               # whether you have matched or not, append the current packet
               qry = qry | {f'res_{i}': info[i] for i in info} # prefix each key in info with 'res_' and pour it into qry
               resolved.append(qry) # ... and add it to the "resolved" list @NOTE: now resolved includes all responses, matching is not required
               """
            elif (
                pkt.getlayer("DNS") and pkt.getlayer("DNS").opcode != 0
            ):  # @TODO What does opcode != 0 mean? What do I have to do in this case?
                logger.error(
                    f'{Fore.RED}{Style.BRIGHT}Packet {i} is a DNS Query but has a non-zero opcode: {pkt.getlayer("DNS").opcode}{Style.RESET_ALL}'
                )
            else:
                logger.error(
                    f"{Fore.RED}{Style.BRIGHT}Packet {i} is not a DNS query.{Style.RESET_ALL}"
                )

        # log results and stats
        logger.info(f"{Style.BRIGHT}Finished analyzing {fname}:{Style.RESET_ALL}")

        # create DataFrame and save it to file
        logger.info("\nCreating DataFrame from resolved queries...")
        df = pd.DataFrame(unresolved)
        logger.info("DataFrame created.")

        logger.info(f"\nSaving DataFrame to {args.output_folder}...")
        if not os.path.isdir(args.output_folder):
            os.makedirs(args.output_folder)
        output_file = os.path.join(args.output_folder, os.path.basename(fname)) + ".csv"
        df.to_csv(output_file)
        logger.info(f"DataFrame saved as {output_file}.")

    # log execution time
    logger.info(
        f"\n{Style.DIM}Execution time: {time.time() - start_time:.2f} seconds.{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main(sys.argv)

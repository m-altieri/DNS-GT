import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def main(host):

   df = pd.read_csv('../../outputs/response-only.csv')
   df = df[['res_ts', 'res_dst_ip', 'res_qry_domain']]
      
   df = df[df['res_dst_ip'] == host]
   print(len(df))
   bin_size=36000
   plt.figure(figsize=(30,10), dpi=120)
   plt.title(host)
   plt.hist(df['res_ts'], bins=bin_size)
   plt.xticks(np.linspace(df.iloc[0]['res_ts'], df.iloc[-1]['res_ts'], 60, dtype=int),
              list(range(60)))
   plt.savefig(f'../../outputs/queries-{host}.png')

   
if __name__ == '__main__':
   assert len(sys.argv) > 0
   host = sys.argv[1]
   main(host)
   

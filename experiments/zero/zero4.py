"""ZERO-4 guarded entry point. Preparation only until explicit budget authorization."""
import argparse, json
def main():
 p=argparse.ArgumentParser(); p.add_argument('--authorize',action='store_true'); a=p.parse_args()
 if not a.authorize: print(json.dumps({'status':'PREPARED_ONLY','candidate_budget':2568,'training_started':False},indent=2)); return
 raise RuntimeError('ZERO4 budget gate: authorization ID and persistent ledger integration required before training')
if __name__=='__main__': main()

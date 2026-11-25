"""ZERO-5 guarded entry point; training requires a new explicit budget authorization."""
import argparse,json
def main():
 p=argparse.ArgumentParser(); p.add_argument('--authorize',action='store_true'); a=p.parse_args(); print(json.dumps({'status':'PREPARED_ONLY' if not a.authorize else 'BUDGET_GATE_REQUIRES_AUTH_ID','candidate_budget':1860,'training_started':False},indent=2))
if __name__=='__main__': main()

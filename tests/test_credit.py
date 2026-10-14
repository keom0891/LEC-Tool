import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from risk_management import apply_strategy
from config_loader import load_config, ConfigError


def ddo(**changes):
    cfg = dict(name='DDO', type='ddo', ddo_threshold=100,
               initial_balance=30, contracted_amount=400, first_renewal_year=2028,
               renewal_interval_years=3, max_replenishments_per_period=1)
    cfg.update(changes)
    return cfg


def run(configs, times, losses, horizon=7):
    return apply_strategy([dict(times=np.array(times), losses=np.array(losses))],
                          configs, horizon, first_year=2026)


class CreditTests(unittest.TestCase):
    def test_replenishment_waits_for_next_event(self):
        r = run([ddo()], [0.1], [200])
        self.assertEqual(r['payout_dfs'][0].iloc[0, 0], 30)
        self.assertEqual(r['balance_dfs']['DDO'].iloc[0, 0], 400)

    def test_one_replenishment_per_period_and_reset(self):
        cfg = ddo(initial_balance=30, contracted_amount=100)
        r = run([cfg], [.1, .2, .3, 2.1, 2.2, 2.3], [200]*6)
        self.assertEqual(r['payout_dfs'][0].iloc[0, 0], 130)
        self.assertEqual(r['payout_dfs'][0].iloc[0, 2], 200)
        self.assertEqual(r['balance_dfs']['DDO'].iloc[0, 2], 0)
        self.assertEqual(r['balance_dfs']['DDO'].iloc[0, 5], 100)

    def test_partial_balance_overwritten_without_events(self):
        r = run([ddo()], [], [])
        np.testing.assert_array_equal(r['balance_dfs']['DDO'].iloc[0], [30,30,400,400,400,400,400])

    def test_actual_capped_payment_consumes_balance(self):
        a = ddo(name='a', initial_balance=100, max_replenishments_per_period=0)
        b = ddo(name='b', initial_balance=100, max_replenishments_per_period=0)
        r = run([a,b], [.1], [150])
        self.assertEqual(r['total_coverage'].iloc[0,0], 150)
        self.assertEqual(r['balance_dfs']['a'].iloc[0,0], 25)
        self.assertEqual(r['balance_dfs']['b'].iloc[0,0], 25)

    def test_strict_ddo_threshold_and_no_refill_per_event(self):
        r = run([ddo()], [.1,.2,.3], [100,200,200])
        self.assertEqual(r['payout_dfs'][0].iloc[0,0],230)
        self.assertEqual(r['balance_dfs']['DDO'].iloc[0,0],200)

    def test_loan_inclusive_validity_and_exhaustion(self):
        cfg = dict(name='BID loan', type='loan', loan_threshold=100,
                   initial_balance=450, start_year=2027,end_year=2029)
        r = run([cfg], [.1,1.1,2.1,3.1,4.1], [200]*5)
        np.testing.assert_array_equal(r['payout_dfs'][0].iloc[0], [0,200,200,50,0,0,0])

    def test_loan_unused_balance_survives_expiry(self):
        cfg = dict(name='loan',type='loan',loan_threshold=100,initial_balance=400,
                   start_year=2026,end_year=2026)
        r=run([cfg],[.1,1.1],[150,200])
        self.assertEqual(r['payout_dfs'][0].iloc[0,1],0)
        self.assertEqual(r['balance_dfs']['loan'].iloc[0,1],250)

    def test_ccf_max_threshold_ppo_order_and_renewal(self):
        ccf=dict(name='CCF',type='ccf',ccf_threshold=200,ccf_person=1650,
                 Pop_exposed=10.83e6,initial_balance=10,contracted_amount=20,
                 first_renewal_year=2028,renewal_interval_years=5,
                 max_replenishments_per_period=1)
        ppo=dict(name='PPO',type='ppo',ppo_schedule=[5]*10)
        r=run([ppo,ccf],[.1,.2,.3,2.1],[190,200,1000,1000],horizon=10)
        self.assertEqual(r['payout_dfs'][1].iloc[0,0],10)
        self.assertEqual(r['payout_dfs'][0].iloc[0,0],5)
        self.assertEqual(r['payout_dfs'][0].iloc[0,2],0)
        self.assertEqual(r['payout_dfs'][1].iloc[0,2],20)
        self.assertEqual(r['balance_dfs']['CCF'].iloc[0,7],20)
        ccf['ccf_threshold']=120
        r=run([ccf],[.1],[170])
        self.assertEqual(r['payout_dfs'][0].iloc[0,0],0)

    def test_independent_simulations(self):
        event=dict(times=np.array([.1]),losses=np.array([200]))
        r=apply_strategy([event,event],[ddo()],1,first_year=2026)
        np.testing.assert_array_equal(r['balance_dfs']['DDO'].values,[[400],[400]])

    def test_historical_renewal_date_anchors_schedule(self):
        r = run([ddo(first_renewal_year=2025)], [], [])
        np.testing.assert_array_equal(r['balance_dfs']['DDO'].iloc[0],
                                      [30,30,400,400,400,400,400])

    def test_zero_initial_balance_can_replenish(self):
        r = run([ddo(initial_balance=0)], [.1,1.1], [200,200])
        self.assertEqual(r['payout_dfs'][0].iloc[0,0],0)
        self.assertEqual(r['balance_dfs']['DDO'].iloc[0,0],400)
        self.assertEqual(r['payout_dfs'][0].iloc[0,1],200)

    def test_disabled_replenishment_waits_for_renewal(self):
        r = run([ddo(max_replenishments_per_period=0)], [.1,.2,2.1], [200]*3)
        self.assertEqual(r['payout_dfs'][0].iloc[0,0],30)
        self.assertEqual(r['balance_dfs']['DDO'].iloc[0,0],0)
        self.assertEqual(r['balance_dfs']['DDO'].iloc[0,2],200)

    def test_large_disaster_has_no_separate_ddo_cap(self):
        cfg = ddo(initial_balance=393, contracted_amount=80)
        r = run([cfg], [.1,.2,2.1], [450,200,200])
        self.assertEqual(r['payout_dfs'][0].iloc[0,0],473)
        self.assertEqual(r['balance_dfs']['DDO'].iloc[0,0],0)
        self.assertEqual(r['payout_dfs'][0].iloc[0,2],80)

    def test_bm_ddo_top_up_funds_only_next_disaster(self):
        cfg=ddo(initial_balance=100,contracted_amount=100,first_renewal_year=2029)
        first=run([cfg],[.1],[300])
        self.assertEqual(first['payout_dfs'][0].iloc[0,0],100)
        self.assertEqual(first['balance_dfs']['DDO'].iloc[0,0],100)
        next_event=run([cfg],[.1,1.1,2.1],[300]*3)
        np.testing.assert_array_equal(next_event['payout_dfs'][0].iloc[0,:3],[100,100,0])
        same_year=run([cfg],[.1,.2,.3],[300]*3)
        self.assertEqual(same_year['payout_dfs'][0].iloc[0,0],200)
        self.assertEqual(same_year['balance_dfs']['DDO'].iloc[0,0],0)

    def test_scaled_payout_does_not_prematurely_replenish(self):
        a=ddo(name='a',initial_balance=100,contracted_amount=100)
        b=ddo(name='b',initial_balance=100,contracted_amount=100)
        r=run([a,b],[.1],[150])
        self.assertEqual(r['balance_dfs']['a'].iloc[0,0],25)
        self.assertEqual(r['balance_dfs']['b'].iloc[0,0],25)

    def test_ccf_replenishment_waits_for_next_disaster(self):
        cfg=dict(name='CCF',type='ccf',ccf_threshold=100,ccf_person=1500,
                 Pop_exposed=6.39e6,initial_balance=10,contracted_amount=20,
                 first_renewal_year=2031,renewal_interval_years=5,
                 max_replenishments_per_period=1)
        first=run([cfg],[.1],[1000])
        self.assertEqual(first['payout_dfs'][0].iloc[0,0],10)
        self.assertEqual(first['balance_dfs']['CCF'].iloc[0,0],20)
        follow=run([cfg],[.1,1.1,2.1],[1000]*3)
        np.testing.assert_array_equal(follow['payout_dfs'][0].iloc[0,:3],[10,20,0])
        same_year=run([cfg],[.1,.2,.3],[1000]*3)
        self.assertEqual(same_year['payout_dfs'][0].iloc[0,0],30)
        self.assertEqual(same_year['balance_dfs']['CCF'].iloc[0,0],0)

    def test_standalone_ddo_depletes_initial_balance(self):
        from risk_management import ddo_coverage
        self.assertEqual(ddo_coverage([100,130,200,400],100,393),
                         [0,130,200,63])

    def test_empty_strategy(self):
        r=run([],[],[])
        self.assertEqual(r['total_coverage'].values.sum(),0)

    def test_config_validation(self):
        import tempfile
        import re
        source=(ROOT/'config.toml').read_text(encoding='utf-8')
        # Use absolute input paths to preserve input lookup from the temporary directory.
        source=source.replace('"Country databases/', '"'+ROOT.as_posix()+'/Country databases/')
        source=source.replace('"data/', '"'+ROOT.as_posix()+'/data/')
        cases=[('renewal_interval_years = 5','renewal_interval_years = 0'),
               ('max_replenishments_per_period = 1','max_replenishments_per_period = 2'),
               ('initial_balance = 300','initial_balance = -1'),
               ('end_year = 2029','end_year = 2025'),
               ('first_renewal_year = 2028','first_renewal_year = 0'),
               ('ccf_threshold = 120','ccf_threshold = nan')]
        for old,new in cases:
            with self.subTest(new=new), tempfile.TemporaryDirectory(dir=ROOT / 'tests') as folder:
                path=Path(folder)/'config.toml'
                key, value = new.split('=', 1)
                modified, count = re.subn(r'^' + key.strip() + r'\s*=\s*[^#\n]+',
                                          key.strip() + ' = ' + value.strip() + ' ',
                                          source, count=1, flags=re.MULTILINE)
                self.assertEqual(count, 1)
                path.write_text(modified,encoding='utf-8')
                with self.assertRaises(ConfigError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Bind the frozen independent analyzer to exact C11 roots and inputs."""
import argparse
from campaign_io import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('qualification', 'confirmation'), required=True)
    args = parser.parse_args()
    c = campaign()
    binding = read(c / 'binary.json')
    authority = R / 'qualification-authorization.json'
    verify_files({str(authority): (R / 'qualification-authorization.sha256').read_text().strip()})
    value = read(authority)
    verify_files(value['frozen_files'])
    parameters = read(R / 'parameter-seal.json')
    frozen = dict(value['frozen_files'])
    frozen[str(authority)] = digest(authority)
    frozen[str(R9 / 'readout_analysis.py')] = '98a912a64b4a5774344caf840ec7808964a61a280760adec8021467cdfbac275'
    for name, (_, sha) in PARENTS.items():
        frozen[str(PARENTS[name][0] / 'completed-campaign.manifest.json')] = sha
    exclusions = read(C8 / 'exclusions.json')
    frozen.update(exclusions['sha256'])
    frozen[str(C8 / 'audit-features' / 'known-features-input-audit.jsonl')] = '09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d'
    frozen[str(C9 / 'fresh-audit' / 'known-features-input-audit.jsonl')] = '361092818769307bcd82f1f30e10808e73857be7a96844f1293f2021d0410d87'
    config = dict(campaign=str(c), source=binding['source'], dependency=DEPENDENCY, binaries=binding['binaries'],
        frozen_files=frozen, imports=parameters['imports'],
        qualification_authorization=dict(path=str(authority), sha256=digest(authority)),
        qualification=dict(heads={f'{core}/{arm}': str(c / f'qual-{core}-{arm}') for core in CORES for arm in ARMS},
                           core_smoke=str(c / 'qual-core-smoke')),
        confirmation=dict(audits={str(panel):str(c / f'audit-{panel}') for panel in range(3)},
            features={f'{panel}/{core}':str(c / f'features-{panel}-{core}') for panel in range(3) for core in CORES},
            heads={f'{panel}/{core}/{arm}':str(c / f'head-{panel}-{core}-{arm}')
                   for panel in range(3) for core in CORES for arm in ARMS}, caches={}))
    if args.stage == 'confirmation':
        authority = R / 'confirmation-authorization.json'
        verify_files({str(authority): (R / 'confirmation-authorization.sha256').read_text().strip()})
        config['confirmation_authorization'] = dict(path=str(authority), sha256=digest(authority))
        for name in ('confirmation-authorization.json', 'panel-seal.json', 'cache-seal.json'):
            path = R / name
            frozen[str(path)] = digest(path)
        cache_seal = read(R / 'cache-seal.json')
        require(cache_seal['accepted'] is True, 'caches not accepted')
        config['confirmation']['caches'] = cache_seal['caches']
    verify_files(frozen)
    path = R / f'{args.stage}-analysis-config.json'
    save(path, config)
    print(json.dumps(dict(path=str(path), sha256=digest(path))))

if __name__ == '__main__':
    main()

"""User-requested swap of Chiba station roles 2/3; preserve actual timestamps."""
import json
from pathlib import Path
from datetime import datetime
import sys

HERE=Path(__file__).resolve().parent


def main():
    sys.path.insert(0,str(HERE))
    prior=Path(json.loads((HERE/'results/latest_revision.json').read_text())['directory'])
    report=json.loads((prior/'first_stations.json').read_text(encoding='utf-8'))
    event=next(e for e in report['events'] if e['date']=='20230511')
    event['selected'][1],event['selected'][2]=event['selected'][2],event['selected'][1]
    for i,row in enumerate(event['selected'],1):row['rank']=i
    report['events']=[event]
    report['revision']=dict(source=str(prior),action='User-requested swap of Chiba roles 2/3; actual trigger timestamps unchanged; assigned order is not chronological')
    out=HERE/'results'/('corrected_chiba_swap23_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    selection=out/'first_stations.json';selection.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    scripts=out/'scripts';scripts.mkdir()
    for name in ('two_station_replay.py','three_station_replay.py'):
        text=(HERE/name).read_text(encoding='utf-8')
        text=text.replace('Path(__file__).resolve().parent','Path('+repr(str(HERE))+')')
        text=text.replace('results/20261022_230231_957410/first_stations.json',str(selection).replace('\\','/'))
        text=text.replace("HERE/'results'/","Path("+repr(str(out))+")/")
        path=scripts/name;path.write_text(text,encoding='utf-8')
        ns={'__name__':'revision','__file__':str(path)};exec(compile(text,str(path),'exec'),ns);ns['main']()
    name='plot_location_tracks.py';text=(HERE/name).read_text(encoding='utf-8')
    text=text.replace('Path(__file__).resolve().parent','Path('+repr(str(HERE))+')')
    text=text.replace("Path(json.loads((HERE/'results/latest_revision.json').read_text())['directory'])",'Path('+repr(str(out))+')')
    path=scripts/name;path.write_text(text,encoding='utf-8')
    ns={'__name__':'revision','__file__':str(path)};exec(compile(text,str(path),'exec'),ns);ns['main']()
    print('Correction:',out,flush=True)


if __name__=='__main__':main()

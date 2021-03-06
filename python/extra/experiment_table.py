from bottle import route, run, template

from pymongo import MongoClient
from bson.objectid import ObjectId

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--db_host', type=str)
args = parser.parse_args()
if args.db_host is not None:
  client = MongoClient(host=args.db_host)
else:
  client = MongoClient()
db = client.timb_experiments_db

@route('/')
def index():
  head = '''<tr>
    <td>id</td>
    <td>when</td>
    <td>name</td>
    <td>time</td>
    <td>rigid?</td>
    <td>disp_cost?</td>
    <td>strain</td>
    <td>norm</td>
  </tr>'''
  body = ''

  for e in db.experiments.find().sort([('datetime', -1)]):
    row = '<tr>'
    row += '<td>' + str(e['_id']) + '</td>'
    row += '<td>' + ('' if 'datetime' not in e else str(e['datetime'])) + '</td>'
    row += '<td>' + str(e['name']) + '</td>'
    row += '<td>' + ('' if 'time_elapsed' not in e else str(e['time_elapsed'])) + '</td>'
    row += '<td>' + ('y' if ('rigid' in e and e['rigid']) else '') + '</td>'
    row += '<td>' + (('y' if 'disp_cost' in e['tracker_params'] and e['tracker_params']['disp_cost'] else '') if ('rigid' in e and e['rigid']) else '') + '</td>'
    row += '<td>' + ('' if 'flow_rigidity_coeff' not in e['tracker_params'] else str(e['tracker_params']['flow_rigidity_coeff'])) + '</td>'
    row += '<td>' + ('' if 'flow_norm_coeff' not in e['tracker_params'] else str(e['tracker_params']['flow_norm_coeff'])) + '</td>'
    row += '</tr>'
    body += row

  return '<table><thead>%s</thead><tbody>%s</tbody></table>' % (head, body)

run(host='localhost', port=8080)

import requests, json, time, os
from datetime import datetime
import pandas as pd
from osm_teams import OSMTeams

TOKEN = os.getenv('OSM_TEAMS_ACCESS_TOKEN')
DEBUG = os.getenv('DEBUG') == 'True'

print(f"DEBUG Status: {DEBUG}")

ym = OSMTeams(token_or_session=TOKEN, organization_id=1, debug=DEBUG)

date_suffix = datetime.now().strftime('%m_%d_%Y')

members_json = f"/tmp/ym_members_{date_suffix}.json"
chapters_json = f"/tmp/ym_chapters_{date_suffix}.json"
youthmappers_json = f"/tmp/youthmappers_{date_suffix}.json"
osm_user_info_json = f"/tmp/osm_user_stats_{date_suffix}.json"

if not os.path.isfile(members_json):
	members = ym.get_all_organization_members(org_attributes=True, org_badges=True)
	members.to_json(members_json)
	
if not os.path.isfile(chapters_json):
	chapters = ym.get_all_organization_teams(members=True, join_link=True, attributes=True, max_count=None)
	chapters.to_json(chapters_json)

members = pd.read_json(members_json)
print(f"Found {len(members)} users in the YouthMappers Organization.")

chapters = pd.read_json(chapters_json)
print(f"Fetched {len(chapters)} teams")


youthmappers = chapters.reset_index().explode('member_uids').rename(
		columns={'index':'team_id', 'member_uids':'uid', 'name':'chapter'}
	)
youthmappers['moderator'] = youthmappers.apply(lambda row: row.uid in row.moderator_uids, axis=1)

youthmappers = youthmappers[(youthmappers.team_id == 2) | ((youthmappers.uid != 1770239)  & (youthmappers.team_id != 2))]

youthmappers = members.merge( youthmappers[
                    ['uid','moderator','chapter','University','City','Country','location', 'team_id']
        ], left_index=True, right_on='uid', how='outer').reset_index(drop=True)

print(f"Completed YM Profiles in OSM Teams: {youthmappers[pd.notnull(youthmappers['Gender'])].uid.nunique()}")
youthmappers.to_json(youthmappers_json)


# # Hit the OSM API for more User Info
if not os.path.isfile(osm_user_info_json):
	osm_user_information = ym.get_mapper_info_from_osm(set(youthmappers.uid))
	osm_user_information.to_json(osm_user_info_json)

osm_user_information = pd.read_json(osm_user_info_json)


# Merge the DFs
y = youthmappers.merge(osm_user_information, left_on='uid', right_index=True, how='outer')
y['changeset_count'] = y.changesets.apply(lambda c: c.get('count'))
y['account_created'] = y['account_created'].apply(pd.Timestamp)
y['alumni'] = y['Alumni'].apply(lambda c: pd.Timestamp(c.get('assigned_at')).date() if type(c) == dict else None)
y['regional_ambassador'] = y['Regional Ambassador'].apply(lambda c: pd.Timestamp(c.get('assigned_at')).date() if type(c) == dict else None)
y['ymsc'] = y['Steering Committee'].apply(lambda c: pd.Timestamp(c.get('assigned_at')).date() if type(c) == dict else None)
y.rename(columns={'Major or Degree Concentration':'major','Gender':'gender','Year Born':'born',
                  'Graduation Date':'graduation', 'Hometown and Country':'hometown', 'Email':'email',
                  'Name':'name','University':'university','City':'city','Country':'country',
                  'location':'chapter_location'}, inplace=True)
y = y[[
    'uid','username','name','gender','email','born','graduation','hometown','changeset_count',
    'account_created','description','chapter','university','city','country','chapter_location',
    'alumni','regional_ambassador','ymsc'
]]

y.to_csv(f"/tmp/youthmappers_{date_suffix}.csv", sep=",", header=True)
y.to_csv("/tmp/youthmappers.tsv", sep="\t", header=False)
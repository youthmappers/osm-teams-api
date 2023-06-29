import requests, json, time, os
import pandas as pd
from osm_teams import OSMTeams

TOKEN = os.getenv('OSM_TEAMS_ACCESS_TOKEN')
DEBUG = os.getenv('DEBUG').lower() == 'true'

print("Beginning YouthMappers Download from OSM Teams")
print(f"  DEBUG Status: {DEBUG}")


ym = OSMTeams(token_or_session=TOKEN, organization_id=1, debug=DEBUG)

if ym.debug or not os.path.isfile("members.json"):
	members = ym.get_all_organization_members(
		org_attributes=True, 
		org_badges=True
	)
	members.to_json("members.json")
	
if ym.debug or not os.path.isfile("chapters.json"):
	chapters = ym.get_all_organization_teams(
		members=True, 
		join_link=True, 
		attributes=True, 
		max_count=None
	)
	chapters.to_json("chapters.json")

members = pd.read_json("members.json")
print(f"Found {len(members)} users in the YouthMappers Organization.")

chapters = pd.read_json("chapters.json")
print(f"Fetched {len(chapters)} teams")

chapters.to_csv("chapters.csv", sep=",", header=True)

youthmappers = chapters.reset_index().explode('member_uids').rename(
		columns={'index':'team_id', 'member_uids':'uid', 'name':'chapter'}
	)
youthmappers['moderator'] = youthmappers.apply(lambda row: row.uid in row.moderator_uids, axis=1)

youthmappers = youthmappers[(youthmappers.team_id == 2) | (
	(youthmappers.team_id != 2) & (
		 	(
				(youthmappers.uid != 1770239) &
				(youthmappers.uid != 6017386) &
				(youthmappers.uid != 2405982) &
				(youthmappers.uid != 2330248)
			)
		)
	)]

youthmappers = members.merge( youthmappers[
                    ['uid','moderator','chapter','University','City','Country','location', 'team_id']
        ], left_index=True, right_on='uid', how='outer').reset_index(drop=True)

print(f"Completed YM Profiles in OSM Teams: {youthmappers[pd.notnull(youthmappers['Gender'])].uid.nunique()}")
youthmappers.to_json("youthmappers.json")


# # Hit the OSM API for more User Info
if ym.debug or not os.path.isfile("osm_user_info.json"):
	osm_user_information = ym.get_mapper_info_from_osm(set(youthmappers.uid))
	osm_user_information.to_json("osm_user_info.json")

osm_user_information = pd.read_json("osm_user_info.json")

try:
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

	y.to_csv("youthmappers.csv", sep=",", header=True)
except:
	print("Failed to Merge the DFs")

print("Done.")
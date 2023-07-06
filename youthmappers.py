import json, os, argparse, base64
import pandas as pd

# Required for Google Drive & Google Sheets
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account

import gspread

from osm_teams import OSMTeams

def main():
	parser = argparse.ArgumentParser(
		description="YouthMappers utility to parse mapper info from OSM Teams"
	)

	parser.add_argument("-g", 
		     "--drive", 
			 action="store_true", 
			 help="Download cached chapter and user list from Google Drive chapter and user lists from OSM Teams"
		)

	parser.add_argument("-t", 
		     "--teams", 
			 action="store_true", 
			 help="Download latest chapter and user lists from OSM Teams"
		)

	parser.add_argument("-u", "--update", 
			action="store_true",
			help="What results to upload to Google"
	)

	parser.add_argument("-o", "--osm", 
			action="store_true",
			help="Hit OSM API"
	)
	
	parser.add_argument("-c",
		"--conflate", 
		action='store_true',
		help="Conflate with previous master list of usernames"
	)

	args = parser.parse_args()

	print(args)

	ym = YouthMappersHandler()

	# Downlod the latest JSON files
	if args.teams:
		ym.download_latest_from_osm_teams()	
	elif args.drive:
		ym.download_latest_from_google_drive()
		
	# Load Chapters and Members dataframes
	ym.read_local_files()

	# Explode the member_uids and build a 1-1 mapper-chapter DF
	ym.build_mappers_df_from_members()

	# Conflate with previous list of known mappers
	if args.conflate:
		ym.fetch_previous_master_list_and_conflate()

	# Get mapper info from OSM?
	if args.osm:
		# ym.download_latest_from_osm()
		ym.merge_df_with_osm()

	# If DF is supposed to 
	
	ym.to_csv()

	if args.update:
		ym.gc = gspread.service_account_from_dict(ym.creds)
		ym.update_latest_youthmapper_roster()
		ym.update_latest_chapter_roster()

	# 
	# 	print("Fetching Previous List from Google Sheets")
	# 	handler.fetch_previous_master_list_and_conflate()
	
	

class YouthMappersHandler():
	global YM_COLUMNS, PII_COLUMNS, BADGE_COLUMNS, OSM_COLUMNS, CHAPTER_COLUMNS
	
	YM_COLUMNS = [
		"username","Name","Gender","Major or Degree Concentration","Graduation Date",
		"Hometown and Country","Role / Position","team_id","all_teams","source","alumni_date"
	]
	PII_COLUMNS = ["Year Born","Email"]
	BADGE_COLUMNS = ["Steering Committee","Regional Ambassador","Alumni","Mentor"]
	OSM_COLUMNS = ["display_name","account_created","description","changeset_count"]
	CHAPTER_COLUMNS = ["Chapter","University","City","Country","chapter_lon","chapter_lat"]


	def __init__(self):
		self.creds = json.loads(base64.b64decode(os.environ.get('YGL_GOOGLE_CREDENTIALS')))
		self.teams = OSMTeams(token_or_session=os.getenv('OSM_TEAMS_ACCESS_TOKEN'), organization_id=1, debug=False)	


	def to_csv(self, filename = 'tmp.csv'):
		self.df.to_csv(filename)


	def download_latest_from_osm_teams(self):
		"""
		Connect to OSM Teams and download all of the organization members and chapters
		"""
		members = self.teams.get_all_organization_members(
			org_attributes=True, 
			org_badges=True
		)
		members.to_json("members.json")

		chapters = self.teams.get_all_organization_teams(
			members=True, 
			join_link=True, 
			attributes=True, 
			max_count=None
		)
		chapters.to_json("chapters.json")

	
	def download_latest_from_google_drive(self):
		"""
		Connect to Google Drive and downloads the latest members.json and chapters.json files
		"""
		g_credentials = service_account.Credentials.from_service_account_info(self.creds)

		drive_service = build(
			'drive', 
			'v3', 
			credentials=g_credentials.with_scopes(
				scopes=['https://www.googleapis.com/auth/drive']
			)
		)
		g_files = drive_service.files()
		print("Successfully obtained credentials from Google")

		members = pd.DataFrame(
			json.loads(
				g_files.get_media(fileId='1tY2KHbVUwqHYxZVVb4dZRsMe8oFSLdMJ').execute()
			)
		)
		members.index = members.index.astype(int)
		members.to_json("members.json")
		
		chapters = pd.DataFrame(
			json.loads(
				g_files.get_media(fileId='1ID2FWaqRdTMF32obGS3Utx5D2F4MPeQ6').execute()
			)
		)
		chapters.index = chapters.index.astype(int)
		chapters.to_json("chapters.json")
	
	
	def read_local_files(self):
		"""
		Load chapters.json and members.json into local dataframes
		"""
		print("\nLoading chapters.json and members.json")
		try:
			self.chapters = pd.read_json("chapters.json")
			self.chapters['chapter_lon'] = self.chapters.location.apply(lambda l: json.loads(l).get('coordinates')[0] if pd.notnull(l) else None)
			self.chapters['chapter_lat'] = self.chapters.location.apply(lambda l: json.loads(l).get('coordinates')[1] if pd.notnull(l) else None)
			print(f" - chapters.json has {len(self.chapters):,} teams")
		except:
			print("\nERROR: No chapters.json file\n")
			raise

		try:
			self.members  = pd.read_json("members.json")
			print(f" - members.json has {len(self.members):,} OSM user IDs from the YouthMappers organization.")
		except:
			print("\nERROR: No members.json file\n")
			raise


	def build_mappers_df_from_members(self):
		print("\nBuilding Mappers DataFrame from members", end="")
		uid_to_chapter_lookup = self.chapters.explode('member_uids').reset_index().rename(
    		columns={'index':'chapter_id_from_teams'}
		).groupby('member_uids').aggregate({'chapter_id_from_teams':[min,list]}).to_dict()

		uid_to_single_chapter = uid_to_chapter_lookup.get(('chapter_id_from_teams','min'))
		uid_to_chapter_list   = uid_to_chapter_lookup.get(('chapter_id_from_teams','list'))

		mappers = self.members

		mappers['team_id']   = mappers.apply(lambda d: uid_to_single_chapter.get(d.name), axis=1)
		mappers['all_teams'] = mappers.apply(lambda d: uid_to_chapter_list.get(d.name), axis=1)
		mappers['source'] = 'OSM Teams'
		mappers['alumni_date'] = mappers.Alumni.apply(
    		lambda j: pd.Timestamp(j.get('assigned_at')).date() if j is not None else None
    	)

		print(f"...read {len(mappers):,} mappers from OSM Teams, {len(mappers[pd.notnull(mappers.Gender)]):,} have profile info")

		self.df = mappers[YM_COLUMNS+PII_COLUMNS+BADGE_COLUMNS]


	def fetch_previous_master_list_and_conflate(self):
		print("Fetching previous user list from Google Drive", end="", flush=True)
		gc = gspread.service_account_from_dict(self.creds)
		previous_master_list_sheet = gc.open_by_key('17EOKwXR8kolG_Lkz-xYH8ysLcwhDl11Zq7scHyOzakI').worksheet('Sheet1')
		previous_master_list = pd.DataFrame(previous_master_list_sheet.get_all_records()).set_index('UID')
		previous_master_list['source'] = 'Old List'
		print(f"...list contains {len(previous_master_list):,} rows")

		df = self.df.join(previous_master_list, rsuffix='_masterlist', how='outer')
		print(f"After joining to previous list, there are {len(df):,} unique YouthMappers")

		df.Gender   = df.apply(lambda row: self.__conflate(row, 'Gender','gender').lower() if self.__conflate(row, 'Gender','gender') is not None else None, axis=1)
		df.username = df.apply(lambda row: self.__conflate(row, 'username','username_masterlist'), axis=1)
		df.Name     = df.apply(lambda row: self.__conflate(row, 'Name','Name_masterlist'), axis=1)
		df.Email    = df.apply(lambda row: self.__conflate(row, 'Email','email'), axis=1)
		df.team_id  = df.apply(lambda row: self.__conflate(row, 'team_id','chapter_id'), axis=1).astype(int)
		df.source   = df.apply(lambda row: self.__conflate(row, 'source','source_masterlist'), axis=1)
		df.alumni_date   = df.apply(lambda row: self.__conflate(row, 'alumni_date','alumni_date_conflated'), axis=1)
		df['Role / Position']    = df.apply(lambda row: self.__conflate(row, 'Role / Position','role'), axis=1)
		print("...Successfully ran conflation")

		self.df = df[YM_COLUMNS+PII_COLUMNS+BADGE_COLUMNS]

	
	def download_latest_from_osm(self):
		print(f"Hitting OSM User API for {len(self.df)} mappers...")
		osm_user_information = self.teams.get_mapper_info_from_osm(set(self.df.index))
		osm_user_information.to_json("osm_user_info.json")
		

	def merge_df_with_osm(self):
		try:
			self.osm = pd.read_json("osm_user_info.json")
		except:
			print("\nERROR: No osm_user_info.json file")

		self.osm['changeset_count'] = self.osm.changesets.apply(lambda c: c.get('count') if pd.notnull(c) else 0)
		self.osm['account_created'] = self.osm['account_created'].apply(pd.Timestamp)

		self.df = self.df.join(self.osm, how='outer')
		self.df = self.df[YM_COLUMNS+PII_COLUMNS+BADGE_COLUMNS+OSM_COLUMNS]


	def update_latest_youthmapper_roster(self):
		OUTPUT_COLUMNS = [
			"uid",
			"username",
			"Name",
			"Gender",
			"Major or Degree Concentration",
			"Graduation Date",
			"Hometown and Country",
			"Role / Position",
			"changeset_count",
			"account_created",
			"team_id",
			"all_teams",
			"source",
			"alumni_date",
			"Alumni",
			"Regional Ambassador",
			"Mentor",
			"Steering Committee",
			"Chapter",
			"University",
			"City",
			"Country",
			"chapter_lon",
			"chapter_lat"
		]

		t = self.df.merge(self.chapters[
    		['name','University','City','Country','chapter_lon','chapter_lat']
                 ].rename(columns={'name':'Chapter'}), left_on='team_id', right_index=True
        ).reset_index().rename(columns={'index':'uid'})[OUTPUT_COLUMNS]

		t.Alumni = t.Alumni.apply(lambda x: pd.Timestamp(x.get('assigned_at')).date() if pd.notnull(x) else '')
		t['Regional Ambassador'] = t['Regional Ambassador'].apply(lambda x: pd.Timestamp(x.get('assigned_at')).date() if pd.notnull(x) else '')
		t['Steering Committee'] = t['Steering Committee'].apply(lambda x: pd.Timestamp(x.get('assigned_at')).date() if pd.notnull(x) else '')
		t.Mentor = t.Mentor.apply(lambda x: pd.Timestamp(x.get('assigned_at')).date() if pd.notnull(x) else '')

		t.to_csv('tmp2.csv')

		# Switch to the primary sheet:
		spreadsheet = self.gc.open_by_key('1IpBO7Kuv75Ij6dNtUQ33t9PUuN7_E4DwHNOrfRY2r1Y')
		ym_sheet = spreadsheet.worksheet("YouthMappers")

		cell_list = ym_sheet.range(f"A2:{self.__get_column_from_int(len(OUTPUT_COLUMNS))}{len(t)+1}")

		OUTPUT_COLUMNS_MAPPING = dict(enumerate(OUTPUT_COLUMNS))
		for c in cell_list:
			df_row, df_col = c.row-2, c.col-1
			val = t.iloc[df_row, df_col]    
			c.value = self.__format_cell_value(val, OUTPUT_COLUMNS_MAPPING.get(df_col))

		ym_sheet.update_cells(cell_list, value_input_option='USER_ENTERED')


	def update_latest_chapter_roster(self):
		CHAPTER_COLUMNS = ['team_id','name','hashtag','bio','privacy','members','City','Country',
                   'Website or Social Media Accounts','Year Established','E-mail','join_link',
                   'chapter_lon','chapter_lat','location']
		
		t = self.chapters.reset_index().rename(columns={'index':'team_id'})[CHAPTER_COLUMNS]

		t.join_link = t.apply(lambda row: f"https://mapping.team/teams/{row.team_id}/invitations/{row.join_link}", axis=1)

		spreadsheet = self.gc.open_by_key('1IpBO7Kuv75Ij6dNtUQ33t9PUuN7_E4DwHNOrfRY2r1Y')
		chapters_sheet = spreadsheet.worksheet("Chapters")
		
		CHAPTER_OUTPUT_COLUMNS_MAPPING = dict(enumerate(CHAPTER_COLUMNS))
		cell_list = chapters_sheet.range(f"A2:{self.__get_column_from_int(len(CHAPTER_COLUMNS))}{len(t)+1}")
		for c in cell_list:
			df_row, df_col = c.row-2, c.col-1
			val = t.iloc[df_row, df_col]    
			c.value = self.__format_cell_value(val, CHAPTER_OUTPUT_COLUMNS_MAPPING.get(df_col))
		chapters_sheet.update_cells(cell_list, value_input_option='USER_ENTERED')



	def __conflate(self, row, priority_field, secondary_field, tertiary_field = None):
		if pd.notnull(row[priority_field]):
			return row[priority_field]
		elif pd.notnull(row[secondary_field]):
			return row[secondary_field]
		elif tertiary_field and pd.notnull(row[tertiary_field]):
			return row[tertiary_field]
		else:
			return None


	def __format_cell_value(self, val, column):
		match column:
			case 'uid':
				return int(val)
			case 'username':
				return f'=HYPERLINK("https://osm.org/user/"&"{val}","{val}")'
			case 'team_id':
				return f'=HYPERLINK("https://mapping.team/teams/"&{int(val)},{int(val)})'
			case 'Year Born':
				return int(val) if pd.notnull(val) else ''
			case 'chapter_lon':
				return float(val) if pd.notnull(val) else ''
			case 'chapter_lat':
				return float(val) if pd.notnull(val) else ''
			case 'all_teams':
				if type(val)==list:
					return "; ".join([str(v) for v in val])
				return ''
			case 'account_created':
				return "'"+val.strftime("%Y-%m-%d") if pd.notnull(val) else ''
			case _:
				return str(val) if pd.notnull(val) else ''
	
	
	def __get_column_from_int(self, number: int):
		alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		idx = number % 26
    
		additional = ''
		if number > 26:
			additional = alphabet[int(number / 26)-1]
			
		return additional + alphabet[idx-1]



if __name__ == "__main__":
	main()